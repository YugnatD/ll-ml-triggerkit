import os
import math
import numpy as np
import tensorflow as tf
import csv
from keras.saving import register_keras_serializable
from ctapipe.instrument import CameraGeometry
import astropy.units as u
import hashlib

from triggerkit.Helper.Quantize import (
    fixed_point_clip_int,
    fixed_point_from_int,
    fixed_point_quantize,
    fixed_point_rescale_int,
    fixed_point_to_int,
    parse_overflow_mode,
    parse_quantization_mode,
    parse_qspec,
    ste_fixed_point,
)

@register_keras_serializable(package="Trigger")
class PerFilterMeanRegularizer(tf.keras.regularizers.Regularizer):
    """Per-filter mean+std penalty over a TDSCAN weight tensor.

        lam      * sum_f (mean_f - target)**2          # center each filter's mean
      + lam_std  * sum_f (std_f  - target_std)**2       # set each filter's spread

    ``mean_f`` / ``std_f`` are the mean and standard deviation of filter f's
    weights, reduced over every axis EXCEPT the filter axis (the last one). Keras
    adds this to the training loss each step and backprops it, shaping each
    filter's weight distribution without touching the inference forward pass. Set
    ``lam=0`` and/or ``lam_std=0`` to disable either term.

    Per-filter (not a single global moment) on purpose: the F filters are
    independent random restarts, so a global mean would let a +mean filter and a
    -mean filter cancel and both stay lopsided. Reducing over all-but-the-last
    axis keeps each filter's penalty -- and its gradient -- independent, exactly
    like the per-column pairwise-AUC loss.

    Why control std too: the post-training FPGA scale (remap_weights) maps both
    mean and std by the same factor s = R/max|w|, so it preserves mean/std but
    amplifies any residual mean error when std is small. Targeting a std keeps s
    bounded, so the centered mean survives scaling. (target mean 0 is the only
    mean a pure scale leaves put, since s*0 = 0.)
    """

    # sqrt(var + eps) keeps the std gradient finite as a filter's spread -> 0.
    _STD_EPS = 1e-12

    def __init__(self, lam=0.0, target=0.0, lam_std=0.0, target_std=0.0):
        self.lam = float(lam)
        self.target = float(target)
        self.lam_std = float(lam_std)
        self.target_std = float(target_std)

    def __call__(self, weights):
        rank = len(weights.shape)
        reduce_axes = list(range(rank - 1))  # everything but the filter axis
        penalty = tf.constant(0.0, dtype=weights.dtype)

        per_filter_mean = tf.reduce_mean(weights, axis=reduce_axes)  # (F,)
        if self.lam > 0.0:
            penalty += self.lam * tf.reduce_sum(tf.square(per_filter_mean - self.target))

        if self.lam_std > 0.0:
            # std_f = sqrt(mean_f((w - mean_f)^2)); eps avoids a div-by-zero grad.
            centered = weights - tf.reshape(per_filter_mean, [1] * (rank - 1) + [-1])
            per_filter_var = tf.reduce_mean(tf.square(centered), axis=reduce_axes)  # (F,)
            per_filter_std = tf.sqrt(per_filter_var + self._STD_EPS)
            penalty += self.lam_std * tf.reduce_sum(tf.square(per_filter_std - self.target_std))

        return penalty

    def get_config(self):
        return {
            "lam": self.lam,
            "target": self.target,
            "lam_std": self.lam_std,
            "target_std": self.target_std,
        }


@register_keras_serializable(package="Trigger")
class TDSCAN(tf.keras.layers.Layer):
    """
    Spatio-temporal conv on fixed hex neighborhood + temporal window.
    Input:  (B, N, T) or (B, N, T, Cin)
    Output: (B, N, T, filters)

    If share_neighbors=True:
      - weights are shared by hexagonal ring (one weight per ring distance)
      - the center slot is inferred from the generated kernel mask (value 0)
    """
    _QUANTIZE_STEP_ORDER = (
        "input",
        "ring_weights",
        "convolution_accumulator",
        "convolution_rescale_shift",
        "temporal_accumulator",
        "temporal_rescale_shift",
    )
    _QUANTIZE_STEP_ALIASES = {
        "input": "input",
        "inputs": "input",
        "waveform": "input",
        "input_waveform": "input",
        "ring_weights": "ring_weights",
        "weights": "ring_weights",
        "kernel_weights": "ring_weights",
        "convolution_accumulator": "convolution_accumulator",
        "spatial_accumulator": "convolution_accumulator",
        "spatial": "convolution_accumulator",
        "convolution_rescale_shift": "convolution_rescale_shift",
        "spatial_rescale_shift": "convolution_rescale_shift",
        "conv_rescale_shift": "convolution_rescale_shift",
        "conv_shift": "convolution_rescale_shift",
        "temporal_accumulator": "temporal_accumulator",
        "temporal": "temporal_accumulator",
        "temporal_rescale_shift": "temporal_rescale_shift",
        "temp_rescale_shift": "temporal_rescale_shift",
        "temp_shift": "temporal_rescale_shift",
    }
    _QUANTIZE_STEP_QSPEC_KEYS = {
        "input",
        "ring_weights",
        "convolution_accumulator",
        "temporal_accumulator",
    }
    _QUANTIZE_STEP_SHIFT_KEYS = {
        "convolution_rescale_shift",
        "temporal_rescale_shift",
    }

    def __init__(
        self,
        neighbors,
        eps_t,
        eps_xy,
        filters,
        initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.5),
        share_neighbors=True,
        input_geometry: CameraGeometry = None,
        quantize_step=None,
        # overflow_mode="AP_WRAP",
        overflow_mode="AP_SAT",
        quantization_mode="AP_TRN",
        rescale_shift=0,
        pad_value=0.0,
        fake_quant_accumulators=False,
        mean_penalty_lambda=0.0,
        mean_penalty_target=0.0,
        std_penalty_lambda=0.0,
        std_penalty_target=0.0,
        **kwargs):
        super().__init__(**kwargs)

        # ---- SAFETY: make weight attributes exist even before build() ----
        self.kernel = None
        self.kernel_rings = None

        # Soft training penalties shaping each filter's weight distribution (see
        # PerFilterMeanRegularizer): pull the per-filter mean toward
        # mean_penalty_target and the per-filter std toward std_penalty_target.
        # Each lambda=0 disables its term -> a no-op, so runs that don't set them
        # are byte-for-byte unchanged. Only active during training (it's a weight
        # regularizer added to the loss).
        self.mean_penalty_lambda = float(mean_penalty_lambda)
        self.mean_penalty_target = float(mean_penalty_target)
        self.std_penalty_lambda = float(std_penalty_lambda)
        self.std_penalty_target = float(std_penalty_target)

        self.eps_t = int(eps_t)
        self.eps_xy = int(eps_xy)
        self.filters = int(filters)
        self.L = 2 * self.eps_t + 1
        self.quantize_step_request = self._normalize_quantize_step(quantize_step)
        self.overflow_mode = parse_overflow_mode(overflow_mode)
        self.fixed_point_quantization_mode = parse_quantization_mode(quantization_mode)
        self.default_rescale_shift = int(rescale_shift)
        if self.default_rescale_shift < 0:
            raise ValueError(f"rescale_shift must be >= 0, got {rescale_shift}.")
        self.pad_value = float(pad_value)
        # When True, the training (STE) forward pass also fake-quantizes the conv
        # and temporal accumulators (round/saturate/rescale) so it tracks the
        # integer inference path even when those registers are lossy (narrowed or
        # rescale-shifted). When False (default) only the weights are STE-quantized,
        # which is already bit-exact with inference whenever the accumulators are
        # the auto-derived lossless widths.
        self.fake_quant_accumulators = bool(fake_quant_accumulators)
        self.initializer = initializer

        # ---- SAFETY: input_geometry is effectively required (get_config uses it) ----
        if input_geometry is None:
            raise ValueError("TDSCAN requires a non-None input_geometry (needed for get_config/from_config).")
        self.input_geometry = input_geometry

        self.output_geometry = self.generate_output_geometry()
        self.list_kernel_mask = self.generate_list_kernel_eps_xy()

        # store neighbors in python form for serialization
        self._neighbors_list = neighbors if isinstance(neighbors, list) else tf.convert_to_tensor(neighbors).numpy().tolist()

        neigh = tf.convert_to_tensor(self._neighbors_list, dtype=tf.int32)  # (N, K)
        self.neigh_idx = neigh
        self.N = int(neigh.shape[0])  # number of pixels
        self.K = int(neigh.shape[1])  # number of spatial neighbors

        self.share_neighbors = bool(share_neighbors)

        if len(self.list_kernel_mask) != self.K:
            raise ValueError(
                f"Kernel mask length {len(self.list_kernel_mask)} does not match neighbor list size {self.K}."
            )

        # validate that there is exactly one center (ring 0)
        self._infer_center_idx()
        self.num_rings = max(self.list_kernel_mask) + 1

        # constant mask tensor reused in the forward pass
        self._mask_tensor = tf.constant(self.list_kernel_mask, dtype=tf.int32)
        self.quantize_step = self._resolve_quantize_step()

    @staticmethod
    def _canonicalize_qspec(qspec):
        if qspec is None:
            return None
        return parse_qspec(qspec)["canonical_qspec"]

    @staticmethod
    def _canonicalize_rescale_shift(shift):
        if shift is None:
            return None
        if isinstance(shift, str):
            raw = shift.strip()
            if not raw or raw.lower() == "none":
                return None
            shift = raw
        try:
            value = int(shift)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid rescale shift {shift!r}; expected an integer or None.")
        if value < 0:
            raise ValueError(f"rescale shift must be >= 0, got {shift!r}.")
        return value

    @classmethod
    def _normalize_quantize_step(cls, quantize_step):
        if quantize_step is None:
            return None
        if not isinstance(quantize_step, dict):
            raise ValueError(
                "quantize_step must be a dict mapping stage names to qspec strings "
                "and optional rescale shift integers."
            )
        if len(quantize_step) == 0:
            return None

        normalized = {key: None for key in cls._QUANTIZE_STEP_ORDER}
        global_rescale_shift = None
        for raw_key, raw_value in quantize_step.items():
            if raw_key == "rescale_shift":
                global_rescale_shift = cls._canonicalize_rescale_shift(raw_value)
                continue
            canonical_key = cls._QUANTIZE_STEP_ALIASES.get(raw_key)
            if canonical_key is None:
                allowed = ", ".join(cls._QUANTIZE_STEP_ORDER + ("rescale_shift",))
                raise ValueError(f"Unsupported quantize_step key '{raw_key}'. Allowed keys: {allowed}.")
            if canonical_key in cls._QUANTIZE_STEP_QSPEC_KEYS:
                normalized[canonical_key] = cls._canonicalize_qspec(raw_value)
            else:
                normalized[canonical_key] = cls._canonicalize_rescale_shift(raw_value)

        if global_rescale_shift is not None:
            if normalized["convolution_rescale_shift"] is None:
                normalized["convolution_rescale_shift"] = global_rescale_shift
            if normalized["temporal_rescale_shift"] is None:
                normalized["temporal_rescale_shift"] = global_rescale_shift
        return normalized

    @staticmethod
    def _growth_bits(n_terms):
        n_terms = int(n_terms)
        if n_terms <= 1:
            return 0
        return int(math.ceil(math.log2(n_terms)))

    @staticmethod
    def _format_qspec(int_bits, frac_bits, signed=True):
        prefix = "SQ" if signed else "UQ"
        return f"{prefix}{int(int_bits)}.{int(frac_bits)}"

    def _derive_convolution_qspec(self, input_qspec, weight_qspec):
        input_spec = parse_qspec(input_qspec)
        weight_spec = parse_qspec(weight_qspec)
        int_bits = input_spec["int_bits"] + weight_spec["int_bits"] + self._growth_bits(self.K)
        frac_bits = input_spec["frac_bits"] + weight_spec["frac_bits"]
        signed = input_spec["signed"] or weight_spec["signed"]
        return self._format_qspec(int_bits=int_bits, frac_bits=frac_bits, signed=signed)

    def _derive_temporal_qspec(self, convolution_qspec):
        conv_spec = parse_qspec(convolution_qspec)
        int_bits = conv_spec["int_bits"] + self._growth_bits(self.L)
        return self._format_qspec(
            int_bits=int_bits,
            frac_bits=conv_spec["frac_bits"],
            signed=conv_spec["signed"],
        )

    def _resolve_quantize_step(self):
        requested = self.quantize_step_request or {}
        input_qspec = requested.get("input")
        if input_qspec is None:
            if any(spec is not None for spec in requested.values()):
                raise ValueError(
                    "quantize_step['input'] must be set when quantization is enabled."
                )
            return None

        weight_qspec = requested.get("ring_weights")
        if weight_qspec is None:
            weight_qspec = input_qspec

        convolution_qspec = requested.get("convolution_accumulator")
        if convolution_qspec is None:
            convolution_qspec = self._derive_convolution_qspec(input_qspec, weight_qspec)

        temporal_qspec = requested.get("temporal_accumulator")
        if temporal_qspec is None:
            temporal_qspec = self._derive_temporal_qspec(convolution_qspec)

        convolution_rescale_shift = requested.get("convolution_rescale_shift")
        if convolution_rescale_shift is None:
            convolution_rescale_shift = self.default_rescale_shift

        temporal_rescale_shift = requested.get("temporal_rescale_shift")
        if temporal_rescale_shift is None:
            temporal_rescale_shift = self.default_rescale_shift

        print(
            f"Quantization enabled with steps: input={input_qspec}, "
            f"ring_weights={weight_qspec}, "
            f"convolution_accumulator={convolution_qspec}, "
            f"temporal_accumulator={temporal_qspec}, "
            f"convolution_rescale_shift={convolution_rescale_shift}, "
            f"temporal_rescale_shift={temporal_rescale_shift}, "
            f"overflow_mode={self.overflow_mode}, "
            f"quantization_mode={self.fixed_point_quantization_mode}"
        )
        return {
            "input": input_qspec,
            "ring_weights": weight_qspec,
            "convolution_accumulator": convolution_qspec,
            "convolution_rescale_shift": convolution_rescale_shift,
            "temporal_accumulator": temporal_qspec,
            "temporal_rescale_shift": temporal_rescale_shift,
        }

    def _quantization_enabled(self):
        return self.quantize_step is not None

    def _weight_quantize_qspec(self):
        if self.quantize_step is None:
            return None
        return self.quantize_step["ring_weights"]

    @staticmethod
    def _requantize_int(
        values,
        src_qspec,
        dst_qspec,
        dtype=tf.int64,
        overflow_mode="AP_WRAP",
        quantization_mode="AP_TRN",
        rescale_shift=0,
    ):
        if dst_qspec is None:
            return tf.cast(values, dtype)
        if src_qspec == dst_qspec:
            return fixed_point_clip_int(
                values,
                qspec=dst_qspec,
                dtype=dtype,
                overflow_mode=overflow_mode,
            )

        return fixed_point_rescale_int(
            values,
            src_qspec=src_qspec,
            dst_qspec=dst_qspec,
            dtype=dtype,
            shift=rescale_shift,
            overflow_mode=overflow_mode,
        )
    
    def generate_output_geometry(self):
        if self.input_geometry is None:
            return None
        if self.input_geometry.name == "DigiCam" or self.input_geometry.name == "DigiCam_R0Alpha" or self.input_geometry.name == "UNKNOWN-7987PX":
            return self.input_geometry
        raise ValueError(f"TDSCAN: camera '{self.input_geometry.name}' not recognized")

    # generate a flat list that labels each neighbor slot by its ring index
    # in the hex kernel (0 = center, 1 = first ring, 2 = second ring, ...).
    # The order follows axial coordinates scanned row by row so that the
    # center ends up in the middle position.
    # ex for eps_xy=1: [1, 1,
    #                  1, 0, 1,
    #                   1, 1]
    # ex for eps_xy=2: [2, 2, 2,
    #                  2, 1, 1, 2,
    #                2, 1, 0, 1, 2,
    #                 2, 1, 1, 2,
    #                   2, 2, 2]
    def generate_list_kernel_eps_xy(self):
        mask = []
        r = self.eps_xy
        for y in range(-r, r + 1):
            qmin = max(-r, -y - r)
            qmax = min(r, -y + r)
            for x in range(qmin, qmax + 1):
                dist = (abs(x) + abs(y) + abs(x + y)) // 2  # hex axial distance
                mask.append(dist)
        return mask

    def _infer_center_idx(self):
        if 0 not in self.list_kernel_mask:
            raise ValueError("Kernel mask does not contain a center (value 0).")
        if self.list_kernel_mask.count(0) != 1:
            raise ValueError("Kernel mask should contain exactly one center (value 0).")
        return self.list_kernel_mask.index(0)

    def stage_name(self):
        # The filename fingerprint (structure + quantization + weights) keeps two
        # runs that differ only in those from overwriting each other. It is NOT
        # the 'id' in get_params: that one is weights-only (see _weights_id).
        return f"tdscan{self.eps_xy}{self.eps_t}h{self._filename_fingerprint()}"

    def stage_type(self):
        return "tdscan"

    def _weights_id(self):
        """md5 of the (quantized) weights only.

        This is the 'id' exposed in get_params(): a short, stable handle that
        means *which weight set*. Structure and quantization are already stored
        as explicit fields, so the id only needs to identify the weights.
        """
        data = b""
        kr = getattr(self, "kernel_rings", None)
        k = getattr(self, "kernel", None)
        if kr is not None:
            data += kr.numpy().tobytes()
        if k is not None:
            data += k.numpy().tobytes()
        return hashlib.md5(data).hexdigest()

    def _filename_fingerprint(self):
        """md5 over structure + quantization + weights.

        Used only to keep output filenames unique when two runs share the same
        structure/quantization/threshold but differ in their weights (the
        weights-only id would collide for same-weights/different-quant runs).
        """
        hash_parts = [
            f"eps_xy={self.eps_xy}",
            f"eps_t={self.eps_t}",
            f"filters={self.filters}",
            f"share_neighbors={self.share_neighbors}",
            f"N={getattr(self, 'N', None)}",
            f"K={getattr(self, 'K', None)}",
        ]
        if self.pad_value != 0.0:
            hash_parts.append(f"pad_value={self.pad_value}")
        if self.quantize_step_request is not None:
            hash_parts.append(f"quantize_step={self.quantize_step}")
            hash_parts.append(f"overflow_mode={self.overflow_mode}")
            hash_parts.append(f"quantization_mode={self.fixed_point_quantization_mode}")
        hash_data = [("|".join(hash_parts)).encode("utf-8")]

        kr = getattr(self, "kernel_rings", None)
        k = getattr(self, "kernel", None)
        if kr is not None:
            hash_data.append(kr.numpy().tobytes())
        if k is not None:
            hash_data.append(k.numpy().tobytes())

        return hashlib.md5(b"".join(hash_data)).hexdigest()

    def get_params(self):
        kr = getattr(self, "kernel_rings", None)
        k = getattr(self, "kernel", None)

        params = {
            "eps_xy": self.eps_xy,
            "eps_t": self.eps_t,
            "filters": self.filters,
            "share_neighbors": self.share_neighbors,
            "id": self._weights_id(),
        }
        # Persist the actual learned weights so a saved run is fully described by
        # its real fields. The tensors are tiny, so we store them inline (nested
        # lists preserve the (L, R, ...) shape).
        if self.share_neighbors:
            if kr is not None:
                params["ring_weights"] = kr.numpy().tolist()
        else:
            if k is not None:
                params["kernel_weights"] = k.numpy().tolist()
        if self.pad_value != 0.0:
            params["pad_value"] = self.pad_value
        if self.quantize_step_request is not None:
            params["quantize_step"] = self.quantize_step
            params["overflow_mode"] = self.overflow_mode
            params["quantization_mode"] = self.fixed_point_quantization_mode
            if self.fake_quant_accumulators:
                params["fake_quant_accumulators"] = self.fake_quant_accumulators
        return params


    def get_stages(self):
        return (self.stage_type(), self.get_params())
    
    def _quantize_weights_like_inference(self, weights):
        """Snap stored weights onto the ring_weights fixed-point grid.

        Returns the input unchanged when quantization is disabled. Otherwise
        applies the same fixed_point_quantize (qspec/overflow/quantization mode)
        that the STE forward pass and call_quantized use, so the result matches
        the values the deployed integer path actually sees.
        """
        weight_qspec = self._weight_quantize_qspec()
        if weight_qspec is None:
            return weights
        return fixed_point_quantize(
            weights,
            qspec=weight_qspec,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )

    def get_flat_quantized_weights(self):
        """Like get_flat_weights, but snapped onto the ring_weights grid.

        Matches the effective weights the quantized inference path uses; equals
        get_flat_weights when quantization is disabled.
        """
        w = self.kernel_rings if self.share_neighbors else self.kernel
        if w is None:
            return []
        return self._quantize_weights_like_inference(w).numpy().reshape(-1).tolist()

    # print the kernel weights in a readable hex layout; handy for debugging
    def print_hexagonal_kernel(self, t_idx=None, in_ch=0, out_ch=0, precision=3, quantized=False):
        """
        Pretty-print spatial weights on the hex grid.

        Args:
            t_idx: temporal index to print (0..L-1). If None, prints all.
            in_ch: input channel index (Cin dimension).
            out_ch: output channel index (Cout dimension).
            precision: decimals in the printed values.
            quantized: if True, snap the weights onto the ring_weights grid first
                (the effective values used by the quantized inference path).
        """
        if t_idx is None:
            t_list = list(range(self.L))
        else:
            t_idx = int(t_idx)
            if not (0 <= t_idx < self.L):
                raise ValueError(f"t_idx must be in [0,{self.L-1}] but got {t_idx}")
            t_list = [t_idx]

        if self.share_neighbors:
            kr = self.kernel_rings
            if quantized:
                kr = self._quantize_weights_like_inference(kr)
            w = kr.numpy()[..., in_ch, out_ch]  # (L, R, 1) squeezed below
            w = np.squeeze(w, axis=2)                         # (L, R)
        else:
            k = self.kernel
            if quantized:
                k = self._quantize_weights_like_inference(k)
            w = k.numpy()[..., in_ch, out_ch]       # (L, K)

        r = self.eps_xy
        for t in t_list:
            print(f"t={t - self.eps_t:+d}:")
            idx = 0
            for y in range(-r, r + 1):
                qmin = max(-r, -y - r)
                qmax = min(r, -y + r)
                row_vals = []
                for _ in range(qmax - qmin + 1):
                    if self.share_neighbors:
                        ring = self.list_kernel_mask[idx]
                        val = w[t, ring]
                    else:
                        val = w[t, idx]
                    row_vals.append(f"{val:.{precision}f}")
                    idx += 1
                print(" ".join(row_vals))
            print()
    
    def set_weights_from_params(self, share_weights, ring_weights=None, kernel_weights=None):
        """
        Set the layer weights from given parameters.
        Depending on share_neighbors, either provide ring_weights (shape L x R x 1 x Cin x Cout),
        or provide kernel_weights for the full kernel.
        """

        # make sure ring_weights is 32 bit
        if ring_weights is not None:
            ring_weights = ring_weights.astype(np.float32)
        # Quantize provided weights if quantization is enabled, to mimic hardware
        weight_qspec = self._weight_quantize_qspec()
        if weight_qspec is not None:
            if share_weights:
                if ring_weights is None:
                    raise ValueError("Must provide ring_weights when share_weights=True.")
                ring_weights = fixed_point_quantize(
                    ring_weights,
                    qspec=weight_qspec,
                    overflow_mode=self.overflow_mode,
                    quantization_mode=self.fixed_point_quantization_mode,
                )
            else:
                if kernel_weights is None:
                    raise ValueError("Must provide kernel_weights when share_weights=False.")
                kernel_weights = fixed_point_quantize(
                    kernel_weights,
                    qspec=weight_qspec,
                    overflow_mode=self.overflow_mode,
                    quantization_mode=self.fixed_point_quantization_mode,
                )
        if share_weights:
            if ring_weights is None:
                raise ValueError("Must provide ring_weights when share_weights=True.")
            target_shape = self.kernel_rings.shape
            ring_weights = tf.reshape(
                tf.convert_to_tensor(ring_weights, dtype=self.kernel_rings.dtype),
                target_shape,
            )
            self.kernel_rings.assign(ring_weights)
        else:
            if kernel_weights is None:
                raise ValueError("Must provide kernel_weights when share_weights=False.")
            # Reshape to the kernel's (L, K, Cin, Cout) shape, mirroring the shared
            # branch above, so a flat/loosely-shaped weight list is accepted too.
            target_shape = self.kernel.shape
            kernel_weights = tf.reshape(
                tf.convert_to_tensor(kernel_weights, dtype=self.kernel.dtype),
                target_shape,
            )
            self.kernel.assign(kernel_weights)

    def get_flat_weights(self):
        """Utility: returns a flat python list of scalar weights for logging/debugging."""
        w = self.kernel_rings if self.share_neighbors else self.kernel
        if w is None:
            return []
        return w.numpy().reshape(-1).tolist()

    def set_trainable(self, trainable: bool):
        """Set the trainable status of the layer and its weights.
        Allows freezing/unfreezing the layer during training.
        """
        self.trainable = trainable
        if getattr(self, "kernel_rings", None) is not None:
            self.kernel_rings._trainable = trainable
        if self.kernel is not None:
            self.kernel._trainable = trainable

    def _mean_penalty_regularizer(self):
        """PerFilterMeanRegularizer for this layer, or None when both terms off."""
        if self.mean_penalty_lambda <= 0.0 and self.std_penalty_lambda <= 0.0:
            return None
        return PerFilterMeanRegularizer(
            lam=self.mean_penalty_lambda, target=self.mean_penalty_target,
            lam_std=self.std_penalty_lambda, target_std=self.std_penalty_target,
        )

    def build(self, input_shape):
        cin = 1 if len(input_shape) == 3 else int(input_shape[-1])
        self.L = 2 * self.eps_t + 1  # number of temporal offsets

        mean_reg = self._mean_penalty_regularizer()

        if self.share_neighbors:
            # One kernel per hex ring (including center at ring 0)
            # Shape: (L, R, 1, Cin, Cout)
            self.kernel_rings = self.add_weight(
                name="kernel_rings",
                shape=(self.L, self.num_rings, 1, cin, self.filters),
                initializer=self.initializer,
                trainable=True,
                regularizer=mean_reg,
                # Keep constraint hook for future if we ever want to store quantized weights
            )
            self.kernel = None
        else:
            # Full kernel: (L, K, Cin, Cout)
            self.kernel = self.add_weight(
                name="kernel",
                shape=(self.L, self.K, cin, self.filters),
                initializer=self.initializer,
                trainable=True,
                regularizer=mean_reg,
            )
            self.kernel_rings = None

        super().build(input_shape)

    def _full_kernel(self):
        """
        Return a (L, K, Cin, Cout) kernel.
        If share_neighbors=True, maps each neighbor slot to its ring kernel
        according to self.list_kernel_mask.
        """
        if not self.share_neighbors:
            return self.kernel

        # Gather ring kernels along the ring axis using the mask (K indices)
        W = tf.gather(self.kernel_rings, self._mask_tensor, axis=1)  # (L, K, 1, Cin, Cout)
        return tf.squeeze(W, axis=2)  # (L, K, Cin, Cout)

    def _training_kernel(self):
        weight_qspec = self._weight_quantize_qspec()
        if weight_qspec is None:
            return self._full_kernel()
        if self.share_neighbors:
            kernel_rings_q = ste_fixed_point(
                self.kernel_rings,
                qspec=weight_qspec,
                overflow_mode=self.overflow_mode,
                quantization_mode=self.fixed_point_quantization_mode,
            )
            return tf.squeeze(tf.gather(kernel_rings_q, self._mask_tensor, axis=1), axis=2)
        return ste_fixed_point(
            self.kernel,
            qspec=weight_qspec,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )

    @staticmethod
    def _resolve_training_flag(training):
        if training is None:
            return False
        if tf.is_tensor(training):
            static_training = tf.get_static_value(tf.cast(training, tf.bool))
            if static_training is not None:
                return bool(static_training)
            return tf.cast(training, tf.bool)
        return bool(training)

    def _call_float_with_kernel(self, x, kernel):
        T = tf.shape(x)[2]

        # (B,N,T,C) -> (B,T,N,C)
        x_perm = tf.transpose(x, [0, 2, 1, 3])

        # -------- Temporal window --------
        dt = tf.range(-self.eps_t, self.eps_t + 1)          # (L,)
        t0 = tf.range(T)[:, tf.newaxis]                     # (T,1)
        tidx = t0 + dt[tf.newaxis, :]                       # (T,L)

        valid_t = tf.logical_and(tidx >= 0, tidx < T)        # (T,L)
        tidx_clip = tf.clip_by_value(tidx, 0, T - 1)         # (T,L)

        # gather time: (B,T,N,C) -> (B,T,L,N,C)
        x_t = tf.gather(x_perm, tidx_clip, axis=1)

        # Pad temporal borders.
        mask_t = valid_t[tf.newaxis, :, :, tf.newaxis, tf.newaxis]  # (1,T,L,1,1)
        pad_value = tf.cast(self.pad_value, x_t.dtype)
        x_t = tf.where(mask_t, x_t, pad_value)

        # -------- Spatial neighborhood --------
        idx = self.neigh_idx                                 # (N,K)
        idx_safe = tf.where(idx < 0, 0, idx)
        valid_n = idx >= 0                                   # (N,K)

        # gather neighbors: (B,T,L,N,C) -> (B,T,L,N,K,C)
        x_tn = tf.gather(x_t, idx_safe, axis=3)
        mask_n = valid_n[tf.newaxis, tf.newaxis, tf.newaxis, :, :, tf.newaxis]  # (1,1,1,N,K,1)
        x_tn = tf.where(mask_n, x_tn, pad_value)

        out = tf.einsum("btlnkc,lkco->btno", x_tn, kernel)
        # (B,T,N,O) -> (B,N,T,O)
        return tf.transpose(out, [0, 2, 1, 3])

    def _fake_quant_accumulator(self, values, src_qspec, dst_qspec, rescale_shift):
        """STE fake-quantization of a real-valued accumulator through the integer
        requantization path.

        The forward output equals the integer inference value (it reuses the same
        ``_requantize_int`` helper as ``call_quantized``), while the gradient passes
        straight through to ``values``. Because the inputs and weights feeding the
        accumulator already sit on their fixed-point grids, ``values`` is an exact
        multiple of ``1 / 2**src_frac``, so recovering the source integer is exact.
        The grid snap is computed in float64 to stay exact for wide registers.
        """
        if dst_qspec is None:
            return values
        src_spec = parse_qspec(src_qspec)
        dst_spec = parse_qspec(dst_qspec)
        src_scale = tf.cast(2 ** int(src_spec["frac_bits"]), tf.float64)
        dst_scale = tf.cast(2 ** int(dst_spec["frac_bits"]), tf.float64)

        src_int = tf.cast(tf.round(tf.cast(values, tf.float64) * src_scale), tf.int64)
        dst_int = self._requantize_int(
            src_int,
            src_qspec=src_spec["canonical_qspec"],
            dst_qspec=dst_spec["canonical_qspec"],
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
            rescale_shift=rescale_shift,
        )
        q = tf.cast(tf.cast(dst_int, tf.float64) / dst_scale, values.dtype)
        return values + tf.stop_gradient(q - values)

    def _call_fake_quant_with_kernel(self, x, kernel):
        """Float STE twin of ``call_quantized`` used for accumulator-aware training.

        Mirrors the integer inference path stage for stage -- input grid snap,
        per-time-slice spatial conv with a conv-accumulator requant, then the
        temporal sum with a temporal-accumulator requant -- but keeps the heavy
        einsum and temporal reduction in float32 so gradients flow (only the
        per-element accumulator requant touches int64, which is GPU-safe). When
        the accumulators are the auto-derived lossless widths the two requants
        reduce to a clip/saturate and this matches the weights-only path; when an
        accumulator is narrowed or rescale-shifted, training now tracks that loss.
        """
        input_qspec = self.quantize_step["input"]
        convolution_qspec = self.quantize_step["convolution_accumulator"]
        convolution_rescale_shift = self.quantize_step["convolution_rescale_shift"]
        temporal_qspec = self.quantize_step["temporal_accumulator"]
        temporal_rescale_shift = self.quantize_step["temporal_rescale_shift"]
        convolution_source_qspec = self._derive_convolution_qspec(
            input_qspec, self.quantize_step["ring_weights"]
        )
        temporal_source_qspec = self._derive_temporal_qspec(convolution_qspec)

        # Stage 1: snap the waveform onto the input grid (STE). A no-op when the
        # input already sits on the grid, e.g. integer score codes into UQ3.0.
        x = ste_fixed_point(
            x,
            qspec=input_qspec,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )
        pad_value = fixed_point_quantize(
            tf.cast(self.pad_value, x.dtype),
            qspec=input_qspec,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )

        T = tf.shape(x)[2]
        x_perm = tf.transpose(x, [0, 2, 1, 3])  # (B,T,N,C)

        # Temporal window with padded borders (same construction as call_quantized).
        dt = tf.range(-self.eps_t, self.eps_t + 1)
        t0 = tf.range(T)[:, tf.newaxis]
        tidx = t0 + dt[tf.newaxis, :]
        valid_t = tf.logical_and(tidx >= 0, tidx < T)
        tidx_clip = tf.clip_by_value(tidx, 0, T - 1)
        x_t = tf.gather(x_perm, tidx_clip, axis=1)  # (B,T,L,N,C)
        mask_t = valid_t[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
        x_t = tf.where(mask_t, x_t, pad_value)

        # Spatial neighborhood with padded invalid slots.
        idx = self.neigh_idx
        idx_safe = tf.where(idx < 0, 0, idx)
        valid_n = idx >= 0
        x_tn = tf.gather(x_t, idx_safe, axis=3)  # (B,T,L,N,K,C)
        mask_n = valid_n[tf.newaxis, tf.newaxis, tf.newaxis, :, :, tf.newaxis]
        x_tn = tf.where(mask_n, x_tn, pad_value)

        # Stage 2: per-time-slice spatial conv, then fake-quant the conv accumulator.
        spatial_acc = tf.einsum("btlnkc,lkco->btlno", x_tn, kernel)  # (B,T,L,N,O)
        spatial_acc = self._fake_quant_accumulator(
            spatial_acc,
            convolution_source_qspec,
            convolution_qspec,
            convolution_rescale_shift,
        )

        # Stage 3: temporal accumulation, then fake-quant the temporal accumulator.
        temporal_acc = tf.reduce_sum(spatial_acc, axis=2)  # (B,T,N,O)
        temporal_acc = self._fake_quant_accumulator(
            temporal_acc,
            temporal_source_qspec,
            temporal_qspec,
            temporal_rescale_shift,
        )

        # (B,T,N,O) -> (B,N,T,O)
        return tf.transpose(temporal_acc, [0, 2, 1, 3])

    def _training_forward(self, x):
        """Training-time forward pass: weight STE always, plus accumulator STE when
        ``fake_quant_accumulators`` is set."""
        kernel = self._training_kernel()
        if self.fake_quant_accumulators:
            return self._call_fake_quant_with_kernel(x, kernel)
        return self._call_float_with_kernel(x, kernel)

    def call(self, inputs, training=None):
        x = inputs
        if x.shape.rank == 3:
            x = x[..., tf.newaxis]  # (B,N,T,1)
        
        x = tf.cast(x, tf.float32)

        if not self._quantization_enabled():
            return self._call_float_with_kernel(x, self._full_kernel())

        training_flag = self._resolve_training_flag(training)
        if tf.is_tensor(training_flag):
            return tf.__internal__.smart_cond.smart_cond(
                training_flag,
                lambda: self._training_forward(x),
                lambda: self.call_quantized(inputs, training=False),
            )
        if training_flag:
            return self._training_forward(x)
        return self.call_quantized(inputs, training=False)

    def call_numpy(self, inputs):
        """
        NumPy reference implementation of the floating-point ``call`` path.
        This mirrors the TensorFlow implementation directly, including the
        temporal border masking and invalid-neighbor masking.
        """
        # Accept the same input layouts as ``call`` and normalize to (B,N,T,C).
        x = np.asarray(inputs)
        if x.ndim == 3:
            x = x[..., np.newaxis]  # (B,N,T,1)
        if x.ndim != 4:
            raise ValueError(f"Expected inputs with rank 3 or 4, got shape {x.shape}.")
        x = x.astype(np.float32, copy=False)

        _, _, num_frames, cin = x.shape
        w_full = self._full_kernel().numpy().astype(np.float32, copy=False)  # (L,K,C,O)
        _, _, cin_kernel, _ = w_full.shape
        if cin != cin_kernel:
            raise ValueError(f"Input channels ({cin}) do not match kernel channels ({cin_kernel}).")

        # Match the TensorFlow layout so time gathers happen on one axis.
        x_perm = np.transpose(x, (0, 2, 1, 3))  # (B,T,N,C)

        # Build the temporal window for every output frame and pad invalid
        # borders exactly like the masked gather in the TF code.
        dt = np.arange(-self.eps_t, self.eps_t + 1, dtype=np.int32)  # (L,)
        t0 = np.arange(num_frames, dtype=np.int32)[:, np.newaxis]  # (T,1)
        tidx = t0 + dt[np.newaxis, :]  # (T,L)
        valid_t = (tidx >= 0) & (tidx < num_frames)
        tidx_clip = np.clip(tidx, 0, num_frames - 1)

        x_t = np.take(x_perm, tidx_clip, axis=1)  # (B,T,L,N,C)
        mask_t = valid_t[np.newaxis, :, :, np.newaxis, np.newaxis]
        x_t = np.where(mask_t, x_t, np.float32(self.pad_value))

        # Expand each pixel to its neighborhood and remove invalid slots.
        neigh_idx = self.neigh_idx.numpy()
        idx_safe = np.where(neigh_idx < 0, 0, neigh_idx)
        valid_n = neigh_idx >= 0

        x_tn = np.take(x_t, idx_safe, axis=3)  # (B,T,L,N,K,C)
        mask_n = valid_n[np.newaxis, np.newaxis, np.newaxis, :, :, np.newaxis]
        x_tn = np.where(mask_n, x_tn, np.float32(self.pad_value))

        # Sum over temporal taps, neighbor slots, and input channels.
        out = np.einsum("btlnkc,lkco->btno", x_tn, w_full, optimize=True)

        # Match the public layer output layout: (B,T,N,O) -> (B,N,T,O).
        return np.transpose(out, (0, 2, 1, 3))
    
    # the fpga call quanitzed
    # 1) quantize from 8 bit to Q9.3 fixed point the all input pixels - stages pixel = Q9.3
    # 2) compute the conv2d on each frame ind for each frame in eps_t then quantize - stages pixels = (Q9.3 * Q9.3) * 7 = Q21.6
    # 3) sum the pixels following eps_t (if eps_t=1, sum 3 frames) - stages pixels = Q21.6 + log2(3) = Q23.6
    # 4) convert to float for thresholding - stages pixels = float (in fpga the thesholding is done in fixed point Q23.6 but here we convert to float for simplicity)
    def call_quantized(self, inputs, training=None):
        if tf.is_tensor(training):
            training = tf.get_static_value(tf.cast(training, tf.bool))
        if training:
            raise NotImplementedError("Quantized call training not implemented yet.")
        if not self._quantization_enabled():
            raise ValueError("call_quantized requires quantization to be enabled.")
        x = inputs
        if x.shape.rank == 3:
            x = x[..., tf.newaxis]  # (B,N,T,1)
        x = tf.cast(x, tf.float32)

        input_qspec = self.quantize_step["input"]
        weight_qspec = self.quantize_step["ring_weights"]
        convolution_qspec = self.quantize_step["convolution_accumulator"]
        convolution_rescale_shift = self.quantize_step["convolution_rescale_shift"]
        temporal_qspec = self.quantize_step["temporal_accumulator"]
        temporal_rescale_shift = self.quantize_step["temporal_rescale_shift"]
        temporal_source_qspec = self._derive_temporal_qspec(convolution_qspec)

        # Stage 1: quantize ADC values into the fixed-point representation used by the FPGA.
        x_int = fixed_point_to_int(
            x,
            qspec=input_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )  # (B,N,T,C)
        pad_int = fixed_point_to_int(
            tf.constant(self.pad_value, dtype=tf.float32),
            qspec=input_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )
        T = tf.shape(x_int)[2]

        # Reorder so temporal gathers happen on a contiguous axis.
        x_perm = tf.transpose(x_int, [0, 2, 1, 3])  # (B,T,N,C)

        # Build the temporal window around each sample and zero out invalid borders.
        dt = tf.range(-self.eps_t, self.eps_t + 1, dtype=tf.int32)  # (L,)
        t0 = tf.range(T, dtype=tf.int32)[:, tf.newaxis]  # (T,1)
        tidx = t0 + dt[tf.newaxis, :]  # (T,L)

        valid_t = tf.logical_and(tidx >= 0, tidx < T)  # (T,L)
        tidx_clip = tf.clip_by_value(tidx, 0, T - 1)  # (T,L)

        x_t = tf.gather(x_perm, tidx_clip, axis=1)  # (B,T,L,N,C)
        mask_t = valid_t[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
        x_t = tf.where(mask_t, x_t, pad_int)

        # Expand to the hex neighborhood and zero invalid edge slots.
        idx = self.neigh_idx  # (N,K)
        idx_safe = tf.where(idx < 0, 0, idx)
        valid_n = idx >= 0

        x_tn = tf.gather(x_t, idx_safe, axis=3)  # (B,T,L,N,K,C)
        mask_n = valid_n[tf.newaxis, tf.newaxis, tf.newaxis, :, :, tf.newaxis]
        x_tn = tf.where(mask_n, x_tn, pad_int)

        w_full_int = fixed_point_to_int(
            self._full_kernel(),
            qspec=weight_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )  # (L,K,C,O)

        # Stage 2: per-time-slice spatial convolution.
        spatial_acc = tf.einsum("btlnkc,lkco->btlno", x_tn, w_full_int)  # (B,T,L,N,O)
        spatial_acc = self._requantize_int(
            spatial_acc,
            src_qspec=self._derive_convolution_qspec(input_qspec, weight_qspec),
            dst_qspec=convolution_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
            rescale_shift=convolution_rescale_shift,
        )

        # Stage 3: temporal accumulation across the window.
        temporal_acc = tf.reduce_sum(spatial_acc, axis=2)  # (B,T,N,O)
        temporal_acc = self._requantize_int(
            temporal_acc,
            src_qspec=temporal_source_qspec,
            dst_qspec=temporal_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
            rescale_shift=temporal_rescale_shift,
        )

        # Stage 4: convert back to float so the threshold layer can operate in software.
        out = fixed_point_from_int(
            temporal_acc,
            qspec=temporal_qspec,
            dtype=tf.float32,
            overflow_mode=self.overflow_mode,
        )
        return tf.transpose(out, [0, 2, 1, 3])

    def call_quantized_numpy(self, inputs):
        """
        NumPy reference implementation of ``call_quantized``.
        This keeps the spatial and temporal accumulation as separate stages so
        the output can be compared directly against the TensorFlow path.
        """
        if not self._quantization_enabled():
            raise ValueError("call_quantized_numpy requires quantization to be enabled.")

        # Accept the same input layouts as ``call`` and normalize to (B,N,T,C).
        x = np.asarray(inputs)
        if x.ndim == 3:
            x = x[..., np.newaxis]  # (B,N,T,1)
        if x.ndim != 4:
            raise ValueError(f"Expected inputs with rank 3 or 4, got shape {x.shape}.")
        x = x.astype(np.float32, copy=False)

        input_qspec = self.quantize_step["input"]
        weight_qspec = self.quantize_step["ring_weights"]
        convolution_qspec = self.quantize_step["convolution_accumulator"]
        convolution_rescale_shift = self.quantize_step["convolution_rescale_shift"]
        temporal_qspec = self.quantize_step["temporal_accumulator"]
        temporal_rescale_shift = self.quantize_step["temporal_rescale_shift"]
        temporal_source_qspec = self._derive_temporal_qspec(convolution_qspec)

        # Stage 1: quantize both the waveform and the kernel into integer
        # fixed-point values so every later operation happens in integer space.
        x_int = fixed_point_to_int(
            x,
            qspec=input_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        ).numpy()  # (B,N,T,C)
        pad_int = int(fixed_point_to_int(
            tf.constant(self.pad_value, dtype=tf.float32),
            qspec=input_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        ).numpy())
        w_full_int = fixed_point_to_int(
            self._full_kernel(),
            qspec=weight_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        ).numpy()  # (L,K,C,O)
        neigh_idx = self.neigh_idx.numpy()

        batch_size, _, num_frames, cin = x_int.shape
        _, _, cin_kernel, cout = w_full_int.shape
        if cin != cin_kernel:
            raise ValueError(f"Input channels ({cin}) do not match kernel channels ({cin_kernel}).")

        # Stage 2 storage: keep one spatially accumulated value per temporal tap
        # ``l`` so we can normalize it after the spatial stage before doing the time sum.
        spatial_acc = np.zeros((batch_size, num_frames, self.L, self.N, cout), dtype=np.int64)

        for l, dt in enumerate(range(-self.eps_t, self.eps_t + 1)):
            # ``dt`` is the temporal offset of the current kernel slice.
            # Build matching source/destination frame ranges. Frames outside
            # [0, T-1] are filled with pad_value exactly like the TensorFlow path.
            if dt >= 0:
                dst_slice = slice(0, num_frames - dt)
                src_slice = slice(dt, num_frames)
            else:
                shift = -dt
                dst_slice = slice(shift, num_frames)
                src_slice = slice(0, num_frames - shift)
            has_valid_temporal_samples = dst_slice.stop > dst_slice.start

            for n in range(self.N): # loop over target pixels; for each one, gather its neighbors and apply the kernel
                for k in range(self.K): # loop over neighbor slots; in the FPGA, these happen in parallel and then get summed into the spatial accumulator for this pixel
                    # Invalid neighborhood slots use pad_value.
                    neighbor = neigh_idx[n, k]

                    # These are the weights of one temporal tap ``l`` and one
                    # spatial slot ``k`` across all input/output channels.
                    weights_ko = w_full_int[l, k]  # (Cin, Cout)
                    if not np.any(weights_ko):
                        continue

                    # Start from pad_value everywhere, then overwrite valid
                    # temporal samples for valid spatial neighbors.
                    samples = np.full((batch_size, num_frames, cin), pad_int, dtype=np.int64)
                    if neighbor >= 0 and has_valid_temporal_samples:
                        samples[:, dst_slice, :] = x_int[:, neighbor, src_slice, :]
                    contrib = np.einsum("btc,co->bto", samples, weights_ko, optimize=True)

                    # Accumulate into the spatial register for target pixel ``n``
                    # and temporal tap ``l``.
                    spatial_acc[:, :, l, n, :] += contrib

        spatial_acc = self._requantize_int(
            spatial_acc,
            src_qspec=self._derive_convolution_qspec(input_qspec, weight_qspec),
            dst_qspec=convolution_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
            rescale_shift=convolution_rescale_shift,
        ).numpy()

        # Stage 3: sum the already-normalized spatial values over the temporal taps
        # and normalize again to the wider temporal accumulator register.
        temporal_acc = np.sum(spatial_acc, axis=2, dtype=np.int64)  # (B,T,N,O)
        temporal_acc = self._requantize_int(
            temporal_acc,
            src_qspec=temporal_source_qspec,
            dst_qspec=temporal_qspec,
            dtype=tf.int64,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
            rescale_shift=temporal_rescale_shift,
        ).numpy()

        # Stage 4: convert the final fixed-point integer accumulator back to
        # float so the result can be compared directly with the TF output.
        out = fixed_point_from_int(
            temporal_acc,
            qspec=temporal_qspec,
            dtype=tf.float32,
            overflow_mode=self.overflow_mode,
        ).numpy()

        # Match the public layer output layout: (B,T,N,O) -> (B,N,T,O).
        return np.transpose(out, (0, 2, 1, 3))



    def _make_decreasing_ring_initializer(self):
        """
        Create initializer where ring 0 > ring 1 > ring 2 > ... across all
        temporal slices, input channels and filters.
        Uses a small truncated-normal jitter around a deterministic 1/(r+1)
        ramp so the order is guaranteed while still giving random variation.
        """
        jitter_std = 0.05  # 5% relative noise keeps ordering across rings

        def init(shape, dtype=None):
            dtype = dtype or tf.keras.backend.floatx()
            _, R, _, _, _ = shape
            base = 1.0 / (tf.range(R, dtype=dtype) + 1.0)  # ring0=1.0, ring1=0.5, ...
            scale = tf.reshape(base, (1, R, 1, 1, 1))
            noise = tf.random.truncated_normal(shape, mean=0.0, stddev=jitter_std, dtype=dtype)
            w = scale * (1.0 + noise)
            return tf.maximum(w, tf.constant(1e-4, dtype=dtype))  # keep positive

        return init

    def get_config(self):
        config = super().get_config()
        config.update({
            "neighbors": self._neighbors_list,
            "eps_t": self.eps_t,
            "eps_xy": self.eps_xy,
            "filters": self.filters,
            "share_neighbors": self.share_neighbors,
            "quantize_step": self.quantize_step_request,
            "overflow_mode": self.overflow_mode,
            "quantization_mode": self.fixed_point_quantization_mode,
            "rescale_shift": self.default_rescale_shift,
            "pad_value": self.pad_value,
            "fake_quant_accumulators": self.fake_quant_accumulators,
            "mean_penalty_lambda": self.mean_penalty_lambda,
            "mean_penalty_target": self.mean_penalty_target,
            "std_penalty_lambda": self.std_penalty_lambda,
            "std_penalty_target": self.std_penalty_target,
        })
        # config.update({"input_geometry": self.input_geometry.to_dict()}) # not serializable
        pix_x = self.input_geometry.pix_x.to_value(u.m).tolist()
        pix_y = self.input_geometry.pix_y.to_value(u.m).tolist()
        pix_area = self.input_geometry.pix_area.to_value(u.m**2).tolist()
        pix_id = self.input_geometry.pix_id.tolist()
        camera_name = self.input_geometry.name
        pix_type = self.input_geometry.pix_type.value
        config.update({"input_geometry": {
            "name": camera_name,
            "pix_id": pix_id,
            "pix_x": pix_x,
            "pix_y": pix_y,
            "pix_area": pix_area,
            "pix_type": pix_type
        }})
        return config

    @classmethod
    def from_config(cls, config):
        config.pop("center_idx", None)
        config.pop("quantize", None)

        input_geometry_dict = config.pop("input_geometry")

        # Convert lists back to numpy arrays before applying astropy units
        pix_x = u.Quantity(np.asarray(input_geometry_dict["pix_x"], dtype=np.float32), u.m)
        pix_y = u.Quantity(np.asarray(input_geometry_dict["pix_y"], dtype=np.float32), u.m)
        pix_area = u.Quantity(np.asarray(input_geometry_dict["pix_area"], dtype=np.float32), u.m**2)

        pix_id = np.asarray(input_geometry_dict["pix_id"], dtype=np.int32)

        # pix_type: try to reconstruct PixelShape if available; else pass through
        pix_type_raw = input_geometry_dict.get("pix_type", None)
        pix_type = pix_type_raw
        try:
            # ctapipe versions differ slightly in where PixelShape lives
            from ctapipe.instrument.camera import PixelShape
            pix_type = PixelShape(pix_type_raw)
        except Exception:
            # If PixelShape import/conversion fails, keep whatever was stored
            pix_type = pix_type_raw

        input_geometry = CameraGeometry(
            name=input_geometry_dict["name"],
            pix_id=pix_id,
            pix_x=pix_x,
            pix_y=pix_y,
            pix_area=pix_area,
            pix_type=pix_type,
        )

        return cls(input_geometry=input_geometry, **config)

    

def getTDSCANNeighbors(eps_xy, camera_name, n_pixels):
    if eps_xy == 0:
        # raise ValueError("eps_xy must be >= 1 for TDSCAN neighbors.")
        return [[i] for i in range(n_pixels)]  # each pixel is its own neighbor
    # Load from CSV
    if camera_name == "DigiCam" or camera_name == "DigiCam_R0Alpha":
        csv_file = 'ConfigFile_SST1M/CTA_SST1M_Pixels_info_epsilon_' + str(eps_xy) + '.csv'
    elif camera_name == "UNKNOWN-7987PX":
        csv_file = 'ConfigFile/CTA_LST_Pixels_info_epsilon_' + str(eps_xy) + '.csv'
    else:
        raise ValueError(f"Unsupported camera geometry: {camera_name}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"TDSCAN neighbor file not found for eps_xy={eps_xy}: {csv_file}")
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # skip the header
        next(reader)
        neighbors = [list(map(int, row)) for row in reader]
    return neighbors
