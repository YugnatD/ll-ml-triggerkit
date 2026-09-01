"""``TDSCANBody``: the fixed, deployed TDSCAN trigger chain as a pluggable body.

This reproduces ``tdscan_chain._add_branch`` / ``build_chain`` from the research
sandbox exactly, including every stage the sandbox could build:

    [fadc?] -> [subtract?] -> [score_quantizer?] -> tdscan -> [digital_sum?]
            -> camera-wide max-pool -> trainable threshold

A multi-branch OR trigger is built when ``branches`` is given. The ``fadc``
front-end is shared by all branches (built once before the branch point), exactly
like the sandbox's shared front-end.

The chain itself is *fixed* -- unlike the CNN, its shape does not change between
experiments -- so it is a concrete body rather than something declared as a
layer list. Knobs mirror the sandbox's ``build_chain`` arguments.

Naming note: the sandbox stage is called ``shift`` but it *subtracts* a scalar
(there is no bit-shift), which is misleading on an FPGA -- so the body exposes it
as ``subtract_*``. It still maps onto the ``shift`` stage internally.
"""

import numpy as np

from triggerkit.models.base import TriggerBody


def generate_lin_space_edges(start, stop, num_bit):
    """Linearly spaced integer edges for the score-quantizer front-end.

    Returns ``2**num_bit - 1`` integer levels evenly spaced over ``[start, stop]``
    -- the default edge placement to hand to :class:`TDSCANBody` as ``edges_func``.
    Pass ``edges_func=None`` (or ``edges_num_bit=0`` / ``stop <= start``) to skip
    the quantizer entirely.
    """
    return np.linspace(start, stop, num=2 ** num_bit - 1).astype(int).tolist()


class TDSCANBody(TriggerBody):
    """Fixed TDSCAN chain: [fadc?] -> [subtract?] -> [score_quantizer?] -> tdscan
    -> [digital_sum?] -> max-pool -> threshold.

    Parameters mirror ``tdscan_chain.build_chain``. Per-branch knobs (everything
    except ``filters``, ``share_neighbors`` and ``fadc``) may be overridden per
    branch via ``branches``; ``filters`` / ``share_neighbors`` are shared across
    branches (the trainer owns the multi-start count) and ``fadc`` is a shared
    front-end.

    Parameters
    ----------
    eps_xy, eps_t, filters, share_neighbors, initializer :
        TDSCAN filter geometry / multi-start / weight init.
    edges_range, edges_num_bit, edges_func :
        Optional score-quantizer front-end (built when ``edges_func`` is given
        and the range/bits are non-trivial).
    subtract_value : float or None
        Scalar subtracted before TDSCAN (the sandbox ``shift``). ``None`` -> no
        subtract stage. Also sets the TDSCAN ``pad_value`` to ``-subtract_value``.
    subtract_quantize_step : dict or None
        Fixed-point spec for the subtract stage, keys ``input`` /
        ``shift_value`` / ``output`` (qspec strings like ``"UQ8.0"``).
    subtract_overflow_mode, subtract_quantization_mode :
        Overflow / rounding modes for the subtract stage (defaults ``AP_WRAP`` /
        ``AP_TRN``, matching the stage).
    quantize_step : dict or None
        TDSCAN inner-accumulator fixed-point spec (input / ring_weights /
        convolution_accumulator / temporal_accumulator + rescale shifts).
    fake_quant_accumulators : bool
        Fake-quantize the inner accumulators in the training (STE) forward pass.
    overflow_mode, quantization_mode, rescale_shift :
        TDSCAN overflow / rounding mode and post-accumulation right-shift
        (defaults ``AP_SAT`` / ``AP_TRN`` / ``0``, matching the stage).
    digital_sum_mode : str or None
        Digital-sum stage applied *after* TDSCAN (e.g. ``"patch7"``). ``None`` ->
        no digital-sum stage.
    ring_weights, kernel_weights, tau, temp, binary_output :
        Pinned TDSCAN weights (shared vs full kernel), threshold seed / sharpness
        / output mode.
    mean_penalty_lambda, mean_penalty_target, std_penalty_lambda, std_penalty_target :
        Soft weight-distribution training penalties.
    fadc : bool
        Add a shared FADC baseline front-end before every branch.
    branches : list of dict or None
        ``None`` -> a single chain (``handles`` values are single layers).
        A list of >=2 per-branch override dicts -> that many parallel branches
        combined with an ``or_merge`` (``handles`` values are lists).
    """

    def __init__(
        self,
        *,
        eps_xy=1,
        eps_t=1,
        filters=1,
        share_neighbors=True,
        edges_range=(0.0, 0.0),
        edges_num_bit=0,
        edges_func=None,
        subtract_value=None,
        subtract_quantize_step=None,
        subtract_overflow_mode="AP_WRAP",
        subtract_quantization_mode="AP_TRN",
        quantize_step=None,
        fake_quant_accumulators=False,
        overflow_mode="AP_SAT",
        quantization_mode="AP_TRN",
        rescale_shift=0,
        digital_sum_mode=None,
        ring_weights=None,
        kernel_weights=None,
        tau=10.0,
        temp=10.0,
        binary_output=True,
        initializer="glorot_uniform",
        mean_penalty_lambda=0.0,
        mean_penalty_target=0.0,
        std_penalty_lambda=0.0,
        std_penalty_target=0.0,
        fadc=False,
        branches=None,
    ):
        self.base_kwargs = dict(
            eps_xy=eps_xy, eps_t=eps_t, filters=filters,
            share_neighbors=share_neighbors,
            edges_range=edges_range, edges_num_bit=edges_num_bit, edges_func=edges_func,
            subtract_value=subtract_value,
            subtract_quantize_step=subtract_quantize_step,
            subtract_overflow_mode=subtract_overflow_mode,
            subtract_quantization_mode=subtract_quantization_mode,
            quantize_step=quantize_step,
            fake_quant_accumulators=fake_quant_accumulators,
            overflow_mode=overflow_mode, quantization_mode=quantization_mode,
            rescale_shift=rescale_shift,
            digital_sum_mode=digital_sum_mode,
            ring_weights=ring_weights, kernel_weights=kernel_weights,
            tau=tau, temp=temp, binary_output=binary_output, initializer=initializer,
            mean_penalty_lambda=mean_penalty_lambda, mean_penalty_target=mean_penalty_target,
            std_penalty_lambda=std_penalty_lambda, std_penalty_target=std_penalty_target,
        )
        # Shared front-end (not per-branch): built once before the branch point.
        self.fadc = fadc
        self.branches = branches

    # ------------------------------------------------------------------ #
    def _add_branch(
        self, chain, *, eps_xy, eps_t, filters, share_neighbors,
        edges_range, edges_num_bit, edges_func,
        subtract_value, subtract_quantize_step,
        subtract_overflow_mode, subtract_quantization_mode,
        quantize_step, fake_quant_accumulators,
        overflow_mode, quantization_mode, rescale_shift,
        digital_sum_mode,
        ring_weights, kernel_weights, tau, temp, binary_output, initializer,
        mean_penalty_lambda, mean_penalty_target,
        std_penalty_lambda, std_penalty_target,
    ):
        """Add one branch from the current cursor; return (tdscan, threshold)."""
        # Subtract front-end (the sandbox "shift" stage -- it subtracts a scalar).
        if subtract_value is not None:
            chain.add_stage(
                "shift", value=subtract_value,
                quantize_step=subtract_quantize_step,
                overflow_mode=subtract_overflow_mode,
                quantization_mode=subtract_quantization_mode)
            pad_value = -subtract_value
        else:
            pad_value = 0.0

        if edges_func is not None and edges_num_bit > 0 and edges_range[1] > edges_range[0]:
            edges = edges_func(start=edges_range[0], stop=edges_range[1], num_bit=edges_num_bit)
            chain.add_stage("score_quantizer", edges=edges)

        tdscan_layer = chain.add_stage(
            "tdscan",
            eps_xy=eps_xy, eps_t=eps_t, filters=filters,
            initializer=initializer, share_neighbors=share_neighbors,
            quantize_step=quantize_step, fake_quant_accumulators=fake_quant_accumulators,
            overflow_mode=overflow_mode, quantization_mode=quantization_mode,
            rescale_shift=rescale_shift, pad_value=pad_value,
            mean_penalty_lambda=mean_penalty_lambda, mean_penalty_target=mean_penalty_target,
            std_penalty_lambda=std_penalty_lambda, std_penalty_target=std_penalty_target,
        )

        if share_neighbors and ring_weights is not None:
            tdscan_layer.set_weights_from_params(share_weights=True, ring_weights=ring_weights)
        elif not share_neighbors and kernel_weights is not None:
            tdscan_layer.set_weights_from_params(share_weights=False, kernel_weights=kernel_weights)
        elif share_neighbors and kernel_weights is not None:
            raise ValueError(
                "kernel_weights provided but share_neighbors=True: use ring_weights instead.")
        elif not share_neighbors and ring_weights is not None:
            raise ValueError(
                "ring_weights provided but share_neighbors=False: use kernel_weights instead. "
                f"Expected a flat array of {tdscan_layer.K * tdscan_layer.L} values "
                f"(L={tdscan_layer.L} time steps x K={tdscan_layer.K} spatial neighbors).")

        # Optional digital-sum stage after TDSCAN (sandbox _add_branch position).
        if digital_sum_mode is not None:
            chain.add_stage("digital_sum", mode=digital_sum_mode)

        chain.add_stage("global_max_pooling_2d")
        threshold_layer = chain.add_stage(
            "threshold", init_tau=tau, temp=temp, binary_output=binary_output)
        return tdscan_layer, threshold_layer

    # ------------------------------------------------------------------ #
    def build(self, chain):
        # Shared FADC front-end, built once so every branch sees it.
        if self.fadc:
            chain.add_stage("fadc")

        if self.branches is None:
            tdscan_layer, threshold_layer = self._add_branch(chain, **self.base_kwargs)
            return {"tdscan": tdscan_layer, "threshold": threshold_layer}

        if len(self.branches) < 2:
            raise ValueError(
                "branches must list at least two branch kwargs dicts "
                "(use None for a single chain).")

        start = chain.branch_point()
        tdscan_layers, threshold_layers, branch_outputs = [], [], []
        for i, branch_kwargs in enumerate(self.branches):
            overrides = dict(branch_kwargs or {})
            for shared in ("filters", "share_neighbors"):
                if shared in overrides and overrides[shared] != self.base_kwargs[shared]:
                    raise ValueError(
                        f"branch {i}: {shared!r} is shared across branches and "
                        "cannot be overridden.")
            resolved = dict(self.base_kwargs)
            resolved.update(overrides)
            resolved["filters"] = self.base_kwargs["filters"]
            resolved["share_neighbors"] = self.base_kwargs["share_neighbors"]

            chain.restore_branch_point(start)
            tdscan_layer, threshold_layer = self._add_branch(chain, **resolved)
            tdscan_layers.append(tdscan_layer)
            threshold_layers.append(threshold_layer)
            branch_outputs.append(threshold_layer.output)

        chain.add_stage("or_merge", branches=branch_outputs)
        return {"tdscan": tdscan_layers, "threshold": threshold_layers}
