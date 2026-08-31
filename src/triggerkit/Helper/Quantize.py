import re
import tensorflow as tf


_OVERFLOW_MODE_ALIASES = {
    "WRAP": "AP_WRAP",
    "AP_WRAP": "AP_WRAP",
    "SAT": "AP_SAT",
    "AP_SAT": "AP_SAT",
}

_QUANTIZATION_MODE_ALIASES = {
    "TRN": "AP_TRN",
    "AP_TRN": "AP_TRN",
    "RND": "AP_RND",
    "AP_RND": "AP_RND",
}


def _build_spec(int_bits, frac_bits, signed):
    int_bits = int(int_bits)
    frac_bits = int(frac_bits)
    signed = bool(signed)

    if signed:
        if int_bits < 1:
            raise ValueError("Signed integer bits (including sign) must be >= 1")
    elif int_bits < 0:
        raise ValueError("Unsigned integer bits must be >= 0")

    if frac_bits < 0:
        raise ValueError("Fractional bits must be >= 0")

    word_bits = int_bits + frac_bits
    if word_bits < 1:
        raise ValueError("Total word length must be >= 1")

    prefix = "SQ" if signed else "UQ"
    return {
        "signed": signed,
        "int_bits": int_bits,
        "frac_bits": frac_bits,
        "word_bits": word_bits,
        "canonical_qspec": f"{prefix}{int_bits}.{frac_bits}",
    }


def _resolve_spec(frac_bits=None, word_bits=None, qspec=None, signed=None):
    """Return a parsed quantization spec dict or None if quantization is disabled."""
    if qspec is not None:
        return parse_qspec(qspec)
    if frac_bits is None or word_bits is None:
        return None
    word_bits = int(word_bits)
    frac_bits = int(frac_bits)
    int_bits = word_bits - frac_bits
    return _build_spec(int_bits=int_bits, frac_bits=frac_bits, signed=True if signed is None else signed)


def parse_qspec(spec):
    """
    Parse strings like ``SQ8.8`` or ``UQ8.8`` into a normalized spec dict.

    Backward compatibility:
    - ``Q8.8`` is accepted as a signed alias for ``SQ8.8``.
    - ``(word_bits, frac_bits)`` is accepted as a signed legacy tuple.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        required = {"signed", "int_bits", "frac_bits", "word_bits", "canonical_qspec"}
        if not required.issubset(spec):
            missing = ", ".join(sorted(required - set(spec)))
            raise ValueError(f"Quantization spec dict is missing keys: {missing}")
        return {
            "signed": bool(spec["signed"]),
            "int_bits": int(spec["int_bits"]),
            "frac_bits": int(spec["frac_bits"]),
            "word_bits": int(spec["word_bits"]),
            "canonical_qspec": str(spec["canonical_qspec"]),
        }
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2:
            word_bits, frac_bits = spec
            return _resolve_spec(word_bits=word_bits, frac_bits=frac_bits, signed=True)
        if len(spec) == 3:
            word_bits, frac_bits, signed = spec
            return _resolve_spec(word_bits=word_bits, frac_bits=frac_bits, signed=signed)
        raise ValueError("Tuple/list quantization specs must be (word_bits, frac_bits) or (word_bits, frac_bits, signed).")
    if not isinstance(spec, str):
        raise ValueError(f"Unsupported quantization spec type: {type(spec)}")

    s = spec.strip()
    if not s:
        return None

    m = re.match(r"^(?:(S|U))?[Qq](\d+)\.(\d+)$", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Quantization format '{spec}' must look like SQ<int>.<frac> or UQ<int>.<frac>, e.g. 'SQ8.8'."
        )

    prefix = m.group(1)
    signed = True if prefix is None else prefix.upper() == "S"
    int_bits = int(m.group(2))
    frac_bits = int(m.group(3))
    return _build_spec(int_bits=int_bits, frac_bits=frac_bits, signed=signed)


def parse_qformat(spec):
    """
    Parse a qspec and return ``(word_bits, frac_bits)``.
    This keeps the legacy API and intentionally omits the signedness bit.
    """
    parsed = parse_qspec(spec)
    if parsed is None:
        return None
    return parsed["word_bits"], parsed["frac_bits"]


def parse_overflow_mode(mode):
    """
    Normalize an HLS overflow mode string.

    Supported modes:
    - ``AP_WRAP``: HLS default wrap-around behavior
    - ``AP_SAT``: saturating behavior
    """
    if mode is None:
        return "AP_WRAP"
    if not isinstance(mode, str):
        raise ValueError(f"Unsupported overflow mode type: {type(mode)}")

    canonical = _OVERFLOW_MODE_ALIASES.get(mode.strip().upper())
    if canonical is None:
        raise ValueError(
            f"Unsupported overflow mode '{mode}'. Expected AP_WRAP or AP_SAT."
        )
    return canonical


def parse_quantization_mode(mode):
    """
    Normalize an HLS quantization mode string.

    Supported modes:
    - ``AP_TRN``: HLS default truncation
    - ``AP_RND``: round to nearest with ties toward +inf
    """
    if mode is None:
        return "AP_TRN"
    if not isinstance(mode, str):
        raise ValueError(f"Unsupported quantization mode type: {type(mode)}")

    canonical = _QUANTIZATION_MODE_ALIASES.get(mode.strip().upper())
    if canonical is None:
        raise ValueError(
            f"Unsupported quantization mode '{mode}'. Expected AP_TRN or AP_RND."
        )
    return canonical


def _fixed_point_limits(word_bits, signed=True):
    if signed:
        min_i = -(2 ** (int(word_bits) - 1))
        max_i = (2 ** (int(word_bits) - 1)) - 1
        return min_i, max_i
    min_i = 0
    max_i = (2 ** int(word_bits)) - 1
    return min_i, max_i


def _fixed_point_max_raw(word_bits):
    return tf.constant((2 ** int(word_bits)) - 1, dtype=tf.int64)


def _fixed_point_to_raw_int(w_int, word_bits):
    modulus = tf.constant(2 ** int(word_bits), dtype=tf.int64)
    return tf.math.floormod(tf.cast(w_int, tf.int64), modulus)


def _fixed_point_from_raw_int(raw_bits, word_bits, signed=True):
    raw_bits = tf.cast(raw_bits, tf.int64)
    if not signed:
        return raw_bits

    modulus = tf.constant(2 ** int(word_bits), dtype=tf.int64)
    sign_threshold = tf.constant(2 ** (int(word_bits) - 1), dtype=tf.int64)
    return tf.where(raw_bits >= sign_threshold, raw_bits - modulus, raw_bits)


def _fixed_point_normalize_int(w_int, word_bits, signed=True, overflow_mode="AP_WRAP"):
    overflow_mode = parse_overflow_mode(overflow_mode)
    values = tf.cast(w_int, tf.int64)

    if overflow_mode == "AP_SAT":
        min_i, max_i = _fixed_point_limits(word_bits, signed=signed)
        return tf.clip_by_value(values, min_i, max_i)

    modulus = tf.constant(2 ** int(word_bits), dtype=tf.int64)
    wrapped = tf.math.floormod(values, modulus)
    if not signed:
        return wrapped

    sign_threshold = tf.constant(2 ** (int(word_bits) - 1), dtype=tf.int64)
    return tf.where(wrapped >= sign_threshold, wrapped - modulus, wrapped)


def _fixed_point_quantize_scaled(values, quantization_mode="AP_TRN"):
    quantization_mode = parse_quantization_mode(quantization_mode)
    values = tf.convert_to_tensor(values)
    half = tf.cast(0.5, values.dtype)

    if quantization_mode == "AP_TRN":
        return tf.math.floor(values)
    return tf.math.floor(values + half)


def fixed_point_clip_int(
    w_int,
    word_bits=None,
    qspec=None,
    dtype=tf.int64,
    signed=True,
    overflow_mode="AP_WRAP",
):
    """
    Normalize an integer tensor to the range of a fixed-point register.

    ``overflow_mode`` follows HLS semantics:
    - ``AP_WRAP``: wrap around on overflow (default, like ``ap_fixed``/``ap_ufixed``)
    - ``AP_SAT``: saturate to min/max representable value
    """
    if qspec is not None:
        spec = parse_qspec(qspec)
        word_bits = spec["word_bits"]
        signed = spec["signed"]
    if word_bits is None:
        return tf.cast(w_int, dtype)

    w_int = _fixed_point_normalize_int(
        w_int,
        word_bits=word_bits,
        signed=signed,
        overflow_mode=overflow_mode,
    )
    return tf.cast(w_int, dtype)


def fixed_point_to_int(
    w,
    frac_bits=None,
    word_bits=None,
    qspec=None,
    dtype=tf.int64,
    signed=True,
    overflow_mode="AP_WRAP",
    quantization_mode="AP_TRN",
):
    """
    Quantize tensor ``w`` and return its fixed-point integer representation.

    ``quantization_mode`` follows HLS semantics:
    - ``AP_TRN``: truncation (default, like ``ap_fixed``/``ap_ufixed``)
    - ``AP_RND``: round to nearest with ties toward +inf
    """
    spec = _resolve_spec(frac_bits=frac_bits, word_bits=word_bits, qspec=qspec, signed=signed)
    if spec is None:
        return tf.cast(w, dtype)

    tensor = tf.convert_to_tensor(w)
    quant_dtype = tensor.dtype if tensor.dtype.is_floating else tf.float64
    tensor = tf.cast(tensor, quant_dtype)
    scale = tf.cast(2 ** int(spec["frac_bits"]), quant_dtype)

    w_int = _fixed_point_quantize_scaled(
        tensor * scale,
        quantization_mode=quantization_mode,
    )
    w_int = _fixed_point_normalize_int(
        w_int,
        word_bits=spec["word_bits"],
        signed=spec["signed"],
        overflow_mode=overflow_mode,
    )
    return tf.cast(w_int, dtype)


def fixed_point_from_int(
    w_int,
    frac_bits=None,
    word_bits=None,
    qspec=None,
    dtype=tf.float32,
    signed=True,
    overflow_mode="AP_WRAP",
):
    """
    Convert fixed-point integers back to floating point values.
    """
    spec = _resolve_spec(frac_bits=frac_bits, word_bits=word_bits, qspec=qspec, signed=signed)
    if spec is None:
        return tf.cast(w_int, dtype)

    w_int = fixed_point_clip_int(
        w_int,
        word_bits=spec["word_bits"],
        signed=spec["signed"],
        dtype=tf.int64,
        overflow_mode=overflow_mode,
    )
    scale = tf.cast(2 ** int(spec["frac_bits"]), dtype)
    return tf.cast(w_int, dtype) / scale


def fixed_point_rescale_int(
    w_int,
    *,
    src_qspec=None,
    dst_qspec=None,
    src_frac_bits=None,
    src_word_bits=None,
    dst_frac_bits=None,
    dst_word_bits=None,
    src_signed=True,
    dst_signed=True,
    dtype=tf.int64,
    shift=0,
    overflow_mode="AP_WRAP",
):
    """
    Rescale a fixed-point integer register by keeping the most significant bits
    of the full source register.

    ``shift`` reduces the amount of right-shift applied before cropping. If it
    makes the selected value wider than the destination register, the result is
    normalized according to ``overflow_mode``.
    """
    overflow_mode = parse_overflow_mode(overflow_mode)
    src_spec = _resolve_spec(
        frac_bits=src_frac_bits,
        word_bits=src_word_bits,
        qspec=src_qspec,
        signed=src_signed,
    )
    dst_spec = _resolve_spec(
        frac_bits=dst_frac_bits,
        word_bits=dst_word_bits,
        qspec=dst_qspec,
        signed=dst_signed,
    )

    if dst_spec is None:
        return tf.cast(w_int, dtype)
    if src_spec is None:
        raise ValueError("fixed_point_rescale_int requires a source quantization spec.")
    shift = int(shift)
    if shift < 0:
        raise ValueError(f"fixed_point_rescale_int shift must be >= 0, got {shift}.")

    values = fixed_point_clip_int(
        w_int,
        qspec=src_spec,
        dtype=tf.int64,
        overflow_mode=overflow_mode,
    )

    register_shift = int(src_spec["word_bits"]) - int(dst_spec["word_bits"]) - shift
    if register_shift >= 0:
        divisor = tf.constant(2 ** register_shift, dtype=tf.int64)
        candidate = tf.math.floordiv(values, divisor)
    else:
        candidate = values * tf.constant(2 ** (-register_shift), dtype=tf.int64)

    dst_int = fixed_point_clip_int(
        candidate,
        qspec=dst_spec,
        dtype=tf.int64,
        overflow_mode=overflow_mode,
    )
    return tf.cast(dst_int, dtype)


def fixed_point_quantize(
    w,
    frac_bits=None,
    word_bits=None,
    qspec=None,
    signed=True,
    overflow_mode="AP_WRAP",
    quantization_mode="AP_TRN",
):
    """
    Quantize tensor ``w`` to fixed-point. If ``qspec`` is None, returns ``w``.
    ``qspec``: string like ``SQ8.8``/``UQ8.8`` or a legacy tuple.
    ``frac_bits``/``word_bits`` are kept for backward compatibility.
    """
    spec = _resolve_spec(frac_bits=frac_bits, word_bits=word_bits, qspec=qspec, signed=signed)
    if spec is None:
        return w

    tensor = tf.convert_to_tensor(w)
    scale = tf.cast(2 ** int(spec["frac_bits"]), tensor.dtype)
    w_int = fixed_point_to_int(
        tensor,
        word_bits=spec["word_bits"],
        frac_bits=spec["frac_bits"],
        signed=spec["signed"],
        overflow_mode=overflow_mode,
        quantization_mode=quantization_mode,
    )
    return tf.cast(w_int, tensor.dtype) / scale


def ste_fixed_point(
    w,
    frac_bits=None,
    word_bits=None,
    qspec=None,
    signed=True,
    overflow_mode="AP_WRAP",
    quantization_mode="AP_TRN",
):
    """Straight-through estimator version of ``fixed_point_quantize``."""
    q = fixed_point_quantize(
        w,
        frac_bits=frac_bits,
        word_bits=word_bits,
        qspec=qspec,
        signed=signed,
        overflow_mode=overflow_mode,
        quantization_mode=quantization_mode,
    )
    return w + tf.stop_gradient(q - w)


class FixedPointConstraint(tf.keras.constraints.Constraint):
    def __init__(
        self,
        qspec=None,
        frac_bits=None,
        word_bits=None,
        signed=True,
        overflow_mode="AP_WRAP",
        quantization_mode="AP_TRN",
    ):
        self.qspec = qspec
        self.frac_bits = frac_bits
        self.word_bits = word_bits
        self.signed = signed
        self.overflow_mode = parse_overflow_mode(overflow_mode)
        self.quantization_mode = parse_quantization_mode(quantization_mode)

    def __call__(self, w):
        return fixed_point_quantize(
            w,
            frac_bits=self.frac_bits,
            word_bits=self.word_bits,
            qspec=self.qspec,
            signed=self.signed,
            overflow_mode=self.overflow_mode,
            quantization_mode=self.quantization_mode,
        )

    def get_config(self):
        return {
            "qspec": self.qspec,
            "frac_bits": self.frac_bits,
            "word_bits": self.word_bits,
            "signed": self.signed,
            "overflow_mode": self.overflow_mode,
            "quantization_mode": self.quantization_mode,
        }
