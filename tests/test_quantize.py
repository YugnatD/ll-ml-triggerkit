import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from triggerkit.Helper.Quantize import (
    fixed_point_quantize,
    fixed_point_rescale_int,
    fixed_point_to_int,
    parse_qspec,
)


def q(x, spec, **kw):
    return fixed_point_quantize(tf.constant(x, tf.float32), qspec=spec, **kw).numpy()


def test_parse_qspec():
    s = parse_qspec("SQ9.3")
    assert (s["signed"], s["int_bits"], s["frac_bits"], s["word_bits"]) == (True, 9, 3, 12)
    assert parse_qspec("UQ4.0")["canonical_qspec"] == "UQ4.0"
    assert parse_qspec("Q8.8")["signed"] is True           # legacy alias
    assert parse_qspec(None) is None
    for bad in ("SQ8", "X4.4", "UQ0.0", 3.5):
        with pytest.raises(ValueError):
            parse_qspec(bad)


def test_wrap_vs_saturate_signed():
    # SQ4.0 holds -8..7
    assert q([9.0, 8.0, -9.0], "SQ4.0", overflow_mode="AP_WRAP").tolist() == [-7.0, -8.0, 7.0]
    assert q([9.0, 8.0, -9.0], "SQ4.0", overflow_mode="AP_SAT").tolist() == [7.0, 7.0, -8.0]


def test_unsigned_saturates_negative_to_zero():
    assert q([-3.0, 20.0], "UQ4.0", overflow_mode="AP_SAT").tolist() == [0.0, 15.0]


def test_truncation_vs_rounding():
    assert q([0.75], "UQ3.1", quantization_mode="AP_TRN").tolist() == [0.5]
    assert q([0.75], "UQ3.1", quantization_mode="AP_RND").tolist() == [1.0]
    assert q([-0.25], "SQ4.1", quantization_mode="AP_TRN").tolist() == [-0.5]   # floor, not toward 0


def test_to_int_is_on_the_scaled_grid():
    got = fixed_point_to_int(tf.constant([1.5, 2.25]), qspec="SQ6.2", dtype=tf.int64).numpy()
    assert got.tolist() == [6, 9]


def test_rescale_keeps_most_significant_bits():
    # SQ8.0 -> SQ4.0 drops 4 LSBs (floor division by 16)
    out = fixed_point_rescale_int(tf.constant([80, -17], tf.int64), src_qspec="SQ8.0", dst_qspec="SQ4.0")
    assert out.numpy().tolist() == [5, -2]
    # shift=2 shifts right by 2 instead of 4 (keeps 2 more bits): 8 -> 2 ...
    out = fixed_point_rescale_int(tf.constant([8, 100], tf.int64), src_qspec="SQ8.0", dst_qspec="SQ4.0",
                                  shift=2, overflow_mode="AP_SAT")
    assert out.numpy().tolist() == [2, 7]          # ... and 100 -> 25 saturates at 7
    with pytest.raises(ValueError):
        fixed_point_rescale_int(tf.constant([1], tf.int64), src_qspec="SQ8.0", dst_qspec="SQ4.0", shift=-1)


def test_no_spec_is_identity():
    x = tf.constant([1.234, -5.6])
    assert np.array_equal(fixed_point_quantize(x).numpy(), x.numpy())
