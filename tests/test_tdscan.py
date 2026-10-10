import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from ctapipe.instrument import CameraGeometry

from triggerkit.Stages.TDSCAN import TDSCAN, getTDSCANNeighbors


@pytest.fixture(scope="module")
def digicam_like():
    g = CameraGeometry.make_rectangular(4, 4)
    return CameraGeometry(name="DigiCam", pix_id=g.pix_id, pix_x=g.pix_x, pix_y=g.pix_y,
                          pix_area=g.pix_area, pix_type=g.pix_type)


def make_layer(geom, **kw):
    nb = getTDSCANNeighbors(1, "DigiCam", 432)
    layer = TDSCAN(nb, eps_t=1, eps_xy=1, filters=2, input_geometry=geom, **kw)
    layer.build((None, 432, 6))
    return layer


def set_weights(layer, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(-6, 7, layer.kernel_rings.shape).astype(np.float32) / 4.0   # on the SQ3.2 grid
    layer.kernel_rings.assign(w)


def adc(seed=1, b=2, t=6):
    return np.random.default_rng(seed).integers(0, 15, (b, 432, t)).astype(np.float32)


def test_float_path_matches_numpy_reference(digicam_like):
    layer = make_layer(digicam_like, pad_value=1.0)
    set_weights(layer)
    x = adc()
    np.testing.assert_allclose(layer(x).numpy(), layer.call_numpy(x), rtol=1e-4, atol=1e-4)


QSTEP = {"input": "UQ4.0", "ring_weights": "SQ3.2"}


def test_quantized_tf_path_matches_numpy_reference(digicam_like):
    layer = make_layer(digicam_like, quantize_step=QSTEP)
    set_weights(layer)
    x = adc()
    np.testing.assert_array_equal(layer.call_quantized(x).numpy(), layer.call_quantized_numpy(x))


def test_inference_uses_the_integer_path_and_equals_float_when_lossless(digicam_like):
    """With on-grid weights/inputs and auto-derived (lossless) accumulators the
    integer path must equal the plain float convolution."""
    q = make_layer(digicam_like, quantize_step=QSTEP)
    f = make_layer(digicam_like)
    set_weights(q); f.kernel_rings.assign(q.kernel_rings.numpy())
    x = adc()
    np.testing.assert_allclose(q(x, training=False).numpy(), f(x).numpy(), atol=1e-4)


def test_training_forward_with_fake_quant_matches_inference(digicam_like):
    """Quantization-aware training must see the deployed arithmetic even when
    an accumulator is narrowed (saturating)."""
    step = dict(QSTEP, convolution_accumulator="SQ5.2", temporal_accumulator="SQ6.2")
    layer = make_layer(digicam_like, quantize_step=step, fake_quant_accumulators=True,
                       overflow_mode="AP_SAT")
    set_weights(layer)
    x = adc()
    infer = layer(x, training=False).numpy()
    train = layer(x, training=True).numpy()
    np.testing.assert_allclose(train, infer, atol=1e-5)
    assert np.abs(infer).max() <= 2 ** 6          # the narrowed register really saturated


def test_gradients_flow_through_quantized_training_path(digicam_like):
    layer = make_layer(digicam_like, quantize_step=QSTEP, fake_quant_accumulators=True,
                       overflow_mode="AP_SAT")
    set_weights(layer)
    x = tf.constant(adc())
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(x, training=True))
    g = tape.gradient(loss, layer.kernel_rings)
    assert g is not None and float(tf.reduce_sum(tf.abs(g))) > 0


def test_quantize_step_validation(digicam_like):
    nb = getTDSCANNeighbors(1, "DigiCam", 432)
    with pytest.raises(ValueError):   # accumulators without an input spec
        TDSCAN(nb, 1, 1, 1, input_geometry=digicam_like, quantize_step={"ring_weights": "SQ3.2"})
    with pytest.raises(ValueError):   # unknown key
        TDSCAN(nb, 1, 1, 1, input_geometry=digicam_like, quantize_step={"input": "UQ4.0", "bogus": 1})
    with pytest.raises(ValueError):   # negative rescale shift
        TDSCAN(nb, 1, 1, 1, input_geometry=digicam_like, rescale_shift=-1)


def test_config_roundtrip_keeps_structure(digicam_like):
    layer = make_layer(digicam_like, quantize_step=QSTEP, pad_value=2.0)
    again = TDSCAN.from_config(layer.get_config())
    assert (again.eps_t, again.eps_xy, again.filters, again.pad_value) == (1, 1, 2, 2.0)
    assert again.quantize_step == layer.quantize_step
