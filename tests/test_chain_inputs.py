"""Which Keras inputs a chain's auxiliary models (calibration, selection) must use."""
import pytest

tf = pytest.importorskip("tensorflow")
from triggerkit.TriggerChain import TriggerChain


def bare_chain(baseline_connected):
    chain = object.__new__(TriggerChain)          # no data files needed for the wiring logic
    chain.input_layer = tf.keras.Input(shape=(4, 5), dtype=tf.uint16, name="waveform")
    chain.input_baseline = tf.keras.Input(shape=(4,), dtype=tf.int32, name="pedestal")
    chain._baseline_connected = baseline_connected
    return chain


def test_single_input_when_baseline_is_not_wired():
    chain = bare_chain(False)
    assert chain.model_inputs() is chain.input_layer
    assert chain.pack_model_inputs("wf", "ped") == "wf"
    # the auxiliary model must build (Keras rejects an unconnected input)
    out = tf.keras.layers.Flatten()(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(chain.input_layer))
    tf.keras.Model(inputs=chain.model_inputs(), outputs=out)
    with pytest.raises(ValueError):
        tf.keras.Model(inputs=[chain.input_layer, chain.input_baseline], outputs=out)


def test_two_inputs_when_fadc_wired_the_baseline():
    chain = bare_chain(True)
    assert chain.model_inputs() == [chain.input_layer, chain.input_baseline]
    assert chain.pack_model_inputs("wf", "ped") == ("wf", "ped")
