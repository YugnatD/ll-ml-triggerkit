"""Integration tests: the real reader / chain / training / cascade on synthetic HDF5 files."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import write_ctapipe_h5
from triggerkit.FileIO.AsyncFileOpener import FileReadError
from triggerkit.FileIO.FileOpenerCTAO import iterate_batches
from triggerkit.TriggerChain import TriggerChain
from triggerkit.models import TDSCANBody


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=30, tels=(1,), seed=i) for i in range(3)]
    nsb = [write_ctapipe_h5(d / f"n{i}.h5", n_events=50, tels=(1,), kind="nsb", seed=10 + i) for i in range(2)]
    return gamma, nsb, d


def tdscan_chain(files, filters=1, **kw):
    gamma, nsb, _ = files
    chain = TriggerChain(gamma, simtel_nsb_path=nsb)
    handles = TDSCANBody(filters=filters, eps_xy=1, eps_t=1, **kw).build(chain)
    chain.compile_chain()
    return chain, handles


def patch7_chain(files):
    gamma, nsb, _ = files
    chain = TriggerChain(gamma, simtel_nsb_path=nsb)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    thr = chain.add_stage("threshold", init_tau=730.0, temp=1.0, binary_output=True)
    chain.compile_chain()
    return chain, thr


# --------------------------------------------------------------------------- spawn
def test_reader_processes_do_not_re_run_the_main_script(tmp_path):
    """Regression: `spawn` re-imported the user's main script in every reader process
    (a full TensorFlow import per opened file). The marker is written at module level,
    so it is written once per import of the script."""
    data = write_ctapipe_h5(tmp_path / "g.h5", n_events=3)
    marker = tmp_path / "marker.txt"
    script = tmp_path / "main_script.py"
    script.write_text(textwrap.dedent(f"""
        open({str(marker)!r}, "a").write("import\\n")        # runs on EVERY import of this file
        from triggerkit.FileIO.AsyncFileOpener import AsyncFileOpenerProcess
        if __name__ == "__main__":
            with AsyncFileOpenerProcess({data!r}) as fo:
                print("EVENTS", sum(1 for _ in fo))
            import sys; print("MAIN_FILE_KEPT", bool(sys.modules["__main__"].__file__))
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert "EVENTS 3" in out.stdout, out.stderr[-800:]
    assert "MAIN_FILE_KEPT True" in out.stdout            # the parent's __main__ is restored afterwards
    assert marker.read_text().count("import") == 1        # not re-imported by the worker


# --------------------------------------------------------------------------- errors
def test_iterate_batches_converts_read_failures_and_passes_others_through():
    def reading_fails():
        yield np.zeros(2, np.float32)
        raise FileReadError("boom: /x/y.h5")

    def other_failure():
        yield np.zeros(2, np.float32)
        raise ValueError("unrelated")

    sig = tf.TensorSpec((2,), tf.float32)
    ds = tf.data.Dataset.from_generator(reading_fails, output_signature=sig)
    with pytest.raises(FileReadError, match="boom"):
        list(iterate_batches(ds))
    ds = tf.data.Dataset.from_generator(other_failure, output_signature=sig)
    with pytest.raises(Exception) as err:
        list(iterate_batches(ds))
    assert not isinstance(err.value, FileReadError)


def test_compute_statistics_raises_a_catchable_error_and_cleans_up(files, tmp_path):
    gamma, nsb, d = files
    bad = d / "corrupt.h5"
    bad.write_bytes(b"not an hdf5 file")
    chain, _ = tdscan_chain(files)
    chain.simtel_path = [gamma[0], str(bad)]
    with pytest.raises(FileReadError):
        chain.compute_statistics(base_name="e", folder=str(tmp_path), batch_size=16, tel_id_only=1,
                                 ignore_errors=False, overwrite=True)
    assert not list(tmp_path.glob("*.h5"))                 # partial statistics file removed
    # ... while ignore_errors=True keeps every readable event
    res = chain.compute_statistics(base_name="e", folder=str(tmp_path), batch_size=16, tel_id_only=1,
                                   ignore_errors=True, overwrite=True)
    assert res["gamma_total"] == 30 and res["nsb_total"] == 100


# --------------------------------------------------------------------------- cascade
def test_cascade_caps_are_exact(files):
    from triggerkit.cascade import TriggerCascade
    chain1, h = tdscan_chain(files)
    h["threshold"].tau.assign(0.0)
    chain2, _ = patch7_chain(files)
    casc = TriggerCascade([chain1, chain2], names=["tdscan", "patch7"])
    stats = casc.compute_statistics(batch_size=32, tel_id_only=1, max_gamma_events=45, max_nsb_events=70)
    assert stats["gamma"]["n_total"] == 45 and stats["nsb"]["n_total"] == 70      # not rounded up to a batch
    scores, n_total, n_up = casc._survivor_scores_for_stage(
        0, files[1], casc._stats_config(32, 1, 0, False, True), 70)
    assert n_total == 70 == n_up == scores.size


# --------------------------------------------------------------------------- autoencoder
def test_reconstruction_loss_matches_the_old_formula_and_is_serializable(tmp_path):
    from triggerkit.models import load_model
    from triggerkit.training.losses import ReconstructionMSE, ReconstructionMSEMetric

    pred = tf.random.normal((3, 5, 4, 1))                    # (B, N, T, C)
    true = tf.random.uniform((3, 5))
    expected = tf.reduce_mean(tf.square(true - tf.squeeze(tf.reduce_sum(pred, axis=2), -1)), axis=-1)
    np.testing.assert_allclose(ReconstructionMSE()(true, pred).numpy(), expected.numpy().mean(), rtol=1e-5)
    m = ReconstructionMSEMetric(); m.update_state(true, pred)
    np.testing.assert_allclose(float(m.result()), expected.numpy().mean(), rtol=1e-5)

    inp = tf.keras.Input((5, 4))
    model = tf.keras.Model(inp, tf.keras.layers.Dense(1)(inp))
    model.compile(optimizer="adam", loss=ReconstructionMSE(), metrics=[ReconstructionMSEMetric()])
    model.fit(np.random.rand(8, 5, 4), np.random.rand(8, 5), verbose=0)
    model.save(tmp_path / "ae.keras")
    again = load_model(str(tmp_path / "ae.keras"))
    assert isinstance(again.loss, ReconstructionMSE)


def test_autoencoder_training_saves_a_reloadable_model(files, tmp_path):
    from triggerkit.models import load_model
    gamma, nsb, _ = files
    chain = TriggerChain(gamma, simtel_nsb_path=nsb)
    chain.add_stage("tdscan", eps_xy=1, eps_t=1, filters=1)
    chain.train_chain_autoencoder(epochs=1, batch_size=16, percent_validation=0.34, tel_id_only=1,
                                  load_ram=True, output_folder=str(tmp_path))
    saved = list(tmp_path.glob("*model.keras"))
    assert len(saved) == 1
    load_model(str(saved[0]))                                # used to raise "Could not locate function"


# --------------------------------------------------------------------------- callbacks
def test_grad_norm_logger_respects_verbose(capsys):
    from triggerkit.Callback.GradNormLogger import GradNormLogger
    model = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.Dense(1)])
    model.compile(optimizer="sgd", loss="mse")
    x, y = np.random.rand(16, 3).astype("float32"), np.random.rand(16, 1).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    model.fit(x, y, epochs=1, verbose=0, callbacks=[GradNormLogger(ds, verbose=0)])
    assert "[grad]" not in capsys.readouterr().out
    model.fit(x, y, epochs=1, verbose=0, callbacks=[GradNormLogger(ds, verbose=1)])
    assert "[grad]" in capsys.readouterr().out


def test_tdscan_controller_attached_whatever_the_layer_is_called(files, monkeypatch):
    from triggerkit.Callback import TDSCANController as module
    gamma, nsb, _ = files
    tdscan_chain(files)                                      # first chain takes the name "tdscan"
    chain, _ = tdscan_chain(files)                           # second chain: "tdscan_1" (or later)
    assert not any(l.name == "tdscan" for l in chain.model.layers)
    created = []
    original = module.TDSCANController
    monkeypatch.setattr(module, "TDSCANController",
                        lambda **kw: (created.append(kw["tdscan_layer"].name), original(**kw))[1])
    from triggerkit.data import TriggerDataset
    ds = TriggerDataset(gamma, nsb, batch_size=16, tel_id_only=1, percent_validation=0.34,
                        max_gamma_samples_train=32, max_nsb_samples_train=32,
                        max_gamma_samples_val=16, max_nsb_samples_val=16, load_ram=True)
    chain.train_chain(epochs=1, dataset=ds, verbose=0, output_folder="/tmp/_tk_ctrl_test")
    assert len(created) == 1 and created[0].startswith("tdscan")
