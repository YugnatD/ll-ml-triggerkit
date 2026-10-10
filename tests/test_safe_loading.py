"""Opening data files must never execute code that is stored inside them."""
import pickle

import h5py
import numpy as np
import pytest

from triggerkit.Helper.history import load_history, load_legacy_npy_history, save_history
from triggerkit.Statistics.StatPlotter import StatPlotter

CHAIN = [("threshold", {"threshold": 2.0})]


class Evil:
    """Unpickling this writes a marker file: proof that code ran."""
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        import pathlib
        return (pathlib.Path(self.marker).write_text, ("pwned",))


def test_history_json_roundtrip_with_numpy_values(tmp_path):
    hist = {"loss": [np.float32(1.5), np.float64(0.5)], "auc": np.array([0.7, 0.8]), "epoch": [np.int64(1)]}
    save_history(str(tmp_path / "h.json"), hist)
    back = load_history(str(tmp_path / "h.json"))
    assert back == {"loss": [1.5, 0.5], "auc": [0.7, 0.8], "epoch": [1]}


def test_stat_plotter_refuses_a_pickled_chain_by_default(tmp_path, capsys):
    marker = tmp_path / "marker.txt"
    with h5py.File(tmp_path / "evil.h5", "w") as f:
        f.attrs["camera_name"] = "DigiCam"
        f.attrs["trigger_chain_pickle"] = np.void(pickle.dumps(Evil(str(marker))))
    with pytest.raises(ValueError, match="not found"):            # the only file was skipped
        StatPlotter(base_reference_config=CHAIN, stat_folder=str(tmp_path))
    assert not marker.exists()                                    # nothing was unpickled
    assert "pickle" in capsys.readouterr().out


def test_stat_plotter_legacy_pickle_needs_an_explicit_opt_in(tmp_path):
    with h5py.File(tmp_path / "old.h5", "w") as f:
        f.attrs["camera_name"] = "DigiCam"
        f.attrs["trigger_chain_pickle"] = np.void(pickle.dumps(CHAIN))
    sp = StatPlotter(base_reference_config=CHAIN, stat_folder=str(tmp_path), allow_legacy_pickle=True)
    assert sp.base_config_result["trigger_chain"] == [("threshold", {"threshold": 2.0})]


def test_legacy_npy_history_is_not_loaded_unless_trusted(tmp_path, capsys):
    pytest.importorskip("tensorflow")
    from synthetic import write_ctapipe_h5
    from triggerkit.TriggerChain import TriggerChain
    from triggerkit.data import TriggerDataset
    from triggerkit.models import TDSCANBody

    g = write_ctapipe_h5(tmp_path / "g.h5", n_events=20, tels=(1,), seed=1)
    n = write_ctapipe_h5(tmp_path / "n.h5", n_events=20, tels=(1,), kind="nsb", seed=2)
    chain = TriggerChain([g], [n])
    TDSCANBody(filters=1, eps_xy=1, eps_t=1).build(chain)
    chain.compile_chain()
    ds = TriggerDataset([g, g], [n], batch_size=16, tel_id_only=1, percent_validation=0.5,
                        max_gamma_samples_train=16, max_nsb_samples_train=16,
                        max_gamma_samples_val=16, max_nsb_samples_val=16, load_ram=True)
    out = tmp_path / "models"
    chain.train_chain(epochs=1, dataset=ds, verbose=0, output_folder=str(out))

    # history is written as JSON next to the model -- and only as JSON
    (model,) = out.glob("*_model.keras")
    stem = str(model)[: -len("_model.keras")]
    assert (out / (model.name.replace("_model.keras", "_history.json"))).exists()
    assert not list(out.glob("*.npy"))
    again = TriggerChain([g], [n])
    again.compile_chain(model_path=str(model))
    assert "loss" in again.model.history.history                 # JSON sidecar loaded

    # a pickled legacy sidecar replaces it: ignored by default, never executed
    (out / (model.name.replace("_model.keras", "_history.json"))).unlink()
    marker = tmp_path / "marker.txt"
    np.save(stem + "_history.npy", {"loss": [1.0], "evil": Evil(str(marker))}, allow_pickle=True)
    capsys.readouterr()
    safe = TriggerChain([g], [n])
    safe.compile_chain(model_path=str(model))
    assert not marker.exists()
    assert "Ignoring legacy history" in capsys.readouterr().out
    assert not getattr(getattr(safe.model, "history", None), "history", None)

    # an explicit opt-in loads a harmless legacy history
    np.save(stem + "_history.npy", {"loss": [1.0]}, allow_pickle=True)
    trusted = TriggerChain([g], [n])
    trusted.compile_chain(model_path=str(model), trust_legacy_history=True)
    assert trusted.model.history.history == {"loss": [1.0]}
