"""The same normalisation through the real classes, on synthetic files."""
import pathlib

import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import write_ctapipe_h5
from triggerkit.FileIO.AsyncFileOpener import FileOpenerCTAO
from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_condition_folds, make_rotation_folds
from triggerkit.data import TriggerDataset
from triggerkit.models import TDSCANBody


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("paths")
    g = write_ctapipe_h5(d / "g.h5", n_events=30, tels=(1,), seed=1)
    n = write_ctapipe_h5(d / "n.h5", n_events=40, tels=(1,), kind="nsb", seed=2)
    return g, n, d


def stats(data, tmp_path, gamma, nsb, **kw):
    chain = TriggerChain(gamma, nsb)
    TDSCANBody(filters=1, eps_xy=1, eps_t=1).build(chain)
    chain.compile_chain()
    res = chain.compute_statistics(base_name="x", folder=str(tmp_path), batch_size=16, tel_id_only=1,
                                   overwrite=True, **kw)
    return chain, res


def test_single_str_paths_read_every_event(data, tmp_path):
    """Regression: a single str path (the declared type) used to be read as 0 events, silently."""
    g, n, _ = data
    chain, res = stats(data, tmp_path, g, n)
    assert (res["gamma_total"], res["nsb_total"]) == (30, 40)
    assert chain.simtel_path == [g] and chain.simtel_nsb_path == [n]


@pytest.mark.parametrize("wrap", [pathlib.Path, lambda p: [pathlib.Path(p)], lambda p: (p,)],
                         ids=["Path", "list[Path]", "tuple"])
def test_other_path_forms(data, tmp_path, wrap):
    g, n, _ = data
    chain, res = stats(data, tmp_path, wrap(g), wrap(n))
    assert (res["gamma_total"], res["nsb_total"]) == (30, 40)
    assert chain.simtel_path == [str(g)] and all(type(x) is str for x in chain.simtel_path)


def test_gamma_only_chain_and_nsb_only_geometry(data, tmp_path):
    g, n, _ = data
    _, res = stats(data, tmp_path, [g], None)                # no NSB files: used to be a TypeError
    assert res["gamma_total"] == 30 and res["nsb_total"] == 0 and res["nsb_rate_hz"] == 0.0
    chain = TriggerChain([], [n])                            # geometry probed from the NSB file
    assert chain.camera_name == "DigiCam_R0Alpha" and chain.simtel_path == []


def test_no_files_at_all_is_a_clear_error():
    with pytest.raises(ValueError, match="at least one gamma"):
        TriggerChain([], None)
    with pytest.raises(TypeError, match="simtel_path"):
        TriggerChain(123)


def test_opener_accepts_path_and_names_unsupported_files(data, tmp_path):
    g, _, d = data
    with FileOpenerCTAO(pathlib.Path(g)) as fo:
        assert fo.camera_name == "DigiCam_R0Alpha"
    with pytest.raises(ValueError, match=r"notes\.txt"):
        FileOpenerCTAO(str(d / "notes.txt"))


def test_dataset_and_folds_normalise_too(data):
    g, n, _ = data
    ds = TriggerDataset(g, n)
    assert ds.gamma_files == [g] and ds.nsb_files == [n]
    assert TriggerDataset(pathlib.Path(g), [pathlib.Path(n)]).gamma_files == [g]
    geom = TriggerChain([g], [n]).geom
    spec = {"gamma_events": 5, "nsb_events": 5, "conditions": ["m"]}
    folds = make_rotation_folds(geom, [spec], {"m": (g, n)})          # bare str per condition
    assert folds[0].gamma_files == [g] and folds[0].nsb_files == [n]
    folds = make_condition_folds(geom, {"m": (pathlib.Path(g), [n], 5, 5)})
    assert folds[0].gamma_files == [g]


def test_calibration_accepts_a_bare_path(data):
    from triggerkit import training
    g, n, _ = data
    chain = TriggerChain([g], [n])
    h = TDSCANBody(filters=1, eps_xy=1, eps_t=1).build(chain)
    tau, temp, sg, sn = training.calibrate_tau(chain, h["threshold"].input, g)
    assert sg.size > 0 and sn.size > 0
    with pytest.raises(ValueError, match="gamma_files"):
        training.calibrate_tau(chain, h["threshold"].input, [])
