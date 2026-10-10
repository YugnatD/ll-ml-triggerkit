import h5py
import numpy as np
import pytest

from triggerkit.Statistics.H5StatsWriter import H5StatsWriter


def _cols(label, triggered):
    return {
        "label": np.asarray(label, np.uint8),
        "triggered": np.asarray(triggered, np.uint8),
        "n_pe": np.ones(len(label), np.float32),
    }


def test_counts_come_from_triggered_column(tmp_path):
    """A chain with no TrainableThreshold (OR chain) must still be counted."""
    p = str(tmp_path / "s.h5")
    w = H5StatsWriter(p, trigger_chain=[("or_merge", {})])
    a, b = w.begin_fold("a"), w.begin_fold("b")
    w.append(_cols([1, 1, 0, 0, 0], [1, 0, 1, 0, 0]), fold_idx=a)
    w.append(_cols([1, 1, 0, 0, 0], [1, 1, 0, 0, 0]), fold_idx=b)
    rate0 = w.close(window_sec=1e-6)

    with h5py.File(p) as f:
        np.testing.assert_allclose(f["folds/gamma_efficiency"][()], [0.5, 1.0])
        # fold a: 1 of 3 NSB fire -> 1/3 / 1us ; fold b: none
        np.testing.assert_allclose(f["folds/trigger_rate_hz"][()], [1 / 3 / 1e-6, 0.0])
        assert f.attrs["trigger_rate_hz"] == pytest.approx(rate0)
        assert list(f["events/fold"][()]) == [0] * 5 + [1] * 5
        assert f.attrs["window_sec"] == pytest.approx(1e-6)


def test_score_threshold_path_still_works(tmp_path):
    p = str(tmp_path / "s.h5")
    w = H5StatsWriter(p, trigger_chain=[("threshold", {})])
    w.f.attrs["pre_threshold_reference_threshold"] = 1.5
    w.begin_fold("all")
    w.append({"label": np.array([1, 0, 0], np.uint8),
              "pre_threshold_score": np.array([2.0, 1.0, 3.0], np.float32)})
    w.close(window_sec=1.0)
    with h5py.File(p) as f:
        assert f["folds/nsb_trig"][()][0] == 1 and f["folds/gamma_trig"][()][0] == 1


def test_column_length_mismatch_rejected(tmp_path):
    w = H5StatsWriter(str(tmp_path / "s.h5"), trigger_chain=[])
    with pytest.raises(ValueError):
        w.append({"label": np.zeros(3, np.uint8), "n_pe": np.zeros(2, np.float32)})
    w.close(window_sec=1.0)
