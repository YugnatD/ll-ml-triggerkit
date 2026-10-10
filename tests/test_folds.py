import importlib.util
import pathlib

import numpy as np
import pytest

from triggerkit.Statistics.folds import auc_standard_error, read_folds_group
from triggerkit.Statistics.H5StatsWriter import H5StatsWriter
from triggerkit.Statistics.metrics import roc_auc_mann_whitney, wilson


def write(path, window=2e-7):
    rng = np.random.default_rng(0)
    w = H5StatsWriter(str(path), trigger_chain=[("threshold", {"threshold": 2.0})], camera_name="DigiCam")
    w.f.attrs["has_pre_threshold_score"] = True
    scores = {}
    for i, name in enumerate(("a", "b", "c")):
        idx = w.begin_fold(name)
        n = 300
        g = rng.normal(2 + i, 1, n).astype(np.float32)
        b = rng.normal(0, 1, n).astype(np.float32)
        scores[name] = (g, b)
        w.append({"label": np.r_[np.ones(n), np.zeros(n)].astype(np.uint8),
                  "triggered": (np.r_[g, b] > 2).astype(np.uint8),
                  "pre_threshold_score": np.r_[g, b].astype(np.float32),
                  "n_pe": np.ones(2 * n, np.float32)}, fold_idx=idx)
    w.close(window_sec=window)
    return scores


def test_read_folds_group(tmp_path):
    p = tmp_path / "s.h5"
    scores = write(p)
    folds = read_folds_group(str(p))
    assert [f["fold"] for f in folds] == ["a", "b", "c"] and [f["index"] for f in folds] == [0, 1, 2]
    for f in folds:
        g, b = scores[f["fold"]]
        assert f["gamma_total"] == f["nsb_total"] == 300
        assert f["gamma_trig"] == int((g > 2).sum()) and f["nsb_trig"] == int((b > 2).sum())
        assert f["eff"] == pytest.approx((g > 2).mean())
        assert f["rate_hz"] == pytest.approx((b > 2).mean() / 2e-7)
        assert f["auc"] == pytest.approx(roc_auc_mann_whitney(g, b))
        assert f["window_sec"] == pytest.approx(2e-7)
        _, lo, hi = wilson(f["nsb_trig"], f["nsb_total"], 0.68)
        assert f["rate_err"] == pytest.approx(max(lo, hi) / 2e-7)
    assert folds[0]["auc"] < folds[1]["auc"] < folds[2]["auc"]


def test_file_without_folds_group_returns_none(tmp_path):
    import h5py
    p = tmp_path / "plain.h5"
    with h5py.File(p, "w") as f:
        f.create_group("events")
    assert read_folds_group(str(p)) is None


def test_auc_standard_error_behaviour():
    assert np.isnan(auc_standard_error(float("nan"), 10, 10))
    assert np.isnan(auc_standard_error(0.7, 0, 10))
    small, large = auc_standard_error(0.7, 50, 50), auc_standard_error(0.7, 5000, 5000)
    assert 0 < large < small
    assert auc_standard_error(1.0, 100, 100) == pytest.approx(0.0, abs=1e-12)


def test_cv_example_uses_the_library_reader(tmp_path):
    path = pathlib.Path(__file__).resolve().parent.parent / "examples" / "stats_cv_report.py"
    spec = importlib.util.spec_from_file_location("stats_cv_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / "s.h5"
    write(p)
    recs = mod._read_file(str(p))
    assert len(recs) == 3 and all(r["camera"] == "DigiCam" and "threshold" in r["chain"] for r in recs)
    assert {"eff", "eff_err", "rate_hz", "rate_err", "auc", "auc_err", "path"} <= set(recs[0])
    groups = mod._collect([str(tmp_path)])
    assert len(groups) == 1
