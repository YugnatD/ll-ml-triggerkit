"""Numpy-only helpers: AUC, ROC and the shared threshold picking."""
import numpy as np
import pytest

from triggerkit.Statistics.metrics import (
    normalize_comparison,
    pick_threshold_from_scores,
    roc_auc_mann_whitney,
    roc_curve,
    threshold_candidates,
    trigger_rate_curve,
)


def _brute_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    d = pos[:, None] - neg[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def test_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 6, 200).astype(float)   # many ties on purpose
    neg = rng.integers(0, 6, 300).astype(float)
    assert roc_auc_mann_whitney(pos, neg) == pytest.approx(_brute_auc(pos, neg))


def test_auc_edge_cases():
    assert np.isnan(roc_auc_mann_whitney([], [1.0]))
    assert roc_auc_mann_whitney([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert roc_auc_mann_whitney([0.0, 1.0], [2.0, 3.0]) == 0.0
    assert roc_auc_mann_whitney([1.0], [1.0]) == 0.5


def test_roc_curve_area_equals_auc():
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1, 1, 500), rng.normal(0, 1, 500)
    fpr, tpr = roc_curve(pos, neg)
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert getattr(np, "trapezoid", getattr(np, "trapz", None))(tpr, fpr) == pytest.approx(roc_auc_mann_whitney(pos, neg), abs=1e-9)


def test_normalize_comparison():
    assert normalize_comparison(None) == "gt"
    assert normalize_comparison(">=") == "ge"
    assert normalize_comparison("GE") == "ge"
    assert normalize_comparison("whatever") == "gt"


@pytest.mark.parametrize("comparison", ["gt", "ge"])
def test_pick_threshold_reported_fraction_is_real(comparison):
    """The fraction returned must be exactly what the deploy-time rule gives."""
    rng = np.random.default_rng(2)
    scores = rng.integers(0, 8, 1000).astype(np.float32)   # heavily quantized
    for want in (0.0, 0.01, 0.1, 0.33, 0.5, 0.9, 1.0):
        tau, frac, mode = pick_threshold_from_scores(scores, want, comparison)
        fired = (scores >= tau) if comparison == "ge" else (scores > tau)
        assert fired.mean() == pytest.approx(frac)


def test_pick_threshold_tie_prefers_include_ties():
    # 0.5 reachable both as "> 1" and ">= 2"-style; include-ties candidate wins
    scores = np.array([0, 0, 1, 1], np.float32)
    tau, frac, mode = pick_threshold_from_scores(scores, 0.5, "gt")
    assert frac == 0.5 and mode == "score >= bin"
    assert (scores > tau).mean() == 0.5


def test_pick_threshold_empty():
    assert pick_threshold_from_scores([], 0.1) == (None, None, None)


def test_rate_curve_is_monotonic_and_consistent():
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 20, 500).astype(np.float32)
    thr, rate = trigger_rate_curve(scores, window_sec=2e-7, comparison="gt")
    assert np.all(np.diff(thr) >= 0)
    assert np.all(np.diff(rate) <= 1e-9)           # higher threshold, lower rate
    for t, r in list(zip(thr, rate))[:: max(1, len(thr) // 15)]:
        assert (scores > t).mean() / 2e-7 == pytest.approx(r)


def test_rate_curve_ge_matches_rule():
    scores = np.array([1, 1, 2, 3, 3, 3], np.float32)
    thr, rate = trigger_rate_curve(scores, 1.0, comparison="ge")
    for t, r in zip(thr, rate):
        assert (scores >= t).mean() == pytest.approx(r)


def test_candidates_shapes():
    taus, fr, modes = threshold_candidates(np.array([1, 2, 2, 5], np.float32))
    assert taus.shape == fr.shape == modes.shape == (6,)
