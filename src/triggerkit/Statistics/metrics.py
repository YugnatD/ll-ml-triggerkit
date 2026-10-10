"""Shared metric definitions used across training, simulation and plotting.

Keeping the AUC in one place guarantees the live simulation readout, the stats
report, and the training metric all mean the same thing: the gamma-vs-NSB ROC
AUC (Wilcoxon-Mann-Whitney), i.e. the probability that a random gamma scores
higher than a random NSB event, with ties counted as 0.5.

This module is intentionally numpy/scipy-only (no TensorFlow) so it can be imported
from the plotting path without pulling in heavy dependencies. It matches the
definition of ``train_utils.PairwiseAUCMetric``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks (1-based), like scipy.stats.rankdata(method='average')."""
    order = np.argsort(values, kind="mergesort")
    sorted_v = values[order]
    n = values.size
    sequential = np.arange(1, n + 1, dtype=np.float64)
    # group id increments whenever the sorted value changes (ties share a group)
    same_as_prev = np.empty(n, dtype=bool)
    same_as_prev[0] = False
    np.not_equal(sorted_v[1:], sorted_v[:-1], out=same_as_prev[1:])
    group = np.cumsum(same_as_prev)  # 0..G-1, ties share a group
    sums = np.bincount(group, weights=sequential)
    counts = np.bincount(group)
    avg_by_group = sums / counts
    ranks_sorted = avg_by_group[group]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc_mann_whitney(pos_scores, neg_scores) -> float:
    """Exact ROC AUC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg).

    Parameters
    ----------
    pos_scores : array-like
        Scores of the positive class (gamma).
    neg_scores : array-like
        Scores of the negative class (NSB).

    Returns
    -------
    float
        AUC in [0, 1] (0.5 = random), or NaN if either class is empty.
    """
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n_pos = pos.size
    n_neg = neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([pos, neg]))
    sum_ranks_pos = float(ranks[:n_pos].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(pos_scores, neg_scores):
    """ROC curve points for a threshold sweep (score > tau triggers).

    Returns ``(fpr, tpr)`` arrays starting at (0, 0) and ending at (1, 1):
    TPR = fraction of positives (gamma) above threshold, FPR = fraction of
    negatives (NSB) above threshold. The area under this curve equals
    :func:`roc_auc_mann_whitney`. Ties are handled by collapsing equal scores.
    """
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n_pos = pos.size
    n_neg = neg.size
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(scores, kind="mergesort")[::-1]  # high score -> low
    scores = scores[order]
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1.0 - labels)
    # keep only the last point of each run of equal scores (proper tie handling)
    keep = np.r_[np.diff(scores) != 0, True]
    tpr = np.r_[0.0, tps[keep] / n_pos]
    fpr = np.r_[0.0, fps[keep] / n_neg]
    return fpr, tpr


# ---------------------------------------------------------------------------
# Threshold helpers (single source of truth: used by TriggerChain, StatPlotter)
# ---------------------------------------------------------------------------

_COMPARISON_ALIASES = {
    ">": "gt", "gt": "gt", "strict": "gt", "score > tau": "gt",
    ">=": "ge", "ge": "ge", "inclusive": "ge", "score >= tau": "ge",
}


def normalize_comparison(comparison) -> str:
    """``"gt"`` or ``"ge"`` from the accepted aliases (unknown / None -> ``"gt"``)."""
    if comparison is None:
        return "gt"
    return _COMPARISON_ALIASES.get(str(comparison).strip().lower(), "gt")


def threshold_candidates(scores, comparison="gt"):
    """All distinct achievable operating points of a ``score > tau`` / ``>=`` rule.

    Returns ``(taus, fractions, modes)`` (concatenated "strict" and
    "include ties" candidates): ``fractions[i]`` is the fraction of ``scores``
    that fire with threshold ``taus[i]``; ``modes[i]`` is 1 for the
    include-ties candidates, else 0. ``scores`` must be non-empty.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    comparison = normalize_comparison(comparison)
    unique_scores, counts = np.unique(scores, return_counts=True)
    counts = counts.astype(np.int64)
    total = int(scores.size)

    counts_gt = total - np.cumsum(counts, dtype=np.int64)
    counts_ge = counts_gt + counts

    if comparison == "ge":
        tau_strict = np.nextafter(unique_scores, np.float32(np.inf))
        tau_include_ties = unique_scores
    else:
        tau_strict = unique_scores
        tau_include_ties = np.nextafter(unique_scores, np.float32(-np.inf))

    taus = np.concatenate([tau_strict, tau_include_ties])
    fractions = np.concatenate([counts_gt / total, counts_ge / total]).astype(np.float64)
    modes = np.concatenate([np.zeros_like(counts_gt, dtype=np.uint8),
                            np.ones_like(counts_ge, dtype=np.uint8)])
    return taus, fractions, modes


def pick_threshold_from_scores(scores, desired_fraction, comparison="gt"):
    """Threshold whose firing fraction on ``scores`` is closest to ``desired_fraction``.

    Returns ``(tau, achieved_fraction, mode)`` where ``mode`` is
    ``"score > bin"`` / ``"score >= bin"``, or ``(None, None, None)`` for empty
    input. On an exact tie between a strict and an include-ties candidate the
    lower tau (include-ties) wins, so a quantized score never undershoots.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return None, None, None
    desired_fraction = float(np.clip(desired_fraction, 0.0, 1.0))
    taus, fractions, modes = threshold_candidates(scores, comparison)

    errors = np.abs(fractions - desired_fraction)
    best_indices = np.flatnonzero(np.isclose(errors, float(errors.min()), rtol=0.0, atol=1e-12))
    include_ties = best_indices[modes[best_indices] == 1]
    best = int(include_ties[0] if include_ties.size > 0 else best_indices[0])
    mode = "score >= bin" if modes[best] == 1 else "score > bin"
    return float(taus[best]), float(fractions[best]), mode


def trigger_rate_curve(scores, window_sec, comparison="gt"):
    """``(thresholds, rates_hz)`` of a score distribution, sorted by threshold."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float64)
    taus, fractions, modes = threshold_candidates(scores, comparison)
    # the curve lists "include ties" points first, then "strict" ones
    n = taus.size // 2
    thresholds = np.concatenate([taus[n:], taus[:n]])
    rates = np.concatenate([fractions[n:], fractions[:n]]) / float(window_sec)
    order = np.argsort(thresholds, kind="mergesort")
    return thresholds[order], rates[order]


def wilson(passed, total, level=0.68):
    #implementation as of: https://root.cern.ch/doc/master/TEfficiency_8cxx_source.html#l03837
    alpha = (1.0 - level) / 2.0
    if total == 0:
        return np.nan,np.nan,np.nan

    average = passed / total
    kappa = norm.ppf(1.0 - alpha)

    mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa)

    delta = (kappa / (total + kappa * kappa)) * np.sqrt(
        total * average * (1.0 - average) + (kappa * kappa) / 4.0
    )

    low = max(0.0, mode - delta)
    high = min(1.0, mode + delta)
    return average, average - low, high - average
