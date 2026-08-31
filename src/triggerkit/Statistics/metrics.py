"""Shared metric definitions used across training, simulation and plotting.

Keeping the AUC in one place guarantees the live simulation readout, the stats
report, and the training metric all mean the same thing: the gamma-vs-NSB ROC
AUC (Wilcoxon-Mann-Whitney), i.e. the probability that a random gamma scores
higher than a random NSB event, with ties counted as 0.5.

This module is intentionally numpy-only (no TensorFlow) so it can be imported
from the plotting path without pulling in heavy dependencies. It matches the
definition of ``train_utils.PairwiseAUCMetric``.
"""

from __future__ import annotations

import numpy as np


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
