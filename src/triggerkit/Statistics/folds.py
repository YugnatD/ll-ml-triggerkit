"""Read the per-fold summary of a statistics HDF5 (``/folds`` + the ``fold`` column).

One implementation shared by ``StatPlotter`` (cross-validation plot) and
``examples/stats_cv_report.py``, so the two can never disagree on how an AUC,
an efficiency or its error is computed. numpy + h5py only (no TensorFlow).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from triggerkit.Statistics.metrics import roc_auc_mann_whitney, wilson


def _s(x) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil standard error of an AUC."""
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def fold_aucs(f: "h5py.File", n_folds: int) -> List[Tuple[float, float]]:
    """Per-fold ``(auc, standard_error)`` from the stored per-event scores.

    All-NaN when the file stores no per-event score (the callers then fall back
    to gamma efficiency).
    """
    out = [(float("nan"), float("nan"))] * n_folds
    grp = f.get("events")
    if grp is None or "pre_threshold_score" not in grp or "fold" not in grp:
        return out
    fold = np.asarray(grp["fold"][()]).reshape(-1)
    label = np.asarray(grp["label"][()]).reshape(-1)
    score = np.asarray(grp["pre_threshold_score"][()]).reshape(-1)
    for i in range(n_folds):
        m = fold == i
        pos = score[m & (label == 1)]
        neg = score[m & (label == 0)]
        auc = roc_auc_mann_whitney(pos, neg)
        out[i] = (auc, auc_standard_error(auc, pos.size, neg.size))
    return out


def read_folds_group(h5_path: str, wilson_level: float = 0.68,
                     default_window_sec: float = 75e-9) -> Optional[List[Dict[str, Any]]]:
    """The ``/folds`` summary table of one stats file, one dict per fold.

    Returns None when the file has no ``/folds`` group (a plain single-fold file).
    Each dict holds the fold ``index`` and ``fold`` name, the raw counts
    (``gamma_trig`` / ``gamma_total`` / ``nsb_trig`` / ``nsb_total``), the
    efficiency ``eff`` and NSB ``rate_hz`` with their Wilson errors
    (``eff_err``, ``rate_err``), the ``window_sec`` and the threshold-free
    ``auc`` (+ ``auc_err``), which the CV plots prefer: gamma efficiency at a
    frozen tau conflates separation power with the working point.
    """
    with h5py.File(h5_path, "r") as f:
        if "folds" not in f:
            return None
        g = f["folds"]
        if "window_sec" in f.attrs:
            window_sec = float(f.attrs["window_sec"])
        elif "window_size_ns" in f.attrs:
            window_sec = float(f.attrs["window_size_ns"]) * 1e-9
        else:
            window_sec = default_window_sec
        names = [_s(x) for x in g["name"][()]]
        gt, gtot = g["gamma_trig"][()], g["gamma_total"][()]
        nt, ntot = g["nsb_trig"][()], g["nsb_total"][()]
        rate = g["trigger_rate_hz"][()]
        aucs = fold_aucs(f, len(names))
        folds = []
        for i, name in enumerate(names):
            eff, elo, ehi = wilson(int(gt[i]), int(gtot[i]), wilson_level)
            _, rlo, rhi = wilson(int(nt[i]), int(ntot[i]), wilson_level)
            folds.append({
                "index": i, "fold": name, "window_sec": window_sec,
                "gamma_trig": int(gt[i]), "gamma_total": int(gtot[i]),
                "nsb_trig": int(nt[i]), "nsb_total": int(ntot[i]),
                "eff": eff, "eff_err": max(elo, ehi),
                "auc": aucs[i][0], "auc_err": aucs[i][1],
                "rate_hz": float(rate[i]),
                "rate_err": max(rlo, rhi) / window_sec,
            })
        return folds
