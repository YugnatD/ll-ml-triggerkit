"""Cross-validation (leakage) report over per-fold statistics files.

The fold stat scripts (``stats_tdscan.py`` / ``stats_hexcnn.py`` with a
``FOLD_SPECS`` list) write ONE HDF5 per run holding every fold: a ``/folds``
group with per-fold counts/rate/efficiency, and a ``fold`` column in ``/events``.
Every fold is evaluated at the SAME frozen ``tau``. This script reads the
``/folds`` group and shows, per model (grouped by the stored trigger chain), how
the separation power (AUC) and the NSB rate move across folds.

Reading the result
-------------------
* AUC is the headline metric because it is THRESHOLD-FREE: it is
  P(score_gamma > score_NSB) over all pairs (Wilcoxon-Mann-Whitney), computed by
  the library's own ``Statistics.metrics.roc_auc_mann_whitney`` on the stored
  per-event ``pre_threshold_score``. Gamma efficiency at a frozen tau conflates
  two different things -- a fold that shifts the NSB score distribution forces
  the working point to move, so the efficiency drops even when the trigger
  separates the classes exactly as well. AUC does not care where tau sits, so it
  compares folds that are not at the same operating point. An honest trigger's
  AUC is flat across every fold; a drift is real loss (or gain) of separation.
* Gamma efficiency (still printed in the table) depends only on the *gamma*
  transform at the frozen tau. A rotation is a physical symmetry, so it should be
  flat; a drift => the model keyed on a specific orientation.
* NSB rate depends only on the *NSB* transform (roll / shuffle) at the frozen
  tau. NSB is spatially ~iid, so a reshuffle is another valid draw and the rate
  should be flat. A drift => the model keyed on specific pixels / spatial NSB
  structure (e.g. a pedestal footprint).

Caveat: folds reuse the SAME events (just reindexed), so the per-fold estimates
are correlated -- the Wilson bars below are single-sample bands, not the spread
of independent draws. The verdict flags a fold whose metric leaves fold-0's band
by more than ``N_SIGMA``; treat it as a screen, not a hypothesis test. The
stronger test (per-event decision-flip rate between folds, joined on event_id)
is noted in the code as future work.

Run it:

    python examples/stats_cv_report.py [FOLDER ...]

Default scans the TDSCAN and hex-CNN fold folders.
"""

import os
import sys

import h5py
import numpy as np

from triggerkit.Statistics.StatPlotter import wilson
from triggerkit.Statistics.metrics import roc_auc_mann_whitney

DEFAULT_FOLDERS = ["simu_sst1m_tel2_tdscan", "simu_sst1m_tel2_hexcnn"]
OUTPUT_DIR = "trigger_report"
N_SIGMA = 3.0   # a fold outside fold-0's band by more than this is flagged


def _auc_se(auc, n_pos, n_neg):
    """Hanley & McNeil standard error of an AUC (exponential-distribution form)."""
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _fold_aucs(f, n_folds):
    """Per-fold AUC + standard error from the per-event pre-threshold scores.

    Returns a list of (auc, se) of length n_folds, all NaN when the file stores
    no pre-threshold score (then the report falls back to gamma efficiency).
    """
    out = [(float("nan"), float("nan"))] * n_folds
    if "events" not in f or not bool(f.attrs.get("has_pre_threshold_score", False)):
        return out
    ev = f["events"]
    if "pre_threshold_score" not in ev:
        return out
    fold = ev["fold"][()]
    label = ev["label"][()]
    score = ev["pre_threshold_score"][()]
    for i in range(n_folds):
        m = fold == i
        pos = score[m & (label == 1)]
        neg = score[m & (label == 0)]
        auc = roc_auc_mann_whitney(pos, neg)
        out[i] = (auc, _auc_se(auc, pos.size, neg.size))
    return out


def _read_file(path):
    """Return the list of per-fold summary dicts stored in one stats HDF5.

    Each statistics run now writes ALL its folds into one file: a `/folds` group
    holds parallel arrays (name, counts, rate, efficiency) indexed by the `fold`
    column in /events. Returns [] for a file with no `/folds` group (e.g. an old
    single-fold file written before this layout)."""
    def s(x):
        return x.decode() if isinstance(x, bytes) else str(x)
    with h5py.File(path, "r") as f:
        if "folds" not in f:
            return []
        g = f["folds"]
        a = f.attrs
        chain = s(a.get("trigger_chain_json", "?"))
        camera = s(a.get("camera_name", "?"))
        window_sec = float(a.get("window_sec", 75e-9))
        names = [s(x) for x in g["name"][()]]
        gt = g["gamma_trig"][()]; gtot = g["gamma_total"][()]
        nt = g["nsb_trig"][()];   ntot = g["nsb_total"][()]
        rate = g["trigger_rate_hz"][()]
        aucs = _fold_aucs(f, len(names))
        recs = []
        for i, name in enumerate(names):
            recs.append({
                "index": i,
                "auc": aucs[i][0],
                "auc_err": aucs[i][1],
                "path": path,
                "fold": name,
                "chain": chain,
                "camera": camera,
                "window_sec": window_sec,
                "gamma_trig": int(gt[i]),
                "gamma_total": int(gtot[i]),
                "nsb_trig": int(nt[i]),
                "nsb_total": int(ntot[i]),
                "rate_hz": float(rate[i]),
            })
        return recs


def _collect(folders):
    """Group per-fold summaries by their trigger chain (one group per model)."""
    groups = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not (fn.endswith(".h5") or fn.endswith(".hdf5")):
                continue
            for rec in _read_file(os.path.join(folder, fn)):
                groups.setdefault(rec["chain"], []).append(rec)
    return groups


def _summarize(folds):
    """Attach efficiency and rate (value + Wilson error) to each fold."""
    for r in folds:
        eff, elo, ehi = wilson(r["gamma_trig"], r["gamma_total"])
        r["eff"], r["eff_err"] = eff, max(elo, ehi)
        # Rate error from the NSB Wilson band scaled by 1/window.
        _, rlo, rhi = wilson(r["nsb_trig"], r["nsb_total"])
        r["rate_err"] = max(rlo, rhi) / r["window_sec"]
    return folds


def _verdict(folds, key, err_key):
    """Flag folds whose metric leaves fold-0's +/- N_SIGMA band."""
    ref = folds[0]
    band = N_SIGMA * ref[err_key]
    flagged = [r for r in folds[1:]
               if abs(r[key] - ref[key]) > band + N_SIGMA * r[err_key]]
    return flagged


def _plot(groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(groups)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    for row, (chain, folds) in enumerate(groups.items()):
        names = [r["fold"] for r in folds]
        xs = np.arange(len(folds))
        cam = folds[0]["camera"]
        # AUC is threshold-free, so it is the honest cross-fold metric. Fall back
        # to gamma efficiency only for files written without per-event scores.
        has_auc = np.isfinite([r["auc"] for r in folds]).any()
        first = (("auc", "auc_err", "AUC (gamma vs NSB)",
                  "Separation power (AUC) vs fold -- threshold-free")
                 if has_auc else
                 ("eff", "eff_err", "gamma efficiency",
                  "Gamma efficiency vs fold (no per-event score stored)"))
        for col, (key, err, label, ax_title) in enumerate([
            first,
            ("rate_hz", "rate_err", "NSB rate [Hz]", "NSB rate vs fold (frozen tau)"),
        ]):
            ax = axes[row][col]
            ax.errorbar(xs, [r[key] for r in folds], yerr=[r[err] for r in folds],
                        fmt="o-", capsize=4)
            ax.axhline(folds[0][key], ls="--", c="gray", alpha=0.6, label="fold-0")
            ax.set_xticks(xs); ax.set_xticklabels(names, rotation=35, ha="right",
                                                 fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(f"{cam}: {ax_title}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Wrote plot {path}")


def main():
    folders = sys.argv[1:] or DEFAULT_FOLDERS
    groups = _collect(folders)
    if not groups:
        sys.exit(f"no stats files with a '/folds' group under {folders}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    any_leak = False
    for chain, folds in groups.items():
        # Sort by the fold's declaration index, NOT its name: folds[0] is the
        # reference fold every verdict is measured against.
        folds.sort(key=lambda r: (r["path"], r["index"]))
        _summarize(folds)
        cam = folds[0]["camera"]
        print(f"\n=== {cam} | {len(folds)} folds ===")
        print(f"{'fold':<32}{'AUC':>18}{'gamma_eff':>16}{'nsb_rate_hz':>18}")
        for r in folds:
            auc_s = ("     nan          " if not np.isfinite(r["auc"])
                     else f"{r['auc']:>10.5f} +/- {r['auc_err']:<.5f}")
            print(f"{r['fold']:<32}{auc_s}"
                  f"{r['eff']*100:>10.3f} +/- {r['eff_err']*100:<.3f}"
                  f"{r['rate_hz']:>12.1f} +/- {r['rate_err']:<.1f}")
        if np.isfinite([r["auc"] for r in folds]).all():
            auc_flag = _verdict(folds, "auc", "auc_err")
            if auc_flag:
                any_leak = True
                print(f"  SEPARATION MOVED (AUC): {[r['fold'] for r in auc_flag]} "
                      f"differ from fold-0 by > {N_SIGMA} sigma -> real change in "
                      f"class separation, not just a working-point shift.")
        eff_flag = _verdict(folds, "eff", "eff_err")
        rate_flag = _verdict(folds, "rate_hz", "rate_err")
        if eff_flag:
            any_leak = True
            print(f"  LEAK SUSPECTED (efficiency): {[r['fold'] for r in eff_flag]} "
                  f"differ from fold-0 by > {N_SIGMA} sigma -> orientation leakage?")
        if rate_flag:
            any_leak = True
            print(f"  LEAK SUSPECTED (rate): {[r['fold'] for r in rate_flag]} "
                  f"differ from fold-0 by > {N_SIGMA} sigma -> pixel/pedestal leakage?")
        if not eff_flag and not rate_flag:
            print(f"  STABLE across folds (within {N_SIGMA} sigma).")

    _plot(groups, os.path.join(OUTPUT_DIR, "cross_validation.png"))
    print("\nOverall:", "LEAK SUSPECTED in >=1 model" if any_leak else "all models STABLE")


if __name__ == "__main__":
    main()
