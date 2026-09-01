"""Cross-validation (leakage) report over per-fold statistics files.

The fold stat scripts (``stats_tdscan.py`` / ``stats_hexcnn.py`` with a
``FOLD_SPECS`` list) write ONE HDF5 per run holding every fold: a ``/folds``
group with per-fold counts/rate/efficiency, and a ``fold`` column in ``/events``.
Every fold is evaluated at the SAME frozen ``tau``. This script reads the
``/folds`` group and shows, per model (grouped by the stored trigger chain), how
the gamma efficiency and NSB rate move across folds.

Reading the result
-------------------
* Gamma efficiency depends only on the *gamma* transform (an exact camera
  rotation). A rotation is a physical symmetry, so an honest trigger's gamma
  efficiency is flat across folds. A drift => the model keyed on a specific
  orientation.
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

DEFAULT_FOLDERS = ["simu_sst1m_tel2_tdscan", "simu_sst1m_tel2_hexcnn"]
OUTPUT_DIR = "trigger_report"
N_SIGMA = 3.0   # a fold outside fold-0's band by more than this is flagged


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
        recs = []
        for i, name in enumerate(names):
            recs.append({
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
        for col, (key, err, label, ax_title) in enumerate([
            ("eff", "eff_err", "gamma efficiency", "Gamma efficiency vs fold"),
            ("rate_hz", "rate_err", "NSB rate [Hz]", "NSB rate vs fold"),
        ]):
            ax = axes[row][col]
            ax.errorbar(xs, [r[key] for r in folds], yerr=[r[err] for r in folds],
                        fmt="o-", capsize=4)
            ax.axhline(folds[0][key], ls="--", c="gray", alpha=0.6, label="fold-0")
            ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel(label)
            ax.set_title(f"{cam}: {ax_title}")
            ax.legend()
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
        folds.sort(key=lambda r: r["fold"])
        _summarize(folds)
        cam = folds[0]["camera"]
        print(f"\n=== {cam} | {len(folds)} folds ===")
        print(f"{'fold':<18}{'gamma_eff':>16}{'nsb_rate_hz':>18}")
        for r in folds:
            print(f"{r['fold']:<18}"
                  f"{r['eff']*100:>10.3f} +/- {r['eff_err']*100:<.3f}"
                  f"{r['rate_hz']:>12.1f} +/- {r['rate_err']:<.1f}")
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
