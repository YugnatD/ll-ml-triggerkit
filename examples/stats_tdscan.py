"""Compute SST-1M trigger statistics for a TDSCAN chain -- on triggerkit.

Port of the sandbox's ``evaluate_perf_tdscan.py`` onto the packaged API. The
chain itself -- eps_xy/eps_t, pinned ring weights, frozen tau -- is defined
ONCE in ``tdscan_chain.py`` (next to this file) and imported here, exactly
like the sandbox's own ``tdscan_chain.build_chain``: ``show_tdscan_chain.py``,
``train_tdscan.py`` and this script all call the SAME function instead of each
keeping its own copy of the same constants. This script only adds its own
concerns on top: the fold specs, the target-rate tau tuning, and writing the
per-event statistics HDF5 that the report generator (``stats_report.py``) reads.

Run it:

    python examples/stats_tdscan.py GAMMA_GLOB NSB_GLOB [OUTPUT_FOLDER]

The output HDF5 lands in OUTPUT_FOLDER (default: simu_sst1m_tel2_tdscan).
"""

import glob
import os
import sys

import tdscan_chain
from triggerkit.augment import make_rotation_folds

# --- Config (mirrors evaluate_perf_tdscan.py) --------------------------------
BASE_NAME = "simu_cross"
TARGET_RATE_HZ = 50_000
SEED = 1337

TOTAL_GAMMAS_EVENTS = 1_000_000
TOTAL_NSB_EVENTS = 400_000

# Cross-validation folds (leakage detector -- see examples/stats_hexcnn.py and
# triggerkit.augment for the rationale). Each fold reindexes gamma by an exact
# camera-rotation symmetry and NSB by a decorrelating reshuffle; tau is tuned
# once and frozen, so a metric shift across folds flags a model leaking on
# orientation / specific pixels. All folds are written into ONE stats HDF5.
#
# How to define folds -- FOLD_SPECS is an arbitrary-length list of dicts, turned
# into augment.Fold objects by make_rotation_folds(chain.geom, FOLD_SPECS,
# seed=SEED). Every key is optional; an empty dict {} is the untouched reference
# fold. The key names are exactly the ones stored in each fold's `config` in the
# output HDF5, so the spec here and the file read the same. A typo raises.
#
#   gamma_deg        (0)          EXACT camera-rotation symmetry applied to the
#                                 gamma rows. This camera is 3-fold, so only
#                                 multiples of 120 (0/120/240) are valid; any
#                                 other angle raises loudly rather than silently
#                                 misplacing pixels.
#   gamma_time_shift (0)          circular roll, in samples, of the GAMMA
#                                 waveform along the time axis. Moves the pulse
#                                 in time to test whether the model leaked its
#                                 absolute temporal position. Keep it small
#                                 (2-5) so the pulse stays inside the 50-sample
#                                 window.
#   nsb_kind         ("original") pixel transform applied to the NSB rows:
#                                   "original" -> identity (no reshuffle)
#                                   "rolled"   -> circular roll of the pixel list
#                                   "shuffle"  -> random permutation
#   nsb_param        (None)       the parameter for nsb_kind: the roll shift for
#                                 "rolled" (default P//2), the seed for
#                                 "shuffle" (default SEED + row index). Ignored
#                                 by "original".
#   nsb_time_shift   (0)          circular roll, in samples, of the NSB waveform
#                                 along the time axis -- the counterpart of
#                                 gamma_time_shift.
#   name             (auto)       explicit fold name. Auto-derived otherwise as
#                                 rot<deg>_<kind><param>[_troll<g>][_ntroll<n>].
#
# On rolling in time: rolling the GAMMAS ALONE moves the signal while tau stays
# tuned on un-rolled NSB, so the two classes are no longer treated alike and the
# efficiency change mixes real temporal leakage with the trigger's own response
# to the window edges. Set gamma_time_shift AND nsb_time_shift to the same value
# for the fair test -- a time-translation-invariant trigger then returns the same
# gamma efficiency AND the same NSB rate as the reference fold.
#
# Add / remove / reorder rows freely. The FIRST row is the reference fold: the
# report compares every other fold against it, and its counts become the file's
# top-level attrs. Set FOLD_SPECS = None to run a single plain (fold-free) pass.
FOLD_SPECS = [
    # --- rotation symmetry ---------------------------------------------------
    {},                                                       # reference fold
    {"gamma_deg": 120},
    {"gamma_deg": 240},
    # --- NSB pixel transforms ------------------------------------------------
    {"nsb_kind": "rolled",  "nsb_param": 1},
    {"nsb_kind": "rolled",  "nsb_param": 42},
    {"nsb_kind": "shuffle", "nsb_param": 2024},
    # --- mixes ---------------------------------------------------------------
    {"gamma_deg": 120, "nsb_kind": "rolled",  "nsb_param": 1},
    {"gamma_deg": 240, "nsb_kind": "rolled",  "nsb_param": 1},
    {"gamma_deg": 120, "nsb_kind": "shuffle", "nsb_param": 2024},
    {"gamma_deg": 240, "nsb_kind": "shuffle", "nsb_param": 2024},
    # --- temporal position: gammas only (the UNFAIR half, kept for reference) -
    {"gamma_time_shift": 2},
    {"gamma_time_shift": 5},
    # --- temporal position: BOTH classes rolled (the fair test) ---------------
    {"gamma_time_shift": 2, "nsb_time_shift": 2},
    {"gamma_time_shift": 5, "nsb_time_shift": 5},
    # --- temporal position: NSB only (isolates the noise side) ---------------
    {"nsb_time_shift": 5, "name": "nsb_only_troll5"},
    {"nsb_time_shift": 2, "name": "nsb_only_troll2"},
]

n_folds = len(FOLD_SPECS)

# Per-fold event caps. Each fold is a full independent pass, so these bound the
# gamma / NSB events processed IN EACH FOLD (not the total across folds). None =
# use everything available. Example: MAX_GAMMA_EVENTS=50_000 + MAX_NSB_EVENTS=
# 100_000 -> every fold sees at most 50k gammas and 100k NSB. Gamma count is
# after the tel_id_only filter; NSB count is source events (augmentation off).
MAX_GAMMA_EVENTS = TOTAL_GAMMAS_EVENTS // n_folds
MAX_NSB_EVENTS = TOTAL_NSB_EVENTS // n_folds

def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB NSB_GLOB [OUTPUT_FOLDER]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2]))
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "simu_sst1m_tel2_tdscan"

    if not gamma_files or not nsb_files:
        sys.exit("no gamma or NSB files matched the given globs.")
    print(f"Found {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")

    # --- The deployed filters=1 TDSCAN chain, shared with show_tdscan_chain.py
    # and train_tdscan.py via tdscan_chain.build_chain (see that module for the
    # actual eps_xy/eps_t/ring_weights/tau knobs). tau=None here: this script
    # re-tunes it below to the target NSB rate rather than keeping the pinned one.
    chain, tdscan_layer, threshold_layer = tdscan_chain.build_chain(
        gamma_files, nsb_files, tau=None)

    # Tune tau ONCE on the nominal NSB and freeze it for every fold, so a rate
    # drift across folds is a real signal rather than a re-tuning artefact.
    tau, predicted_rate = chain.find_threshold_for_target_rate(
        target_rate_hz=TARGET_RATE_HZ,
        tolerance_hz=2,
        N_event_esimate_threshold=25_000,
        batch_size=1024,
    )
    print(f"tau={tau}  predicted_rate={predicted_rate} Hz (frozen for all folds)")
    threshold_layer.tau.assign(tau)

    os.makedirs(output_folder, exist_ok=True)

    # All folds go into ONE statistics HDF5 (each event tagged with a `fold`
    # column, per-fold summaries in the /folds group). The same pixel permutation
    # drives the TDSCAN (pixel-list) chain here that drives the CNN grid chain in
    # stats_hexcnn.py. FOLD_SPECS=None -> a single fold-free pass.
    #
    # IMPORTANT with folds: keep the DATASET-level NSB augmentation OFF
    # (nsb_roll_copies=0, nsb_skip_original_events=False) so each NSB event is
    # yielded EXACTLY ONCE, untouched. The fold's own nsb_index is then the sole
    # NSB transform -- fold ("rolled", 50) means each NSB event rolled by 50 and
    # nothing else (no original kept, no extra dataset roll stacked on top). Every
    # fold is a full pass over the same events, so all folds have identical counts.
    folds = None if FOLD_SPECS is None else make_rotation_folds(chain.geom, FOLD_SPECS, seed=SEED)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=output_folder, batch_size=512,
        tel_id_only=1, nsb_roll_copies=0, nsb_skip_original_events=False,
        ignore_errors=False, folds=folds,
        max_gamma_events=MAX_GAMMA_EVENTS, max_nsb_events=MAX_NSB_EVENTS)
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
