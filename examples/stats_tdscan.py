"""Compute SST-1M trigger statistics for a TDSCAN chain -- on triggerkit.

Port of the sandbox's ``evaluate_perf_tdscan.py`` onto the packaged API. The
chain itself -- eps_xy/eps_t, pinned ring weights, frozen tau -- is defined
ONCE in ``tdscan_chain.py`` (next to this file) and imported here, exactly
like the sandbox's own ``tdscan_chain.build_chain``: ``show_tdscan_chain.py``,
``train_tdscan.py`` and this script all call the SAME function instead of each
keeping its own copy of the same constants. This script only adds its own
concerns on top: the fold specs, the target-rate tau tuning, and writing the
per-event statistics HDF5 that the report generator (``stats_report.py``) reads.

Run it (quote the globs; repeat --condition for several NSB conditions, the
first one is the reference):

    python examples/stats_tdscan.py --condition medium "GAMMA_GLOB" "NSB_GLOB" [--output FOLDER] [--quick]

The output HDF5 lands in --output (default: simu_sst1m_tel2_tdscan).
"""

import os

import stats_common
import tdscan_chain
from triggerkit.augment import make_rotation_folds

# --- Config (mirrors evaluate_perf_tdscan.py) --------------------------------
BASE_NAME = "simu_cross"
TARGET_RATE_HZ = 50_000
SEED = 1337

# Cross-validation folds (leakage detector, see triggerkit.augment): each fold
# reindexes the gammas by an exact camera rotation and/or the NSB by a
# decorrelating reshuffle or time roll. tau is tuned once and frozen, so a
# metric shift across folds flags a model leaking on orientation, specific
# pixels or the pulse time. stats_common.STANDARD_FOLDS describes each key; it
# is shared with stats_patch7.py so the two runs compare fold by fold. All folds
# are written into ONE stats HDF5.
FOLD_ROWS = stats_common.STANDARD_FOLDS


def main():
    args = stats_common.argument_parser(__doc__.splitlines()[0], "simu_sst1m_tel2_tdscan").parse_args()
    conditions = stats_common.condition_files(args)
    gamma_files, nsb_files = next(iter(conditions.values()))  # reference condition
    output_folder = args.output

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
        N_event_esimate_threshold=stats_common.threshold_events(args),
        batch_size=1024,
        # same NSB sample compute_statistics evaluates below: the originals,
        # not the rolled copies find_threshold_for_target_rate defaults to
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
    )
    print(f"tau={tau}  predicted_rate={predicted_rate} Hz (frozen for all folds)")
    threshold_layer.tau.assign(tau)

    os.makedirs(output_folder, exist_ok=True)

    # All folds go into ONE statistics HDF5 (each event tagged with a `fold`
    # column, per-fold summaries in the /folds group). The same pixel permutation
    # drives the TDSCAN (pixel-list) chain here that drives the CNN grid chain in
    # stats_hexcnn.py.
    #
    # IMPORTANT with folds: keep the DATASET-level NSB augmentation OFF
    # (nsb_roll_copies=0, nsb_skip_original_events=False) so each NSB event is
    # yielded EXACTLY ONCE, untouched. The fold's own nsb_index is then the sole
    # NSB transform -- fold ("rolled", 50) means each NSB event rolled by 50 and
    # nothing else (no original kept, no extra dataset roll stacked on top). Every
    # fold is a full pass over the same events, so all folds have identical counts.
    specs = stats_common.fold_specs(FOLD_ROWS, list(conditions), args.quick)
    folds = make_rotation_folds(chain.geom, specs, conditions, seed=SEED)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=output_folder, batch_size=512,
        tel_id_only=1, nsb_roll_copies=0, nsb_skip_original_events=False,
        ignore_errors=False, folds=folds)
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
