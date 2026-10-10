"""Compute SST-1M trigger statistics for the REAL patch7 telescope trigger.

This is the deployed hardware trigger with the research TDSCAN filter removed:

    [fadc?] -> [subtract?] -> digital_sum(patch7) -> camera-wide max-pool
            -> trainable threshold

i.e. the DigiCam digital-sum trigger. Each 7-pixel patch is summed per time
slice, the camera-wide maximum over all patches and all time slices is taken,
and the event fires when that maximum crosses ``tau``. There is NO learned
layer -- ``tau`` is the only free parameter, tuned once to the target NSB rate
and frozen. Same threshold-tuning / statistics / fold plumbing as
``stats_tdscan.py``; only the body differs (digital_sum instead of tdscan).

Run it (quote the globs; repeat --condition for several NSB conditions, the
first one is the reference):

    python examples/stats_patch7.py --condition medium "GAMMA_GLOB" "NSB_GLOB" [--output FOLDER] [--quick]

The output HDF5 lands in --output (default: simu_sst1m_tel2_patch7).
"""

import os

import stats_common
from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds

# --- Config ------------------------------------------------------------------
BASE_NAME = "simu_cross"
TARGET_RATE_HZ = 50_000
SEED = 1337

# Threshold seed / sharpness. tau is retuned to TARGET_RATE_HZ below; TAU_INIT is
# only the starting point. binary_output=True -> hard fire/no-fire decision.
TAU_INIT = 10.0
TAU_TEMP = 10.0

# Optional front-end stages before the digital sum (both off = the plain patch7
# trigger on the raw waveform). Set FADC=True to add the shared FADC baseline
# front-end; set SUBTRACT_VALUE to a scalar to subtract a pedestal before summing.
FADC = False
SUBTRACT_VALUE = None

# Cross-validation folds: stats_common.STANDARD_FOLDS, identical to
# stats_tdscan.py so the two runs compare fold by fold. patch7 learns nothing,
# so they are a pure consistency check: gamma efficiency must not change under
# a camera rotation, nor the NSB rate under an NSB reshuffle or time roll.
FOLD_ROWS = stats_common.STANDARD_FOLDS


def main():
    args = stats_common.argument_parser(__doc__.splitlines()[0], "simu_sst1m_tel2_patch7").parse_args()
    conditions = stats_common.condition_files(args)
    gamma_files, nsb_files = next(iter(conditions.values()))  # reference condition
    output_folder = args.output

    # --- The real patch7 trigger, built directly on the chain (no body) ------
    # digital_sum(patch7) needs no learned weights, so there is nothing to build
    # a dedicated body for -- three add_stage calls are the whole trigger.
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    if FADC:
        chain.add_stage("fadc")
    if SUBTRACT_VALUE is not None:
        chain.add_stage("shift", value=SUBTRACT_VALUE)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    threshold_layer = chain.add_stage(
        "threshold", init_tau=TAU_INIT, temp=TAU_TEMP, binary_output=True)

    # find_threshold_for_target_rate needs chain.model to locate the threshold.
    chain.compile_chain()

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
    # column, per-fold summaries in the /folds group). The pixel permutation is
    # applied to the raw waveform + pedestal before the digital sum, so folds
    # work here exactly as for TDSCAN.
    #
    # DATASET-level NSB augmentation stays OFF (nsb_roll_copies=0,
    # nsb_skip_original_events=False): each NSB event is yielded exactly once,
    # untouched, so the fold's own nsb_index is the ONLY NSB transform.
    specs = stats_common.fold_specs(FOLD_ROWS, list(conditions), args.quick)
    folds = make_rotation_folds(chain.geom, specs, conditions, seed=SEED)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=output_folder, batch_size=512,
        tel_id_only=1, nsb_roll_copies=0, nsb_skip_original_events=False,
        ignore_errors=False, folds=folds)
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
