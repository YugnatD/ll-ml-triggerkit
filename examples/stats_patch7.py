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

Run it:

    python examples/stats_patch7.py GAMMA_GLOB NSB_GLOB [OUTPUT_FOLDER]

The output HDF5 lands in OUTPUT_FOLDER (default: simu_sst1m_tel2_patch7).

Note: DigitalSumChannelList reads ConfigFile_SST1M/CTA_SST1M_Pixels_info_trigger.csv
via a RELATIVE path, so run this from a directory where that file is reachable
(same requirement as the sandbox).
"""

import glob
import os
import sys

from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds

# --- Config ------------------------------------------------------------------
BASE_NAME = "simu"
TARGET_RATE_HZ = 50_000
SEED = 1337

TOTAL_GAMMAS_EVENTS = 400_000
TOTAL_NSB_EVENTS = 200_000

# Threshold seed / sharpness. tau is retuned to TARGET_RATE_HZ below; TAU_INIT is
# only the starting point. binary_output=True -> hard fire/no-fire decision.
TAU_INIT = 10.0
TAU_TEMP = 10.0

# Optional front-end stages before the digital sum (both off = the plain patch7
# trigger on the raw waveform). Set FADC=True to add the shared FADC baseline
# front-end; set SUBTRACT_VALUE to a scalar to subtract a pedestal before summing.
FADC = False
SUBTRACT_VALUE = None

# Cross-validation folds (leakage detector -- see triggerkit.augment). For the
# patch7 trigger there is nothing learned, so folds are a pure consistency check:
# gamma efficiency MUST be invariant under a camera-rotation symmetry, and the
# NSB rate MUST be invariant under an NSB reshuffle. Any drift here is a bug in
# the geometry / patch map, not model leakage. Set FOLD_SPECS = None to run a
# single plain (fold-free) pass. See stats_tdscan.py for the row syntax.
FOLD_SPECS = [
    # rotation symmetry (efficiency must stay flat)
    (0,   "original"),          # reference: no rotation, no NSB reshuffle
    (120, "original"),          # +120 deg
    (240, "original"),          # +240 deg
    # NSB reshuffle (rate must stay flat)
    (0,   ("rolled", 1)),
    (0,   ("rolled", 42)),
    (0,   ("shuffle", 2024)),
]

n_folds = len(FOLD_SPECS) if FOLD_SPECS is not None else 1

# Per-fold event caps (each fold is a full independent pass; these bound events
# processed IN EACH FOLD). None = use everything.
MAX_GAMMA_EVENTS = TOTAL_GAMMAS_EVENTS // n_folds
MAX_NSB_EVENTS = TOTAL_NSB_EVENTS // n_folds


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB NSB_GLOB [OUTPUT_FOLDER]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2]))
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "simu_sst1m_tel2_patch7"
    if not gamma_files or not nsb_files:
        sys.exit("no gamma or NSB files matched the given globs.")
    print(f"Found {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")

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
        N_event_esimate_threshold=25_000,
        batch_size=1024,
    )
    print(f"tau={tau}  predicted_rate={predicted_rate} Hz (frozen for all folds)")
    threshold_layer.tau.assign(tau)

    os.makedirs(output_folder, exist_ok=True)

    # All folds go into ONE statistics HDF5 (each event tagged with a `fold`
    # column, per-fold summaries in the /folds group). The pixel permutation is
    # applied to the raw waveform + pedestal before the digital sum, so folds
    # work here exactly as for TDSCAN. FOLD_SPECS=None -> a single fold-free pass.
    #
    # DATASET-level NSB augmentation stays OFF (nsb_roll_copies=0,
    # nsb_skip_original_events=False): each NSB event is yielded exactly once,
    # untouched, so the fold's own nsb_index is the ONLY NSB transform.
    folds = None if FOLD_SPECS is None else make_rotation_folds(chain.geom, FOLD_SPECS, seed=SEED)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=output_folder, batch_size=512,
        tel_id_only=1, nsb_roll_copies=0, nsb_skip_original_events=False,
        ignore_errors=False, folds=folds,
        max_gamma_events=MAX_GAMMA_EVENTS, max_nsb_events=MAX_NSB_EVENTS)
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
