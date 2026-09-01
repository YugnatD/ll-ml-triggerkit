"""Compute SST-1M trigger statistics for a TDSCAN chain -- on triggerkit.

Port of the sandbox's ``evaluate_perf_tdscan.py`` onto the packaged API. It
builds the deployed (filters=1) TDSCAN chain, optionally pins a set of ring
weights, tunes the threshold ``tau`` to a target NSB rate, then writes the
per-event statistics HDF5 that the report generator (``stats_report.py``) reads.

    sandbox                            triggerkit
    ----------------------------------------------------------------
    tdscan_chain.build_chain(...)   -> TDSCANBody(...).build(chain)
    (the rest -- find_threshold_for_target_rate, compute_statistics -- are
     TriggerChain methods and are unchanged)

Run it:

    python examples/stats_tdscan.py GAMMA_GLOB NSB_GLOB [OUTPUT_FOLDER]

The output HDF5 lands in OUTPUT_FOLDER (default: simu_sst1m_tel2_tdscan).
"""

import glob
import os
import sys

import numpy as np

from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds
from triggerkit.models import TDSCANBody, generate_lin_space_edges

# --- Config (mirrors evaluate_perf_tdscan.py) --------------------------------
BASE_NAME = "simu"
TARGET_RATE_HZ = 50_000
SEED = 1337

TOTAL_GAMMAS_EVENTS = 400_000
TOTAL_NSB_EVENTS = 200_000

# Cross-validation folds (leakage detector -- see examples/stats_hexcnn.py and
# triggerkit.augment for the rationale). Each fold reindexes gamma by an exact
# camera-rotation symmetry and NSB by a decorrelating reshuffle; tau is tuned
# once and frozen, so a metric shift across folds flags a model leaking on
# orientation / specific pixels. All folds are written into ONE stats HDF5.
#
# How to define folds -- FOLD_SPECS is an arbitrary-length list of
# (gamma_deg, nsb_kind) rows, turned into augment.Fold objects by
# make_rotation_folds(chain.geom, FOLD_SPECS, seed=SEED). Each row:
#   gamma_deg : an EXACT camera-rotation symmetry. For this 3-fold camera that
#               is a multiple of 120 (0/120/240); any other angle raises loudly
#               rather than silently misplacing pixels. 0 = no rotation.
#   nsb_kind  : how the NSB pixels are reshuffled at the frozen tau --
#       "original"            -> identity (no reshuffle)
#       "rolled"              -> circular roll of the pixel list, default shift P//2
#       "shuffle"             -> random permutation, default seed = SEED + row_index
#     Override the parameter by passing a tuple instead of the bare string:
#       ("rolled", 50)        -> roll by exactly 50 pixels
#       ("shuffle", 2024)     -> shuffle with a pinned seed (reproducible)
#
# Add / remove / reorder rows freely. The FIRST row is the reference fold: the
# report compares every other fold against it, and its counts become the file's
# top-level attrs. Set FOLD_SPECS = None to run a single plain (fold-free) pass.
FOLD_SPECS = [
    # check the rotation symmetry
    (0,   "original"),          # reference: no rotation, no NSB reshuffle
    (120, "original"),          # +120 deg, no NSB reshuffle
    (240, "original"),          # +240 deg, no NSB reshuffle
    # now check nsb
    (0,   ("rolled", 1)),   
    (0,   ("rolled", 42)),   
    (0, ("shuffle", 2024)),
    # some mix
    (120, ("rolled", 1)),
    (240, ("rolled", 1)),
    (120, ("shuffle", 2024)),
    (240, ("shuffle", 2024)),
]

n_folds = len(FOLD_SPECS)

# Per-fold event caps. Each fold is a full independent pass, so these bound the
# gamma / NSB events processed IN EACH FOLD (not the total across folds). None =
# use everything available. Example: MAX_GAMMA_EVENTS=50_000 + MAX_NSB_EVENTS=
# 100_000 -> every fold sees at most 50k gammas and 100k NSB. Gamma count is
# after the tel_id_only filter; NSB count is source events (augmentation off).
MAX_GAMMA_EVENTS = TOTAL_GAMMAS_EVENTS // n_folds
MAX_NSB_EVENTS = TOTAL_NSB_EVENTS // n_folds

# Score-quantizer front-end: bucket each pixel's input into 2**EDGES_NUM_BIT - 1
# integer levels spaced over EDGES_RANGE=(lo, hi) before TDSCAN (mimics the FPGA
# ADC quantization of the waveform). To DISABLE it, set EDGES_FUNC = None (like
# every other None-able knob); it's also skipped when EDGES_NUM_BIT == 0 or
# hi <= lo. Don't set EDGES_RANGE = None -- that would crash on edges_range[1].
EDGES_RANGE = (16, 128)   # (lo, hi) span the quantizer edges cover
EDGES_NUM_BIT = 4         # bit depth -> 2**4 - 1 = 15 levels
EDGES_FUNC = generate_lin_space_edges   # edge-placement fn (from triggerkit.models); None disables the quantizer

# Inner accumulator quantization (the FPGA fixed-point path inside the TDSCAN
# filter). Each qspec is "<U|S>Q<int_bits>.<frac_bits>": UQ4.0 = unsigned 4-bit
# integer (0..15), SQ9.0 = signed 9-bit integer, UQ3.1 = unsigned 3.1 fixed point.
#   input                 -> waveform fed into the filter
#   ring_weights          -> the learned weights
#   convolution_accumulator / convolution_rescale_shift -> the spatial (per-ring)
#       accumulator qspec and its post-accumulation right-shift
#   temporal_accumulator  / temporal_rescale_shift       -> same for the time sum
# Set to None to run in full float (no inner quantization).
QUANTIZE_STEP = {
    "input": "UQ4.0",
    "ring_weights": "UQ3.1",
    "convolution_accumulator": "SQ9.0",
    "convolution_rescale_shift": 0,
    "temporal_accumulator": "SQ9.0",
    "temporal_rescale_shift": 0,
}

# TDSCAN accumulator overflow / rounding + post-accumulation right-shift.
OVERFLOW_MODE = "AP_SAT"        # saturate on overflow
QUANTIZATION_MODE = "AP_TRN"    # truncate on rounding
RESCALE_SHIFT = 0               # right-shift (÷2**n) the accumulator after summation to rescale back into range; 0 = no rescale

# Optional front-/back-end stages (all off by default -> the deployed chain).
SUBTRACT_VALUE = None           # scalar subtracted before TDSCAN (FPGA pedestal/"shift" subtraction); None = no subtract stage
SUBTRACT_QUANTIZE_STEP = None   # fixed-point for the subtract stage, keys input/shift_value/output, e.g. {"input": "UQ8.0", "shift_value": "UQ8.0", "output": "SQ9.0"}
SUBTRACT_OVERFLOW_MODE = "AP_WRAP"      # subtract-stage overflow behaviour
SUBTRACT_QUANTIZATION_MODE = "AP_TRN"   # subtract-stage rounding behaviour
DIGITAL_SUM_MODE = None         # digital-sum stage after TDSCAN summing pixel scores over a patch, e.g. "patch7" (7-pixel patch trigger); None = off
FADC = False                    # shared FADC baseline-subtraction front-end before TDSCAN; False = off

# Flat ring weights to pin (share_neighbors=True). Length must match the kernel:
# eps_t=1 -> 6, eps_t=2 -> 10, ... Set to None to keep the build-time init.
RING_WEIGHTS = np.array(
    [0.5000, 0.0625, -0.5000, -0.0039, -1.0000, -0.2500, 1.0000, 0.1250, 0.5000, 0.2500])


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB NSB_GLOB [OUTPUT_FOLDER]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2]))
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "simu_sst1m_tel2_tdscan"
    if not gamma_files or not nsb_files:
        sys.exit("no gamma or NSB files matched the given globs.")
    print(f"Found {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")

    # --- The deployed filters=1 TDSCAN chain, via the packaged body ----------
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    handles = TDSCANBody(
        filters=1,
        edges_range=EDGES_RANGE,
        edges_num_bit=EDGES_NUM_BIT,
        edges_func=EDGES_FUNC,
        quantize_step=QUANTIZE_STEP,
        overflow_mode=OVERFLOW_MODE,
        quantization_mode=QUANTIZATION_MODE,
        rescale_shift=RESCALE_SHIFT,
        subtract_value=SUBTRACT_VALUE,
        subtract_quantize_step=SUBTRACT_QUANTIZE_STEP,
        subtract_overflow_mode=SUBTRACT_OVERFLOW_MODE,
        subtract_quantization_mode=SUBTRACT_QUANTIZATION_MODE,
        digital_sum_mode=DIGITAL_SUM_MODE,
        fadc=FADC,
    ).build(chain)
    tdscan_layer, threshold_layer = handles["tdscan"], handles["threshold"]

    # Pin (and freeze) the TDSCAN weights: shared -> ring_weights.
    if RING_WEIGHTS is not None:
        if tdscan_layer.share_neighbors:
            print(f"Setting TDSCAN ring weights to {RING_WEIGHTS}")
            tdscan_layer.set_weights_from_params(share_weights=True, ring_weights=RING_WEIGHTS)
        else:
            print(f"Setting TDSCAN kernel weights to {RING_WEIGHTS}")
            tdscan_layer.set_weights_from_params(share_weights=False, kernel_weights=RING_WEIGHTS)

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
