"""Compute SST-1M trigger statistics for a TDSCAN chain -- on triggerkit.

Port of the sandbox's ``evaluate_perf_tdscan.py`` onto the packaged API. It
builds the deployed (filters=1) TDSCAN chain, optionally pins a set of ring
weights, tunes the threshold ``tau`` to a target NSB rate, then writes the
per-event statistics HDF5 that the report generator (``stats_report.py``) reads.

    sandbox                            triggerkit
    ----------------------------------------------------------------
    tdscan_chain.build_chain(...)   -> get_body("tdscan", ...).build(chain)
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
from triggerkit.models import get_body

# --- Config (mirrors evaluate_perf_tdscan.py) --------------------------------
BASE_NAME = "simu"
TARGET_RATE_HZ = 50_000
EDGES_RANGE = (16, 128)
EDGES_NUM_BIT = 4

# Flat ring weights to pin (share_neighbors=True). Length must match the kernel:
# eps_t=1 -> 6, eps_t=2 -> 10, ... Set to None to keep the build-time init.
RING_WEIGHTS = np.array(
    [0.5000, 0.0625, -0.5000, -0.0039, -1.0000, -0.2500, 1.0000, 0.1250, 0.5000, 0.2500])


def generate_lin_space_edges(start, stop, num_bit):
    """Linearly spaced integer edges for the score quantizer (2**num_bit - 1)."""
    return np.linspace(start, stop, num=2 ** num_bit - 1).astype(int).tolist()


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
    handles = get_body(
        "tdscan",
        filters=1,
        edges_range=EDGES_RANGE,
        edges_num_bit=EDGES_NUM_BIT,
        edges_func=generate_lin_space_edges,
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

    tau, predicted_rate = chain.find_threshold_for_target_rate(
        target_rate_hz=TARGET_RATE_HZ,
        tolerance_hz=2,
        N_event_esimate_threshold=25_000,
        batch_size=1024,
    )
    print(f"tau={tau}  predicted_rate={predicted_rate} Hz")
    threshold_layer.tau.assign(tau)

    os.makedirs(output_folder, exist_ok=True)
    chain.compute_statistics(
        base_name=BASE_NAME,
        folder=output_folder,
        batch_size=512,
        tel_id_only=1,
        nsb_roll_copies=1,
        nsb_skip_original_events=True,
        ignore_errors=False,
    )
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
