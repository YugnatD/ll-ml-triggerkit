"""Compute SST-1M trigger statistics for the hex 3D-CNN chain -- on triggerkit.

Port of the sandbox's ``evaluate_perf_hexcnn.py`` onto the packaged API. Where
the sandbox hand-built a scatter-to-grid adapter and imported an external Keras
port of the CNN, here the whole thing -- adapter + Conv3D temporal blocks +
hexagdly spatial blocks + classifier + threshold -- is one packaged body
(``hex3d_hybrid``), so no external ``predict_keras`` / npz is needed.

    sandbox                                    triggerkit
    --------------------------------------------------------------------------
    build_adapter(...) + predict_keras.*   -> get_body("hex3d_hybrid", ...).build(chain)
    (find_threshold_for_target_rate, compute_statistics are TriggerChain
     methods, unchanged)

Requires the ``[hexcnn]`` extra (``pip install '.[hexcnn]'`` -- pulls in
keras_hexagdly).

Run it:

    python examples/stats_hexcnn.py GAMMA_GLOB NSB_GLOB [WEIGHTS.keras [OUTPUT_FOLDER]]

WEIGHTS.keras is a model saved by examples/train_hexcnn.py; omit it to run on a
random-init CNN (a plumbing smoke test only -- the rates/efficiencies are
meaningless without trained weights).
"""

import glob
import os
import sys

from triggerkit.TriggerChain import TriggerChain
from triggerkit.models import get_body

BASE_NAME = "simu"
TARGET_RATE_HZ = 50_000
TIME_SKIP = 0       # frames dropped from the front
TIME_WINDOW = 32    # frames kept -> the CNN's trained time dimension


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB NSB_GLOB [WEIGHTS.keras [OUTPUT_FOLDER]]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2]))
    weights_path = sys.argv[3] if len(sys.argv) > 3 else None
    output_folder = sys.argv[4] if len(sys.argv) > 4 else "simu_sst1m_tel2_hexcnn"
    if not gamma_files or not nsb_files:
        sys.exit("no gamma or NSB files matched the given globs.")
    print(f"Found {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")

    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    print(f"camera={chain.camera_name}  num_pixels={chain.num_pixels}  "
          f"num_samples={chain.num_samples}  window_size={chain.window_size:.3e}s")
    if chain.num_samples < TIME_SKIP + TIME_WINDOW:
        raise ValueError(
            f"waveform has {chain.num_samples} samples; the adapter needs at least "
            f"{TIME_SKIP + TIME_WINDOW}.")

    # Adapter + CNN + threshold, all in one body. scatter_to_grid reads the
    # camera geometry from the chain, so no manual GridTransform wiring here.
    get_body(
        "hex3d_hybrid",
        filters=1,
        time_skip=TIME_SKIP,
        time_window=TIME_WINDOW,
    ).build(chain)

    # find_threshold_for_target_rate needs chain.model to locate the threshold.
    chain.compile_chain()
    chain.model.summary()

    if weights_path:
        chain.model.load_weights(weights_path)
        print(f"Loaded trained weights from {weights_path}")
    else:
        print("WARNING: no weights given -- running on random init (smoke test only).")

    tau, predicted_rate = chain.find_threshold_for_target_rate(
        target_rate_hz=TARGET_RATE_HZ,
        tolerance_hz=2,
        N_event_esimate_threshold=25_000,
        batch_size=1024,
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
    )
    print(f"tau={tau}  predicted_rate={predicted_rate} Hz")
    chain._get_last_trainable_threshold_layer().tau.assign(tau)

    os.makedirs(output_folder, exist_ok=True)
    chain.compute_statistics(
        base_name=BASE_NAME,
        folder=output_folder,
        batch_size=512,
        tel_id_only=1,
        nsb_roll_copies=0,
        nsb_skip_original_events=False,
        ignore_errors=False,
    )
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
