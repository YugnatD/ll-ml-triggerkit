"""Compute SST-1M trigger statistics for the hex 3D-CNN chain -- on triggerkit.

Port of the sandbox's ``evaluate_perf_hexcnn.py`` onto the packaged API.

Reopening the trained CNN is trivial: examples/train_hexcnn.py saves the whole
threshold-terminated model as one ``.keras`` file, and every custom layer
(adapter + Conv3D + hexagdly + classifier + threshold) is
``register_keras_serializable``. So ``chain.compile_chain(model_path=...)`` just
calls ``tf.keras.models.load_model`` under the hood and rebuilds the exact graph
-- NO redefining the architecture, no ``make_body``, and no ``keras_hexagdly``
import needed here. It even reloads the ``_history.npy`` sidecar if present.

    sandbox                                    triggerkit
    --------------------------------------------------------------------------
    build_adapter(...) + predict_keras.*   -> chain.compile_chain(model_path=MODEL.keras)
    (find_threshold_for_target_rate, compute_statistics are TriggerChain
     methods, unchanged)

Run it:

    python examples/stats_hexcnn.py GAMMA_GLOB NSB_GLOB MODEL.keras [OUTPUT_FOLDER]

MODEL.keras is a model saved by examples/train_hexcnn.py.
"""

import glob
import os
import sys

from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds

BASE_NAME = "simu"
TARGET_RATE_HZ = 50_000
SEED = 1337

# Per-fold event caps. Each fold is a full independent pass, so these bound the
# gamma / NSB events processed IN EACH FOLD (not the total across folds). None =
# use everything. Gamma count is after the tel_id_only filter; NSB count is
# source events (dataset augmentation off).
MAX_GAMMA_EVENTS = None
MAX_NSB_EVENTS = None

# Cross-validation folds: a leakage detector. Each fold reindexes gamma by an
# exact camera-rotation symmetry and NSB by a decorrelating reshuffle. If the
# model learned the physics, efficiency + rate stay flat across folds; if it
# cheated (fixed orientation, hot pixels, pedestal artefact), they shift -- and
# that shift is what the per-fold report exposes. Arbitrary length: add/remove
# rows freely. Gamma degrees must be exact symmetries (multiples of 120 for this
# 3-fold camera); non-symmetric angles raise loudly rather than misplace pixels.
# nsb_kind is "original" / "rolled" / "shuffle" (or ("rolled", shift) /
# ("shuffle", seed) to pin the parameter). Set to None to run fold-free.
FOLD_SPECS = [
    (0,   "original"),
    (120, "rolled"),
    (240, "shuffle"),
]


def main():
    if len(sys.argv) < 4:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB NSB_GLOB MODEL.keras [OUTPUT_FOLDER]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2]))
    model_path = sys.argv[3]
    output_folder = sys.argv[4] if len(sys.argv) > 4 else "simu_sst1m_tel2_hexcnn"
    if not gamma_files or not nsb_files:
        sys.exit("no gamma or NSB files matched the given globs.")
    if not os.path.exists(model_path):
        sys.exit(f"model not found: {model_path}")
    print(f"Found {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")

    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    print(f"camera={chain.camera_name}  num_pixels={chain.num_pixels}  "
          f"num_samples={chain.num_samples}  window_size={chain.window_size:.3e}s")

    # Reopen the trained model as-is (architecture + weights + tau). compile_chain
    # sees an existing model_path, load_model's it (custom layers auto-resolve via
    # their register_keras_serializable registration), and reloads the history.
    chain.compile_chain(model_path=model_path)
    print(f"Loaded trained model from {model_path}")

    # Tune tau ONCE on the nominal (fold-free) NSB and freeze it -- every fold is
    # evaluated at the same operating point, so a rate drift across folds is a
    # real signal, not a re-tuning artefact.
    tau, predicted_rate = chain.find_threshold_for_target_rate(
        target_rate_hz=TARGET_RATE_HZ,
        tolerance_hz=2,
        N_event_esimate_threshold=25_000,
        batch_size=1024,
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
    )
    print(f"tau={tau}  predicted_rate={predicted_rate} Hz (frozen for all folds)")
    chain._get_last_trainable_threshold_layer().tau.assign(tau)

    os.makedirs(output_folder, exist_ok=True)

    # All folds go into ONE statistics HDF5 (each event tagged with a `fold`
    # column, per-fold summaries in the /folds group). Point the report at that
    # single file to compare metrics across folds. FOLD_SPECS=None -> a single
    # fold-free pass.
    #
    # DATASET-level NSB augmentation stays OFF (nsb_roll_copies=0,
    # nsb_skip_original_events=False): each NSB event is yielded exactly once,
    # untouched, so the fold's own nsb_index is the ONLY NSB transform (a
    # ("rolled", 50) fold = each NSB event rolled by 50, no original, no stacked
    # dataset roll). Every fold is a full pass over the same events -> identical
    # per-fold counts.
    folds = None if FOLD_SPECS is None else make_rotation_folds(chain.geom, FOLD_SPECS, seed=SEED)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=output_folder, batch_size=512,
        tel_id_only=1, nsb_roll_copies=0, nsb_skip_original_events=False,
        ignore_errors=False, folds=folds,
        max_gamma_events=MAX_GAMMA_EVENTS, max_nsb_events=MAX_NSB_EVENTS)
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
