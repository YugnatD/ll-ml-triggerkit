"""Compute SST-1M trigger statistics for the hex 3D-CNN chain -- on triggerkit.

Port of the sandbox's ``evaluate_perf_hexcnn.py`` onto the packaged API.

Reopening the trained CNN is trivial: examples/train_hexcnn.py saves the whole
threshold-terminated model as one ``.keras`` file, and every custom layer
(adapter + Conv3D + hexagdly + classifier + threshold) is
``register_keras_serializable``. So ``chain.compile_chain(model_path=...)`` just
calls ``tf.keras.models.load_model`` under the hood and rebuilds the exact graph
-- NO redefining the architecture, no ``make_body``, and no ``keras_hexagdly``
import needed here. It even reloads the ``_history.json`` sidecar if present.

    sandbox                                    triggerkit
    --------------------------------------------------------------------------
    build_adapter(...) + predict_keras.*   -> chain.compile_chain(model_path=MODEL.keras)
    (find_threshold_for_target_rate, compute_statistics are TriggerChain
     methods, unchanged)

Run it (quote the globs; repeat --condition for several NSB conditions, the
first one is the reference):

    python examples/stats_hexcnn.py MODEL.keras --condition medium "GAMMA_GLOB" "NSB_GLOB" [--output FOLDER] [--quick]

MODEL.keras is a model saved by examples/train_hexcnn.py.
"""

import os

import stats_common
from triggerkit.TriggerChain import TriggerChain
from triggerkit.augment import make_rotation_folds

BASE_NAME = "simu"
TARGET_RATE_HZ = 50_000
SEED = 1337

# Event budget of every fold: (gamma_events, nsb_events), None = every event.
FOLD_EVENTS = (None, None)

# Cross-validation folds: a leakage detector. Each fold reindexes gamma by an
# exact camera-rotation symmetry and NSB by a decorrelating reshuffle. If the
# model learned the physics, efficiency + rate stay flat across folds; if it
# cheated (fixed orientation, hot pixels, pedestal artefact), they shift -- and
# that shift is what the per-fold report exposes. Arbitrary length: add/remove
# rows freely.
#
# Each row is a dict; every key is optional and {} is the untouched reference
# fold (run under every --condition, the others under the reference one only).
# Keys (defaults in brackets):
#   gamma_deg        [0]           camera rotation on the gamma rows. Must be an
#                                  exact symmetry -- a multiple of 120 for this
#                                  3-fold camera; other angles raise loudly
#                                  rather than misplace pixels.
#   gamma_time_shift [0]           edge-hold time shift of the GAMMA waveform in samples
#   nsb_kind         ["original"]  "original" / "rolled" / "shuffle"
#   nsb_param        [None]        roll shift or shuffle seed (kind default if None)
#   nsb_time_shift   [0]           edge-hold time shift of the NSB waveform in samples
#   name             [auto]        explicit fold name
# An unknown key raises, so typos surface immediately.
#
# The time-shift folds test whether the CNN leaked the absolute temporal position
# of the pulse. Shift the gammas alone and the signal moves while tau stays tuned
# on unshifted NSB -- the classes are no longer treated alike. Shift BOTH by the
# same amount for the fair test: a time-translation-invariant trigger returns the
# same gamma efficiency AND the same NSB rate as the reference fold.
FOLD_ROWS = [
    {},                                                # reference fold
    {"gamma_deg": 120, "nsb_kind": "rolled"},
    {"gamma_deg": 240, "nsb_kind": "shuffle"},
    {"gamma_time_shift": 5},                           # gammas only
    {"gamma_time_shift": 5, "nsb_time_shift": 5},      # both -> fair test
]


def main():
    parser = stats_common.argument_parser(__doc__.splitlines()[0], "simu_sst1m_tel2_hexcnn")
    parser.add_argument("model_path", metavar="MODEL.keras", help="model saved by examples/train_hexcnn.py")
    args = parser.parse_args()
    model_path, output_folder = args.model_path, args.output
    if not os.path.exists(model_path):
        raise SystemExit(f"model not found: {model_path}")
    conditions = stats_common.condition_files(args)
    gamma_files, nsb_files = next(iter(conditions.values()))  # reference condition

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
        N_event_esimate_threshold=stats_common.threshold_events(args),
        batch_size=1024,
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
    )
    print(f"tau={tau}  predicted_rate={predicted_rate} Hz (frozen for all folds)")
    chain._get_last_trainable_threshold_layer().tau.assign(tau)

    os.makedirs(output_folder, exist_ok=True)

    # All folds go into ONE statistics HDF5 (each event tagged with a `fold`
    # column, per-fold summaries in the /folds group). Point the report at that
    # single file to compare metrics across folds.
    #
    # DATASET-level NSB augmentation stays OFF (nsb_roll_copies=0,
    # nsb_skip_original_events=False): each NSB event is yielded exactly once,
    # untouched, so the fold's own nsb_index is the ONLY NSB transform (a
    # ("rolled", 50) fold = each NSB event rolled by 50, no original, no stacked
    # dataset roll). Every fold is a full pass over the same events -> identical
    # per-fold counts.
    specs = stats_common.fold_specs(FOLD_ROWS, list(conditions), args.quick,
                                    reference_events=FOLD_EVENTS, fold_events=FOLD_EVENTS)
    folds = make_rotation_folds(chain.geom, specs, conditions, seed=SEED)
    chain.compute_statistics(
        base_name=BASE_NAME, folder=output_folder, batch_size=512,
        tel_id_only=1, nsb_roll_copies=0, nsb_skip_original_events=False,
        ignore_errors=False, folds=folds)
    print(f"Wrote per-event statistics under {output_folder}/")


if __name__ == "__main__":
    main()
