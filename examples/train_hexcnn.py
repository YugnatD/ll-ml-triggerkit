"""Train the hex 3D-CNN trigger with a pairwise-AUC loss -- on triggerkit.

Port of the sandbox's ``train_hex_cnn.py`` onto the packaged API. The recipe is
identical to ``examples/train_tdscan.py`` -- pairwise-AUC multi-start training,
best restart kept, tau re-calibrated -- only the body differs: instead of a
TDSCAN filter it is the ``hex3d_hybrid`` backbone (scatter-to-grid adapter +
Conv3D temporal blocks + hexagdly spatial blocks + Dense classifier + threshold).

One important difference from TDSCAN restarts: the backbone is SHARED across the
``N_FILTERS`` restarts; only the final Dense ``classifier`` has ``N_FILTERS``
independent output columns. Collapsing therefore keeps the (single) trained
backbone and just the winning classifier column.

Requires the ``[hexcnn]`` extra (``pip install '.[hexcnn]'`` -- pulls in
keras_hexagdly).

Run it:

    python examples/train_hexcnn.py GAMMA_GLOB NSB_GLOB [SELECT_GAMMA_GLOB]
"""

import glob
import sys

import numpy as np
import tensorflow as tf

from triggerkit import training
from triggerkit.TriggerChain import TriggerChain
from triggerkit.data import TriggerDataset
from triggerkit.models import get_body

# --- Config (mirrors train_hex_cnn.py) ---------------------------------------
PERCENT_VALIDATION = 0.2
N_FILTERS = 4             # parallel classifier-head restarts (shared backbone)
BATCH_SIZE = 16
EPOCHS = 100
TARGET_RATE_HZ = 50_000
SEED = 1337

TIME_SKIP = 0
TIME_WINDOW = 32

# Backbone (weight-carrying) layer names in the hex3d_hybrid body; the ReLU /
# scatter / pool layers have no weights and are rebuilt fresh on collapse.
BACKBONE_LAYER_NAMES = ("temporal_0", "temporal_2", "spatial_0", "spatial_2")


def _train_inputs(chain):
    if chain.camera_name == "DigiCam_R0Alpha":
        return chain.input_layer
    return [chain.input_layer, chain.input_baseline]


def _build_body(chain, filters):
    """Build the hex3d_hybrid body onto ``chain``; return its handles dict."""
    return get_body(
        "hex3d_hybrid",
        filters=filters,
        time_skip=TIME_SKIP,
        time_window=TIME_WINDOW,
    ).build(chain)


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB NSB_GLOB [SELECT_GAMMA_GLOB]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2]))
    select_files = sorted(glob.glob(sys.argv[3])) if len(sys.argv) > 3 else gamma_files
    if not gamma_files or not nsb_files:
        sys.exit("no gamma or NSB files matched the given globs.")

    tf.keras.utils.set_random_seed(SEED)
    print(f"Global RNG seed set to {SEED}.")
    print(f"Found {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")

    # --- The training chain: N_FILTERS classifier heads (collapsed to 1 below).
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    handles = _build_body(chain, N_FILTERS)
    classifier = handles["classifier"]
    threshold_layer = next(h for h in handles.values() if hasattr(h, "tau"))
    score_tensor = classifier.output  # pre-threshold score, shape (B, N_FILTERS)

    # --- Calibrate tau / temp on the best-filter envelope --------------------
    calib_score = tf.keras.layers.Lambda(
        lambda s: tf.reduce_max(s, axis=-1), name="calib_best_filter"
    )(score_tensor)
    init_tau, temp, scores_g, scores_n = training.calibrate_tau(
        chain, calib_score, gamma_files)
    print(f"Calibration: gamma mean={float(scores_g.mean()):.2f} "
          f"=> init_tau={init_tau:.2f}, temp={temp:.4g}")
    print(f"  gamma score range=[{scores_g.min():.2f}, {scores_g.max():.2f}] (n={scores_g.size})")
    print(f"  nsb   score range=[{scores_n.min():.2f}, {scores_n.max():.2f}] (n={scores_n.size})")

    threshold_layer.tau.assign(init_tau)
    threshold_layer.set_trainable(False)

    # --- AUC training model on the pre-threshold score -----------------------
    auc_sharpness = float(temp)
    print(f"Pairwise AUC sharpness = {auc_sharpness:.4g} (from calibrated temp)")

    chain.model = tf.keras.Model(inputs=_train_inputs(chain), outputs=score_tensor)
    chain.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=training.make_pairwise_auc_loss_multi(sharpness=auc_sharpness),
        metrics=training.make_per_filter_auc_metrics(N_FILTERS),
    )
    chain.model.summary()

    # No WeightStatsLogger: it reads TDSCAN-specific layer.kernel/kernel_rings.
    train_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_best", mode="max", patience=30,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_best", mode="max", factor=0.5, patience=10,
            min_lr=1e-5, verbose=1),
        training.PerFilterAUCLogger(N_FILTERS),
    ]

    dataset = TriggerDataset(
        gamma_files, nsb_files,
        batch_size=BATCH_SIZE,
        percent_validation=PERCENT_VALIDATION,
        tel_id_only=1,
        max_gamma_samples_train=10_000,
        max_gamma_samples_val=10_000,
        max_nsb_samples_train=10_000,
        max_nsb_samples_val=10_000,
        load_ram=True,
        seed=SEED,
    )
    history = chain.train_chain(
        epochs=EPOCHS,
        callbacks=train_callbacks,
        verbose=0,
        dataset=dataset,
    )

    # --- Pick the winning classifier column ----------------------------------
    hist = history.history if history is not None else {}
    print(f"Selecting best filter on {len(select_files)} held-out gamma files "
          f"at target NSB rate {TARGET_RATE_HZ} Hz.")
    best_f, sel_info = training.select_best_filter(
        chain, score_tensor, select_files, target_rate_hz=TARGET_RATE_HZ)
    eff = sel_info["efficiency"]
    print(f"Held-out gamma efficiency @ {TARGET_RATE_HZ} Hz "
          f"(n_gamma={sel_info['n_gamma']}, n_nsb={sel_info['n_nsb']}): "
          + ", ".join(f"f{k}={eff[k]:.4f}" for k in range(N_FILTERS)))
    print(f"Keeping filter {best_f} (efficiency={eff[best_f]:.4f}); "
          "collapsing to filters=1.")

    # Snapshot the shared trained backbone + the winning classifier column.
    backbone_weights = {n: handles[n].get_weights() for n in BACKBONE_LAYER_NAMES}
    clf_kernel, clf_bias = classifier.get_weights()
    best_kernel = clf_kernel[:, best_f:best_f + 1]
    best_bias = clf_bias[best_f:best_f + 1]

    # --- Rebuild the filters=1 chain and transplant the trained weights ------
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    handles = _build_body(chain, 1)
    classifier = handles["classifier"]
    threshold_layer = next(h for h in handles.values() if hasattr(h, "tau"))
    for n in BACKBONE_LAYER_NAMES:
        handles[n].set_weights(backbone_weights[n])
    classifier.set_weights([best_kernel, best_bias])
    score_tensor = classifier.output  # now shape (B, 1)

    # --- Post-training tau re-calibration ------------------------------------
    post_tau, _, post_g, _post_n = training.calibrate_tau(
        chain, score_tensor, gamma_files, compute_temp=False)
    threshold_layer.tau.assign(post_tau)
    print(f"Post-training calibration: gamma mean={float(post_g.mean()):.2f}, "
          f"new tau={post_tau:.2f}")

    # --- Save the collapsed filters=1 chain ----------------------------------
    chain.model = tf.keras.Model(inputs=_train_inputs(chain), outputs=score_tensor)
    model_path = chain.generate_output_filename(
        folder="trained_models", base_name="hexcnn_trigger", suffix="model.keras")
    chain.model.save(model_path)
    print(f"Saved collapsed filters=1 model to {model_path}")

    if hist:
        history_path = chain.generate_output_filename(
            folder="trained_models", base_name="hexcnn_trigger", suffix="history.npy")
        np.save(history_path, hist)
        print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
