"""Train the TDSCAN trigger chain with a pairwise-AUC loss -- on triggerkit.

This is the sandbox's ``main_train_tdscan_bce.py`` ported onto the packaged API.
Nothing about the training recipe changed; only the imports did:

    sandbox                              triggerkit
    -------------------------------------------------------------------
    tdscan_chain.build_chain(...)     -> get_body("tdscan", ...).build(chain)
    train_utils.<fn>                  -> triggerkit.training.<fn>
    (inline dataset kwargs)           -> TriggerDataset(...) + train_chain(dataset=)
    dataset_config.get_*_files        -> gamma/NSB file lists you pass in

Multi-start training: the TDSCAN model is tiny (6-15 params), so we train
``N_FILTERS`` independent filters in parallel in one forward pass (random
restarts). The pairwise-AUC loss is summed per filter so their gradients stay
independent, and a per-filter ``auc_fk`` metric (plus ``auc_best``) is reported
each epoch. After training we keep only the best filter's weights and collapse
them into a normal ``filters=1`` chain -- so everything downstream (tau
calibration, save) is unchanged.

Run it:

    python examples/train_tdscan.py GAMMA_GLOB NSB_GLOB [SELECT_GAMMA_GLOB]

e.g.

    python examples/train_tdscan.py '/data/gamma/*.simtel.gz' '/data/nsb.simtel.gz'

If SELECT_GAMMA_GLOB is omitted, the held-out best-filter selection reuses the
training gammas (fine for a smoke test; use a disjoint set for a real run).
"""

import glob
import sys

import numpy as np
import tensorflow as tf

from triggerkit import training
from triggerkit.TriggerChain import TriggerChain
from triggerkit.data import TriggerDataset
from triggerkit.models import get_body

# --- Config (mirrors the sandbox script) -------------------------------------
PERCENT_VALIDATION = 0.2
N_FILTERS = 20            # parallel random restarts; best one is kept
BATCH_SIZE = 32
EPOCHS = 400
TARGET_RATE_HZ = 50_000   # operating point the held-out filter selection uses
SEED = 1337

# Score-quantizer edge placement, as in tdscan_chain.py.
EDGES_RANGE = (16, 128)
EDGES_NUM_BIT = 4


def generate_lin_space_edges(start, stop, num_bit):
    """Linearly spaced integer edges for the score quantizer (2**num_bit - 1)."""
    num_edges = 2 ** num_bit - 1
    return np.linspace(start, stop, num=num_edges).astype(int).tolist()


def _train_inputs(chain):
    """The chain's model inputs (1 for DigiCam_R0Alpha, else waveform+baseline)."""
    if chain.camera_name == "DigiCam_R0Alpha":
        return chain.input_layer
    return [chain.input_layer, chain.input_baseline]


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
    print(f"Found {len(gamma_files)} training gamma files, {len(nsb_files)} NSB files.")

    # --- The training chain: N_FILTERS parallel restarts (collapsed to 1 below).
    # get_body(...).build(chain) replaces tdscan_chain.build_chain: the TDSCAN body
    # appends the exact same stages ([score_quantizer] -> tdscan -> max-pool ->
    # threshold) and hands back the tdscan/threshold layer handles.
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    body = get_body(
        "tdscan",
        filters=N_FILTERS,
        edges_range=EDGES_RANGE,
        edges_num_bit=EDGES_NUM_BIT,
        edges_func=generate_lin_space_edges,
    )
    handles = body.build(chain)
    tdscan_layer, threshold_layer = handles["tdscan"], handles["threshold"]
    score_tensor = threshold_layer.input  # pre-threshold score, shape (B, N_FILTERS)

    # --- Calibrate tau / temp from a real forward pass -----------------------
    # One column per filter; calibration wants one score per event, so calibrate
    # on the best-filter envelope (max over filters). This only seeds the loss
    # sharpness; tau is re-calibrated on the collapsed chain after training. The
    # reduction must be a Keras layer (score_tensor is a symbolic KerasTensor).
    calib_score = tf.keras.layers.Lambda(
        lambda s: tf.reduce_max(s, axis=-1), name="calib_best_filter"
    )(score_tensor)
    init_tau, temp, scores_g, scores_n = training.calibrate_tau(
        chain, calib_score, gamma_files
    )
    print(f"Calibration: gamma mean={float(scores_g.mean()):.2f} "
          f"=> init_tau={init_tau:.2f}, temp={temp:.4g}")
    print(f"  gamma score range=[{scores_g.min():.2f}, {scores_g.max():.2f}] (n={scores_g.size})")
    print(f"  nsb   score range=[{scores_n.min():.2f}, {scores_n.max():.2f}] (n={scores_n.size})")

    # The pairwise-AUC loss trains the raw score and ignores tau, so we just seed
    # the threshold's tau and freeze it; it is re-calibrated after training.
    threshold_layer.tau.assign(init_tau)
    tdscan_layer.set_trainable(True)
    threshold_layer.set_trainable(False)

    # --- Build the AUC training model directly on the pre-threshold score -----
    # train_chain reads self.model.fit(...), so we point chain.model at the score.
    auc_sharpness = float(temp)
    print(f"Pairwise AUC sharpness = {auc_sharpness:.4g} (from calibrated temp)")

    chain.model = tf.keras.Model(inputs=_train_inputs(chain), outputs=score_tensor)
    chain.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=training.make_pairwise_auc_loss_multi(sharpness=auc_sharpness),
        metrics=training.make_per_filter_auc_metrics(N_FILTERS),
    )
    chain.model.summary()

    train_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_best", mode="max", patience=100,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_best", mode="max", factor=0.5, patience=20,
            min_lr=1e-5, verbose=1),
        training.PerFilterAUCLogger(N_FILTERS),
        training.WeightStatsLogger(tdscan_layer),
    ]

    # --- Data source: one TriggerDataset, injected into train_chain -----------
    # This replaces the pile of inline dataset kwargs the sandbox passed to
    # train_chain; train_chain(dataset=...) uses it verbatim.
    dataset = TriggerDataset(
        gamma_files, nsb_files,
        batch_size=BATCH_SIZE,
        percent_validation=PERCENT_VALIDATION,
        tel_id_only=1,
        max_gamma_samples_train=50_000,
        max_gamma_samples_val=15_000,
        max_nsb_samples_train=50_000,
        max_nsb_samples_val=15_000,
        load_ram=True,
        seed=SEED,
    )
    history = chain.train_chain(
        epochs=EPOCHS,
        callbacks=train_callbacks,
        verbose=0,  # PerFilterAUCLogger prints the per-epoch summary
        dataset=dataset,
    )

    # --- Collapse the N_FILTERS restarts to the single best filter ------------
    # EarlyStopping already restored the kernel at its best epoch. Pick the filter
    # on a held-out set, ranked by gamma efficiency at the target NSB rate.
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
          "collapsing chain to filters=1.")

    # The filter axis is last in both weight layouts, so slicing it is uniform.
    shared = tdscan_layer.share_neighbors
    if shared:
        best_weights = tdscan_layer.kernel_rings.numpy()[..., best_f:best_f + 1]
    else:
        best_weights = tdscan_layer.kernel.numpy()[..., best_f:best_f + 1]

    # Rebuild the normal filters=1 chain and copy the winning weights in.
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    handles = get_body(
        "tdscan",
        filters=1,
        edges_range=EDGES_RANGE,
        edges_num_bit=EDGES_NUM_BIT,
        edges_func=generate_lin_space_edges,
    ).build(chain)
    tdscan_layer, threshold_layer = handles["tdscan"], handles["threshold"]
    score_tensor = threshold_layer.input  # now shape (B, 1)
    if shared:
        tdscan_layer.set_weights_from_params(share_weights=True, ring_weights=best_weights)
    else:
        tdscan_layer.set_weights_from_params(share_weights=False, kernel_weights=best_weights)

    # --- Post-training tau re-calibration ------------------------------------
    post_tau, _, post_g, _post_n = training.calibrate_tau(
        chain, score_tensor, gamma_files, compute_temp=False)
    threshold_layer.tau.assign(post_tau)
    print(f"Post-training calibration: gamma mean={float(post_g.mean()):.2f}, "
          f"new tau={post_tau:.2f}")

    # --- Save the collapsed filters=1 chain ----------------------------------
    chain.model = tf.keras.Model(inputs=_train_inputs(chain), outputs=score_tensor)
    model_path = chain.generate_output_filename(
        folder="trained_models", base_name="trigger_chain", suffix="model.keras")
    chain.model.save(model_path)
    print(f"Saved collapsed filters=1 model to {model_path}")

    if hist:
        history_path = chain.generate_output_filename(
            folder="trained_models", base_name="trigger_chain", suffix="history.npy")
        np.save(history_path, hist)
        print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
