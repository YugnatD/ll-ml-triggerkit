"""Threshold (tau/temp) calibration from a real forward pass.

Runs the chain up to its pre-threshold score on a small gamma+NSB sample and
picks ``tau`` (and optionally ``temp``) from the resulting score split. Split out
of the sandbox's ``train_utils.py``; behaviour unchanged.
"""

import numpy as np
import tensorflow as tf

from triggerkit.FileIO.FileOpenerCTAO import (
    AsyncFileOpenerProcess,
    SimTelTFDataset,
    SimTelTFDatasetConfig,
)


def _default_calib_config():
    return SimTelTFDatasetConfig(
        batch_size=256,
        shuffle_samples=True,
        sample_shuffle_buffer=2000,
        seed=1337,
        load_ram=True,
        interleave_files=True,
        waveform_level="r0",
        gamma_tel_id_only=1,
        gamma_n_pe_max=None,
        gamma_n_pe_min=50,
        gamma_skip_if_missing_n_pe=True,
        include_event_features=True,
        event_feature_keys=("n_pe", "energy"),
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
        nsb_roll_axis=1,
        # Own seed, distinct from train_chain's cfg_train/cfg_val, so the
        # calibration pass's NSB pixel rotations aren't correlated with training.
        nsb_roll_augment=True,
        nsb_roll_seed=2024,
        max_gamma_samples_total=400,
        max_nsb_samples_total=400,
        repeat=False,
        ignore_errors=True,
    )


def _score_distributions(chain, score_tensor, gamma_files, config):
    """Run a forward pass and split pre-threshold scores into gamma / NSB."""
    if chain.camera_name == "DigiCam_R0Alpha":
        pre_inputs = chain.input_layer
        pack_inputs = lambda wf, ped: wf
    else:
        pre_inputs = [chain.input_layer, chain.input_baseline]
        pack_inputs = lambda wf, ped: (wf, ped)
    pre_threshold_model = tf.keras.Model(inputs=pre_inputs, outputs=score_tensor)

    ds = SimTelTFDataset(
        gamma_files=gamma_files[: max(2, min(4, len(gamma_files)))],
        nsb_files=chain.simtel_nsb_path,
        opener_cls=AsyncFileOpenerProcess,
        config=config,
    ).dataset()

    scores_g, scores_n = [], []
    for feat, lbl in ds:
        wf = tf.reshape(tf.cast(feat["waveform"], tf.uint16),
                        (-1, chain.num_pixels, chain.num_samples))
        ped = tf.reshape(tf.cast(feat["pedestal"], tf.int32), (-1, chain.num_pixels))
        scores = pre_threshold_model(pack_inputs(wf, ped), training=False).numpy().reshape(-1)
        l = lbl.numpy().reshape(-1)
        scores_g.extend(scores[l == 1].tolist())
        scores_n.extend(scores[l == 0].tolist())

    return np.asarray(scores_g, dtype=np.float32), np.asarray(scores_n, dtype=np.float32)


def calibrate_tau(chain, score_tensor, gamma_files, config=None, compute_temp=True):
    """Pick ``tau`` (and optionally ``temp``) from the pre-threshold score split.

    ``tau`` is the midpoint between the NSB 95th percentile (rare-fire) and the
    gamma median (typical-fire), placing the sigmoid's most-sensitive region in
    the discriminative band. ``temp`` makes the sigmoid transition over that band
    (~3/temp wide), not over a single ADC count.

    Returns ``(init_tau, temp_or_None, scores_g, scores_n)``.
    """
    config = config or _default_calib_config()
    scores_g, scores_n = _score_distributions(chain, score_tensor, gamma_files, config)
    if scores_g.size == 0 or scores_n.size == 0:
        raise RuntimeError(
            f"Tau calibration got too few samples (gamma={scores_g.size}, nsb={scores_n.size})."
        )

    nsb_hi = float(np.percentile(scores_n, 95))
    gam_md = float(np.median(scores_g))
    init_tau = 0.5 * (nsb_hi + gam_md)

    temp = None
    if compute_temp:
        band = max(abs(gam_md - nsb_hi), 1.0)
        temp = 3.0 / band

    return init_tau, temp, scores_g, scores_n
