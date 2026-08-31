"""Held-out best-filter (and best-OR-column) selection.

After training, score every filter once on a held-out set the early-stopping
split never saw and rank by the metric that matters -- gamma efficiency at the
target NSB rate -- with a bootstrap stability tiebreak. Also keeps the older
history-based picker for reference. Split out of the sandbox's ``train_utils.py``;
behaviour unchanged.
"""

import numpy as np
import tensorflow as tf

from triggerkit.FileIO.FileOpenerCTAO import (
    AsyncFileOpenerProcess,
    SimTelTFDataset,
    SimTelTFDatasetConfig,
)


def pick_best_filter_from_history(history_dict, n_filters):
    """Return ``(best_filter, best_epoch, per_filter_auc)`` from a training history.

    EarlyStopping(restore_best_weights=True, monitor='val_auc_best') leaves the
    weights at the epoch where the best filter peaked. We therefore pick that same
    epoch and, among the F filters at that epoch, the one with the highest
    validation AUC -- so the chosen filter is consistent with the kept weights.
    Falls back to the train-side metrics if no validation curves are present.
    """
    prefix = "val_" if f"val_auc_f0" in history_dict else ""
    best_curve = np.asarray(history_dict[f"{prefix}auc_best"], dtype=float)
    best_epoch = int(np.argmax(best_curve))
    per_filter = np.array(
        [float(history_dict[f"{prefix}auc_f{k}"][best_epoch]) for k in range(n_filters)]
    )
    return int(np.argmax(per_filter)), best_epoch, per_filter


# --- Best-filter selection on held-out data ----------------------------------
# pick_best_filter_from_history (above) reads the training history: it takes the
# epoch that maximized val_auc_best and, among filters at that epoch, the highest
# val_auc. Two biases come with that: (1) the metric pairs gamma vs NSB only
# *within* each batch, so it approximates -- not equals -- the global AUC; and
# (2) taking the max over F filters x many epochs is a winner's-curse (the lucky
# noise draw wins). It also ranks on threshold-free AUC, while the trigger only
# ever runs at one operating point (a fixed NSB rate).
#
# The functions here replace that with an honest selection: after training (when
# EarlyStopping has already frozen the weights), score every filter once on a
# *held-out* set the early-stopping split never saw, and rank by the metric that
# matters -- gamma efficiency at the target NSB rate -- with a bootstrap stability
# tiebreak. Nothing about training changes; only how the winner is chosen.


def _selection_config(max_gamma=25_000, max_nsb=25_000, batch_size=128):
    """Dataset config for held-out filter selection (more samples than calib)."""
    return SimTelTFDatasetConfig(
        batch_size=batch_size,
        shuffle_samples=False,
        sample_shuffle_buffer=2000,
        seed=1337,
        load_ram=False,
        interleave_files=True,
        waveform_level="r0",
        gamma_tel_id_only=1,
        gamma_n_pe_max=None,
        gamma_n_pe_min=None,
        gamma_skip_if_missing_n_pe=True,
        include_event_features=True,
        event_feature_keys=("n_pe", "energy"),
        # Was a fixed +1 shift applied to every NSB event (nsb_roll_copies=1),
        # the same rotation dataset-wide. Replaced with a random per-batch,
        # per-row rotation -- own seed so this held-out/"test" split's
        # rotations are independent of train_chain's and calibration's.
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
        nsb_roll_axis=1,
        nsb_roll_augment=True,
        nsb_roll_seed=9001,
        max_gamma_samples_total=max_gamma,
        max_nsb_samples_total=max_nsb,
        repeat=False,
        ignore_errors=True,
    )


def collect_multi_filter_scores(chain, score_outputs, gamma_files, config=None):
    """Per-filter event scores on a held-out gamma+NSB set, split by class.

    ``score_outputs`` is the multi-filter pre-threshold score tensor ``(B, F)``
    (single chain) or a list of such tensors (one per OR branch). Runs ONE forward
    pass over ``gamma_files`` + the chain's NSB files and returns, per output, the
    gamma and NSB score arrays of shape ``(n_events, F)``.

    Returns ``(scores_g, scores_n)`` for a single output, or a list of such pairs
    (output order preserved) when ``score_outputs`` is a list. Pass gamma files the
    training/early-stopping split never used so the selection is unbiased.
    """
    config = config or _selection_config()
    single = not isinstance(score_outputs, (list, tuple))
    outputs = [score_outputs] if single else list(score_outputs)

    if chain.camera_name == "DigiCam_R0Alpha":
        pre_inputs = chain.input_layer
        pack_inputs = lambda wf, ped: wf
    else:
        pre_inputs = [chain.input_layer, chain.input_baseline]
        pack_inputs = lambda wf, ped: (wf, ped)
    score_model = tf.keras.Model(inputs=pre_inputs, outputs=outputs)

    ds = SimTelTFDataset(
        gamma_files=gamma_files,
        nsb_files=chain.simtel_nsb_path,
        opener_cls=AsyncFileOpenerProcess,
        config=config,
    ).dataset()

    g_chunks = [[] for _ in outputs]
    n_chunks = [[] for _ in outputs]
    for feat, lbl in ds:
        wf = tf.reshape(tf.cast(feat["waveform"], tf.uint16),
                        (-1, chain.num_pixels, chain.num_samples))
        ped = tf.reshape(tf.cast(feat["pedestal"], tf.int32), (-1, chain.num_pixels))
        preds = score_model(pack_inputs(wf, ped), training=False)
        if single:
            preds = [preds]
        l = lbl.numpy().reshape(-1)
        is_g, is_n = l == 1, l == 0
        for i, p in enumerate(preds):
            arr = np.asarray(p).reshape(l.shape[0], -1)  # (B, F)
            g_chunks[i].append(arr[is_g])
            n_chunks[i].append(arr[is_n])

    pairs = []
    for i in range(len(outputs)):
        g = np.concatenate(g_chunks[i], axis=0) if g_chunks[i] else np.empty((0, 1), np.float32)
        n = np.concatenate(n_chunks[i], axis=0) if n_chunks[i] else np.empty((0, 1), np.float32)
        pairs.append((g.astype(np.float32), n.astype(np.float32)))
    return pairs[0] if single else pairs


def _efficiency_at_rate(scores_g, scores_n, target_fraction):
    """Gamma efficiency of a single score column at a target NSB fire fraction.

    Sets tau so ``P(nsb > tau) == target_fraction`` (the deploy-time ``score > tau``
    rule), then returns ``(tau, gamma_eff, achieved_fraction)``. tau is the
    ``(1 - target_fraction)`` quantile of the NSB scores; ties make the achieved
    fraction land near, not exactly on, the target.
    """
    scores_g = np.asarray(scores_g, np.float32).ravel()
    scores_n = np.asarray(scores_n, np.float32).ravel()
    if scores_n.size == 0 or scores_g.size == 0:
        return float("nan"), float("nan"), float("nan")
    target_fraction = float(np.clip(target_fraction, 0.0, 1.0))
    tau = float(np.quantile(scores_n, 1.0 - target_fraction))
    achieved = float(np.mean(scores_n > tau))
    gamma_eff = float(np.mean(scores_g > tau))
    return tau, gamma_eff, achieved


def select_best_filter(chain, score_tensor, selection_gamma_files,
                       target_rate_hz, config=None,
                       n_bootstrap=200, tie_margin=0.005, seed=1337):
    """Pick the filter that generalizes best, on held-out data, at the target rate.

    Scores all F filters once on ``selection_gamma_files`` (use files the training
    /early-stopping split never saw), then ranks each filter by its gamma
    efficiency when its threshold is set to the target NSB rate -- the actual
    operating point, not threshold-free AUC. Among filters within ``tie_margin`` of
    the best efficiency, a bootstrap over events breaks the tie in favour of the
    most *stable* filter (highest mean efficiency across resamples), guarding
    against a single lucky draw winning.

    Returns ``(best_filter, info)`` where ``info`` has per-filter ``efficiency``,
    ``tau``, ``achieved_fraction`` and the bootstrap ``mean``/``std`` for tiebreak
    candidates. Does not modify the chain.
    """
    scores_g, scores_n = collect_multi_filter_scores(
        chain, score_tensor, selection_gamma_files, config=config
    )
    if scores_g.size == 0 or scores_n.size == 0:
        raise RuntimeError(
            f"select_best_filter got too few held-out samples "
            f"(gamma={scores_g.shape}, nsb={scores_n.shape})."
        )
    n_filters = scores_g.shape[1]
    target_fraction = float(target_rate_hz) * chain.window_size

    eff = np.full(n_filters, np.nan)
    taus = np.full(n_filters, np.nan)
    achieved = np.full(n_filters, np.nan)
    for f in range(n_filters):
        taus[f], eff[f], achieved[f] = _efficiency_at_rate(
            scores_g[:, f], scores_n[:, f], target_fraction
        )

    best_eff = float(np.nanmax(eff))
    candidates = np.flatnonzero(eff >= best_eff - tie_margin)

    boot_mean = {f: float(eff[f]) for f in candidates}
    boot_std = {f: 0.0 for f in candidates}
    if candidates.size > 1 and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        ng, nn = scores_g.shape[0], scores_n.shape[0]
        per_filter_samples = {int(f): [] for f in candidates}
        for _ in range(n_bootstrap):
            gi = rng.integers(0, ng, ng)
            ni = rng.integers(0, nn, nn)
            for f in candidates:
                _, e, _ = _efficiency_at_rate(
                    scores_g[gi, f], scores_n[ni, f], target_fraction
                )
                per_filter_samples[int(f)].append(e)
        for f in candidates:
            arr = np.asarray(per_filter_samples[int(f)], np.float32)
            boot_mean[int(f)] = float(np.nanmean(arr))
            boot_std[int(f)] = float(np.nanstd(arr))
        best_f = int(max(candidates, key=lambda f: boot_mean[int(f)]))
    else:
        best_f = int(candidates[0])

    info = {
        "efficiency": eff,
        "tau": taus,
        "achieved_fraction": achieved,
        "target_fraction": target_fraction,
        "candidates": candidates.tolist(),
        "bootstrap_mean": boot_mean,
        "bootstrap_std": boot_std,
        "n_gamma": int(scores_g.shape[0]),
        "n_nsb": int(scores_n.shape[0]),
    }
    return best_f, info


def _paired_efficiency_at_rate(branch_g, branch_n, target_fraction, n_cand=48):
    """Best combined gamma efficiency of an OR of branches at the target NSB rate.

    ``branch_g``/``branch_n`` are lists (one per branch) of 1-D score arrays for a
    single filter column. Searches per-branch taus so the OR's NSB fire fraction is
    within reach of ``target_fraction`` and gamma efficiency is maximized. Full grid
    for 2 branches, coordinate ascent for more. Returns ``(taus, gamma_eff,
    achieved_fraction)``.
    """
    n_branches = len(branch_n)
    if any(n.size == 0 for n in branch_n) or any(g.size == 0 for g in branch_g):
        return [float("nan")] * n_branches, float("nan"), float("nan")
    qs = np.linspace(0.0, 1.0, n_cand)
    cand = [np.unique(np.quantile(branch_n[i], qs).astype(np.float32)) for i in range(n_branches)]

    def or_frac(scores_list, taus):
        fired = np.zeros(scores_list[0].shape[0], dtype=bool)
        for i, t in enumerate(taus):
            fired |= scores_list[i] > t
        return float(np.mean(fired))

    best = None  # (in_tol, gamma_eff, -|frac-target|), taus, eff, frac
    if n_branches == 2:
        for ta in cand[0]:
            fa_n = branch_n[0] > ta
            fa_g = branch_g[0] > ta
            for tb in cand[1]:
                frac = float(np.mean(fa_n | (branch_n[1] > tb)))
                eff = float(np.mean(fa_g | (branch_g[1] > tb)))
                in_tol = frac <= target_fraction * 1.05  # at or below target rate
                key = (1 if in_tol else 0,
                       eff if in_tol else -abs(frac - target_fraction),
                       float(ta + tb))
                if best is None or key > best[0]:
                    best = (key, [float(ta), float(tb)], eff, frac)
        _, taus, eff, frac = best
        return taus, eff, frac
    # coordinate ascent for >2 branches — same key convention as the 2-branch grid:
    # prefer at-or-below target rate (in_tol), maximize efficiency within tolerance.
    taus = [float(np.quantile(branch_n[i], 1.0 - target_fraction / n_branches))
            for i in range(n_branches)]
    for _ in range(3):
        for i in range(n_branches):
            best_i = None
            for t in cand[i]:
                trial = list(taus); trial[i] = float(t)
                frac = or_frac(branch_n, trial)
                eff = or_frac(branch_g, trial)
                in_tol = frac <= target_fraction * 1.05
                key = (1 if in_tol else 0,
                       eff if in_tol else -abs(frac - target_fraction),
                       float(t))
                if best_i is None or key > best_i[0]:
                    best_i = (key, float(t), eff, frac)
            taus[i] = best_i[1]
    return taus, or_frac(branch_g, taus), or_frac(branch_n, taus)


def select_best_joint_column(chain, branch_score_tensors, selection_gamma_files,
                             target_rate_hz, config=None):
    """Pick the OR joint column that generalizes best, on held-out data.

    ``branch_score_tensors`` is the list of per-branch multi-filter pre-threshold
    scores ``(B, F)``. For each filter column f the branches train as a *pair*
    (column f of every branch), so we rank columns by the combined gamma efficiency
    of their OR when the per-branch thresholds are tuned to the target NSB rate.
    Returns ``(best_column, info)``; does not modify the chain.
    """
    pairs = collect_multi_filter_scores(
        chain, branch_score_tensors, selection_gamma_files, config=config
    )
    branch_g = [g for g, _ in pairs]
    branch_n = [n for _, n in pairs]
    n_filters = branch_g[0].shape[1]
    target_fraction = float(target_rate_hz) * chain.window_size

    eff = np.full(n_filters, np.nan)
    fracs = np.full(n_filters, np.nan)
    for f in range(n_filters):
        _, eff[f], fracs[f] = _paired_efficiency_at_rate(
            [g[:, f] for g in branch_g],
            [n[:, f] for n in branch_n],
            target_fraction,
        )
    best_col = int(np.nanargmax(eff))
    info = {
        "efficiency": eff,
        "achieved_fraction": fracs,
        "target_fraction": target_fraction,
        "n_gamma": int(branch_g[0].shape[0]),
        "n_nsb": int(branch_n[0].shape[0]),
    }
    return best_col, info
