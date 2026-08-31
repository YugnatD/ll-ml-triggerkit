"""Cascade of frozen trigger chains: TDSCAN -> CNN -> ...

A ``TriggerCascade`` chains several already-built ``TriggerChain`` stages. Each
stage is a full trigger (it ends in a threshold / fire decision); an event
reaches stage ``k+1`` only if stages ``0..k`` all fired on it. Joint training is
NOT done here -- each stage is trained on its own. What this module gives you is
the two things the cascade needs in practice:

* :meth:`TriggerCascade.calibrate` -- set each stage's threshold so its *output*
  NSB rate hits a per-stage target, with every downstream stage calibrated only
  on the NSB events that survived the stages above it.
* :meth:`TriggerCascade.compute_statistics` -- the per-stage rates (NSB) and
  efficiencies (gamma): how many events enter each stage and how many leave.

Both operate at the forward-pass / numpy level (like
``TriggerChain.find_threshold_for_target_rate``), so no differentiable gate is
needed -- the stages stay independent Keras models.
"""

import numpy as np
import tensorflow as tf

from triggerkit.FileIO.FileOpenerCTAO import (
    AsyncFileOpenerProcess,
    SimTelTFDataset,
    SimTelTFDatasetConfig,
)


def _collapse_to_event(score):
    """Reduce a per-pixel / per-filter score tensor to one value per event (max)."""
    score = tf.cast(score, tf.float32)
    rank = score.shape.rank
    if rank is None:
        score = tf.reshape(score, (tf.shape(score)[0], -1))
        return tf.reduce_max(score, axis=1).numpy()
    if rank == 1:
        return score.numpy().reshape(-1)
    axes = tuple(range(1, rank))
    return tf.reduce_max(score, axis=axes).numpy().reshape(-1)


class TriggerCascade:
    """An ordered cascade of frozen :class:`TriggerChain` stages.

    Parameters
    ----------
    stages : sequence of TriggerChain
        Built + compiled chains, in trigger order. Each must end in a fire
        decision (its ``model`` output collapses to a per-event 0/1 / probability).
    names : sequence of str or None
        Human labels for reporting; defaults to ``stage0, stage1, ...``.

    All stages are assumed to share the camera geometry / waveform shape and the
    same gamma & NSB files (taken from ``stages[0]`` unless overridden per call).
    """

    def __init__(self, stages, names=None):
        self.stages = list(stages)
        if len(self.stages) < 2:
            raise ValueError("A cascade needs at least two stages.")
        self.names = list(names) if names is not None else [
            f"stage{i}" for i in range(len(self.stages))
        ]
        if len(self.names) != len(self.stages):
            raise ValueError("names must match the number of stages.")

    @property
    def window_size(self):
        return self.stages[0].window_size

    # ------------------------------------------------------------------ #
    def _stats_config(self, batch_size, tel_id_only, nsb_roll_copies,
                      nsb_skip_original_events, ignore_errors):
        return SimTelTFDatasetConfig(
            batch_size=batch_size,
            shuffle_samples=False,
            sample_shuffle_buffer=10000,
            seed=1337,
            load_ram=False,
            interleave_files=True,
            waveform_level="r0",
            gamma_tel_id_only=tel_id_only,
            gamma_n_pe_max=None,
            gamma_n_pe_min=None,
            gamma_skip_if_missing_n_pe=True,
            include_event_features=True,
            event_feature_keys=("n_pe", "energy"),
            nsb_skip_original_events=nsb_skip_original_events,
            nsb_roll_copies=nsb_roll_copies,
            nsb_roll_axis=1,
            repeat=False,
            ignore_errors=ignore_errors,
        )

    def _dataset(self, gamma_files, nsb_files, config):
        return SimTelTFDataset(
            gamma_files=gamma_files,
            nsb_files=nsb_files,
            opener_cls=AsyncFileOpenerProcess,
            config=config,
        ).dataset()

    @staticmethod
    def _pack_inputs(stage, wf, ped):
        return wf if len(stage.model.inputs) == 1 else (wf, ped)

    def _fire_mask(self, stage, wf, ped):
        """Boolean (B,) mask: which events this stage fires on."""
        out = stage.model(self._pack_inputs(stage, wf, ped), training=False)
        return _collapse_to_event(out) > 0.5

    def _pre_threshold_model(self, stage):
        """Keras model outputting the stage's score just before its last threshold."""
        thr = stage._get_last_trainable_threshold_layer()
        if thr is None:
            raise ValueError(
                f"stage has no TrainableThreshold layer; cannot calibrate its rate.")
        return tf.keras.Model(inputs=stage.model.inputs, outputs=thr.input), thr

    def _tensors(self, stage, feat):
        wf = tf.reshape(tf.cast(feat["waveform"], tf.uint16),
                        (-1, stage.num_pixels, stage.num_samples))
        ped = tf.reshape(tf.cast(feat["pedestal"], tf.int32), (-1, stage.num_pixels))
        return wf, ped

    # ------------------------------------------------------------------ #
    def _survival_counts(self, gamma_files, nsb_files, label, config, max_events):
        """Count events entering the cascade and surviving each stage, for one class.

        ``label`` selects gamma (1) or NSB (0). Returns ``(n_total, n_pass)`` with
        ``n_pass[k]`` = events that fired on stages ``0..k`` (cumulative).
        """
        gfiles = gamma_files if label == 1 else []
        nfiles = nsb_files if label == 0 else []
        ds = self._dataset(gfiles, nfiles, config)

        n_total = 0
        n_pass = [0] * len(self.stages)
        s0 = self.stages[0]
        for feat, lbl in ds:
            l = tf.reshape(tf.cast(lbl, tf.int32), (-1,)).numpy()
            keep = l == label
            if not np.any(keep):
                continue
            wf, ped = self._tensors(s0, feat)
            wf = tf.boolean_mask(wf, keep)
            ped = tf.boolean_mask(ped, keep)
            n_total += int(wf.shape[0])

            surviving = tf.ones((wf.shape[0],), dtype=tf.bool).numpy()
            for k, stage in enumerate(self.stages):
                if not np.any(surviving):
                    break
                wf_s = tf.boolean_mask(wf, surviving)
                ped_s = tf.boolean_mask(ped, surviving)
                fired = self._fire_mask(stage, wf_s, ped_s)
                # Map the survivors' fire result back onto the full-batch mask.
                idx = np.flatnonzero(surviving)
                surviving = np.zeros_like(surviving)
                surviving[idx[fired]] = True
                n_pass[k] += int(fired.sum())

            if max_events is not None and n_total >= max_events:
                break
        return n_total, n_pass

    # ------------------------------------------------------------------ #
    def compute_statistics(self, gamma_files=None, nsb_files=None, *,
                           batch_size=4096, tel_id_only=2, nsb_roll_copies=1,
                           nsb_skip_original_events=True, ignore_errors=True,
                           max_gamma_events=None, max_nsb_events=None):
        """Per-stage NSB rates and gamma efficiencies through the cascade.

        Returns a dict with, per class, the number of events entering the cascade
        and the cumulative count surviving each stage, plus derived rates. Rates
        are ``fraction_of_input_windows_firing / window_size`` (Hz); efficiencies
        are cumulative pass fractions.
        """
        gamma_files = gamma_files if gamma_files is not None else self.stages[0].simtel_path
        nsb_files = nsb_files if nsb_files is not None else self.stages[0].simtel_nsb_path
        config = self._stats_config(
            batch_size, tel_id_only, nsb_roll_copies,
            nsb_skip_original_events, ignore_errors)

        g_total, g_pass = self._survival_counts(
            gamma_files, nsb_files, 1, config, max_gamma_events)
        n_total, n_pass = self._survival_counts(
            gamma_files, nsb_files, 0, config, max_nsb_events)

        ws = self.window_size

        def _per_stage(total, passed):
            rows = []
            prev = total
            for k in range(len(self.stages)):
                cum_frac = (passed[k] / total) if total else float("nan")
                cond_frac = (passed[k] / prev) if prev else float("nan")
                rows.append({
                    "stage": self.names[k],
                    "entered": int(prev),
                    "passed": int(passed[k]),
                    "cumulative_fraction": cum_frac,
                    "conditional_fraction": cond_frac,
                    "cumulative_rate_hz": cum_frac / ws,
                })
                prev = passed[k]
            return rows

        return {
            "window_size_s": ws,
            "gamma": {"n_total": int(g_total), "stages": _per_stage(g_total, g_pass)},
            "nsb": {"n_total": int(n_total), "stages": _per_stage(n_total, n_pass)},
        }

    # ------------------------------------------------------------------ #
    def _survivor_scores_for_stage(self, k, nsb_files, config, max_events):
        """NSB pre-threshold scores at stage ``k`` for events that passed 0..k-1.

        Returns ``(scores, n_total, n_survivors_upstream)`` where ``scores`` are
        the stage-``k`` per-event scores of the upstream survivors.
        """
        ds = self._dataset([], nsb_files, config)
        pre_model, _thr = self._pre_threshold_model(self.stages[k])

        scores = []
        n_total = 0
        n_up = 0
        s0 = self.stages[0]
        for feat, lbl in ds:
            l = tf.reshape(tf.cast(lbl, tf.int32), (-1,)).numpy()
            keep = l == 0
            if not np.any(keep):
                continue
            wf, ped = self._tensors(s0, feat)
            wf = tf.boolean_mask(wf, keep)
            ped = tf.boolean_mask(ped, keep)
            n_total += int(wf.shape[0])

            surviving = np.ones((wf.shape[0],), dtype=bool)
            for j in range(k):
                if not np.any(surviving):
                    break
                wf_s = tf.boolean_mask(wf, surviving)
                ped_s = tf.boolean_mask(ped, surviving)
                fired = self._fire_mask(self.stages[j], wf_s, ped_s)
                idx = np.flatnonzero(surviving)
                surviving = np.zeros_like(surviving)
                surviving[idx[fired]] = True

            if not np.any(surviving):
                continue
            wf_s = tf.boolean_mask(wf, surviving)
            ped_s = tf.boolean_mask(ped, surviving)
            sc = _collapse_to_event(pre_model(
                self._pack_inputs(self.stages[k], wf_s, ped_s), training=False))
            scores.append(sc.astype(np.float32))
            n_up += int(surviving.sum())

            if max_events is not None and n_total >= max_events:
                break

        all_scores = np.concatenate(scores) if scores else np.empty((0,), np.float32)
        return all_scores, n_total, n_up

    def calibrate(self, target_rates_hz, nsb_files=None, *,
                  batch_size=4096, tel_id_only=1, nsb_roll_copies=1,
                  nsb_skip_original_events=True, ignore_errors=True,
                  max_events=25_000, verbose=True):
        """Set each stage's threshold so its cascade *output* NSB rate hits its target.

        ``target_rates_hz[k]`` is the desired NSB rate at the *output* of stage
        ``k`` (i.e. events firing stages ``0..k``), in Hz. Stages are calibrated in
        order; stage ``k`` is calibrated only on the NSB events that survived the
        (already-calibrated) stages above it. Assigns ``tau`` in place on each
        stage's last threshold layer.

        Returns a list of per-stage dicts (target/achieved rate, tau, survivor count).
        """
        if len(target_rates_hz) != len(self.stages):
            raise ValueError("target_rates_hz must have one entry per stage.")
        nsb_files = nsb_files if nsb_files is not None else self.stages[0].simtel_nsb_path
        config = self._stats_config(
            batch_size, tel_id_only, nsb_roll_copies,
            nsb_skip_original_events, ignore_errors)
        ws = self.window_size

        results = []
        for k, stage in enumerate(self.stages):
            scores, n_total, n_up = self._survivor_scores_for_stage(
                k, nsb_files, config, max_events)
            if scores.size == 0 or n_total == 0:
                raise RuntimeError(
                    f"stage {k} ({self.names[k]}): no NSB survivors upstream to "
                    "calibrate on; loosen the upstream target rate.")

            # Desired fraction over ALL nsb windows, converted to a conditional
            # fraction over the upstream survivors (what this stage's scores span).
            target_frac_total = float(np.clip(target_rates_hz[k] * ws, 0.0, 1.0))
            upstream_frac = n_up / n_total
            cond_frac = float(np.clip(target_frac_total / max(upstream_frac, 1e-12), 0.0, 1.0))

            _pre, thr = self._pre_threshold_model(stage)
            comparison = getattr(thr, "comparison", "gt")
            tau, achieved_cond, mode = stage._pick_tau_from_empirical_scores(
                scores, desired_fraction=cond_frac, comparison=comparison)
            if tau is None:
                raise RuntimeError(f"stage {k}: tau selection failed (empty scores).")
            thr.tau.assign(tau)

            achieved_rate = (achieved_cond * upstream_frac) / ws
            info = {
                "stage": self.names[k],
                "target_rate_hz": float(target_rates_hz[k]),
                "achieved_rate_hz": float(achieved_rate),
                "tau": float(tau),
                "conditional_fraction": float(achieved_cond),
                "upstream_survivor_fraction": float(upstream_frac),
                "n_survivors": int(n_up),
                "selection_mode": mode,
            }
            results.append(info)
            if verbose:
                print(f"[{self.names[k]}] tau={tau:.6f} -> output NSB rate "
                      f"{achieved_rate:.1f} Hz (target {target_rates_hz[k]:.1f} Hz, "
                      f"{n_up}/{n_total} upstream survivors, {mode}).")
        return results
