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

Time windows
------------
By default every stage sees the whole datacube of the files. ``windows=`` gives
each stage the samples it looks at, in sample indices of the files' datacube:

* :class:`Full` -- the whole datacube (default).
* :class:`Fixed` ``(start, length)`` -- always the same samples, e.g. the first 32
  of 50 (what a simulated chain that ignores the trigger time does), or the central
  50 of an 82-sample datacube for the first stage.
* :class:`AlignedClamped` ``(length, offset)`` -- ``length`` samples starting
  ``offset`` samples before the trigger time of a stage above, shifted back inside
  the datacube when they would leave it. No sample is ever invented: the pulse is in
  the window, but not always at the same place.
* :class:`AlignedStrict` ``(length, offset)`` -- exactly ``offset`` samples before
  the trigger time, taken from the real samples of a datacube longer than the upstream
  windows (e.g. 82 samples with the first stage on the central 50). The cascade
  checks at construction that the window can never leave the datacube and raises
  otherwise.

The trigger time of a stage is read on its score *before* the global pooling (the
per-sample map the threshold is applied to): ``rising`` = first sample above the
threshold, ``falling`` = last sample of that first pulse, ``middle`` =
``(rising + falling) // 2``. Only the first pulse of the window counts. A stage
without a time axis in front of its pooling (e.g. a CNN) cannot be the reference of
an aligned window: pick an earlier stage with ``ref_stage=``.

A stage that sees ``L`` samples must be built for ``L`` samples:
``TriggerChain(..., num_samples=L)`` (a reloaded model already has its input
length; the cascade checks it). Rates are normalised to the first stage's window
(``stages[0].window_size``), i.e. to the part of the datacube the cascade actually
triggers on.

Per-event output
----------------
``compute_statistics(folder=...)`` also writes one standard stats file
(:class:`H5StatsWriter`) per stage, readable by ``StatPlotter``: the file of stage
``k`` holds the *cumulative* output of the cascade up to stage ``k`` (its
``triggered`` column), so each stage is a "config" that can be plotted against
n_pe / energy like any chain. :meth:`TriggerCascade.stats_config` gives the config
to look it up. Every file also carries, for every stage ``j``: ``reached_stage{j}``,
``fired_stage{j}``, ``score_stage{j}`` (pre-threshold score, NaN where the event did
not reach the stage), ``t_trig_stage{j}`` and ``window_start_stage{j}`` (datacube
samples, -1 where undefined).
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

from triggerkit.FileIO.FileOpenerCTAO import (
    AsyncFileOpenerProcess,
    SimTelTFDataset,
    iterate_batches,
    SimTelTFDatasetConfig,
)
from triggerkit.Stages.OrMerge import OrMerge
from triggerkit.Stages.TrainSoftMaxPool2D import TrainSoftMaxPool2D
from triggerkit.Statistics.H5StatsWriter import H5StatsWriter


# --------------------------------------------------------------------- windows
@dataclass(frozen=True)
class Full:
    """The whole datacube of the files."""


@dataclass(frozen=True)
class Fixed:
    """Samples ``start .. start + length - 1`` of the datacube, for every event."""
    start: int
    length: int


@dataclass(frozen=True)
class AlignedClamped:
    """``length`` samples from ``t_trig - offset``, shifted back inside the datacube.

    ``ref_stage``: index of the stage above whose trigger time is used; ``None`` =
    the closest stage above that has a time axis.
    """
    length: int
    offset: int
    ref_stage: Optional[int] = None


@dataclass(frozen=True)
class AlignedStrict:
    """``length`` samples from exactly ``t_trig - offset``; never leaves the datacube.

    Needs a datacube longer than the upstream windows; the cascade refuses a
    configuration in which the window could leave it. ``ref_stage`` as for
    :class:`AlignedClamped`.
    """
    length: int
    offset: int
    ref_stage: Optional[int] = None


TRIG_TIME_MODES = ("rising", "falling", "middle")


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


def trigger_times(above, mode="rising"):
    """Trigger time of each row of a boolean ``(n, T)`` above-threshold map.

    Only the first pulse (first run of consecutive ``True``) counts. Every row must
    contain at least one ``True``.
    """
    if mode not in TRIG_TIME_MODES:
        raise ValueError(f"trigger time mode must be one of {TRIG_TIME_MODES}, got {mode!r}.")
    above = np.asarray(above, dtype=bool)
    if above.ndim != 2 or not above.any(axis=1).all():
        raise ValueError("trigger_times needs an (n, T) map with a sample above threshold in every row.")
    n_t = above.shape[1]
    rising = above.argmax(axis=1)
    if mode == "rising":
        return rising
    after = np.arange(n_t)[None, :] > rising[:, None]
    ended = after & ~above                             # first sample back below threshold
    falling = np.where(ended.any(axis=1), ended.argmax(axis=1) - 1, n_t - 1)
    if mode == "falling":
        return falling
    return (rising + falling) // 2


def _crop(wf, starts, length):
    """``wf[i, :, starts[i] : starts[i] + length]`` for every event ``i`` (numpy)."""
    n, n_pix, n_t = wf.shape
    if length == n_t and not np.any(starts):
        return wf
    idx = starts[:, None] + np.arange(length)[None, :]
    idx = np.broadcast_to(idx[:, None, :], (n, n_pix, length))
    return np.take_along_axis(wf, idx, axis=2)


class TriggerCascade:
    """An ordered cascade of frozen :class:`TriggerChain` stages.

    Parameters
    ----------
    stages : sequence of TriggerChain
        Built + compiled chains, in trigger order. Each must end in a fire
        decision (its ``model`` output collapses to a per-event 0/1 / probability).
    names : sequence of str or None
        Human labels for reporting; defaults to ``stage0, stage1, ...``.
    windows : sequence of window specs or None
        One of :class:`Full`, :class:`Fixed`, :class:`AlignedClamped`,
        :class:`AlignedStrict` per stage (see the module docstring). ``None`` = every
        stage sees the whole datacube.
    trig_time : str or sequence of str
        How the trigger time of a stage is defined when a later window is aligned on
        it: ``"rising"`` (default), ``"falling"`` or ``"middle"``; one value for all
        stages or one per stage.

    All stages are assumed to share the camera geometry / waveform shape and the
    same gamma & NSB files (taken from ``stages[0]`` unless overridden per call).
    """

    def __init__(self, stages, names=None, windows=None, trig_time="rising"):
        self.stages = list(stages)
        if len(self.stages) < 2:
            raise ValueError("A cascade needs at least two stages.")
        n = len(self.stages)
        self.names = list(names) if names is not None else [f"stage{i}" for i in range(n)]
        if len(self.names) != n:
            raise ValueError("names must match the number of stages.")

        for k, st in enumerate(self.stages):
            if getattr(st, "model", None) is None:
                raise ValueError(f"stage {k} ({self.names[k]}) has no model: call compile_chain() first.")
            model_len = st.model.inputs[0].shape[2]
            if model_len is not None and model_len != st.num_samples:
                raise ValueError(f"stage {k} ({self.names[k]}): its model takes {model_len} samples "
                                 f"but the chain says {st.num_samples} (a reloaded model trained on "
                                 "another length?).")
        self.num_pixels = self.stages[0].num_pixels
        self.data_num_samples = self._file_samples(self.stages[0])
        for k, st in enumerate(self.stages):
            if st.num_pixels != self.num_pixels:
                raise ValueError(f"stage {k} ({self.names[k]}) takes {st.num_pixels} pixels, "
                                 f"stage 0 takes {self.num_pixels}.")
            if self._file_samples(st) != self.data_num_samples:
                raise ValueError(f"stage {k} ({self.names[k]}) was built on files with "
                                 f"{self._file_samples(st)} samples, stage 0 on "
                                 f"{self.data_num_samples}.")

        self.windows = list(windows) if windows is not None else [Full()] * n
        if len(self.windows) != n:
            raise ValueError("windows must have one entry per stage.")
        self.trig_time = [trig_time] * n if isinstance(trig_time, str) else list(trig_time)
        if len(self.trig_time) != n:
            raise ValueError("trig_time must be one string or one entry per stage.")
        for mode in self.trig_time:
            if mode not in TRIG_TIME_MODES:
                raise ValueError(f"trig_time must be one of {TRIG_TIME_MODES}, got {mode!r}.")

        self._probes = {}      # stage -> one model giving fire output, score and time map
        self.ref_stage = [None] * n
        self.sample_ranges = []
        self._check_windows()
        self._ref_stages = {r for r in self.ref_stage if r is not None}

    @staticmethod
    def _file_samples(stage):
        return int(getattr(stage, "file_num_samples", stage.num_samples))

    def _label(self, k):
        return f"stage {k} ({self.names[k]})"

    # ------------------------------------------------------------------ #
    def _check_windows(self):
        """Validate every window and compute the samples each stage can look at.

        ``sample_ranges[k] = (lo, hi)``: the first and last datacube sample stage ``k``
        can see over all events. An aligned window inherits the range of its reference
        stage (its trigger time lies in it), which is how ``AlignedStrict`` is proven
        never to leave the datacube before any event is read.
        """
        n_t = self.data_num_samples
        for k, (stage, w) in enumerate(zip(self.stages, self.windows)):
            if not isinstance(w, (Full, Fixed, AlignedClamped, AlignedStrict)):
                raise TypeError(f"{self._label(k)}: unknown window {w!r}.")
            length = n_t if isinstance(w, Full) else w.length
            if length is None or not 1 <= int(length) <= n_t:
                raise ValueError(f"{self._label(k)}: window length {length} must be between 1 "
                                 f"and the datacube length {n_t}.")
            if stage.num_samples != length:
                raise ValueError(f"{self._label(k)}: the window has {length} samples but the "
                                 f"stage takes {stage.num_samples}; build it with "
                                 f"TriggerChain(..., num_samples={length}).")

            if isinstance(w, Full):
                self.sample_ranges.append((0, n_t - 1))
            elif isinstance(w, Fixed):
                if w.start < 0 or w.start + w.length > n_t:
                    raise ValueError(f"{self._label(k)}: Fixed({w.start}, {w.length}) leaves the "
                                     f"datacube (samples 0..{n_t - 1}).")
                self.sample_ranges.append((w.start, w.start + w.length - 1))
            else:
                ref = self._resolve_ref(k, w.ref_stage)
                self.ref_stage[k] = ref
                lo_ref, hi_ref = self.sample_ranges[ref]
                start_lo, start_hi = lo_ref - w.offset, hi_ref - w.offset
                if isinstance(w, AlignedClamped):
                    start_lo = int(np.clip(start_lo, 0, n_t - w.length))
                    start_hi = int(np.clip(start_hi, 0, n_t - w.length))
                elif start_lo < 0 or start_hi + w.length > n_t:
                    raise ValueError(
                        f"{self._label(k)}: AlignedStrict(length={w.length}, offset={w.offset}) on "
                        f"the trigger of {self._label(ref)}, which can fire on samples "
                        f"{lo_ref}..{hi_ref}, needs samples {start_lo}..{start_hi + w.length - 1}, "
                        f"but the datacube holds 0..{n_t - 1}. Use a longer datacube or a "
                        "narrower upstream window.")
                self.sample_ranges.append((start_lo, start_hi + w.length - 1))

    def _resolve_ref(self, k, ref):
        """Reference stage of an aligned window of stage ``k`` (explicit or the closest one)."""
        if k == 0:
            raise ValueError("stage 0 has no stage above it: its window cannot be aligned.")
        if ref is None:
            for j in range(k - 1, -1, -1):
                if self._time_map_layer(j) is not None:
                    return j
            raise ValueError(f"{self._label(k)}: no stage above it has a time axis to align on.")
        if not 0 <= ref < k:
            raise ValueError(f"{self._label(k)}: ref_stage={ref} must be a stage above it (0..{k - 1}).")
        if self._time_map_layer(ref) is None:
            raise ValueError(f"{self._label(k)}: {self._label(ref)} has no per-sample score in "
                             "front of its global pooling, so it has no trigger time.")
        return ref

    def _time_map_layer(self, k):
        """The global pooling layer whose input is stage ``k``'s (B, N, T, F) score map, or None.

        Only when that pooling feeds the stage's last threshold directly: then "a sample
        of the map is above tau" is exactly what makes the stage fire. Any layer in
        between (rescaling, ...) or an OR of several branches would make the comparison
        meaningless, so such a stage has no trigger time.
        """
        stage = self.stages[k]
        thr = stage._get_last_trainable_threshold_layer()
        if thr is None or any(isinstance(layer, OrMerge) for layer in stage.model.layers):
            return None
        history = getattr(thr.input, "_keras_history", None)
        pool = history[0] if history else None
        if not isinstance(pool, (tf.keras.layers.GlobalMaxPooling2D, TrainSoftMaxPool2D)):
            return None
        shape = tuple(pool.input.shape)
        if len(shape) != 4 or shape[2] != stage.num_samples:
            return None
        return pool

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

    def _probe(self, k):
        """One model per stage: (fire output, pre-threshold score, per-sample map), cached.

        Score and map are ``None`` when the stage has no threshold / no time axis.
        Returns ``(model, threshold_layer, has_score, has_map)``.
        """
        if k not in self._probes:
            stage = self.stages[k]
            thr = stage._get_last_trainable_threshold_layer()
            pool = self._time_map_layer(k)            # None unless thr is fed by it
            outputs = [stage.model.outputs[0]]
            if thr is not None:
                outputs.append(thr.input)
            if pool is not None:
                outputs.append(pool.input)
            model = tf.keras.Model(inputs=stage.model.inputs, outputs=outputs)
            self._probes[k] = (model, thr, thr is not None, pool is not None)
        return self._probes[k]

    def _run_stage(self, k, inputs):
        """Fire mask, pre-threshold score and trigger time (in the stage's window) of stage ``k``."""
        model, thr, has_score, has_map = self._probe(k)
        res = model(inputs, training=False)
        res = res if isinstance(res, (list, tuple)) else [res]
        fired = _collapse_to_event(res[0]) > 0.5
        n = fired.shape[0]
        score = (_collapse_to_event(res[1]).astype(np.float32) if has_score
                 else np.full(n, np.nan, np.float32))
        times = np.full(n, -1, dtype=np.int64)
        if has_map:
            per_sample = tf.cast(res[2], tf.float32).numpy().max(axis=(1, 3))     # (n, T)
            tau = float(thr.tau.numpy().flatten()[0])
            above = per_sample >= tau if thr.comparison == "ge" else per_sample > tau
            ok = fired & above.any(axis=1)
            if k in self._ref_stages and not np.array_equal(ok, fired):
                raise RuntimeError(
                    f"{self._label(k)} fired on events with no sample above its threshold in the "
                    "map in front of its global pooling: that map is not the score the threshold "
                    "cuts, so its trigger time cannot be derived.")
            if ok.any():
                times[ok] = trigger_times(above[ok], self.trig_time[k])
        return fired, score, times

    def _fire_and_times(self, k, inputs):
        """Fire mask of stage ``k`` and, for the fired events, the trigger time in its window."""
        fired, _score, times = self._run_stage(k, inputs)
        return fired, times

    def _window_starts(self, k, rows, t_abs):
        """Datacube index of the first sample stage ``k`` sees, for the events ``rows``."""
        w, n_t = self.windows[k], self.data_num_samples
        if isinstance(w, Full):
            return np.zeros(rows.size, dtype=np.int64)
        if isinstance(w, Fixed):
            return np.full(rows.size, w.start, dtype=np.int64)
        t_ref = t_abs[self.ref_stage[k]][rows]
        if np.any(t_ref < 0):
            raise RuntimeError(f"{self._label(k)}: an event reached it without a trigger time "
                               f"from {self._label(self.ref_stage[k])}.")
        starts = t_ref - w.offset
        if isinstance(w, AlignedClamped):
            return np.clip(starts, 0, n_t - w.length)
        bad = (starts < 0) | (starts + w.length > n_t)
        if np.any(bad):
            raise RuntimeError(f"{self._label(k)}: the AlignedStrict window left the datacube "
                               f"(trigger times {np.unique(t_ref[bad])[:5]}); the files do not "
                               "have the datacube length the cascade was built for.")
        return starts

    def _propagate(self, wf, ped, stop_at=None):
        """Run one batch through the cascade.

        Returns ``(passed, scores, record)``:

        * ``passed[k]`` = events of the batch that fired on stages ``0..k``;
        * with ``stop_at=k`` the propagation stops in front of stage ``k`` and
          ``scores`` are its pre-threshold scores on the events that reached it
          (``record`` is then ``None``);
        * otherwise ``record`` holds per-event ``(n_stages, B)`` arrays: ``reached``,
          ``fired`` (fired on stage k, among the events that reached it), ``score``
          (NaN where not reached), ``t_trig`` and ``start`` (datacube samples, -1).
        """
        n_st, n_ev = len(self.stages), wf.shape[0]
        rec = {"reached": np.zeros((n_st, n_ev), bool), "fired": np.zeros((n_st, n_ev), bool),
               "score": np.full((n_st, n_ev), np.nan, np.float32),
               "t_trig": np.full((n_st, n_ev), -1, np.int64),
               "start": np.full((n_st, n_ev), -1, np.int64)}
        surviving = np.ones(n_ev, dtype=bool)
        passed = [0] * n_st
        for k, stage in enumerate(self.stages):
            rows = np.flatnonzero(surviving)
            if rows.size == 0:
                if stop_at is not None:
                    return passed, np.empty((0,), np.float32), None
                break
            starts = self._window_starts(k, rows, rec["t_trig"])
            wf_k = tf.convert_to_tensor(_crop(wf[rows], starts, stage.num_samples))
            inputs = self._pack_inputs(stage, wf_k, tf.convert_to_tensor(ped[rows]))
            fired, score, t_rel = self._run_stage(k, inputs)
            if k == stop_at:
                if not self._probe(k)[2]:
                    raise ValueError("stage has no TrainableThreshold layer; cannot calibrate its rate.")
                return passed, score, None
            rec["reached"][k, rows] = True
            rec["fired"][k, rows] = fired
            rec["score"][k, rows] = score
            rec["start"][k, rows] = starts
            timed = t_rel >= 0
            rec["t_trig"][k, rows[timed]] = starts[timed] + t_rel[timed]
            surviving = np.zeros(n_ev, dtype=bool)
            surviving[rows[fired]] = True
            passed[k] = int(fired.sum())
        if stop_at is not None:
            raise ValueError(f"stop_at={stop_at} is not a stage index (0..{n_st - 1}).")
        return passed, None, rec

    def _batches(self, gamma_files, nsb_files, label, config, max_events):
        """``(wf, ped, extras)`` numpy batches of one class, whole datacubes, capped at ``max_events``."""
        ds = self._dataset(gamma_files if label == 1 else [],
                           nsb_files if label == 0 else [], config)
        n_total = 0
        for feat, lbl in iterate_batches(ds):
            keep = tf.reshape(tf.cast(lbl, tf.int32), (-1,)).numpy() == label
            if not np.any(keep):
                continue
            wf = tf.reshape(tf.cast(feat["waveform"], tf.uint16),
                            (-1, self.num_pixels, self.data_num_samples)).numpy()[keep]
            ped = tf.reshape(tf.cast(feat["pedestal"], tf.int32), (-1, self.num_pixels)).numpy()[keep]
            extras = {key: np.asarray(feat[key]).reshape(-1)[keep]
                      for key in ("event_id", "n_pe", "energy") if key in feat}
            if max_events is not None:
                # exact cap: keep only the events that still fit in the budget
                room = max_events - n_total
                if room <= 0:
                    return
                wf, ped = wf[:room], ped[:room]
                extras = {key: v[:room] for key, v in extras.items()}
            n_total += wf.shape[0]
            yield wf, ped, extras
            if max_events is not None and n_total >= max_events:
                return

    # ------------------------------------------------------------------ #
    def stats_config(self, k, base_name="cascade"):
        """Config identifying stage ``k``'s stats file in ``StatPlotter`` (``add_plot``, ``get_results``)."""
        return [("cascade_stage", {"cascade": str(base_name), "stage": int(k), "name": self.names[k]})]

    def _description(self):
        """Everything needed to rebuild the cascade, stored in each stats file."""
        stages = []
        for k, stage in enumerate(self.stages):
            thr = stage._get_last_trainable_threshold_layer()
            stages.append({
                "name": self.names[k],
                "chain": stage.generate_chain_list(),
                "num_samples": int(stage.num_samples),
                "window": {"type": type(self.windows[k]).__name__, **vars(self.windows[k])},
                "ref_stage": self.ref_stage[k],
                "trig_time": self.trig_time[k],
                "tau": None if thr is None else float(thr.tau.numpy().flatten()[0]),
                "comparison": None if thr is None else thr.comparison,
            })
        return {"datacube_samples": self.data_num_samples, "stages": stages}

    def _stats_paths(self, folder, base_name):
        def clean(text):
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
        return [os.path.join(folder, f"{clean(base_name)}_stage{k}_{clean(name)}_stats.h5")
                for k, name in enumerate(self.names)]

    def _survival_counts(self, gamma_files, nsb_files, label, config, max_events, writers=None):
        """Count events entering the cascade and surviving each stage, for one class.

        ``label`` selects gamma (1) or NSB (0). Returns ``(n_total, n_pass)`` with
        ``n_pass[k]`` = events that fired on stages ``0..k`` (cumulative). With
        ``writers`` (one :class:`H5StatsWriter` per stage) every event is also written.
        """
        n_total = 0
        n_pass = [0] * len(self.stages)
        for wf, ped, extras in self._batches(gamma_files, nsb_files, label, config, max_events):
            n_ev = wf.shape[0]
            n_total += n_ev
            passed, _, rec = self._propagate(wf, ped)
            n_pass = [a + b for a, b in zip(n_pass, passed)]
            if writers:
                cols = {
                    "label": np.full(n_ev, label, np.uint8),
                    "event_id": extras.get("event_id", np.full(n_ev, -1)).astype(np.int64),
                    "n_pe": extras.get("n_pe", np.full(n_ev, np.nan)).astype(np.float32),
                    "energy": extras.get("energy", np.full(n_ev, np.nan)).astype(np.float32),
                }
                for j in range(len(self.stages)):
                    cols[f"reached_stage{j}"] = rec["reached"][j].astype(np.uint8)
                    cols[f"fired_stage{j}"] = rec["fired"][j].astype(np.uint8)
                    cols[f"score_stage{j}"] = rec["score"][j]
                    cols[f"t_trig_stage{j}"] = rec["t_trig"][j]
                    cols[f"window_start_stage{j}"] = rec["start"][j]
                passed_up_to = np.logical_and.accumulate(rec["fired"], axis=0)
                for k, writer in enumerate(writers):
                    writer.append({**cols, "triggered": passed_up_to[k].astype(np.uint8)})
        return n_total, n_pass

    def compute_statistics(self, gamma_files=None, nsb_files=None, *,
                           batch_size=4096, tel_id_only=1, nsb_roll_copies=0,
                           nsb_skip_original_events=False, ignore_errors=True,
                           max_gamma_events=None, max_nsb_events=None,
                           folder=None, base_name="cascade", overwrite=False):
        """Per-stage NSB rates and gamma efficiencies through the cascade.

        Returns a dict with, per class, the number of events entering the cascade
        and the cumulative count surviving each stage, plus derived rates. Rates
        are ``fraction_of_input_windows_firing / window_size`` (Hz), ``window_size``
        being the first stage's window; efficiencies are cumulative pass fractions.

        With ``folder``, also writes one stats file per stage (see the module
        docstring); ``overwrite=False`` refuses to replace existing files. The paths
        are returned under ``"files"``. Defaults match ``TriggerChain.compute_statistics``
        for the NSB (every original event, no rolled copies) and read telescope 1.
        """
        if nsb_skip_original_events and int(nsb_roll_copies) <= 0:
            raise ValueError(
                "nsb_skip_original_events=True with nsb_roll_copies=0 yields NO NSB "
                "events. Set nsb_skip_original_events=False, or nsb_roll_copies>=1.")
        gamma_files = gamma_files if gamma_files is not None else self.stages[0].simtel_path
        nsb_files = nsb_files if nsb_files is not None else self.stages[0].simtel_nsb_path
        config = self._stats_config(
            batch_size, tel_id_only, nsb_roll_copies,
            nsb_skip_original_events, ignore_errors)
        ws = self.window_size

        writers, paths = [], []
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            paths = self._stats_paths(folder, base_name)
            existing = [p for p in paths if os.path.exists(p)]
            if existing and not overwrite:
                raise FileExistsError(f"{existing} already exist; pass overwrite=True to replace them.")
            description = self._description()
            camera = getattr(self.stages[0], "camera_name", "unknown")
            for k, path in enumerate(paths):
                writer = H5StatsWriter(path, trigger_chain=self.stats_config(k, base_name),
                                       camera_name=camera)
                writer.f.attrs["cascade_json"] = json.dumps(
                    H5StatsWriter._to_jsonable(description))
                writer.f.attrs["cascade_stage"] = k
                writers.append(writer)
        try:
            g_total, g_pass = self._survival_counts(
                gamma_files, nsb_files, 1, config, max_gamma_events, writers)
            n_total, n_pass = self._survival_counts(
                gamma_files, nsb_files, 0, config, max_nsb_events, writers)
            for writer in writers:
                writer.close(window_sec=ws)
        except BaseException:
            for writer, path in zip(writers, paths):     # never leave partial files behind
                try:
                    writer.f.close()
                except Exception:
                    pass
                if os.path.exists(path):
                    os.remove(path)
            raise

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
            "files": paths,
        }

    # ------------------------------------------------------------------ #
    def _survivor_scores_for_stage(self, k, nsb_files, config, max_events):
        """NSB pre-threshold scores at stage ``k`` for events that passed 0..k-1.

        Returns ``(scores, n_total, n_survivors_upstream)`` where ``scores`` are
        the stage-``k`` per-event scores of the upstream survivors.
        """
        scores = []
        n_total = 0
        for wf, ped, _extras in self._batches([], nsb_files, 0, config, max_events):
            n_total += wf.shape[0]
            _passed, sc, _rec = self._propagate(wf, ped, stop_at=k)
            scores.append(sc)
        all_scores = np.concatenate(scores) if scores else np.empty((0,), np.float32)
        return all_scores, n_total, int(all_scores.size)

    def calibrate(self, target_rates_hz, nsb_files=None, *,
                  batch_size=4096, tel_id_only=1, nsb_roll_copies=0,
                  nsb_skip_original_events=False, ignore_errors=True,
                  max_events=25_000, verbose=True):
        """Set each stage's threshold so its cascade *output* NSB rate hits its target.

        ``target_rates_hz[k]`` is the desired NSB rate at the *output* of stage
        ``k`` (i.e. events firing stages ``0..k``), in Hz. Stages are calibrated in
        order; stage ``k`` is calibrated only on the NSB events that survived the
        (already-calibrated) stages above it, each seen through its own window.
        Assigns ``tau`` in place on each stage's last threshold layer.

        Returns a list of per-stage dicts (target/achieved rate, tau, survivor count).
        """
        if len(target_rates_hz) != len(self.stages):
            raise ValueError("target_rates_hz must have one entry per stage.")
        if nsb_skip_original_events and int(nsb_roll_copies) <= 0:
            raise ValueError(
                "nsb_skip_original_events=True with nsb_roll_copies=0 yields NO NSB "
                "events. Set nsb_skip_original_events=False, or nsb_roll_copies>=1.")
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

            _model, thr, _has_score, _has_map = self._probe(k)
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
