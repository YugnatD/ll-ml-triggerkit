from __future__ import annotations

import os
import enum
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from triggerkit.FileIO.FileOpenerCTAOHDF5 import FileOpenerCTAOHDF5
from triggerkit.FileIO.FileOpenerCTAOSimtel import FileOpenerCTAOSimtel

class FileType(enum.Enum):
    SIMTEL = 1  # .simtel.gz files
    H5 = 2  # .h5 files


class FileOpenerCTAO:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file_type = self._detect_file_type(filepath)
        if self.file_type == FileType.SIMTEL:
            self._impl = FileOpenerCTAOSimtel(filepath)
        elif self.file_type == FileType.H5:
            self._impl = FileOpenerCTAOHDF5(filepath)
        else:
            raise ValueError("Unsupported file format.")

    @staticmethod
    def _detect_file_type(filepath: str) -> FileType:
        if filepath.endswith(".simtel.gz") or filepath.endswith(".simtel"):  # compressed or uncompressed
            return FileType.SIMTEL
        if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
            return FileType.H5
        return None

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def __enter__(self):
        self._impl.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._impl.__exit__(exc_type, exc_value, traceback)

    def _open(self):
        return self._impl._open()

    def _close(self):
        return self._impl._close()

    def __iter__(self):
        self._impl.__iter__()
        return self

    def __next__(self):
        return next(self._impl)


from setproctitle import setproctitle

class AsyncFileOpenerProcess:
    def __init__(self, filepath, max_queue_size=200,
                 waveform_level=None, keep_dl0=True, keep_dl1=True, keep_true_image=True):
        """
        waveform_level / keep_dl0 / keep_dl1 / keep_true_image let a caller
        that only needs part of each event's payload tell the producer to
        drop the rest BEFORE it's pickled through the Queue, instead of
        paying to serialize and transfer arrays the consumer immediately
        discards. Defaults keep everything (unchanged behavior) since
        AsyncFileOpenerProcess is shared by several consumers with different
        needs -- e.g. TriggerChain._iter_samples wants both waveform levels
        plus true_image, TriggerChain's event-finder wants dl0/dl1 too. Only
        SimTelTFDataset (which knows it wants exactly one waveform_level and
        never touches dl0/dl1/true_image) opts into trimming.
        """
        # print(f"Starting AsyncFileOpenerProcess for {filepath}")
        self.filepath = filepath
        self._queue = mp.Queue(maxsize=max_queue_size)
        self._sentinel = None  # value used to signal end of stream
        self._proc = mp.Process(
            target=self._producer,
            args=(self.filepath, self._queue, waveform_level, keep_dl0, keep_dl1, keep_true_image),
            name="CTAO Async File Opener Process",
            daemon=True,
        )
        self._proc.start()
        self._finished = False

    @staticmethod
    def _producer(filepath, q, waveform_level=None, keep_dl0=True, keep_dl1=True, keep_true_image=True):
        setproctitle("CTAO Async File Opener Process")
        try:
            for item in FileOpenerCTAO(filepath):
                (tel_ids_list, wf_r0_list, wf_r1_list, dl0_list, dl1_list,
                 true_image_list, pedestal_per_sample_list, event_stat_list, i_event) = item
                # Drop whichever fields the caller said it doesn't need before
                # they get serialized across the process boundary -- ctapipe
                # still decodes all of them (that cost is unavoidable here),
                # but this avoids pickling/transferring arrays that would be
                # thrown away on the other side of the queue.
                if waveform_level == "r0":
                    wf_r1_list = None
                elif waveform_level == "r1":
                    wf_r0_list = None
                if not keep_dl0:
                    dl0_list = None
                if not keep_dl1:
                    dl1_list = None
                if not keep_true_image:
                    true_image_list = None
                q.put((tel_ids_list, wf_r0_list, wf_r1_list, dl0_list, dl1_list,
                       true_image_list, pedestal_per_sample_list, event_stat_list, i_event))
        finally:
            # signal completion
            q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration

        item = self._queue.get()
        if item is self._sentinel:
            # make sure background process has terminated
            if self._proc.is_alive():
                self._proc.join()
            self._finished = True
            raise StopIteration

        return item
    
    def close(self):
        """Explicitly clean up the child process."""
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc.join()
        # Don't let the Queue's background feeder thread try to flush
        # whatever was left buffered when we just terminated the producer;
        # release the pipe/semaphore now instead of waiting on GC/atexit.
        self._queue.close()
        self._queue.cancel_join_thread()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



@dataclass
class SimTelTFDatasetConfig:
    batch_size: int = 32
    # Global caps across all files (None = no cap)
    max_gamma_samples_total: Optional[int] = None
    max_nsb_samples_total: Optional[int] = None

    # Shuffling
    shuffle_files: bool = True
    shuffle_samples: bool = True
    sample_shuffle_buffer: int = 10_000
    seed: int = 1337

    load_ram: bool = False  # If True, load all data into RAM at start (may use lots of memory) but faster training.

    # If True, read from all files in a round-robin/random fashion instead of
    # consuming one file completely before moving to the next. Keep true
    # to have a better mixing of gamma/nsb samples within each batch.
    interleave_files: bool = True

    # Cap on how many files (each an AsyncFileOpenerProcess subprocess + Queue)
    # are open at once when interleave_files=True. Previously every file in
    # the gamma+nsb list was opened up front and round-robined forever, so
    # e.g. 40 gamma files + 1 nsb file meant 41 concurrent subprocesses/queues
    # for the whole dataset build. A small rolling window gives essentially
    # the same gamma/nsb mixing (as long as it's >= 2) at a fraction of the
    # concurrent resource footprint; exhausted files are replaced from the
    # backlog as they finish.
    interleave_max_concurrent_files: int = 8

    # Which waveform to use: "r0" or "r1"
    waveform_level: str = "r0"  # "r0" or "r1"

    # If set, keep only this tel_id for gamma files (label=1)
    gamma_tel_id_only: Optional[int] = 2

    # -------- NEW: gamma filter by n_pe --------
    # Keep gamma only if (n_pe_min < n_pe < n_pe_max), when bounds are set.
    gamma_n_pe_max: Optional[float] = None  # e.g. 350.0
    gamma_n_pe_min: Optional[float] = None
    gamma_skip_if_missing_n_pe: bool = True

    # If True, yield extra scalar features from stat_event dict
    include_event_features: bool = True

    # Which scalar keys to include (must exist in your stat_event dict)
    event_feature_keys: Tuple[str, ...] = (
        "n_pe", "ev_time", "energy", "azimuth", "altitude",
        "h_first_int", "xmax", "xcore", "ycore"
    )

    # -------- NEW: NSB augmentation via rolling --------
    # For each NSB sample, yield N additional rolled copies (in addition to the original).
    nsb_skip_original_events: bool = False  # If True, skip the original NSB sample, keep only the rolled copies.
    nsb_roll_copies: int = 0  # N additional samples per NSB telescope-sample
    nsb_roll_axis: int = 1    # axis=1 corresponds to P in [C, P, S]

    # -------- NEW: NSB augmentation via a random per-sample circular pixel roll --------
    # Independent of nsb_roll_copies above (which duplicates samples with a
    # fixed +1,+2,+3... shift -- the same "watermark" rotation sequence for
    # every NSB event in the dataset). This instead gives every NSB row in a
    # batch its own random circular pixel shift, redrawn each time a batch is
    # produced (waveform and pedestal are shifted together so pixel identity
    # stays consistent). No duplication: one NSB event in, one rotated NSB
    # event out. Gamma rows are untouched. Give train/val/test configs
    # different nsb_roll_seed values so their rotations aren't correlated.
    nsb_roll_augment: bool = False
    nsb_roll_seed: int = 1337

    # If you want to repeat forever (useful with steps_per_epoch)
    repeat: bool = True

    # If True, skip samples that error (corrupt event, unexpected shapes, etc.)
    ignore_errors: bool = True


class SimTelTFDataset:
    """
    Streams simtel-derived waveforms into a tf.data.Dataset using:
      - gamma_files => label 1
      - nsb_file(s) => label 0

    Pattern 1: one telescope = one training sample.
    """

    def __init__(
        self,
        gamma_files: Sequence[str],
        nsb_files: Union[str, os.PathLike, Sequence[str]],  # can be a single file path
        opener_cls: Callable[..., Any],  # e.g. AsyncFileOpenerProcess
        config: Optional[SimTelTFDatasetConfig] = None,
    ):
        self.gamma_files = list(gamma_files)

        # -------- NEW: accept nsb_files as a single string/path safely --------
        if isinstance(nsb_files, (str, os.PathLike)):
            self.nsb_files = [str(nsb_files)]
        else:
            self.nsb_files = list(nsb_files)

        self.opener_cls = opener_cls
        self.cfg = config or SimTelTFDatasetConfig()
        # global quota trackers (set/reset in dataset())
        self._remaining_gamma: Optional[int] = None
        self._remaining_nsb: Optional[int] = None

        if self.cfg.waveform_level not in ("r0", "r1"):
            raise ValueError("config.waveform_level must be 'r0' or 'r1'")

        # Used when load_ram=True to cache all samples in memory.
        self._ram_cache: Optional[Tuple[Dict[str, np.ndarray], np.ndarray]] = None

    # ---------- Public API ----------

    def dataset(self) -> tf.data.Dataset:
        # reset global quotas each time a dataset pipeline is built
        self._remaining_gamma = (
            None if self.cfg.max_gamma_samples_total is None
            else int(self.cfg.max_gamma_samples_total)
        )
        self._remaining_nsb = (
            None if self.cfg.max_nsb_samples_total is None
            else int(self.cfg.max_nsb_samples_total)
        )

        rng = np.random.default_rng(self.cfg.seed)

        file_label_pairs: List[Tuple[str, int]] = (
            [(f, 1) for f in self.gamma_files] +
            [(f, 0) for f in self.nsb_files]
        )

        if self.cfg.load_ram:
            ds = self._dataset_from_ram(file_label_pairs, rng)
            ds = ds.padded_batch(
                self.cfg.batch_size,
                padded_shapes=self._padded_shapes(),
                padding_values=self._padding_values(),
                drop_remainder=False
            )
            ds = self._maybe_apply_nsb_roll(ds)
            return ds.prefetch(tf.data.AUTOTUNE)

        def infinite_generator() -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
            while True:
                if self._quotas_exhausted():
                    break
                pairs = list(file_label_pairs)
                if self.cfg.shuffle_files:
                    rng.shuffle(pairs)

                if self.cfg.interleave_files:
                    yield from self._interleaved_files(pairs, rng)
                else:
                    for filepath, label in pairs:
                        yield from self._iter_one_file(filepath, label, rng)

                if not self.cfg.repeat:
                    break
                if self._quotas_exhausted():
                    break

        ds = tf.data.Dataset.from_generator(
            infinite_generator,
            output_signature=self._output_signature()
        )

        if self.cfg.ignore_errors:
            # WARNING:tensorflow:From /home/tanguy/Desktop/tm-ctao/Code/Stat/triggerkit/FileIO/FileOpenerCTAO.py:704: ignore_errors (from tensorflow.python.data.experimental.ops.error_ops) is deprecated and will be removed in a future version.
            # ds = ds.apply(tf.data.experimental.ignore_errors())
            ds = ds.ignore_errors()

        if self.cfg.shuffle_samples:
            ds = ds.shuffle(
                self.cfg.sample_shuffle_buffer,
                seed=self.cfg.seed,
                reshuffle_each_iteration=True
            )

        ds = ds.padded_batch(
            self.cfg.batch_size,
            padded_shapes=self._padded_shapes(),
            padding_values=self._padding_values(),
            drop_remainder=False
        )
        ds = self._maybe_apply_nsb_roll(ds)

        return ds.prefetch(tf.data.AUTOTUNE)

    def _maybe_apply_nsb_roll(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        """Give each NSB row a fresh, independent random pixel-index rotation.

        Runs after padded_batch so the shift is drawn per batch (not once for
        the whole dataset like nsb_roll_copies' fixed +1,+2,+3... sequence),
        and per row within the batch, so two NSB events in the same batch get
        different rotations. tf.random.Generator.from_seed makes the sequence
        of draws reproducible for a given nsb_roll_seed across runs, while
        still varying batch to batch within one run.
        """
        if not self.cfg.nsb_roll_augment:
            return ds
        rng = tf.random.Generator.from_seed(self.cfg.nsb_roll_seed)

        def _roll_batch(features, label):
            wf = features["waveform"]   # (B, C, P, S)
            ped = features["pedestal"]  # (B, P)
            P = tf.shape(wf)[2]
            B = tf.shape(wf)[0]

            # One random shift per row; same shift applied to waveform and
            # pedestal so a pixel's ADC trace and its pedestal move together.
            shifts = rng.uniform((B,), minval=0, maxval=P, dtype=tf.int32)
            roll_idx = tf.math.floormod(tf.range(P)[None, :] - shifts[:, None], P)  # (B, P)

            wf_rolled = tf.gather(wf, roll_idx, axis=2, batch_dims=1)
            ped_rolled = tf.gather(ped, roll_idx, batch_dims=1)

            is_nsb = tf.equal(label, 0)
            wf_out = tf.where(is_nsb[:, None, None, None], wf_rolled, wf)
            ped_out = tf.where(is_nsb[:, None], ped_rolled, ped)

            features = dict(features)
            features["waveform"] = wf_out
            features["pedestal"] = ped_out
            return features, label

        return ds.map(_roll_batch, num_parallel_calls=tf.data.AUTOTUNE)

    # ---------- Internals ----------

    @staticmethod
    def _to_float_or_nan(v: Any) -> float:
        try:
            a = np.asarray(v)
            if a.size == 0:
                return float("nan")
            return float(a.reshape(-1)[0])
        except Exception:
            return float("nan")

    def _passes_gamma_n_pe_filter(self, sd: Dict[str, Any]) -> bool:
        # If no bounds set, always pass.
        if self.cfg.gamma_n_pe_max is None and self.cfg.gamma_n_pe_min is None:
            return True

        n_pe = self._to_float_or_nan(sd.get("n_pe", float("nan")))
        if np.isnan(n_pe):
            return not self.cfg.gamma_skip_if_missing_n_pe

        if self.cfg.gamma_n_pe_max is not None and not (n_pe < float(self.cfg.gamma_n_pe_max)):
            return False
        if self.cfg.gamma_n_pe_min is not None and not (n_pe > float(self.cfg.gamma_n_pe_min)):
            return False
        return True

    def _interleaved_files(
        self,
        pairs: List[Tuple[str, int]],
        rng: np.random.Generator,
    ) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """
        Pull samples from multiple files in a random round-robin fashion to avoid
        long stretches of a single label, while keeping at most
        interleave_max_concurrent_files open (each open file is a live
        AsyncFileOpenerProcess subprocess + Queue) instead of opening every
        file in `pairs` at once. Exhausted files are replaced from the backlog
        as they finish, so the active window stays roughly constant.
        """
        max_active = max(1, int(self.cfg.interleave_max_concurrent_files))
        backlog = list(pairs)
        active_streams: List[Iterator] = []

        def _refill():
            while backlog and len(active_streams) < max_active:
                fp, lbl = backlog.pop(0)
                active_streams.append(self._iter_one_file(fp, lbl, rng))

        _refill()
        try:
            while active_streams:
                if self._quotas_exhausted():
                    break
                idx = int(rng.integers(len(active_streams))) if len(active_streams) > 1 else 0
                try:
                    yield next(active_streams[idx])
                except StopIteration:
                    active_streams.pop(idx)
                    _refill()
        finally:
            # Deterministically close any streams left open (quota exhausted,
            # or this generator itself abandoned early) instead of leaving
            # their subprocess+Queue cleanup to whenever the GC gets to them.
            # close() throws GeneratorExit at the suspended yield inside each
            # stream's `with self.opener_cls(...) as fo:`, which runs its
            # __exit__ right away.
            for stream in active_streams:
                stream.close()

    def _one_pass_stream(
        self,
        pairs: List[Tuple[str, int]],
        rng: np.random.Generator,
    ) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        if self.cfg.interleave_files:
            return iter(self._interleaved_files(pairs, rng))

        def sequential():
            for filepath, label in pairs:
                if self._quota_exhausted_label(label) and self._quotas_exhausted():
                    break
                yield from self._iter_one_file(filepath, label, rng)

        return iter(sequential())

    def _load_all_into_memory(
        self,
        pairs: List[Tuple[str, int]],
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        feat_lists: Dict[str, List[np.ndarray]] = {}
        labels: List[np.ndarray] = []

        stream = self._one_pass_stream(pairs, rng)
        while True:
            try:
                features, label = next(stream)
            except StopIteration:
                break
            except Exception:
                if self.cfg.ignore_errors:
                    continue
                raise

            for k, v in features.items():
                feat_lists.setdefault(k, []).append(np.array(v))
            labels.append(np.array(label, dtype=np.int32))

        if not labels:
            return {}, np.array([], dtype=np.int32)

        feat_arrays: Dict[str, np.ndarray] = {}
        for k, vals in feat_lists.items():
            try:
                feat_arrays[k] = np.stack(vals, axis=0)
            except ValueError as exc:
                raise ValueError(
                    f"Inconsistent shapes for feature '{k}' while using load_ram; "
                    "disable load_ram or ensure fixed shapes."
                ) from exc

        labels_arr = np.array(labels, dtype=np.int32)
        return feat_arrays, labels_arr

    def _dataset_from_ram(
        self,
        file_label_pairs: List[Tuple[str, int]],
        rng: np.random.Generator,
    ) -> tf.data.Dataset:
        # ensure quotas exist when dataset() was bypassed (defensive)
        if self._remaining_gamma is None and self.cfg.max_gamma_samples_total is not None:
            self._remaining_gamma = int(self.cfg.max_gamma_samples_total)
        if self._remaining_nsb is None and self.cfg.max_nsb_samples_total is not None:
            self._remaining_nsb = int(self.cfg.max_nsb_samples_total)

        pairs = list(file_label_pairs)
        if self.cfg.shuffle_files:
            rng.shuffle(pairs)

        if self._ram_cache is None:
            feat_arrays, labels_arr = self._load_all_into_memory(pairs, rng)
            if labels_arr.size == 0:
                raise ValueError(
                    "No samples loaded into RAM; check input files, filters, or disable load_ram."
                )
            self._ram_cache = (feat_arrays, labels_arr)

        feat_arrays, labels_arr = self._ram_cache
        ds = tf.data.Dataset.from_tensor_slices((feat_arrays, labels_arr))

        if self.cfg.repeat:
            ds = ds.repeat()

        if self.cfg.shuffle_samples:
            ds = ds.shuffle(
                self.cfg.sample_shuffle_buffer,
                seed=self.cfg.seed,
                reshuffle_each_iteration=True
            )

        return ds

    def _iter_one_file(
        self,
        simtel_file: str,
        label: int,
        rng: np.random.Generator,
    ) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        # bail out early if this label quota is already exhausted
        if self._quota_exhausted_label(label):
            return

        # This is the only opener_cls call site that knows upfront exactly what
        # it needs (one waveform level, never dl0/dl1/true_image -- see below),
        # so it's the only one that opts into AsyncFileOpenerProcess's payload
        # trimming; the other TriggerChain call sites keep the untrimmed default.
        with self.opener_cls(
            simtel_file,
            waveform_level=self.cfg.waveform_level,
            keep_dl0=False,
            keep_dl1=False,
            keep_true_image=False,
        ) as fo:
            for (
                tel_ids_list, wf_r0_list, wf_r1_list, dl0_list, dl1_list, true_image_list,
                pedestal_per_sample_list, event_stat_list, i_event
            ) in fo:
                if self._quota_exhausted_label(label):
                    break

                if not event_stat_list:
                    continue

                stats_by_tel: Dict[int, Dict[str, Any]] = {}
                for d in event_stat_list:
                    if isinstance(d, dict) and "telescope" in d:
                        try:
                            stats_by_tel[int(d["telescope"])] = d
                        except Exception:
                            pass

                ev0 = event_stat_list[0] if isinstance(event_stat_list[0], dict) else {}
                event_id = np.int64(ev0.get("event_id", -1))

                wf_list = wf_r0_list if self.cfg.waveform_level == "r0" else wf_r1_list
                ped_list = pedestal_per_sample_list

                for tel_id, wf, ped in zip(tel_ids_list, wf_list, ped_list):
                    tel_id_i = int(tel_id)

                    # Gamma-only tel_id filter
                    if label == 1 and self.cfg.gamma_tel_id_only is not None:
                        if tel_id_i != int(self.cfg.gamma_tel_id_only):
                            continue

                    sd = stats_by_tel.get(tel_id_i, ev0)

                    # -------- NEW: gamma filter by n_pe --------
                    if label == 1 and not self._passes_gamma_n_pe_filter(sd):
                        continue

                    x = np.asarray(wf, dtype=np.uint16)

                    # Normalize waveform shape to [C, P, S]
                    # C = channels, P = pixels, S = samples
                    if x.ndim == 2:
                        x = x[None, ...]
                    elif x.ndim != 3:
                        raise ValueError(f"Unexpected waveform ndim={x.ndim}, shape={x.shape}")

                    if ped is None:
                        ped = np.zeros(x.shape[1], dtype=np.float32)  # assume shape [C, P, S] -> ped per pixel
                    ped_arr = np.array(ped, dtype=np.int32)

                    def make_features(wf_arr: np.ndarray, ped_vec: np.ndarray) -> Dict[str, np.ndarray]:
                        features: Dict[str, np.ndarray] = {
                            "waveform": wf_arr,
                            "pedestal": ped_vec,
                            "event_id": np.array(event_id, dtype=np.int64),
                            "n_pe": np.array(sd.get("n_pe", -1), dtype=np.float32),
                            "tel_id": np.array(tel_id_i, dtype=np.int32),
                        }
                        if self.cfg.include_event_features:
                            for k in self.cfg.event_feature_keys:
                                v = sd.get(k, -1.0)
                                features[k] = np.array(v, dtype=np.float32)
                        return features

                    y = np.array(label, dtype=np.int32)
                
                    # Yield original
                    if label == 1:  # gamma
                        if self._consume_quota(label):
                            yield make_features(x, ped_arr), y
                    else:
                        if not self.cfg.nsb_skip_original_events and self._consume_quota(label):
                            yield make_features(x, ped_arr), y

                    if label == 0 and self.cfg.nsb_roll_copies > 0:
                        n = int(self.cfg.nsb_roll_copies)
                        wf_aug = x
                        ped_aug = ped_arr
                        # print(ped_aug.shape) # (1296,)
                        # print(wf_aug.shape)  # (1,1296,50) -> (batch, pixels, samples)
                        for i in range(n):
                            wf_aug = np.roll(wf_aug, shift=1, axis=int(self.cfg.nsb_roll_axis)).copy()
                            ped_aug = np.roll(ped_aug, shift=1, axis=0).copy()
                            if self._consume_quota(label):
                                yield make_features(wf_aug, ped_aug), y
                            else:
                                break

                if self._quota_exhausted_label(label):
                    break

    def _output_signature(self):
        feat = {
            "waveform": tf.TensorSpec(shape=(None, None, None), dtype=tf.uint16),
            "pedestal": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "n_pe": tf.TensorSpec(shape=(), dtype=tf.float32),
            "event_id": tf.TensorSpec(shape=(), dtype=tf.int64),
            "tel_id": tf.TensorSpec(shape=(), dtype=tf.int32),
        }
        if self.cfg.include_event_features:
            for k in self.cfg.event_feature_keys:
                feat[k] = tf.TensorSpec(shape=(), dtype=tf.float32)

        return (feat, tf.TensorSpec(shape=(), dtype=tf.int32))

    def _padded_shapes(self):
        feat_shapes = {
            "waveform": [None, None, None],
            "pedestal": [None],
            "n_pe": [],
            "event_id": [],
            "tel_id": [],
        }
        if self.cfg.include_event_features:
            for k in self.cfg.event_feature_keys:
                feat_shapes[k] = []
        return (feat_shapes, [])

    def _padding_values(self):
        feat_vals = {
            "waveform": tf.constant(0, tf.uint16),
            "pedestal": tf.constant(0, tf.int32),
            "n_pe": tf.constant(-1.0, tf.float32),
            "event_id": tf.constant(-1, tf.int64),
            "tel_id": tf.constant(-1, tf.int32),
        }
        if self.cfg.include_event_features:
            for k in self.cfg.event_feature_keys:
                feat_vals[k] = tf.constant(0.0, tf.float32)
        return (feat_vals, tf.constant(0, tf.int32))

    # ---------- Global quota helpers ----------
    def _consume_quota(self, label: int) -> bool:
        if label == 1:
            if self._remaining_gamma is None:
                return True
            if self._remaining_gamma <= 0:
                return False
            self._remaining_gamma -= 1
            return True
        else:
            if self._remaining_nsb is None:
                return True
            if self._remaining_nsb <= 0:
                return False
            self._remaining_nsb -= 1
            return True

    def _quota_exhausted_label(self, label: int) -> bool:
        if label == 1:
            return self._remaining_gamma is not None and self._remaining_gamma <= 0
        return self._remaining_nsb is not None and self._remaining_nsb <= 0

    def _quotas_exhausted(self) -> bool:
        gamma_done = self._remaining_gamma is not None and self._remaining_gamma <= 0
        nsb_done = self._remaining_nsb is not None and self._remaining_nsb <= 0
        return gamma_done and nsb_done
