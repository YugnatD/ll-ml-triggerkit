"""Low-level per-file opener + its background-process wrapper.

Deliberately TensorFlow-free (unlike FileOpenerCTAO.py, which also defines
SimTelTFDataset and therefore imports tensorflow at module level). That
matters here specifically: AsyncFileOpenerProcess spawns a NEW child every
time the interleave window in SimTelTFDataset opens another file -- i.e.
repeatedly, throughout a run, long after the parent process has already
initialized TensorFlow/CUDA (model built, GPU context up, background threads
running). Forking at that point (Python's multiprocessing default on Linux)
duplicates the parent's memory including whatever a TF/CUDA-internal thread
happened to be holding at that exact instant; only the calling thread
survives the fork, so a lock inherited as "held" by a now-nonexistent thread
can never be released -> the child (or the parent, whose own threading state
the fork operation can also disturb) hangs forever in some unrelated-looking
C call, immune to SIGTERM. Observed in practice: a stats_tdscan.py run stuck
with every worker blocked in `futex_wait_queue`, confirmed via `py-spy dump`
to be hanging in `AsyncFileOpenerProcess.close()`'s `self._proc.join()`.

The fix is to spawn (not fork) these workers, which starts each one from a
clean interpreter that never inherits the parent's TF/CUDA state. The reason
this class lives in its own module rather than just switching the context in
FileOpenerCTAO.py: spawn re-imports whatever module defines the target
function in the new interpreter, and that module is `import tensorflow`-free
here, so re-importing it (to resolve `AsyncFileOpenerProcess._producer`) only
pulls in numpy/h5py/ctapipe -- not a second full TensorFlow+CUDA init per
file. Splitting it out keeps the fix (no hang) from costing a full TF import
per spawned worker (which would otherwise partly undo it -- files get opened
repeatedly over a run, so that cost is paid over and over).
"""

from __future__ import annotations

import enum
import queue
import warnings
import multiprocessing as mp

from setproctitle import setproctitle

from triggerkit.FileIO.FileOpenerCTAOHDF5 import FileOpenerCTAOHDF5
from triggerkit.FileIO.FileOpenerCTAOSimtel import FileOpenerCTAOSimtel

#: All AsyncFileOpenerProcess workers use this context (see module docstring)
#: instead of the platform default (fork on Linux).
_SPAWN_CTX = mp.get_context("spawn")


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


class AsyncFileOpenerProcess:
    def __init__(self, filepath, max_queue_size=200,
                 waveform_level=None, keep_dl0=True, keep_dl1=True,
                 keep_true_image=True, keep_peak_time=False):
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
        self._queue = _SPAWN_CTX.Queue(maxsize=max_queue_size)
        self._sentinel = None  # value used to signal end of stream
        self._proc = _SPAWN_CTX.Process(
            target=self._producer,
            args=(self.filepath, self._queue, waveform_level, keep_dl0,
                  keep_dl1, keep_true_image, keep_peak_time),
            name="CTAO Async File Opener Process",
            daemon=True,
        )
        self._proc.start()
        self._finished = False

    @staticmethod
    def _producer(filepath, q, waveform_level=None, keep_dl0=True,
                  keep_dl1=True, keep_true_image=True, keep_peak_time=False):
        setproctitle("CTAO Async File Opener Process")
        try:
            for item in FileOpenerCTAO(filepath):
                (tel_ids_list, wf_r0_list, wf_r1_list, dl0_list, dl1_list,
                 true_image_list, peak_time_list, pedestal_per_sample_list,
                 event_stat_list, i_event) = item
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
                if not keep_peak_time:
                    peak_time_list = None

                q.put((tel_ids_list, wf_r0_list, wf_r1_list, dl0_list, dl1_list,
                       true_image_list, peak_time_list, pedestal_per_sample_list,
                       event_stat_list, i_event))
        finally:
            # signal completion
            q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration

        # A plain blocking self._queue.get() hangs FOREVER if the producer
        # dies without reaching its `finally: q.put(None)` -- e.g. OOM-killed
        # by the kernel, or any other SIGKILL/segfault, which skips Python
        # cleanup entirely. Since _interleaved_files' round-robin is a single
        # generator serving the whole tf.data pipeline, one stream blocked
        # like this freezes ALL of it (every other file too), which looks
        # like the whole run silently stopped computing rather than one file
        # erroring out. Poll with a timeout instead, and only give up once
        # the producer process has actually died -- a merely slow (but alive)
        # producer keeps waiting normally, no behavior change for that case.
        while True:
            try:
                item = self._queue.get(timeout=30)
                break
            except queue.Empty:
                if not self._proc.is_alive():
                    self._finished = True
                    warnings.warn(
                        f"AsyncFileOpenerProcess for {self.filepath!r} died "
                        "without signaling completion (likely OOM-killed or "
                        "crashed); treating as end of stream instead of "
                        "hanging forever.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    raise StopIteration
                # still alive, just slow (large/slow file) -- keep waiting.

        if item is self._sentinel:
            # make sure background process has terminated
            if self._proc.is_alive():
                self._proc.join()
            self._finished = True
            raise StopIteration

        return item

    def close(self):
        """Explicitly clean up the child process.

        Bounded, with a SIGKILL escalation: a plain terminate() (SIGTERM) can
        fail to actually end the process promptly -- e.g. the exact fork-time
        hang this module's docstring describes leaves a child wedged in a C
        call that never returns to check for pending signals. The unbounded
        `self._proc.join()` that used to follow terminate() unconditionally
        then hangs forever too (confirmed via `py-spy dump` on a stuck run).
        Escalating to SIGKILL after a short wait makes this method actually
        bounded no matter what state the child is in.
        """
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                warnings.warn(
                    f"AsyncFileOpenerProcess for {self.filepath!r} did not "
                    "exit within 10s of SIGTERM; sending SIGKILL.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._proc.kill()
                self._proc.join(timeout=10)
        # Don't let the Queue's background feeder thread try to flush
        # whatever was left buffered when we just terminated the producer;
        # release the pipe/semaphore now instead of waiting on GC/atexit.
        self._queue.close()
        self._queue.cancel_join_thread()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
