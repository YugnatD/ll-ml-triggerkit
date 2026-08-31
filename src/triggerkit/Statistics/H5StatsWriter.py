import json
from collections.abc import Mapping

import h5py
import numpy as np

PRE_THRESHOLD_SCORE_DATASET = "pre_threshold_score"
PRE_THRESHOLD_REFERENCE_ATTR = "pre_threshold_reference_threshold"
PRE_THRESHOLD_AVAILABLE_ATTR = "has_pre_threshold_score"
PRE_THRESHOLD_COMPARISON_ATTR = "pre_threshold_comparison"


class H5StatsWriter:
    """
    One-file-per-config stats store.
    Layout:
      /meta attrs: camera_name, trigger_chain_json, trigger_rate_hz, window_sec, ...
      /events/<col>: 1D datasets (extendable)
        label: uint8 (1=gamma, 0=nsb)
        event_id: int64
        tel_id: int32
        n_pe: float32
        energy: float32
        ev_time, azimuth, altitude, h_first_int, xmax, xcore, ycore: float32
        tel_pos_x, tel_pos_y, tel_pos_z: float32
        n_clusters: uint8   (optional legacy hard trigger)
        p_trig: float32 (optional, threshold-layer output probability)
        pre_threshold_score: float32 (optional, scalar score immediately before TrainableThreshold)
      attrs:
        pre_threshold_comparison: "gt" or "ge" for the hard threshold rule
    """
    def __init__(self, path, trigger_chain, camera_name="unknown",
                 compression="lzf", chunk_rows=200_000):
        self.path = path
        self.f = h5py.File(path, "w")
        self.g = self.f.create_group("events")

        # Meta
        self.f.attrs["format"] = "hdf5_stats_v1"
        self.f.attrs["camera_name"] = camera_name
        # store trigger_chain as JSON (tuples become lists; ok for matching if you rebuild tuples later)
        self.f.attrs["trigger_chain_json"] = json.dumps(self._to_jsonable(trigger_chain))

        self.compression = compression
        self.chunk_rows = int(chunk_rows)
        self.n_rows = 0

        # Counters for NSB rate
        self.nsb_total = 0
        self.nsb_trig = 0
        self.gamma_total = 0

        # Optional min/max tracking for fast plotting (gamma only)
        self.gamma_npe_min = np.inf
        self.gamma_npe_max = -np.inf

        self._datasets = {}

    @classmethod
    def _to_jsonable(cls, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_jsonable(v) for v in value]
        if hasattr(value, "tolist"):
            return cls._to_jsonable(value.tolist())
        return str(value)

    def _require_ds(self, name, dtype):
        if name in self._datasets:
            return self._datasets[name]
        ds = self.g.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(self.chunk_rows,),
            compression=self.compression
        )
        self._datasets[name] = ds
        return ds

    def append(self, cols: dict):
        """
        cols: dict of {name: 1D numpy array}, all same length.
        """
        # basic checks
        keys = list(cols.keys())
        n = len(cols[keys[0]])
        for k in keys[1:]:
            if len(cols[k]) != n:
                raise ValueError(f"Column {k} has len {len(cols[k])} != {n}")

        old = self.n_rows
        new = old + n

        # create/resize/write each column
        for name, arr in cols.items():
            arr = np.asarray(arr)
            ds = self._require_ds(name, arr.dtype)
            ds.resize((new,))
            ds[old:new] = arr

        self.n_rows = new

        # Update counters/minmax.
        label = cols.get("label", None)
        n_clusters = cols.get("n_clusters", None)
        pre_threshold_score = cols.get(PRE_THRESHOLD_SCORE_DATASET, None)
        n_pe = cols.get("n_pe", None)

        if label is not None:
            label = np.asarray(label)
            if n_clusters is not None:
                n_clusters = np.asarray(n_clusters)
            if pre_threshold_score is not None:
                pre_threshold_score = np.asarray(pre_threshold_score, dtype=np.float32)
            if n_pe is not None:
                n_pe = np.asarray(n_pe)

            nsb_mask = (label == 0)
            gam_mask = (label == 1)

            self.nsb_total += int(nsb_mask.sum())
            if n_clusters is not None:
                self.nsb_trig += int((n_clusters[nsb_mask] > 0).sum())
            elif pre_threshold_score is not None and PRE_THRESHOLD_REFERENCE_ATTR in self.f.attrs:
                ref_threshold = float(self.f.attrs[PRE_THRESHOLD_REFERENCE_ATTR])
                comparison = self.f.attrs.get(PRE_THRESHOLD_COMPARISON_ATTR, "gt")
                if isinstance(comparison, bytes):
                    comparison = comparison.decode("utf-8")
                if str(comparison).lower() == "ge":
                    self.nsb_trig += int((pre_threshold_score[nsb_mask] >= ref_threshold).sum())
                else:
                    self.nsb_trig += int((pre_threshold_score[nsb_mask] > ref_threshold).sum())

            self.gamma_total += int(gam_mask.sum())
            if n_pe is not None and gam_mask.any():
                self.gamma_npe_min = min(self.gamma_npe_min, float(n_pe[gam_mask].min()))
                self.gamma_npe_max = max(self.gamma_npe_max, float(n_pe[gam_mask].max()))

    def close(self, window_sec: float):
        window_sec = float(window_sec)
        self.f.attrs["window_sec"] = window_sec

        # Trigger rate from NSB: p_bg/window
        p_bg = (self.nsb_trig / self.nsb_total) if self.nsb_total > 0 else 0.0
        trigger_rate_hz = p_bg / window_sec

        self.f.attrs["trigger_rate_hz"] = float(trigger_rate_hz)
        self.f.attrs["num_events_gamma"] = int(self.gamma_total)
        self.f.attrs["num_events_nsb"] = int(self.nsb_total)
        self.f.attrs["nsb_trig"] = int(self.nsb_trig)

        if np.isfinite(self.gamma_npe_min):
            self.f.attrs["gamma_n_pe_min"] = float(self.gamma_npe_min)
            self.f.attrs["gamma_n_pe_max"] = float(self.gamma_npe_max)

        self.f.close()
        return trigger_rate_hz
