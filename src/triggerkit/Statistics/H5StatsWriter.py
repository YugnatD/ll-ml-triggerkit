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
        fold: uint8   (cross-validation fold index; 0 for a plain single run)
      /folds group (one entry per cross-validation fold; `fold` column indexes it)
        attrs: n_folds
        name, nsb_total, nsb_trig, gamma_total, gamma_trig,
        gamma_efficiency, trigger_rate_hz : 1D arrays over folds
        /folds/config/<key> : per-fold build config as strings (gamma_deg, nsb_kind, ...)
      attrs:
        pre_threshold_comparison: "gt" or "ge" for the hard threshold rule
        top-level counts (num_events_*, nsb_trig, gamma_trig, trigger_rate_hz,
        gamma_efficiency) describe fold 0 (the nominal fold) for backward compat
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

        # Per-fold counters. Every appended row is tagged with the current fold
        # index (a `fold` column in /events), and its counts land in that fold's
        # bucket. A plain (fold-free) run is simply a single fold named "all", so
        # the layout is uniform and the top-level attrs stay backward-compatible.
        self.folds = []          # list of per-fold dicts (see begin_fold)
        self._fold_idx = -1      # index of the fold currently being written

        self._datasets = {}

    def begin_fold(self, name, config=None):
        """Start a new fold; subsequent append() rows are tagged with its index."""
        self.folds.append({
            "name": str(name),
            "config": dict(config or {}),
            "nsb_total": 0, "nsb_trig": 0,
            "gamma_total": 0, "gamma_trig": 0,
            "npe_min": np.inf, "npe_max": -np.inf,
        })
        self._fold_idx = len(self.folds) - 1
        return self._fold_idx

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

        # Tag every row with the current fold index (auto-open a default fold so a
        # caller that never calls begin_fold still gets a valid single-fold file).
        if self._fold_idx < 0:
            self.begin_fold("all")
        cols = dict(cols)
        cols["fold"] = np.full(n, self._fold_idx, dtype=np.uint8)

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
            cur = self.folds[self._fold_idx]
            label = np.asarray(label)
            if n_clusters is not None:
                n_clusters = np.asarray(n_clusters)
            if pre_threshold_score is not None:
                pre_threshold_score = np.asarray(pre_threshold_score, dtype=np.float32)
            if n_pe is not None:
                n_pe = np.asarray(n_pe)

            nsb_mask = (label == 0)
            gam_mask = (label == 1)

            def _n_trig(mask):
                # Prefer the legacy hard-trigger cluster count; otherwise threshold
                # the pre-threshold score against the stored reference.
                if n_clusters is not None:
                    return int((n_clusters[mask] > 0).sum())
                if pre_threshold_score is not None and PRE_THRESHOLD_REFERENCE_ATTR in self.f.attrs:
                    ref = float(self.f.attrs[PRE_THRESHOLD_REFERENCE_ATTR])
                    comparison = self.f.attrs.get(PRE_THRESHOLD_COMPARISON_ATTR, "gt")
                    if isinstance(comparison, bytes):
                        comparison = comparison.decode("utf-8")
                    if str(comparison).lower() == "ge":
                        return int((pre_threshold_score[mask] >= ref).sum())
                    return int((pre_threshold_score[mask] > ref).sum())
                return 0

            cur["nsb_total"] += int(nsb_mask.sum())
            cur["nsb_trig"] += _n_trig(nsb_mask)
            cur["gamma_total"] += int(gam_mask.sum())
            cur["gamma_trig"] += _n_trig(gam_mask)

            if n_pe is not None and gam_mask.any():
                cur["npe_min"] = min(cur["npe_min"], float(n_pe[gam_mask].min()))
                cur["npe_max"] = max(cur["npe_max"], float(n_pe[gam_mask].max()))

    def close(self, window_sec: float):
        window_sec = float(window_sec)
        self.f.attrs["window_sec"] = window_sec

        if not self.folds:      # nothing was ever appended
            self.begin_fold("all")

        def _rate(d):
            return (d["nsb_trig"] / d["nsb_total"] / window_sec) if d["nsb_total"] > 0 else 0.0

        def _eff(d):
            return (d["gamma_trig"] / d["gamma_total"]) if d["gamma_total"] > 0 else 0.0

        # Per-fold summary table so the file is self-describing: which folds it
        # holds, how each was built (config), and each fold's counts/rate/eff.
        # The `fold` column in /events indexes into these arrays.
        grp = self.f.create_group("folds")
        grp.attrs["n_folds"] = len(self.folds)
        str_dt = h5py.string_dtype()
        grp.create_dataset("name", data=np.array([d["name"] for d in self.folds], dtype=str_dt))
        for k in ("nsb_total", "nsb_trig", "gamma_total", "gamma_trig"):
            grp.create_dataset(k, data=np.array([d[k] for d in self.folds], dtype=np.int64))
        grp.create_dataset("gamma_efficiency", data=np.array([_eff(d) for d in self.folds], dtype=np.float64))
        grp.create_dataset("trigger_rate_hz", data=np.array([_rate(d) for d in self.folds], dtype=np.float64))
        # Fold build config (union of keys across folds), stored as strings so
        # mixed/None values round-trip without dtype headaches.
        cfg_keys = sorted({k for d in self.folds for k in d["config"]})
        if cfg_keys:
            cg = grp.create_group("config")
            for key in cfg_keys:
                vals = [d["config"].get(key) for d in self.folds]
                cg.create_dataset(
                    key,
                    data=np.array(["" if v is None else str(v) for v in vals], dtype=str_dt))

        # Top-level attrs describe fold 0 (the nominal fold) so single-fold files
        # and existing StatPlotter code keep working unchanged. Multi-fold detail
        # lives in /folds and the /events `fold` column.
        f0 = self.folds[0]
        trigger_rate_hz = _rate(f0)
        self.f.attrs["trigger_rate_hz"] = float(trigger_rate_hz)
        self.f.attrs["num_events_gamma"] = int(f0["gamma_total"])
        self.f.attrs["num_events_nsb"] = int(f0["nsb_total"])
        self.f.attrs["nsb_trig"] = int(f0["nsb_trig"])
        self.f.attrs["gamma_trig"] = int(f0["gamma_trig"])
        self.f.attrs["gamma_efficiency"] = float(_eff(f0))

        # gamma n_pe range spans all folds (a plotting hint only).
        npe_min = min(d["npe_min"] for d in self.folds)
        npe_max = max(d["npe_max"] for d in self.folds)
        if np.isfinite(npe_min):
            self.f.attrs["gamma_n_pe_min"] = float(npe_min)
            self.f.attrs["gamma_n_pe_max"] = float(npe_max)

        self.f.close()
        return trigger_rate_hz
