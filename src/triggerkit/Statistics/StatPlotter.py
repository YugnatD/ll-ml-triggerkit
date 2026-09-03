#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StatPlotter (HDF5 only)

HDF5 expectations (your current writer output):
- file attrs (examples):
  - trigger_chain_json : JSON string like [["baselinesubstractor", {}], ["digital_sum", {"threshold_flower": 322, "mode": "triplet"}]]
  - trigger_rate_hz    : float
  - window_sec         : float (seconds per NSB window)
  - num_events_gamma   : int
  - num_events_nsb     : int
  - nsb_trig           : int (number of triggered nsb events)
  - gamma_n_pe_min/max : floats (optional)
- datasets (two supported layouts):

Layout A (recommended, produced by the writer we discussed):
  /events/label        (0=nsb, 1=gamma)   OPTIONAL but strongly recommended
  /events/event_id     OPTIONAL
  /events/n_pe         metric dataset
  /events/energy       optional metric dataset
  /events/n_clusters   OR /events/triggered

Layout B:
  /gamma/n_pe, /gamma/n_clusters (or triggered), ...
  /nsb/n_pe,   /nsb/n_clusters   (or triggered), ...

Effective-area plotting supports analytic thrown-event expectations for
CORSIKA/sim_telarray style gamma productions.
"""

from __future__ import annotations

import os
import json
import pickle
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import norm

from triggerkit.Statistics.H5StatsWriter import (
    PRE_THRESHOLD_AVAILABLE_ATTR,
    PRE_THRESHOLD_COMPARISON_ATTR,
    PRE_THRESHOLD_REFERENCE_ATTR,
    PRE_THRESHOLD_SCORE_DATASET,
)
from triggerkit.Statistics.metrics import roc_auc_mann_whitney, roc_curve

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


ConfigType = List[Tuple[str, Dict[str, Any]]]


# Default effective-area generation setup for the current SST-1M production:
#   NSHOW = 3000
#   CSCAT reuse count = 20
#   total thrown = 60000
#   ESLOPE = -2.0  -> dN/dE ~ E^-2
#   ERANGE = 200 GeV -> 800000 GeV
#   A_GEN = 2705489.3277890724 m^2
EA_DEFAULT_EMIN_TEV = 0.2
EA_DEFAULT_EMAX_TEV = 800.0
EA_DEFAULT_TOTAL_THROWN = 60_000
EA_DEFAULT_SLOPE = 2.0
EA_DEFAULT_A_GEN_M2 = 2_705_489.3277890724


def _as_str(x: Any) -> str:
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return str(x)


def wilson(passed, total, level=0.68):
    #implementation as of: https://root.cern.ch/doc/master/TEfficiency_8cxx_source.html#l03837
    alpha = (1.0 - level) / 2.0
    if total == 0:
        return np.nan,np.nan,np.nan

    average = passed / total
    kappa = norm.ppf(1.0 - alpha)

    mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa)

    delta = (kappa / (total + kappa * kappa)) * np.sqrt(
        total * average * (1.0 - average) + (kappa * kappa) / 4.0
    )

    low = max(0.0, mode - delta)
    high = min(1.0, mode + delta)
    return average, average - low, high - average


def compute_efficiency_wilson(xaxis_list, trigger_list,binning):
    """
    Compute efficiency vs energy with Wilson interval errors.

    Parameters
    ----------
    xaxis_list : array-like
        X axis to slice on.
    trigger_list : array-like (bool)
        Trigger flag per event (True = triggered).
    n_bins : int
        Number of logarithmic bins.
    conf : float
        Confidence level (default 0.68 ≈ 1σ).

    Returns
    -------
    bin_centers : np.ndarray
    efficiency : np.ndarray
    err_low : np.ndarray
    err_high : np.ndarray
    """

    # --- bins ---
    bin_centers = np.sqrt(binning[:-1] * binning[1:])

    efficiency = []
    err_low = []
    err_high = []

    # --- loop over bins ---
    for i in range(len(binning) - 1):
        mask = (xaxis_list >= binning[i]) & (xaxis_list < binning[i+1])

        n = np.sum(mask)
        k = np.sum(trigger_list[mask])

        eff, low, high = wilson(k,n)

        efficiency.append(eff)
        err_low.append(low)
        err_high.append(high)

    efficiency = np.array(efficiency)
    err_low = np.array(err_low)
    err_high = np.array(err_high)
    #clip negative uncertainty - edge cases
    err_low = np.clip(err_low, 0, None)
    err_high = np.clip(err_high, 0, None)

    return {
        "bin_centers": bin_centers,
        "binning": binning,
        "efficiency": efficiency,
        "err_low": err_low,
        "err_high": err_high
    }


def compute_ratio_with_errors(model_base, model):
    """
    Compute ratio of efficiencies with error propagation.

    Parameters
    ----------
    model1, model2 : dict
        Must contain:
        - "bin_centers"
        - "efficiency"
        - "err_low"
        - "err_high"

    Returns
    -------
    dict with same structure
    """

    eff1 = np.asarray(model_base["efficiency"])
    eff2 = np.asarray(model["efficiency"])

    err1_low = np.asarray(model_base["err_low"])
    err1_high = np.asarray(model_base["err_high"])

    err2_low = np.asarray(model["err_low"])
    err2_high = np.asarray(model["err_high"])

    ratio = np.full_like(eff1, np.nan, dtype=float)
    err_low = np.full_like(eff1, np.nan, dtype=float)
    err_high = np.full_like(eff1, np.nan, dtype=float)

    # --- compute ratio + propagated errors ---
    for i in range(len(eff1)):
        if eff2[i] > 0 and eff1[i] > 0:
            r = eff2[i] / eff1[i]

            # relative errors
            rel_low = np.sqrt(
                (err1_high[i] / eff1[i])**2 +
                (err2_low[i] / eff2[i])**2
            )

            #print(err1_high[i], eff1[i], err1_high[i] / eff1[i], err2_low[i],eff2[i],err2_low[i] / eff2[i])

            rel_high = np.sqrt(
                (err1_low[i] / eff1[i])**2 +
                (err2_high[i] / eff2[i])**2
            )

            ratio[i] = r
            err_low[i] = r * rel_low
            err_high[i] = r * rel_high
            #print(r, rel_low, rel_high, r * rel_low, r * rel_high)

    return {
        "bin_centers": model_base["bin_centers"],
        "efficiency": ratio,
        "binning": model_base["binning"],
        "err_low": err_low,
        "err_high": err_high
    }


class StatPlotter:
    def __init__(
        self,
        base_reference_config: ConfigType = [("digital_sum", {"threshold_flower": 2229, "mode": "flower"})],
        stat_folder: Union[str, List[str]] = "output_gamma",
        allow_h5: bool = True,
        float_atol: float = 1e-6,
        float_rtol: float = 1e-5,
        h5_chunk_rows: int = 200_000,
        fold: Optional[str] = None,
    ):
        # Accept either a single folder string or a list of folders.
        folders = [stat_folder] if isinstance(stat_folder, str) else list(stat_folder)
        self.stat_folder = folders[0]  # kept for backwards-compat (e.g. report output dir)
        self.base_reference_config = base_reference_config

        # Optional cross-validation fold selection. A stats file may hold several
        # folds (a `fold` column in /events + a /folds table). fold=None reads
        # every event = the aggregate over all folds (default, unchanged). Setting
        # fold="rot120_rolled" restricts every event read to that single fold, so
        # the same plots can be drawn per fold to inspect leakage in detail.
        self._fold_filter = fold

        self.float_atol = float_atol
        self.float_rtol = float_rtol
        self.h5_chunk_rows = int(h5_chunk_rows)
        self.wilson_level = 0.68

        self.all_results: List[Tuple[str, Dict[str, Any]]] = []

        if allow_h5:
            for folder in folders:
                for fn in sorted(os.listdir(folder)):
                    if fn.endswith(".h5") or fn.endswith(".hdf5"):
                        path = os.path.join(folder, fn)
                        try:
                            self.all_results.append((fn, self._load_h5_meta(path)))
                        except (OSError, BlockingIOError, RuntimeError, ValueError) as exc:
                            # Skip files that are unreadable (e.g. being written by a
                            # concurrent job, truncated, or corrupt) instead of failing
                            # the whole session.
                            print(f"Skipping unreadable stats file '{fn}': {exc}")

        # Resolve base config
        self.base_config_result = self.get_results(base_reference_config)
        if self.base_config_result is None:
            raise ValueError("Base reference configuration not found in the provided stat folder.")

        # plot state
        self.metrics = "n_pe"
        self.x_range = None
        self.nbins = 50
        self.mode = "efficiency"
        self.target_rate_hz = None
        self.score_threshold = None
        self._distinct_plot_palette = [
            "#2E91E5",
            "#E15F99",
            "#1CA71C",
            "#FB0D0D",
            "#DA16FF",
            "#B68100",
            "#750D86",
            "#00A08B",
            "#FC0080",
            "#6C7C32",
            "#778AAE",
            "#862A16",
        ]

        self._roc_auc_cache: Dict[str, float] = {}
        self._score_threshold_cache: Dict[Tuple[str, float, str], Dict[str, Any]] = {}
        self._score_rate_cache: Dict[Tuple[str, float, str], float] = {}
        self._warned_messages: set[str] = set()
        self._queued_plot_colors: Dict[str, str] = {}
        self._active_plot_colors: Dict[str, str] = {}

        # base trigger-rate cache
        if isinstance(self.base_config_result.get("trigger_rate"), tuple):
            self.base_ref_trigger_rate = self.base_config_result["trigger_rate"][0]
            self.base_ref_trigger_rate_std = self.base_config_result["trigger_rate"][1]
        else:
            self.base_ref_trigger_rate = self.base_config_result.get("trigger_rate", 0.0)
            self.base_ref_trigger_rate_std = -1.0

        # ----------------------------
        # Generic plotting API (queue-based)
        #
        # Motivation:
        #   Let users write one plotting script with:
        #     - init_plot(...)  (store plot options)
        #     - add_plot(...)   (queue configs)
        #     - showPlot(plot_type=...)  (render the chosen plot)
        #   so they don't have to switch between addPlotAbsolute/addPlotRatio/addPlotEffectiveArea.
        #
        # Notes:
        #   - The legacy initPlotXXX/addPlotXXX API is still available.
        #   - If you call showPlot() without plot_type, it behaves like before (just saves/shows the current figure).
        self._plot_init_kwargs: Dict[str, Any] = {}
        self._queued_plots: List[Dict[str, Any]] = []

    def _normalize_metric_name(self, metric: str) -> str:
        metric_name = _as_str(metric).strip()
        aliases = {
            "npe": "n_pe",
            "n_pe": "n_pe",
            "energy": "energy",
            "tenergy": "energy",
        }
        return aliases.get(metric_name.lower(), metric_name)

    def _config_color_key(self, config: ConfigType) -> str:
        try:
            return json.dumps(config, sort_keys=True, default=str)
        except Exception:
            return repr(config)

    def _color_for_palette_index(self, index: int) -> str:
        if index < len(self._distinct_plot_palette):
            return self._distinct_plot_palette[index]
        hue = (float(index) * 0.618033988749895) % 1.0
        rgb = colors.hsv_to_rgb((hue, 0.68, 0.88))
        return colors.to_hex(rgb)

    def _get_config_plot_color(self, config: ConfigType) -> str:
        config_key = self._config_color_key(config)
        queued_color = self._queued_plot_colors.get(config_key)
        if queued_color is not None:
            self._active_plot_colors.setdefault(config_key, queued_color)
            return queued_color
        if config_key not in self._active_plot_colors:
            self._active_plot_colors[config_key] = self._color_for_palette_index(len(self._active_plot_colors))
        return self._active_plot_colors[config_key]

    def _metric_xlabel(self, metric: str) -> str:
        metric_name = self._normalize_metric_name(metric)
        if metric_name == "n_pe":
            return "Number of Photoelectrons (n_pe)"
        if metric_name == "energy":
            return "Energy (TeV)"
        return metric_name

    # ----------------------------
    # Loading / matching configs
    # ----------------------------

    def _load_h5_meta(self, h5_path: str) -> Dict[str, Any]:
        if h5py is None:
            raise RuntimeError("h5py is not installed but allow_h5=True was requested.")
        meta: Dict[str, Any] = {"_format": "h5", "_path": h5_path}

        with h5py.File(h5_path, "r") as f:
            attrs = f.attrs

            meta["camera_name"] = _as_str(attrs.get("camera_name", "unknown"))
            meta["format"] = _as_str(attrs.get("format", "h5stats-v1"))

            # Trigger chain: preferred JSON
            chain = self._read_trigger_chain_from_h5(attrs)
            meta["trigger_chain"] = chain

            # rates / counters
            if "trigger_rate_hz" in attrs:
                meta["trigger_rate"] = float(attrs["trigger_rate_hz"])
            elif "trigger_rate" in attrs:
                meta["trigger_rate"] = float(attrs["trigger_rate"])
            else:
                meta["trigger_rate"] = None

            meta["window_sec"] = self._read_window_sec(attrs)

            meta["num_events_gamma"] = int(attrs.get("num_events_gamma", attrs.get("num_events", 0)))
            meta["num_events_nsb"] = int(attrs.get("num_events_nsb", 0))
            meta["nsb_trig"] = int(attrs.get("nsb_trig", -1))

            # ranges (optional)
            if "gamma_n_pe_min" in attrs:
                meta["gamma_n_pe_min"] = float(attrs["gamma_n_pe_min"])
            if "gamma_n_pe_max" in attrs:
                meta["gamma_n_pe_max"] = float(attrs["gamma_n_pe_max"])

            # quick layout hint
            if "events" in f:
                meta["_layout"] = "events"
            elif "gamma" in f and "nsb" in f:
                meta["_layout"] = "split"
            else:
                meta["_layout"] = "unknown"

            meta["has_pre_threshold_score"] = bool(
                attrs.get(PRE_THRESHOLD_AVAILABLE_ATTR, False)
                or ("events" in f and PRE_THRESHOLD_SCORE_DATASET in f["events"])
                or ("gamma" in f and PRE_THRESHOLD_SCORE_DATASET in f["gamma"])
                or ("nsb" in f and PRE_THRESHOLD_SCORE_DATASET in f["nsb"])
            )
            if PRE_THRESHOLD_REFERENCE_ATTR in attrs:
                meta["reference_threshold"] = float(attrs[PRE_THRESHOLD_REFERENCE_ATTR])
            else:
                meta["reference_threshold"] = self._extract_threshold_from_chain(chain)
            if PRE_THRESHOLD_COMPARISON_ATTR in attrs:
                meta["threshold_comparison"] = self._normalize_threshold_comparison(
                    _as_str(attrs[PRE_THRESHOLD_COMPARISON_ATTR])
                )
            else:
                meta["threshold_comparison"] = self._extract_threshold_comparison_from_chain(chain)

        return meta

    def _read_window_sec(self, attrs) -> float:
        if "window_sec" in attrs:
            return float(attrs["window_sec"])
        if "window_size_ns" in attrs:
            return float(attrs["window_size_ns"]) * 1e-9
        # fallback (your typical window)
        return 75e-9

    def _read_trigger_chain_from_h5(self, attrs) -> ConfigType:
        # Preferred JSON
        if "trigger_chain_json" in attrs:
            raw = attrs["trigger_chain_json"]
            raw = _as_str(raw)
            chain_ll = json.loads(raw)  # list of [stage, params]
            return [(str(stage), dict(params)) for stage, params in chain_ll]

        # Backward-compatible pickle attrs
        for k in ("trigger_chain_pickle", "trigger_chain_pkl"):
            if k in attrs:
                v = attrs[k]
                b = v.tobytes() if hasattr(v, "tobytes") else bytes(v)
                chain = pickle.loads(b)
                # normalize
                return [(str(stage), dict(params)) for stage, params in chain]

        return []

    def _extract_threshold_from_chain(self, chain: ConfigType) -> Optional[float]:
        last_threshold = None
        for stage, raw_params in chain:
            if str(stage).lower() != "threshold":
                continue
            params = self._as_mapping(raw_params) or {}
            if "threshold" not in params:
                continue
            try:
                last_threshold = float(params["threshold"])
            except Exception:
                continue
        return last_threshold

    @staticmethod
    def _normalize_threshold_comparison(comparison: Optional[str]) -> str:
        if comparison is None:
            return "gt"
        aliases = {
            ">": "gt",
            "gt": "gt",
            "strict": "gt",
            "score > tau": "gt",
            ">=": "ge",
            "ge": "ge",
            "inclusive": "ge",
            "score >= tau": "ge",
        }
        return aliases.get(str(comparison).strip().lower(), "gt")

    def _extract_threshold_comparison_from_chain(self, chain: ConfigType) -> str:
        comparison = "gt"
        for stage, raw_params in chain:
            if str(stage).lower() != "threshold":
                continue
            params = self._as_mapping(raw_params) or {}
            comparison = self._normalize_threshold_comparison(params.get("comparison", comparison))
        return comparison

    @classmethod
    def _score_trigger_mask(
        cls,
        scores: np.ndarray,
        threshold: float,
        comparison: Optional[str] = "gt",
    ) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        if cls._normalize_threshold_comparison(comparison) == "ge":
            return scores >= float(threshold)
        return scores > float(threshold)

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_messages:
            return
        self._warned_messages.add(key)
        print(message)

    # Descriptive keys describe a run rather than distinguish it. They are
    # ignored during matching unless a reference config explicitly pins them
    # (e.g. to disambiguate two variants that differ only by their weights).
    _IGNORED_MATCH_KEYS = ("id", "ring_weights", "kernel_weights")
    # Weight arrays get a shape-insensitive, looser comparison (see _weights_equal).
    _WEIGHT_MATCH_KEYS = ("ring_weights", "kernel_weights")
    # Absolute tolerance for pinned weight matching. Generous enough to accept
    # the rounded weights printed in the "Trigger chain info" log (4 decimals),
    # tight enough to still tell genuinely different weight sets apart.
    _weight_match_atol = 1e-3

    def _value_equal(self, a: Any, b: Any) -> bool:
        # numeric tolerance
        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
            return bool(np.isclose(float(a), float(b), atol=self.float_atol, rtol=self.float_rtol))
        # list/array tolerance (e.g. score_quantizer edges, tdscan ring weights)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            try:
                aa = np.asarray(a, dtype=float)
                bb = np.asarray(b, dtype=float)
            except (TypeError, ValueError):
                return a == b
            if aa.shape != bb.shape:
                return False
            return bool(np.allclose(aa, bb, atol=self.float_atol, rtol=self.float_rtol))
        return a == b

    def _weights_equal(self, a: Any, b: Any) -> bool:
        """Compare pinned tdscan weights shape-insensitively.

        Stored weights keep their full ``(L, R, 1, Cin, Cout)`` shape, while a
        config usually pins the squeezed/rounded form printed in the log. Both
        flatten to the same element order, so we compare the flattened arrays
        with a tolerance that tolerates the printed rounding.
        """
        try:
            aa = np.asarray(a, dtype=float).ravel()
            bb = np.asarray(b, dtype=float).ravel()
        except (TypeError, ValueError):
            return a == b
        if aa.shape != bb.shape:
            return False
        return bool(np.allclose(aa, bb, atol=self._weight_match_atol, rtol=0.0))

    def _match_chain(self, chain: ConfigType, ref: ConfigType) -> bool:
        if len(chain) != len(ref):
            return False
        for (st, p), (rst, rp) in zip(chain, ref):
            if str(st).lower() != str(rst).lower():
                return False
            p = dict(p)
            rp = dict(rp)
            # Drop descriptive keys from the stored chain unless the reference
            # pins them, so an under-specified ref can't accidentally match a
            # more specific chain (e.g. a float config matching a quantized one).
            for k in self._IGNORED_MATCH_KEYS:
                if k not in rp:
                    p.pop(k, None)
            if set(p.keys()) != set(rp.keys()):
                return False
            for k, v in rp.items():
                if k in self._WEIGHT_MATCH_KEYS:
                    ok = self._weights_equal(p[k], v)
                else:
                    ok = self._value_equal(p[k], v)
                if not ok:
                    return False
        return True

    def get_results(self, to_find_config: ConfigType) -> Optional[Dict[str, Any]]:
        result_config = None
        for filename, result in self.all_results:
            trigger_chain = result.get("trigger_chain") or result.get("trigger_chain", [])
            if trigger_chain is None:
                continue

            # normalize chain elements (lists -> tuples can happen with JSON)
            chain_norm: ConfigType = [(str(s), dict(p)) for s, p in trigger_chain]

            if self._match_chain(chain_norm, to_find_config):
                result_config = dict(result)  # shallow copy
                result_config["trigger_chain"] = chain_norm
                result_config["_filename"] = filename
                break

        if result_config is None:
            return None

        return result_config

    def _is_base_config(
        self,
        chain: Optional[ConfigType] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if result is self.base_config_result:
            return True
        if result is not None:
            chain = result.get("trigger_chain")
        if not chain:
            return False

        chain_norm: ConfigType = []
        for stage, params in chain:
            chain_norm.append((str(stage), self._as_mapping(params) or {}))
        return self._match_chain(chain_norm, self.base_reference_config)

    def _format_counts_line(self) -> str:
        """Return a short 'Gamma events: X | NSB events: Y' string based on base config metadata."""
        def _fmt(v: Any) -> str:
            if v is None:
                return "NA"
            try:
                iv = int(v)
                return f"{iv:,}"
            except Exception:
                try:
                    return f"{float(v):g}"
                except Exception:
                    return str(v)

        g = self.base_config_result.get("num_events_gamma")
        n = self.base_config_result.get("num_events_nsb")
        return f"Gamma events: {_fmt(g)} | NSB events: {_fmt(n)}"

    def _as_mapping(self, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (str, bytes)):
            return None
        if hasattr(value, "items"):
            try:
                return {str(k): v for k, v in value.items()}
            except Exception:
                pass
        try:
            return dict(value)
        except Exception:
            return None

    @staticmethod
    def _format_label_value(value: Any, precision: int = 6) -> str:
        if value is None:
            return "None"
        try:
            return f"{float(value):.{precision}g}"
        except Exception:
            return str(value)

    @staticmethod
    def _format_hash_short(value: Any) -> Optional[str]:
        if value is None:
            return None
        value_str = str(value)
        if not value_str:
            return None
        if len(value_str) <= 6:
            return value_str
        return f"{value_str[:2]}{value_str[-2:]}"

    def _format_quantize_step_label(self, quantize_step: Any) -> str:
        qmap = self._as_mapping(quantize_step)
        if not qmap:
            return ""

        order = [
            ("input", "i"),
            ("ring_weights", "w"),
            ("convolution_accumulator", "c"),
            ("convolution_rescale_shift", "cs"),
            ("temporal_accumulator", "t"),
            ("temporal_rescale_shift", "ts"),
            ("shift_value", "v"),
            ("output", "o"),
        ]
        parts = []
        seen = set()

        for key, short in order:
            value = qmap.get(key)
            if value is None:
                continue
            parts.append(f"{short}={value}")
            seen.add(key)

        for key, value in qmap.items():
            if key in seen or value is None:
                continue
            parts.append(f"{key}={value}")

        if not parts:
            return ""
        return f" q({','.join(parts)})"

    # ----------------------------
    # HDF5 streaming helpers
    # ----------------------------

    def _is_h5(self, result: Dict[str, Any]) -> bool:
        return result.get("_format") == "h5"

    def _resolve_h5_group(self, f: "h5py.File", kind: str) -> Tuple[Optional["h5py.Group"], Optional["h5py.Dataset"]]:
        """
        Returns:
          - group containing the data
          - optional label dataset (only for /events layout)
        """
        if "events" in f:
            grp = f["events"]
            label_ds = grp["label"] if "label" in grp else None
            return grp, label_ds

        # split layout
        if kind in f:
            return f[kind], None

        raise KeyError(f"HDF5 layout not recognized for kind='{kind}'. Expected /events or /{kind}.")

    def _resolve_triggered_ds(self, grp: "h5py.Group") -> "h5py.Dataset":
        if "triggered" in grp:
            return grp["triggered"]
        if "n_clusters" in grp:
            return grp["n_clusters"]
        if "p_trig" in grp:
            return grp["p_trig"]
        raise KeyError("Could not find 'triggered', 'n_clusters', or 'p_trig' dataset to determine triggered events.")

    def _resolve_score_ds(self, grp: "h5py.Group") -> Optional["h5py.Dataset"]:
        if PRE_THRESHOLD_SCORE_DATASET in grp:
            return grp[PRE_THRESHOLD_SCORE_DATASET]
        return None

    def _fold_row_mask(self, grp: "h5py.Group", sl: slice) -> Optional[np.ndarray]:
        """Boolean keep-mask over the sliced rows for the selected fold.

        Returns None when no fold filtering applies (fold=None, or the file has no
        `fold` column -- e.g. a legacy single-fold file). Raises if the requested
        fold name is not present in the file's /folds table."""
        if self._fold_filter is None or grp is None or "fold" not in grp:
            return None
        f = grp.file
        names = None
        if "folds" in f and "name" in f["folds"]:
            names = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in f["folds"]["name"][()]]
        if not names or self._fold_filter not in names:
            raise KeyError(
                f"fold {self._fold_filter!r} not found in {f.filename} "
                f"(available: {names})")
        idx = names.index(self._fold_filter)
        return np.asarray(grp["fold"][sl]).reshape(-1) == idx

    def _label_mask(self, label_chunk: np.ndarray, kind: str) -> np.ndarray:
        # label is typically 0/1 integers, but we handle strings too
        if label_chunk.dtype.kind in ("S", "U", "O"):
            s = label_chunk.astype(str)
            if kind == "gamma":
                return (s == "gamma") | (s == "1")
            return (s == "nsb") | (s == "0")
        else:
            target = 1 if kind == "gamma" else 0
            return label_chunk == target

    def _collect_metric_values(
        self,
        result: Dict[str, Any],
        metric: str,
        kind: str,
        chunk_rows: Optional[int] = None,
    ) -> np.ndarray:
        if not self._is_h5(result):
            raise ValueError("_collect_metric_values called on non-h5 result.")

        if h5py is None:
            raise RuntimeError("h5py is not installed.")

        chunk_rows = int(chunk_rows or self.h5_chunk_rows)
        values: List[np.ndarray] = []

        with h5py.File(result["_path"], "r") as f:
            grp, label_ds = self._resolve_h5_group(f, kind if kind in ("gamma", "nsb") else kind)
            if grp is None:
                raise KeyError("Could not resolve HDF5 group for metric collection.")
            if metric not in grp:
                raise KeyError(f"Metric dataset '{metric}' not found in group '{grp.name}' in {result['_path']}.")

            ds = grp[metric]
            n = int(ds.shape[0])
            for start in range(0, n, chunk_rows):
                end = min(n, start + chunk_rows)
                sl = slice(start, end)

                x = np.asarray(ds[sl]).reshape(-1)
                keep = self._fold_row_mask(grp, sl)
                if label_ds is not None:
                    lab = np.asarray(label_ds[sl]).reshape(-1)
                    m = self._label_mask(lab, kind="gamma" if kind == "gamma" else "nsb")
                    if keep is not None:
                        m = m & keep
                    x = x[m]
                elif keep is not None:
                    x = x[keep]

                if x.size == 0:
                    continue

                mfinite = np.isfinite(x)
                x = x[mfinite]
                if x.size == 0:
                    continue

                values.append(x.astype(np.float32, copy=False))

        if not values:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(values, axis=0)

    @classmethod
    def _pick_threshold_from_empirical_scores(
        cls,
        scores: np.ndarray,
        desired_fraction: float,
        comparison: Optional[str] = "gt",
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return None, None, None

        desired_fraction = float(np.clip(desired_fraction, 0.0, 1.0))
        comparison = cls._normalize_threshold_comparison(comparison)
        unique_scores, counts = np.unique(scores, return_counts=True)
        counts = counts.astype(np.int64)
        total = int(scores.size)

        cumulative = np.cumsum(counts, dtype=np.int64)
        counts_gt = total - cumulative
        counts_ge = counts_gt + counts

        if comparison == "ge":
            tau_strict = np.nextafter(unique_scores.astype(np.float32), np.float32(np.inf))
            tau_include_ties = unique_scores.astype(np.float32)
        else:
            tau_strict = unique_scores.astype(np.float32)
            tau_include_ties = np.nextafter(unique_scores.astype(np.float32), np.float32(-np.inf))

        candidate_taus = np.concatenate([tau_strict, tau_include_ties])
        candidate_fractions = np.concatenate([
            counts_gt.astype(np.float64) / total,
            counts_ge.astype(np.float64) / total,
        ])
        candidate_modes = np.concatenate([
            np.zeros_like(counts_gt, dtype=np.uint8),
            np.ones_like(counts_ge, dtype=np.uint8),
        ])

        errors = np.abs(candidate_fractions - desired_fraction)
        best_error = float(np.min(errors))
        best_indices = np.flatnonzero(np.isclose(errors, best_error, rtol=0.0, atol=1e-12))
        best_include_ties = best_indices[candidate_modes[best_indices] == 1]
        best_idx = int(best_include_ties[0] if best_include_ties.size > 0 else best_indices[0])

        mode = "score >= bin" if candidate_modes[best_idx] == 1 else "score > bin"
        return float(candidate_taus[best_idx]), float(candidate_fractions[best_idx]), mode

    def _get_result_trigger_rate_hz(self, result: Dict[str, Any]) -> float:
        if result.get("trigger_rate") is not None:
            rate = result["trigger_rate"]
            if isinstance(rate, tuple):
                return float(rate[0])
            return float(rate)

        if result.get("has_pre_threshold_score", False) and result.get("reference_threshold") is not None:
            comparison = result.get("threshold_comparison", "gt")
            cache_key = (result["_path"], float(result["reference_threshold"]), comparison)
            if cache_key in self._score_rate_cache:
                return float(self._score_rate_cache[cache_key])

            nsb_scores = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
            if nsb_scores.size == 0:
                return 0.0
            rate = float(
                np.count_nonzero(
                    self._score_trigger_mask(
                        nsb_scores,
                        float(result["reference_threshold"]),
                        comparison=comparison,
                    )
                )
                / nsb_scores.size
                / float(result.get("window_sec", 75e-9))
            )
            self._score_rate_cache[cache_key] = rate
            return rate

        nsb_trig = result.get("nsb_trig", -1)
        n_nsb = result.get("num_events_nsb", 0)
        ws = result.get("window_sec", 75e-9)
        if nsb_trig >= 0 and n_nsb > 0 and ws > 0:
            return float(nsb_trig) / float(n_nsb) / float(ws)
        return float("nan")

    def _resolve_trigger_strategy(
        self,
        result: Dict[str, Any],
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        if target_rate_hz is not None and score_threshold is not None:
            raise ValueError("Use either target_rate_hz or score_threshold, not both.")

        if score_threshold is not None:
            threshold = float(score_threshold)
            comparison = result.get("threshold_comparison", "gt")
            cache_key = (result["_path"], threshold, comparison)
            if cache_key in self._score_rate_cache:
                rate_hz = self._score_rate_cache[cache_key]
            else:
                if not result.get("has_pre_threshold_score", False):
                    self._warn_once(
                        f"missing-score-threshold:{result['_path']}",
                        f"{os.path.basename(result['_path'])} has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset; falling back to stored trigger decisions.",
                    )
                    return {
                        "mode": "stored",
                        "score_threshold": None,
                        "trigger_rate_hz": self._get_result_trigger_rate_hz(result),
                        "reference_threshold": result.get("reference_threshold"),
                    }
                nsb_scores = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
                if nsb_scores.size == 0:
                    rate_hz = 0.0
                else:
                    rate_hz = float(
                        np.count_nonzero(
                            self._score_trigger_mask(
                                nsb_scores,
                                threshold,
                                comparison=comparison,
                            )
                        )
                        / nsb_scores.size
                        / result.get("window_sec", 75e-9)
                    )
                self._score_rate_cache[cache_key] = rate_hz
            return {
                "mode": "score",
                "score_threshold": threshold,
                "trigger_rate_hz": rate_hz,
                "reference_threshold": threshold,
            }

        if target_rate_hz is not None:
            desired_rate = float(target_rate_hz)
            comparison = result.get("threshold_comparison", "gt")
            cache_key = (result["_path"], desired_rate, comparison)
            cached = self._score_threshold_cache.get(cache_key)
            if cached is not None:
                return dict(cached)

            if not result.get("has_pre_threshold_score", False):
                self._warn_once(
                    f"missing-score-rate:{result['_path']}",
                    f"{os.path.basename(result['_path'])} has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset; falling back to stored trigger decisions.",
                )
                return {
                    "mode": "stored",
                    "score_threshold": None,
                    "trigger_rate_hz": self._get_result_trigger_rate_hz(result),
                    "reference_threshold": result.get("reference_threshold"),
                }

            nsb_scores = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
            if nsb_scores.size == 0:
                resolved = {
                    "mode": "score",
                    "score_threshold": None,
                    "trigger_rate_hz": 0.0,
                    "reference_threshold": None,
                    "target_rate_hz": desired_rate,
                    "selection_mode": None,
                }
            else:
                desired_fraction = float(np.clip(desired_rate * result.get("window_sec", 75e-9), 0.0, 1.0))
                threshold, predicted_fraction, selection_mode = self._pick_threshold_from_empirical_scores(
                    nsb_scores,
                    desired_fraction=desired_fraction,
                    comparison=comparison,
                )
                rate_hz = 0.0 if predicted_fraction is None else float(predicted_fraction / result.get("window_sec", 75e-9))
                resolved = {
                    "mode": "score",
                    "score_threshold": threshold,
                    "trigger_rate_hz": rate_hz,
                    "reference_threshold": threshold,
                    "target_rate_hz": desired_rate,
                    "selection_mode": selection_mode,
                }

            self._score_threshold_cache[cache_key] = dict(resolved)
            return resolved

        return {
            "mode": "score" if result.get("has_pre_threshold_score", False) and result.get("reference_threshold") is not None else "stored",
            "score_threshold": result.get("reference_threshold") if result.get("has_pre_threshold_score", False) else None,
            "trigger_rate_hz": self._get_result_trigger_rate_hz(result),
            "reference_threshold": result.get("reference_threshold"),
        }

    def _get_default_range(self, result: Dict[str, Any], metric: str, kind: str) -> Tuple[float, float]:
        metric = self._normalize_metric_name(metric)

        # Use stored ranges when available (best for streaming)
        if metric == "n_pe" and kind == "gamma":
            lo = float(result.get("gamma_n_pe_min", 1.0))
            hi = float(result.get("gamma_n_pe_max", 1e3))
            # guard
            if not np.isfinite(lo) or lo <= 0:
                lo = 1.0
            if not np.isfinite(hi) or hi <= lo:
                hi = max(lo * 10.0, 100.0)
            return lo, hi

        # If no attrs, pick a safe range
        if metric == "n_pe":
            return 1.0, 2000.0
        if metric == "energy":
            return 0.005, 50.0
        return 1.0, 1000.0

    def _make_log_bins(self, lo: float, hi: float, nbins: int) -> np.ndarray:
        eps = 1e-3
        lo = max(float(lo), eps)
        hi = float(hi)
        if hi <= lo:
            hi = lo * 10.0
        return np.logspace(np.log10(lo), np.log10(hi + eps), nbins)

    def _histogram_stream(
        self,
        result: Dict[str, Any],
        metric: str,
        kind: str,
        bins: np.ndarray,
        chunk_rows: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          hist_all, hist_triggered
        """
        if not self._is_h5(result):
            raise ValueError("_histogram_stream called on non-h5 result.")

        if h5py is None:
            raise RuntimeError("h5py is not installed.")

        chunk_rows = int(chunk_rows or self.h5_chunk_rows)
        hist_all = np.zeros(len(bins) - 1, dtype=np.int64)
        hist_trig = np.zeros(len(bins) - 1, dtype=np.int64)

        path = result["_path"]
        with h5py.File(path, "r") as f:
            grp, label_ds = self._resolve_h5_group(f, kind if kind in ("gamma", "nsb") else kind)
            if grp is None:
                raise KeyError("Could not resolve HDF5 group for streaming.")

            if metric not in grp:
                raise KeyError(f"Metric dataset '{metric}' not found in group '{grp.name}' in {path}.")

            x_ds = grp[metric]
            resolved_score_threshold = score_threshold
            score_ds = self._resolve_score_ds(grp)
            if score_ds is not None and resolved_score_threshold is None:
                resolved_score_threshold = result.get("reference_threshold")
            trig_ds = None if resolved_score_threshold is not None and score_ds is not None else self._resolve_triggered_ds(grp)

            n = int(x_ds.shape[0])
            for start in range(0, n, chunk_rows):
                end = min(n, start + chunk_rows)
                sl = slice(start, end)

                x = np.asarray(x_ds[sl])
                # ensure 1D
                x = x.reshape(-1)
                score_raw = None

                keep = self._fold_row_mask(grp, sl)
                if label_ds is not None:
                    lab = np.asarray(label_ds[sl]).reshape(-1)
                    m = self._label_mask(lab, kind="gamma" if kind == "gamma" else "nsb")
                    if keep is not None:
                        m = m & keep
                    x = x[m]
                    if score_ds is not None and resolved_score_threshold is not None:
                        score_raw = np.asarray(score_ds[sl]).reshape(-1)[m]
                    else:
                        trig_raw = np.asarray(trig_ds[sl]).reshape(-1)[m]
                else:
                    if keep is not None:
                        x = x[keep]
                    if score_ds is not None and resolved_score_threshold is not None:
                        score_raw = np.asarray(score_ds[sl]).reshape(-1)
                        if keep is not None:
                            score_raw = score_raw[keep]
                    else:
                        trig_raw = np.asarray(trig_ds[sl]).reshape(-1)
                        if keep is not None:
                            trig_raw = trig_raw[keep]

                if score_raw is not None:
                    trig = self._score_trigger_mask(
                        score_raw,
                        float(resolved_score_threshold),
                        comparison=result.get("threshold_comparison", "gt"),
                    )
                elif trig_raw.dtype == np.bool_:
                    trig = trig_raw
                else:
                    trig = trig_raw > 0

                # filter to bin range for stability / speed
                lo, hi = bins[0], bins[-1]
                rmask = (x >= lo) & (x <= hi)
                x = x[rmask]
                trig = trig[rmask]

                if x.size == 0:
                    continue

                hist_all += np.histogram(x, bins=bins)[0]
                if np.any(trig):
                    hist_trig += np.histogram(x[trig], bins=bins)[0]

        return hist_all, hist_trig

    def _histogram2d_stream(
        self,
        result: Dict[str, Any],
        x_metric: str,
        y_metric: str,
        kind: str,
        x_bins: np.ndarray,
        y_bins: np.ndarray,
        chunk_rows: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns:
          hist2d (shape: len(x_bins)-1, len(y_bins)-1)
        """
        if not self._is_h5(result):
            raise ValueError("_histogram2d_stream called on non-h5 result.")

        if h5py is None:
            raise RuntimeError("h5py is not installed.")

        chunk_rows = int(chunk_rows or self.h5_chunk_rows)
        hist2d = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=np.int64)

        path = result["_path"]
        with h5py.File(path, "r") as f:
            grp, label_ds = self._resolve_h5_group(f, kind if kind in ("gamma", "nsb") else kind)
            if grp is None:
                raise KeyError("Could not resolve HDF5 group for streaming.")

            if x_metric not in grp:
                raise KeyError(f"Metric dataset '{x_metric}' not found in group '{grp.name}' in {path}.")
            if y_metric not in grp:
                raise KeyError(f"Metric dataset '{y_metric}' not found in group '{grp.name}' in {path}.")

            x_ds = grp[x_metric]
            y_ds = grp[y_metric]

            n = int(x_ds.shape[0])
            for start in range(0, n, chunk_rows):
                end = min(n, start + chunk_rows)
                sl = slice(start, end)

                x = np.asarray(x_ds[sl]).reshape(-1)
                y = np.asarray(y_ds[sl]).reshape(-1)

                keep = self._fold_row_mask(grp, sl)
                if label_ds is not None:
                    lab = np.asarray(label_ds[sl]).reshape(-1)
                    m = self._label_mask(lab, kind="gamma" if kind == "gamma" else "nsb")
                    if keep is not None:
                        m = m & keep
                    x = x[m]
                    y = y[m]
                elif keep is not None:
                    x = x[keep]
                    y = y[keep]

                # drop NaNs/infs
                mfinite = np.isfinite(x) & np.isfinite(y)
                x = x[mfinite]
                y = y[mfinite]

                # filter to bin range for stability / speed
                xlo, xhi = x_bins[0], x_bins[-1]
                ylo, yhi = y_bins[0], y_bins[-1]
                rmask = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
                x = x[rmask]
                y = y[rmask]

                if x.size == 0:
                    continue

                hist2d += np.histogram2d(x, y, bins=[x_bins, y_bins])[0].astype(np.int64)

        return hist2d

    # ----------------------------
    # Original computations (now H5-aware)
    # ----------------------------

    def compute_efficiency_ratio(
        self,
        to_compare_result: Dict[str, Any],
        base_target_rate_hz: Optional[float] = None,
        base_score_threshold: Optional[float] = None,
        compare_target_rate_hz: Optional[float] = None,
        compare_score_threshold: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        # streaming path (HDF5)
        if self.x_range is None:
            lo, hi = self._get_default_range(self.base_config_result, self.metrics, kind="gamma")
        else:
            lo, hi = self.x_range
        bins = self._make_log_bins(lo, hi, self.nbins)

        base_model = self._compute_efficiency_absolute_stats(
            self.base_config_result,
            binning=bins,
            target_rate_hz=base_target_rate_hz,
            score_threshold=base_score_threshold,
        )
        cmp_model = self._compute_efficiency_absolute_stats(
            to_compare_result,
            binning=bins,
            target_rate_hz=compare_target_rate_hz,
            score_threshold=compare_score_threshold,
        )
        return compute_ratio_with_errors(base_model, cmp_model)

    def _efficiency_from_counts(self, all_h: np.ndarray, trig_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        efficiency = []
        err_low = []
        err_high = []

        for total, passed in zip(all_h, trig_h):
            eff, low, high = wilson(int(passed), int(total), level=self.wilson_level)
            efficiency.append(eff)
            err_low.append(low)
            err_high.append(high)

        efficiency = np.asarray(efficiency, dtype=np.float64)
        err_low = np.asarray(err_low, dtype=np.float64)
        err_high = np.asarray(err_high, dtype=np.float64)

        efficiency = np.nan_to_num(efficiency, nan=0.0, posinf=0.0, neginf=0.0)
        err_low = np.clip(np.nan_to_num(err_low, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        err_high = np.clip(np.nan_to_num(err_high, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        eff_err = np.vstack([err_low, err_high])
        return efficiency, eff_err

    def _compute_efficiency_absolute_hist(
        self,
        to_compare_result: Dict[str, Any],
        binning: Optional[np.ndarray] = None,
        score_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if binning is None:
            if self.x_range is None:
                lo, hi = self._get_default_range(to_compare_result, self.metrics, kind="gamma")
            else:
                lo, hi = self.x_range
            binning = self._make_log_bins(lo, hi, self.nbins)

        all_h, trig_h = self._histogram_stream(
            to_compare_result,
            self.metrics,
            kind="gamma",
            bins=binning,
            score_threshold=score_threshold,
        )
        return binning, all_h, trig_h

    def _compute_efficiency_absolute_stats(
        self,
        to_compare_result: Dict[str, Any],
        binning: Optional[np.ndarray] = None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        trigger_strategy = self._resolve_trigger_strategy(
            to_compare_result,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
        )
        binning, all_h, trig_h = self._compute_efficiency_absolute_hist(
            to_compare_result,
            binning=binning,
            score_threshold=trigger_strategy.get("score_threshold"),
        )
        efficiency, eff_err = self._efficiency_from_counts(all_h, trig_h)
        return {
            "bin_centers": np.sqrt(binning[:-1] * binning[1:]),
            "binning": binning,
            "efficiency": efficiency,
            "err_low": eff_err[0],
            "err_high": eff_err[1],
            "all_h": all_h,
            "trig_h": trig_h,
            "trigger_strategy": trigger_strategy,
        }

    def compute_efficiency_absolute(
        self,
        to_compare_result: Dict[str, Any],
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stats = self._compute_efficiency_absolute_stats(
            to_compare_result,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
        )
        eff_err = np.vstack([stats["err_low"], stats["err_high"]])
        return stats["bin_centers"], stats["efficiency"], eff_err

    # ----------------------------
    # Trigger-rate / NSB helpers
    # ----------------------------

    def compute_ratio_nsb_gamma(self, n_event_NSB: int, n_event_gamma: int, window_size_ns: float = 75e-9) -> float:
        # legacy heuristic
        time_simulated_nsb = n_event_NSB * window_size_ns
        time_simulated_gamma = n_event_gamma * 0.5
        if time_simulated_nsb == 0:
            return 0.0
        return time_simulated_gamma / time_simulated_nsb

    def retrieve_nsb_trigged_from_trigger_rate(self, trigger_rate_hz: Union[float, Tuple[float, float]], num_events: int, ws: float = 75e-9) -> float:
        if isinstance(trigger_rate_hz, tuple):
            trigger_rate_hz = trigger_rate_hz[0]
        return float(trigger_rate_hz) * float(ws) * float(num_events)

    def get_nsb_trigger_rate_hz(
        self,
        to_compare_config: Optional[ConfigType] = None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> float:
        """
        Returns the NSB trigger rate (Hz) for a config.
        Uses stored attribute trigger_rate_hz if present; otherwise derives it from nsb_trig/num_events_nsb.
        """
        if to_compare_config is None:
            r = self.base_config_result
        else:
            r = self.get_results(to_compare_config)
            if r is None:
                return float("nan")
        strategy = self._resolve_trigger_strategy(
            r,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
        )
        return float(strategy.get("trigger_rate_hz", float("nan")))

    # ----------------------------
    # Plot interface (kept similar)
    # ----------------------------

    # ----------------------------
    # Generic plot API: init_plot / add_plot / showPlot(plot_type=...)
    # ----------------------------

    def init_plot(self, **kwargs) -> "StatPlotter":
        """
        Generic init that stores plot options (but does not render anything yet).

        Typical usage:
            plotter.init_plot(title="My plot", metrics="energy", x_range=(0.2, 50), y_range=(0, 1), bins=50)
            plotter.add_plot(cfg1)
            plotter.add_plot(cfg2)
            plotter.showPlot(plot_type="absolute")
            # ... or ... plotter.showPlot(plot_type="effective_area")
            # ... or ... plotter.showPlot(plot_type="effective_area_counts")

        Parameters are stored and later filtered depending on plot_type.
        Unknown keys are ignored for a given plot_type.
        """
        # Keep a copy so caller dict mutations don't affect us
        self._plot_init_kwargs = dict(kwargs)
        self._queued_plots = []
        self._queued_plot_colors = {}
        return self

    def add_plot(self, to_compare_config: ConfigType, label: Optional[str] = None, **kwargs) -> "StatPlotter":
        """
        Queue a configuration to be plotted.

        Any extra kwargs are stored and passed to the underlying addPlotXXX method
        when you call showPlot(plot_type=...).

        Examples:
            plotter.add_plot(cfg, label="threshold 300")
            plotter.add_plot(cfg, mul_trig_rate=1000.0)  # for ratio plots
            plotter.add_plot(cfg)  # for effective-area or count plots
        """
        config_key = self._config_color_key(to_compare_config)
        if config_key not in self._queued_plot_colors:
            self._queued_plot_colors[config_key] = self._color_for_palette_index(len(self._queued_plot_colors))
        self._queued_plots.append({"config": to_compare_config, "label": label, "kwargs": dict(kwargs)})
        return self

    def _normalize_plot_type(self, plot_type: str) -> str:
        pt = (plot_type or "").strip().lower()
        aliases = {
            # absolute efficiency
            "abs": "absolute",
            "absolute_efficiency": "absolute",
            "absoluteefficiency": "absolute",
            "absolute": "absolute",
            # ratio efficiency
            "rel": "ratio",
            "relative": "ratio",
            "ratio": "ratio",
            "efficiency_ratio": "ratio",
            # effective area
            "aeff": "effective_area",
            "effectivearea": "effective_area",
            "effective_area": "effective_area",
            "effective-area": "effective_area",
            "aeff_counts": "effective_area_counts",
            "effective_area_counts": "effective_area_counts",
            "effective-area-counts": "effective_area_counts",
            "effectiveareacounts": "effective_area_counts",
            # trigger rate scan
            "trigger_rate_vs_threshold": "trigger_rate_vs_threshold",
            "trigger-rate-vs-threshold": "trigger_rate_vs_threshold",
            "trigger_rate_scan": "trigger_rate_vs_threshold",
            "rate_vs_threshold": "trigger_rate_vs_threshold",
            "threshold_scan": "trigger_rate_vs_threshold",
            # gamma-efficiency vs rate tradeoff
            "efficiency_vs_trigger_rate": "efficiency_vs_trigger_rate",
            "efficiency-vs-trigger-rate": "efficiency_vs_trigger_rate",
            "efficiency_vs_rate": "efficiency_vs_trigger_rate",
            "performance_vs_rate": "efficiency_vs_trigger_rate",
            "tradeoff": "efficiency_vs_trigger_rate",
        }
        return aliases.get(pt, pt)

    def _render_queued_plot(self, plot_type: str, **overrides) -> None:
        """Render the queued configs as the requested plot_type."""
        pt = self._normalize_plot_type(plot_type)

        # Merge stored init kwargs with per-call overrides (overrides win)
        init_kwargs = dict(getattr(self, "_plot_init_kwargs", {}) or {})
        init_kwargs.update(overrides or {})

        def pick(keys: Iterable[str]) -> Dict[str, Any]:
            return {k: init_kwargs[k] for k in keys if k in init_kwargs and init_kwargs[k] is not None}

        if pt == "absolute":
            # initPlotAbsolute uses 'bins' (not nbins)
            abs_kwargs = pick(["title", "metrics", "mode", "x_range", "y_range", "target_rate_hz", "score_threshold"])
            if "bins" in init_kwargs and init_kwargs["bins"] is not None:
                abs_kwargs["bins"] = int(init_kwargs["bins"])
            elif "nbins" in init_kwargs and init_kwargs["nbins"] is not None:
                abs_kwargs["bins"] = int(init_kwargs["nbins"])
            self.initPlotAbsolute(**abs_kwargs)
            for item in getattr(self, "_queued_plots", []):
                kw = dict(item.get("kwargs") or {})
                self.addPlotAbsolute(
                    to_compare_config=item["config"],
                    label=item.get("label"),
                    mul_trig_rate=float(kw.pop("mul_trig_rate", 1.0)),
                    target_rate_hz=kw.pop("target_rate_hz", None),
                    score_threshold=kw.pop("score_threshold", None),
                )

        elif pt == "ratio":
            # initPlotRatio uses 'nbins'
            ratio_kwargs = pick(["title", "metrics", "x_range", "y_range", "target_rate_hz", "score_threshold"])
            if "bins" in init_kwargs and init_kwargs["bins"] is not None:
                ratio_kwargs["nbins"] = int(init_kwargs["bins"])
            elif "nbins" in init_kwargs and init_kwargs["nbins"] is not None:
                ratio_kwargs["nbins"] = int(init_kwargs["nbins"])
            self.initPlotRatio(**ratio_kwargs)
            for item in getattr(self, "_queued_plots", []):
                kw = dict(item.get("kwargs") or {})
                self.addPlotRatio(
                    to_compare_config=item["config"],
                    label=item.get("label"),
                    mul_trig_rate=float(kw.pop("mul_trig_rate", 1.0)),
                    target_rate_hz=kw.pop("target_rate_hz", None),
                    score_threshold=kw.pop("score_threshold", None),
                )

        elif pt == "effective_area_counts":
            ea_counts_kwargs = pick([
                "title", "emin_tev", "emax_tev", "nbins", "show_expected_powerlaw",
                "expected_N", "expected_slope", "x_range", "y_range_counts",
                "target_rate_hz", "score_threshold",
            ])
            # allow 'bins' as alias for nbins in EA
            if "nbins" not in ea_counts_kwargs:
                if "bins" in init_kwargs and init_kwargs["bins"] is not None:
                    ea_counts_kwargs["nbins"] = int(init_kwargs["bins"])
            self.initPlotEffectiveAreaCounts(**ea_counts_kwargs)
            for item in getattr(self, "_queued_plots", []):
                kw = dict(item.get("kwargs") or {})
                self.addPlotEffectiveAreaCounts(
                    to_compare_config=item["config"],
                    label=item.get("label"),
                    target_rate_hz=kw.pop("target_rate_hz", None),
                    score_threshold=kw.pop("score_threshold", None),
                )

        elif pt == "effective_area":
            ea_kwargs = pick([
                "title", "emin_tev", "emax_tev", "nbins", "A_gen_m2", "use_base_thrown",
                "use_theoretical_thrown", "plot_errors", "expected_N", "expected_slope",
                "x_range", "y_range_aeff", "target_rate_hz", "score_threshold",
            ])
            if "nbins" not in ea_kwargs:
                if "bins" in init_kwargs and init_kwargs["bins"] is not None:
                    ea_kwargs["nbins"] = int(init_kwargs["bins"])
            self.initPlotEffectiveArea(**ea_kwargs)
            for item in getattr(self, "_queued_plots", []):
                kw = dict(item.get("kwargs") or {})
                self.addPlotEffectiveArea(
                    to_compare_config=item["config"],
                    label=item.get("label"),
                    A_gen_m2=kw.pop("A_gen_m2", None),
                    use_base_thrown=kw.pop("use_base_thrown", None),
                    use_theoretical_thrown=kw.pop("use_theoretical_thrown", None),
                    plot_errors=kw.pop("plot_errors", None),
                    target_rate_hz=kw.pop("target_rate_hz", None),
                    score_threshold=kw.pop("score_threshold", None),
                )

        elif pt == "trigger_rate_vs_threshold":
            rate_kwargs = pick(["title", "x_range", "y_range"])
            self.initPlotTriggerRateVsThreshold(**rate_kwargs)
            drawn_target_rates: set = set()
            for item in getattr(self, "_queued_plots", []):
                kw = dict(item.get("kwargs") or {})
                item_target_rate_hz = kw.pop("target_rate_hz", init_kwargs.get("target_rate_hz"))
                item_max_points = kw.pop("max_points", init_kwargs.get("max_points", 20_000))
                target_key = None if item_target_rate_hz is None else float(item_target_rate_hz)
                draw_target_rate_line = target_key is not None and target_key not in drawn_target_rates
                self.addPlotTriggerRateVsThreshold(
                    to_compare_config=item["config"],
                    label=item.get("label"),
                    target_rate_hz=item_target_rate_hz,
                    max_points=item_max_points,
                    draw_target_rate_line=draw_target_rate_line,
                )
                if target_key is not None:
                    drawn_target_rates.add(target_key)

        elif pt == "efficiency_vs_trigger_rate":
            eff_rate_kwargs = pick(["title", "x_range", "y_range", "target_rate_hz", "max_points"])
            eff_rate_kwargs["metric"] = self._normalize_metric_name(init_kwargs.get("metrics", "energy"))
            if "metric_bins" in init_kwargs:
                eff_rate_kwargs["metric_bins"] = init_kwargs.get("metric_bins")
            self.initPlotEfficiencyVsTriggerRate(**eff_rate_kwargs)
            drawn_target_rates: set = set()
            for item in getattr(self, "_queued_plots", []):
                kw = dict(item.get("kwargs") or {})
                item_target_rate_hz = kw.pop("target_rate_hz", init_kwargs.get("target_rate_hz"))
                item_max_points = kw.pop("max_points", init_kwargs.get("max_points", 4_000))
                item_metric = self._normalize_metric_name(kw.pop("metrics", init_kwargs.get("metrics", "energy")))
                item_metric_bins = kw.pop("metric_bins", init_kwargs.get("metric_bins"))
                target_key = None if item_target_rate_hz is None else float(item_target_rate_hz)
                draw_target_rate_line = target_key is not None and target_key not in drawn_target_rates
                self.addPlotEfficiencyVsTriggerRate(
                    to_compare_config=item["config"],
                    label=item.get("label"),
                    metric=item_metric,
                    metric_bins=item_metric_bins,
                    target_rate_hz=item_target_rate_hz,
                    max_points=item_max_points,
                    draw_target_rate_line=draw_target_rate_line,
                )
                if target_key is not None:
                    drawn_target_rates.add(target_key)

        else:
            raise ValueError(
                f"Unknown plot_type='{plot_type}'. Supported: 'absolute', 'ratio', 'effective_area', 'effective_area_counts', 'trigger_rate_vs_threshold', 'efficiency_vs_trigger_rate'."
            )

    # Convenience alias (snake_case):
    def show_plot(self, *args, **kwargs):
        return self.showPlot(*args, **kwargs)

    def showTriggerRateVsThreshold(
        self,
        filename: Optional[str] = None,
        show: bool = True,
        location: str = "outside bottom",
        **plot_kwargs,
    ):
        return self.showPlot(
            filename=filename,
            show=show,
            location=location,
            plot_type="trigger_rate_vs_threshold",
            **plot_kwargs,
        )

    def show_trigger_rate_vs_threshold(self, *args, **kwargs):
        return self.showTriggerRateVsThreshold(*args, **kwargs)

    def showEfficiencyVsTriggerRate(
        self,
        filename: Optional[str] = None,
        show: bool = True,
        location: str = "best",
        **plot_kwargs,
    ):
        return self.showPlot(
            filename=filename,
            show=show,
            location=location,
            plot_type="efficiency_vs_trigger_rate",
            **plot_kwargs,
        )

    def show_efficiency_vs_trigger_rate(self, *args, **kwargs):
        return self.showEfficiencyVsTriggerRate(*args, **kwargs)

    def _report_config_label(self, config: ConfigType, index: Optional[int] = None) -> str:
        label = self._generate_config_label_text(
            config,
            threshold_override=None,
            include_threshold=True,
        )
        if label:
            return label
        if index is not None:
            return f"config {index + 1}"
        return "config"

    def _report_config_slug(self, config: ConfigType, index: int) -> str:
        result = self.get_results(config)
        if result is not None:
            filename = result.get("_filename")
            if filename:
                stem = os.path.splitext(os.path.basename(str(filename)))[0]
                if stem:
                    return f"{index + 1:02d}_{stem}"

        label = self._report_config_label(config, index=index).lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in label)
        slug = "_".join(part for part in slug.split("_") if part)
        if not slug:
            slug = f"config_{index + 1:02d}"
        return f"{index + 1:02d}_{slug[:96]}"

    def _report_config_short_name(self, config: ConfigType, index: int) -> str:
        prefix = f"C{index + 1}"
        for stage_type, raw_params in config:
            params = self._as_mapping(raw_params) or {}
            st = str(stage_type).lower()
            if st == "digital_sum":
                mode = params.get("mode", "")
                return f"{prefix} DS {mode}".strip()
            if st == "tdscan":
                tdscan_id = params.get("id")
                tdscan_id_short = self._format_hash_short(tdscan_id) or "td"
                qmap = self._as_mapping(params.get("quantize_step")) or {}
                conv_q = qmap.get("convolution_accumulator")
                temp_q = qmap.get("temporal_accumulator")
                if conv_q is not None and temp_q is not None:
                    return f"{prefix} TD h{tdscan_id_short} [{conv_q}/{temp_q}]"
                return f"{prefix} TD h{tdscan_id_short}"
        return f"{prefix} Config"

    def _report_chain_lines(self, config: ConfigType) -> List[Tuple[str, str]]:
        """Readable per-stage summary of a config's chain for the report.

        Weights are read from the matched stored result (full precision) then
        squeezed/rounded for display, so the tdscan weights show up even when the
        config selected the file by id alone.
        """
        result = self.get_results(config)
        chain = (result.get("trigger_chain") if result else None) or config

        def fmt_num(v: Any) -> str:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return str(v)
            return str(int(fv)) if fv.is_integer() else f"{fv:g}"

        lines: List[Tuple[str, str]] = []
        for stage, raw in chain:
            params = self._as_mapping(raw) or {}
            st = str(stage).lower()
            if st == "tdscan":
                parts = [f"eps_xy={params.get('eps_xy')}", f"eps_t={params.get('eps_t')}"]
                weights = params.get("ring_weights", params.get("kernel_weights"))
                if weights is not None:
                    try:
                        arr = np.round(np.squeeze(np.asarray(weights, dtype=float)), 4)
                        parts.append(f"weights={arr.tolist()}")
                    except (TypeError, ValueError):
                        pass
                qlabel = self._format_quantize_step_label(params.get("quantize_step")).strip()
                if qlabel:
                    parts.append(qlabel)
                if params.get("pad_value"):
                    parts.append(f"pad={fmt_num(params.get('pad_value'))}")
                if params.get("id"):
                    parts.append(f"id={self._format_hash_short(params.get('id'))}")
                lines.append(("tdscan", ", ".join(parts)))
            elif st == "score_quantizer":
                edges = params.get("edges") or []
                lines.append(("score_quantizer", "edges=[" + ", ".join(fmt_num(e) for e in edges) + "]"))
            elif st == "threshold":
                thr = params.get("threshold")
                bits = [fmt_num(thr) if thr is not None else "NA"]
                if params.get("comparison"):
                    bits.append(str(params.get("comparison")))
                if params.get("binary"):
                    bits.append("binary")
                lines.append(("threshold", " ".join(bits)))
            elif st == "shift":
                lines.append(("shift", f"value={fmt_num(params.get('value'))}"))
            elif st == "digital_sum":
                lines.append(("digital_sum", f"mode={params.get('mode')}"))
            else:
                kv = ", ".join(
                    f"{k}={v}" for k, v in params.items() if k not in self._IGNORED_MATCH_KEYS
                )
                lines.append((st, kv))
        return lines

    def _render_distribution_plot(self, metric, x_range, bins, title, full_path, show, kind="gamma"):
        """Render a histogram of all `kind` events for `metric` (trigger-independent).

        Uses the base config's stored events (the gamma sample is shared across
        configs), so this is the distribution of the whole dataset.
        """
        metric = self._normalize_metric_name(metric)
        try:
            values = self._collect_metric_values(self.base_config_result, metric, kind=kind)
        except KeyError:
            return False
        if values.size == 0:
            return False
        lo, hi = x_range
        edges = self._make_log_bins(lo, hi, int(bins))
        with self._report_style_context():
            plt.close("all")
            plt.figure()
            plt.hist(values, bins=edges, color="#2E91E5", alpha=0.85,
                     edgecolor="white", linewidth=0.4)
            plt.xscale("log")
            plt.xlabel(self._metric_xlabel(metric))
            plt.ylabel("Number of events")
            plt.title(title)
            plt.minorticks_on()
            plt.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.4)
            plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
            plt.xlim(max(lo, 1e-3), hi)
            self._report_style_current_figure(title=title)
            self._report_stash_vector_figure(full_path)
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    def _collect_event_columns(self, result, names, kind="gamma", chunk_rows=None):
        """Stream several event datasets together, aligned, for one label kind.

        Unlike `_collect_metric_values`, no per-field finite filtering is applied,
        so derived quantities (e.g. impact distance from xcore/ycore/tel_pos) stay
        row-aligned. The caller is responsible for a joint finite mask.
        """
        if not self._is_h5(result):
            raise ValueError("_collect_event_columns called on non-h5 result.")
        if h5py is None:
            raise RuntimeError("h5py is not installed.")
        chunk_rows = int(chunk_rows or self.h5_chunk_rows)
        out = {nm: [] for nm in names}
        with h5py.File(result["_path"], "r") as f:
            grp, label_ds = self._resolve_h5_group(f, kind)
            for nm in names:
                if nm not in grp:
                    raise KeyError(nm)
            n = int(grp[names[0]].shape[0])
            for start in range(0, n, chunk_rows):
                sl = slice(start, min(n, start + chunk_rows))
                mask = None
                if label_ds is not None:
                    mask = self._label_mask(np.asarray(label_ds[sl]).reshape(-1), kind=kind)
                keep = self._fold_row_mask(grp, sl)
                if keep is not None:
                    mask = keep if mask is None else (mask & keep)
                for nm in names:
                    a = np.asarray(grp[nm][sl]).reshape(-1)
                    out[nm].append(a[mask] if mask is not None else a)
        return {nm: (np.concatenate(v) if v else np.empty((0,), dtype=np.float32))
                for nm, v in out.items()}

    def _gamma_impact_columns(self, result, with_energy=False):
        """Return aligned (impact_distance, score[, energy]) arrays for gamma events."""
        names = ["xcore", "ycore", "tel_pos_x", "tel_pos_y", PRE_THRESHOLD_SCORE_DATASET]
        if with_energy:
            names.append("energy")
        cols = self._collect_event_columns(result, names, kind="gamma")
        r = np.sqrt((cols["xcore"] - cols["tel_pos_x"]) ** 2 + (cols["ycore"] - cols["tel_pos_y"]) ** 2)
        score = cols[PRE_THRESHOLD_SCORE_DATASET]
        finite = np.isfinite(r) & np.isfinite(score)
        if with_energy:
            energy = cols["energy"]
            finite &= np.isfinite(energy)
            return r[finite], score[finite], energy[finite]
        return r[finite], score[finite]

    def _auto_impact_distance_range(self):
        try:
            r, _ = self._gamma_impact_columns(self.base_config_result)
        except (KeyError, OSError, ValueError):
            return (0.0, 500.0)
        if r.size == 0:
            return (0.0, 500.0)
        hi = float(np.percentile(r, 99.0))
        if not np.isfinite(hi) or hi <= 0:
            hi = float(np.max(r)) or 500.0
        return (0.0, hi)

    def _render_score_distribution(self, cfg, label, target_rate_hz, full_path, show, bins=200):
        result = self.get_results(cfg)
        if result is None or not result.get("has_pre_threshold_score", False):
            return False
        gamma = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="gamma")
        nsb = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
        if gamma.size == 0 and nsb.size == 0:
            return False
        all_scores = np.concatenate([a for a in (gamma, nsb) if a.size > 0])
        lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
        if np.isclose(lo, hi):
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        gh = np.histogram(gamma, bins=edges)[0] if gamma.size else np.zeros(bins)
        nh = np.histogram(nsb, bins=edges)[0] if nsb.size else np.zeros(bins)
        with self._report_style_context():
            plt.close("all")
            plt.figure()
            plt.yscale("log")
            if gamma.size:
                plt.step(centers, gh, where="mid", label="Gamma", color="#1CA71C")
            if nsb.size:
                plt.step(centers, nh, where="mid", label="NSB", color="#FB0D0D")
            strat = self._resolve_trigger_strategy(result, target_rate_hz=target_rate_hz)
            tau = strat.get("score_threshold")
            rate = float(strat.get("trigger_rate_hz", float("nan")))
            if tau is not None:
                plt.axvline(tau, color="black", linestyle="--",
                            label=f"threshold={tau:.6g}, rate={rate:.0f} Hz")
            roc = self._config_roc_auc(result)
            auc_txt = f"; AUC={roc * 100.0:.1f}%" if np.isfinite(roc) else ""
            plt.xlabel("Pre-threshold score")
            plt.ylabel("Events / bin")
            plt.title(f"Pre-threshold score distribution\n{label}; {self._rate_suffix(rate)}{auc_txt}")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend(loc="best", fontsize=9)
            self._report_style_current_figure()
            self._report_stash_vector_figure(full_path)
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    def _render_roc_curves(self, plot_items, full_path, show):
        """Overlay the gamma-vs-NSB ROC curve of every config (area = ROC AUC)."""
        drew = False
        with self._report_style_context():
            plt.close("all")
            plt.figure()
            for cfg, label in plot_items:
                result = self.get_results(cfg)
                if result is None or not result.get("has_pre_threshold_score", False):
                    continue
                gamma = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="gamma")
                nsb = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
                if gamma.size == 0 or nsb.size == 0:
                    continue
                fpr, tpr = roc_curve(gamma, nsb)
                auc = self._config_roc_auc(result)
                auc_txt = f"; AUC={auc * 100.0:.1f}%" if np.isfinite(auc) else ""
                plt.plot(fpr, tpr, color=self._get_config_plot_color(cfg), label=f"{label}{auc_txt}")
                drew = True
            if not drew:
                plt.close("all")
                return False
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, label="random")
            plt.xlabel("NSB acceptance (false positive rate)")
            plt.ylabel("Gamma efficiency (true positive rate)")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.02)
            plt.grid(True, alpha=0.3)
            plt.title("ROC curve (gamma vs NSB, pre-threshold score)")
            plt.legend(loc="lower right", fontsize=9)
            self._report_style_current_figure()
            self._report_stash_vector_figure(full_path)
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    def _render_efficiency_vs_impact_distance(self, plot_items, target_rate_hz, r_range, bins, full_path, show):
        edges = np.linspace(r_range[0], r_range[1], int(bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        drew = False
        with self._report_style_context():
            plt.close("all")
            plt.figure()
            for cfg, label in plot_items:
                result = self.get_results(cfg)
                if result is None or not result.get("has_pre_threshold_score", False):
                    continue
                strategy = self._resolve_trigger_strategy(result, target_rate_hz=target_rate_hz)
                tau = strategy.get("score_threshold")
                if tau is None:
                    continue
                try:
                    r, score = self._gamma_impact_columns(result)
                except (KeyError, OSError, ValueError):
                    continue
                if r.size == 0:
                    continue
                trig = self._score_trigger_mask(score, float(tau), comparison=result.get("threshold_comparison", "gt"))
                inr = (r >= edges[0]) & (r <= edges[-1])
                all_h = np.histogram(r[inr], bins=edges)[0]
                trig_h = np.histogram(r[inr & trig], bins=edges)[0]
                eff, eff_err = self._efficiency_from_counts(all_h, trig_h)
                curve_label = f"{label}; {self._rate_suffix(strategy.get('trigger_rate_hz', 0.0))}"
                plt.errorbar(centers, eff, yerr=eff_err, fmt="o-", markersize=3, capsize=2,
                             color=self._get_config_plot_color(cfg), label=curve_label)
                drew = True
            if not drew:
                plt.close("all")
                return False
            plt.xlabel("Impact distance r [m]")
            plt.ylabel("Trigger efficiency")
            plt.ylim(0.0, 1.05)
            plt.grid(True, alpha=0.3)
            rate_txt = f"  (target {float(target_rate_hz):.0f} Hz)" if target_rate_hz is not None else ""
            plt.title(f"Gamma trigger efficiency vs impact distance{rate_txt}")
            plt.legend(loc="best", fontsize=9)
            self._report_style_current_figure()
            self._report_stash_vector_figure(full_path)
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    def _render_efficiency_map_energy_impact(self, cfg, label, target_rate_hz, energy_range, r_range,
                                             full_path, show, e_bins=25, r_bins=20):
        result = self.get_results(cfg)
        if result is None or not result.get("has_pre_threshold_score", False):
            return False
        strategy = self._resolve_trigger_strategy(result, target_rate_hz=target_rate_hz)
        tau = strategy.get("score_threshold")
        if tau is None:
            return False
        try:
            r, score, energy = self._gamma_impact_columns(result, with_energy=True)
        except (KeyError, OSError, ValueError):
            return False
        if r.size == 0:
            return False
        trig = self._score_trigger_mask(score, float(tau), comparison=result.get("threshold_comparison", "gt"))
        e_edges = np.logspace(np.log10(max(energy_range[0], 1e-4)), np.log10(energy_range[1]), e_bins + 1)
        r_edges = np.linspace(r_range[0], r_range[1], r_bins + 1)
        all2d = np.histogram2d(energy, r, bins=[e_edges, r_edges])[0]
        trig2d = np.histogram2d(energy[trig], r[trig], bins=[e_edges, r_edges])[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            eff = np.where(all2d > 0, trig2d / all2d, np.nan)
        if not np.isfinite(eff).any():
            return False
        with self._report_style_context():
            plt.close("all")
            plt.figure()
            mesh = plt.pcolormesh(e_edges, r_edges, eff.T, vmin=0.0, vmax=1.0, cmap="viridis", shading="auto")
            plt.colorbar(mesh, label="Trigger efficiency")
            plt.xscale("log")
            plt.xlabel("Energy (TeV)")
            plt.ylabel("Impact distance r [m]")
            plt.title(f"Trigger efficiency map (E × impact distance)\n{label}; {self._rate_suffix(strategy.get('trigger_rate_hz', 0.0))}")
            self._report_style_current_figure()
            self._report_stash_vector_figure(full_path)
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    def _render_npe_vs_energy(self, energy_range, n_pe_range, full_path, show, e_bins=60, npe_bins=60):
        result = self.base_config_result
        e_edges = np.logspace(np.log10(max(energy_range[0], 1e-4)), np.log10(energy_range[1]), e_bins)
        npe_edges = np.logspace(np.log10(max(n_pe_range[0], 1e-3)), np.log10(n_pe_range[1]), npe_bins)
        try:
            counts = self._histogram2d_stream(result, "energy", "n_pe", kind="gamma",
                                              x_bins=e_edges, y_bins=npe_edges)
        except (KeyError, OSError, ValueError):
            return False
        if counts.size == 0 or float(np.nanmax(counts)) <= 0:
            return False
        with self._report_style_context():
            plt.close("all")
            plt.figure()
            norm = colors.LogNorm(vmin=1, vmax=float(np.nanmax(counts)))
            mesh = plt.pcolormesh(e_edges, npe_edges, counts.T, norm=norm, cmap="viridis", shading="auto")
            plt.colorbar(mesh, label="Number of events")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Energy (TeV)")
            plt.ylabel("Number of Photoelectrons (n_pe)")
            plt.title("n_pe vs Energy (all gamma events)")
            self._report_style_current_figure()
            self._report_stash_vector_figure(full_path)
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    @staticmethod
    def _draw_score_quantizer_inset(fig, edges, rect, color="#2E91E5", ink="#2f2f2f", muted="#6b6b6b"):
        """Draw the score-quantizer staircase transfer function as an inset.

        Convention (see ScoreQuantizer): code = count(score >= edge), i.e. a step
        function rising by one at every edge, from 0 up to len(edges).
        """
        try:
            edges = [float(e) for e in edges]
        except (TypeError, ValueError):
            return
        if not edges:
            return
        n = len(edges)
        # Input is an 8-bit value, so always show the full 0..255 range.
        x_min, x_max = 0.0, 255.0
        xs = [x_min] + edges + [x_max]
        ys = [0] + list(range(1, n + 1)) + [n]

        ax = fig.add_axes(rect)
        ax.step(xs, ys, where="post", color=color, linewidth=1.4)
        ax.plot(edges, list(range(1, n + 1)), "o", color=color, markersize=2.5)
        ax.set_title("Score quantizer transfer fn", fontsize=7, color=ink, pad=2)
        ax.set_xlabel("input", fontsize=6, color=muted, labelpad=1)
        ax.set_ylabel("output", fontsize=6, color=muted, labelpad=1)
        ax.tick_params(labelsize=6, colors=muted, length=2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, n + 0.5)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    def _config_score_quantizer_edges(self, cfg: ConfigType) -> Optional[List[float]]:
        """Return the score_quantizer edges for a config, or None if it has none.

        Prefers the matched stored chain (full precision) and falls back to the
        config as written, mirroring the inset lookup in the PDF details page.
        """
        stored_chain = (self.get_results(cfg) or {}).get("trigger_chain") or cfg
        for stage, raw in stored_chain:
            if str(stage).lower() == "score_quantizer":
                return (self._as_mapping(raw) or {}).get("edges")
        return None

    def _render_score_quantizer(self, cfg, label, full_path, show, presentation_svg=False):
        """Render the score-quantizer transfer function as a standalone plot.

        Draws the same staircase as the PDF inset (see
        :meth:`_draw_score_quantizer_inset`) but as a full-size figure, then
        stashes it as a vector figure and saves the PNG, exactly like every
        other report plot. Returns False if the config has no score_quantizer.

        When ``presentation_svg`` is set, the standalone ``.svg`` is rendered in
        a polished, slide-ready style (large type, filled staircase, fonts
        embedded as paths so Keynote/PowerPoint render it identically regardless
        of installed fonts). The report PNG/PDF stay in the standard style.
        """
        edges = self._config_score_quantizer_edges(cfg)
        if not edges:
            return False
        try:
            edges = [float(e) for e in edges]
        except (TypeError, ValueError):
            return False
        if not edges:
            return False

        n = len(edges)
        # Input is an 8-bit value, so always show the full 0..255 range.
        x_min, x_max = 0.0, 255.0
        xs = [x_min] + edges + [x_max]
        ys = [0] + list(range(1, n + 1)) + [n]

        color = self._get_config_plot_color(cfg)
        title = f"Score quantizer transfer function\n{label}"
        with self._report_style_context():
            plt.close("all")
            plt.figure()
            ax = plt.gca()
            ax.step(xs, ys, where="post", color=color, linewidth=2.0)
            ax.plot(edges, list(range(1, n + 1)), "o", color=color, markersize=6)
            ax.set_xlabel("Input (8-bit score)")
            ax.set_ylabel("Output (quantized code)")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.5, n + 0.5)
            ax.grid(True, which="major", alpha=0.3)
            plt.title(title)
            self._report_style_current_figure(title=title)
            self._report_stash_vector_figure(
                full_path,
                presentation_svg_renderer=(
                    (lambda svg_path: self._render_score_quantizer_presentation(
                        edges, label, color, svg_path))
                    if presentation_svg else None
                ),
            )
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    def _render_score_quantizer_presentation(self, edges, label, color, svg_path):
        """Write a slide-ready vector SVG of the score-quantizer transfer fn.

        Standalone from the report styling: bigger fonts, a filled staircase,
        and fonts embedded as paths (``svg.fonttype='path'``) so the file looks
        identical in Keynote/PowerPoint even without the rendering font.
        """
        edges = [float(e) for e in edges]
        n = len(edges)
        x_min, x_max = 0.0, 255.0
        xs = [x_min] + edges + [x_max]
        ys = [0] + list(range(1, n + 1)) + [n]

        with plt.rc_context({
            "svg.fonttype": "path",        # embed glyphs as paths -> font-independent
            "figure.figsize": (12.0, 7.5),
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.6,
            "axes.titlesize": 26,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "font.size": 16,
        }):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Filled staircase + crisp outline for a bold slide read.
            ax.fill_between(xs, ys, step="post", color=color, alpha=0.18, linewidth=0)
            ax.step(xs, ys, where="post", color=color, linewidth=3.5,
                    solid_capstyle="round", solid_joinstyle="round")
            ax.plot(edges, list(range(1, n + 1)), "o", color=color,
                    markersize=11, markeredgecolor="white", markeredgewidth=1.6,
                    zorder=5)
            ax.set_xlabel("Input (8-bit score)")
            ax.set_ylabel("Output (quantized code)")
            ax.set_title(f"Score quantizer transfer function\n{label}", pad=16)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.5, n + 0.5)
            ax.grid(True, which="major", color="#cccccc", linestyle="--",
                    linewidth=1.0, alpha=0.8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.margins(x=0.0)
            fig.tight_layout()
            fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    def _report_stash_vector_figure(self, full_path: str, presentation_svg_renderer=None) -> None:
        """Stash the current figure so the PDF/HTML can embed it as vector graphics.

        When vector mode is active (see generateReport(vector_plots=...)),
        the PDF renderer draws the live Matplotlib figure straight onto the
        page — sharp at any zoom, like the score-quantizer transfer-function
        inset — instead of pasting the rasterized PNG. No-op otherwise.

        When generateReport(save_svg=True), this also writes a vector ``.svg``
        next to the ``.png`` the caller is about to save, so each plot is
        available as a standalone scalable file too.

        ``presentation_svg_renderer``, when given, is called as
        ``renderer(svg_path)`` to write the standalone ``.svg`` instead of the
        default ``fig.savefig`` — letting a caller emit a slide-ready variant
        while the stashed PDF/HTML figure stays in the report style.

        The figure object survives the helper's trailing ``plt.close("all")``
        (close only detaches it from pyplot), so it is safe to call this just
        before that close.
        """
        figs = getattr(self, "_report_vector_figs", None)
        if figs is None:
            return
        fig = plt.gcf()
        figs[os.path.abspath(full_path)] = fig
        if getattr(self, "_report_save_svg", False):
            svg_path = os.path.splitext(full_path)[0] + ".svg"
            try:
                if presentation_svg_renderer is not None:
                    presentation_svg_renderer(svg_path)
                else:
                    fig.savefig(svg_path, bbox_inches="tight", facecolor="#fbfbf8")
            except Exception as exc:  # pragma: no cover - SVG is best-effort
                self._warn_once(
                    "report-svg-failed",
                    f"Failed to write SVG plot ({exc!r}); PNG still available.",
                )

    @contextmanager
    def _report_style_context(self):
        with plt.rc_context({
            "figure.figsize": (11.5, 7.0),
            "figure.facecolor": "#fbfbf8",
            "savefig.facecolor": "#fbfbf8",
            "axes.facecolor": "#fffdfa",
            "axes.edgecolor": "#2f2f2f",
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "#fffdfa",
            "legend.edgecolor": "#d9d3c7",
            "legend.fancybox": True,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.3,
            "lines.markersize": 6.5,
            "errorbar.capsize": 3.5,
        }):
            yield

    def _report_style_current_figure(self, title: Optional[str] = None) -> None:
        fig = plt.gcf()
        for ax in fig.get_axes():
            try:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            except Exception:
                pass
            if title:
                try:
                    current_title = ax.get_title()
                    if not current_title:
                        ax.set_title(title, fontsize=17, fontweight="bold")
                except Exception:
                    pass
            try:
                ax.grid(True, which="major", alpha=0.22, linestyle="--")
                ax.grid(True, which="minor", alpha=0.08, linestyle=":")
            except Exception:
                pass

    def _report_render_queued_plot(
        self,
        plot_items: List[Tuple[ConfigType, str]],
        *,
        plot_type: str,
        filename: str,
        show: bool,
        location: str = "best",
        title: Optional[str] = None,
        **plot_kwargs,
    ) -> bool:
        if not plot_items:
            return False
        with self._report_style_context():
            plt.close("all")
            self.init_plot()
            for cfg, label in plot_items:
                self.add_plot(cfg, label=label)
            self.showPlot(
                filename=filename,
                show=show,
                location=location,
                plot_type=plot_type,
                **plot_kwargs,
            )
            self._report_style_current_figure(title=title)
            self._report_stash_vector_figure(filename)
            plt.gcf().savefig(filename, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
            return os.path.exists(filename)

    # ------------------------------------------------------------------ #
    # Cross-validation (per-fold leakage) plot
    # ------------------------------------------------------------------ #
    def _cv_label_for_chain(self, chain: ConfigType) -> str:
        """Short human label for a chain (stage types + threshold)."""
        try:
            stages = [str(stage[0]) for stage in chain]
        except Exception:
            return "chain"
        thr = self._extract_threshold_from_chain(chain)
        core = "+".join(s for s in stages if s != "threshold") or "chain"
        return f"{core} (thr={thr:.3g})" if thr is not None else core

    def _read_folds_group(self, h5_path: str) -> Optional[List[Dict[str, Any]]]:
        """Read the `/folds` summary table of one stats file.

        Returns a list of per-fold dicts (name, counts, rate, window_sec) with
        gamma efficiency + NSB rate Wilson errors already attached, or None when
        the file has no `/folds` group (a plain single-fold / pre-fold file).
        """
        if h5py is None:
            raise RuntimeError("h5py is not installed but a fold read was requested.")

        def s(x):
            return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

        with h5py.File(h5_path, "r") as f:
            if "folds" not in f:
                return None
            g = f["folds"]
            window_sec = self._read_window_sec(f.attrs)
            names = [s(x) for x in g["name"][()]]
            gt = g["gamma_trig"][()]; gtot = g["gamma_total"][()]
            nt = g["nsb_trig"][()];   ntot = g["nsb_total"][()]
            rate = g["trigger_rate_hz"][()]
            folds = []
            for i, name in enumerate(names):
                eff, elo, ehi = wilson(int(gt[i]), int(gtot[i]), self.wilson_level)
                _, rlo, rhi = wilson(int(nt[i]), int(ntot[i]), self.wilson_level)
                folds.append({
                    "fold": name,
                    "eff": eff, "eff_err": max(elo, ehi),
                    "rate_hz": float(rate[i]),
                    "rate_err": max(rlo, rhi) / window_sec,
                })
            return folds

    def _resolve_cv_rows(
        self,
        configs: Optional[List[ConfigType]],
        legend_overrides: Optional[List[Optional[str]]],
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Resolve the ``(label, folds)`` rows to draw for the CV plot.

        ``configs=None`` picks every loaded stats file carrying a ``/folds``
        group; otherwise each config is matched to its file. Files without a
        ``/folds`` group are silently skipped (they are plain single-pass runs).
        """
        rows: List[Tuple[str, List[Dict[str, Any]]]] = []
        if configs is None:
            seen = set()
            for _fn, res in self.all_results:
                path = res.get("_path")
                if not path or path in seen:
                    continue
                seen.add(path)
                folds = self._read_folds_group(path)
                if folds:
                    rows.append((self._cv_label_for_chain(res.get("trigger_chain", [])), folds))
        else:
            for i, config in enumerate(configs):
                res = self.get_results(config)
                if res is None or not self._is_h5(res):
                    self._warn_once(f"cv-miss-{i}", f"cross-validation: no stats file matched config #{i}; skipping.")
                    continue
                folds = self._read_folds_group(res["_path"])
                if not folds:
                    continue  # plain single-pass file -- nothing to draw
                label = None
                if legend_overrides and i < len(legend_overrides):
                    label = legend_overrides[i]
                rows.append((label or self._cv_label_for_chain(res.get("trigger_chain", [])), folds))
        return rows

    def _draw_cross_validation(
        self,
        rows: List[Tuple[str, List[Dict[str, Any]]]],
        n_sigma: float = 3.0,
    ) -> "plt.Figure":
        """Draw the per-fold eff + NSB-rate grid onto a fresh figure and return it.

        One row per config: gamma efficiency vs fold (left), NSB rate vs fold
        (right), Wilson error bars, a dashed fold-0 reference and an ``n_sigma``
        band around it. Flat across folds = no leak.
        """
        n = len(rows)
        fig, axes = plt.subplots(n, 2, figsize=(11.5, 3.3 * n), squeeze=False)
        for r, (label, folds) in enumerate(rows):
            names = [d["fold"] for d in folds]
            xs = np.arange(len(folds))
            for c, (key, err, ylab, sub) in enumerate([
                ("eff", "eff_err", "gamma efficiency", "Gamma efficiency vs fold"),
                ("rate_hz", "rate_err", "NSB rate [Hz]", "NSB rate vs fold"),
            ]):
                ax = axes[r][c]
                ax.errorbar(xs, [d[key] for d in folds], yerr=[d[err] for d in folds],
                            fmt="o-", capsize=4)
                ref = folds[0]
                ax.axhline(ref[key], ls="--", c="gray", alpha=0.6, label="fold-0")
                band = n_sigma * ref[err]
                ax.axhspan(ref[key] - band, ref[key] + band, color="gray", alpha=0.10,
                           label=f"±{n_sigma:g}σ")
                ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right")
                ax.set_ylabel(ylab)
                ax.set_title(f"{label}: {sub}")
                ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _render_cross_validation(
        self,
        configs: Optional[List[ConfigType]],
        legend_overrides: Optional[List[Optional[str]]],
        full_path: str,
        show: bool,
        n_sigma: float = 3.0,
    ) -> bool:
        """Report-integrated CV render: PNG into ``full_path`` + stash for the PDF.

        Mirrors the other ``_render_*`` helpers so the cross-validation graph
        becomes one section INSIDE the report (report.md image + a vector PDF
        page), rather than a standalone file. Returns False (drawing nothing)
        when no config has a ``/folds`` group.
        """
        rows = self._resolve_cv_rows(configs, legend_overrides)
        if not rows:
            return False
        with self._report_style_context():
            plt.close("all")
            self._draw_cross_validation(rows, n_sigma=n_sigma)
            self._report_style_current_figure()
            self._report_stash_vector_figure(full_path)
            plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
            if show:
                plt.show()
            plt.close("all")
        return os.path.exists(full_path)

    def plot_cross_validation(
        self,
        configs: Optional[List[ConfigType]] = None,
        output_dir: str = "trigger_report",
        filename: str = "cross_validation",
        n_sigma: float = 3.0,
        legend_overrides: Optional[List[Optional[str]]] = None,
        formats: Tuple[str, ...] = ("png",),
        show: bool = False,
    ) -> Optional[str]:
        """Standalone per-fold gamma efficiency + NSB rate cross-validation plot.

        Renders the leakage-detector graph from the ``/folds`` group written by
        ``compute_statistics(folds=...)``: one row per config, gamma efficiency
        vs fold (left) and NSB rate vs fold (right), with Wilson error bars and a
        dashed fold-0 reference. A physically-honest trigger is FLAT across folds
        (efficiency invariant under camera rotation, rate invariant under an NSB
        reshuffle); a drift beyond ``n_sigma`` of fold-0's band is a leak.

        This writes a STANDALONE file (one per entry in ``formats``). The same
        graph is embedded automatically as a section of ``generateReport`` when
        any plotted config carries folds, so you rarely need to call this
        directly. ``configs=None`` picks every loaded file that has folds.
        Returns the first written path, or None if no config had folds.
        """
        rows = self._resolve_cv_rows(configs, legend_overrides)
        if not rows:
            print("plot_cross_validation: no config with a /folds group; nothing to plot.")
            return None

        fig = self._draw_cross_validation(rows, n_sigma=n_sigma)
        os.makedirs(output_dir, exist_ok=True)
        written = []
        for ext in formats:
            path = os.path.join(output_dir, f"{filename}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=200)
            written.append(path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"plot_cross_validation: wrote {', '.join(written)}")
        return written[0] if written else None

    def generateReport(
        self,
        configs: Optional[List[ConfigType]] = None,
        output_dir: str = "stat_report",
        report_name: str = "report.md",
        title: str = "StatPlotter Report",
        target_rate_hz: Optional[float] = None,
        show: bool = False,
        include_base_config: bool = False,
        legend_overrides: Optional[List[Optional[str]]] = None,
        generate_pdf: bool = True,
        pdf_name: str = "report.pdf",
        vector_plots: bool = True,
        generate_html: bool = True,
        html_name: str = "report.html",
        save_svg: bool = True,
        presentation_svg: bool = False,
        individual_rate_scan_max_points: int = 20_000,
        effective_area_emin_tev: float = EA_DEFAULT_EMIN_TEV,
        effective_area_emax_tev: float = EA_DEFAULT_EMAX_TEV,
        effective_area_expected_N: Optional[float] = None,
        effective_area_expected_slope: float = EA_DEFAULT_SLOPE,
        effective_area_a_gen_m2: float = EA_DEFAULT_A_GEN_M2,
        effective_area_nbins: int = 50,
        n_pe_x_range: Tuple[float, float] = (20, 350),
        n_pe_absolute_y_range: Tuple[float, float] = (0.0, 1.05),
        n_pe_ratio_y_range: Tuple[float, float] = (0.6, 2.05),
        n_pe_bins: int = 60,
        energy_x_range: Tuple[float, float] = (0.1, 800),
        energy_absolute_y_range: Tuple[float, float] = (0.0, 1.05),
        energy_ratio_y_range: Tuple[float, float] = (0.0, 2.05),
        energy_bins: int = 70,
        impact_distance_range: Optional[Tuple[float, float]] = None,
        impact_distance_bins: int = 40,
        cross_validation: bool = True,
        cross_validation_n_sigma: float = 3.0,
    ) -> str:
        """
        Generate a markdown report plus the main plots commonly used in the
        SST-1M TDScan studies.

        The report contains:
        - combined plots for the provided configurations
        - one independent trigger-rate-vs-threshold scan per configuration

        ``presentation_svg`` (requires ``save_svg``): emit the standalone
        score-quantizer ``.svg`` in a polished, slide-ready style (large type,
        filled staircase, fonts embedded as paths for Keynote/PowerPoint). The
        report PNG/PDF keep the standard style. No effect when a config has no
        score_quantizer stage.

        Returns the absolute path to the generated markdown report.
        """
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        saved_plot_init_kwargs = dict(getattr(self, "_plot_init_kwargs", {}) or {})
        saved_queue = [dict(item) for item in getattr(self, "_queued_plots", [])]

        # Helpers stash their live figures here (keyed by absolute output path) so
        # the PDF can embed them as vector graphics (instead of the rasterized
        # PNGs) and the HTML report can inline them as zoomable SVG. The PNGs are
        # still written for the markdown report; only the PDF/HTML benefit.
        # When save_svg is on, each helper also drops a .svg next to its .png.
        need_figs = (generate_pdf and vector_plots) or generate_html or save_svg
        self._report_vector_figs = {} if need_figs else None
        self._report_save_svg = bool(save_svg)

        try:
            if configs is None:
                configs = [dict(item).get("config") for item in saved_queue if dict(item).get("config") is not None]
            else:
                configs = list(configs)

            legend_overrides = list(legend_overrides) if legend_overrides is not None else None

            if include_base_config:
                if self.base_reference_config not in configs:
                    configs = [self.base_reference_config] + configs
                    if legend_overrides is not None:
                        legend_overrides = [None] + legend_overrides

            if not configs:
                raise ValueError("generateReport requires at least one configuration.")

            if legend_overrides is not None and len(legend_overrides) != len(configs):
                self._warn_once(
                    "legend-overrides-length",
                    f"legend_overrides has {len(legend_overrides)} entries but {len(configs)} "
                    "configs were provided; extra/missing entries fall back to auto labels.",
                )

            resolved_configs: List[ConfigType] = []
            skipped_configs: List[str] = []
            config_descriptions: List[Tuple[ConfigType, str, str, str]] = []

            for idx, cfg in enumerate(configs):
                label = self._report_config_label(cfg, index=idx)
                short_name = self._report_config_short_name(cfg, index=idx)
                if legend_overrides is not None and idx < len(legend_overrides):
                    override = legend_overrides[idx]
                    if override:
                        short_name = str(override)
                result = self.get_results(cfg)
                if result is None:
                    skipped_configs.append(f"{label}: configuration not found in stat folder")
                    continue
                slug = self._report_config_slug(cfg, idx)
                resolved_configs.append(cfg)
                config_descriptions.append((cfg, label, slug, short_name))

            if not resolved_configs:
                raise ValueError("None of the requested configurations were found in the stat folder.")

            combined_sections: List[Tuple[str, str]] = []
            report_plot_items = [(cfg, short_name) for cfg, _label, _slug, short_name in config_descriptions]

            def add_combined_plot(section_title: str, filename: str, plot_type: str, location: str = "best", **kwargs) -> None:
                full_path = os.path.join(output_dir, filename)
                if self._report_render_queued_plot(
                    report_plot_items,
                    plot_type=plot_type,
                    filename=full_path,
                    show=show,
                    location=location,
                    title=section_title,
                    **kwargs,
                ):
                    combined_sections.append((section_title, filename))

            add_combined_plot(
                "NSB Trigger Rate vs Threshold",
                "nsb_trigger_rate_vs_threshold_all.png",
                "trigger_rate_vs_threshold",
                location="best",
                target_rate_hz=target_rate_hz,
            )
            add_combined_plot(
                "Gamma Efficiency vs NSB Trigger Rate",
                "gamma_efficiency_vs_rate_all.png",
                "efficiency_vs_trigger_rate",
                location="upper left",
                target_rate_hz=target_rate_hz,
            )

            ea_common_kwargs: Dict[str, Any] = {
                "emin_tev": effective_area_emin_tev,
                "emax_tev": effective_area_emax_tev,
                "expected_slope": effective_area_expected_slope,
                "nbins": effective_area_nbins,
                "target_rate_hz": target_rate_hz,
            }
            if effective_area_expected_N is not None:
                ea_common_kwargs["expected_N"] = effective_area_expected_N

            add_combined_plot(
                "Effective Area Counts",
                "effective_area_counts.png",
                "effective_area_counts",
                location="upper left",
                **ea_common_kwargs,
            )
            add_combined_plot(
                "Effective Area",
                "effective_area.png",
                "effective_area",
                location="upper left",
                A_gen_m2=effective_area_a_gen_m2,
                **ea_common_kwargs,
            )
            add_combined_plot(
                "Absolute Efficiency vs Npe",
                "absolute_efficiency_npe.png",
                "absolute",
                metrics="n_pe",
                x_range=n_pe_x_range,
                y_range=n_pe_absolute_y_range,
                bins=n_pe_bins,
                target_rate_hz=target_rate_hz,
            )
            add_combined_plot(
                "Efficiency Ratio vs Npe",
                "ratio_efficiency_npe.png",
                "ratio",
                metrics="n_pe",
                x_range=n_pe_x_range,
                y_range=n_pe_ratio_y_range,
                bins=n_pe_bins,
                target_rate_hz=target_rate_hz,
            )
            add_combined_plot(
                "Absolute Efficiency vs Energy",
                "absolute_efficiency_energy.png",
                "absolute",
                metrics="energy",
                x_range=energy_x_range,
                y_range=energy_absolute_y_range,
                bins=energy_bins,
                target_rate_hz=target_rate_hz,
            )
            add_combined_plot(
                "Efficiency Ratio vs Energy",
                "ratio_efficiency_energy.png",
                "ratio",
                metrics="energy",
                x_range=energy_x_range,
                y_range=energy_ratio_y_range,
                bins=energy_bins,
                target_rate_hz=target_rate_hz,
            )

            # Distributions of the whole dataset (all gamma events, trigger-independent).
            def add_distribution(section_title: str, filename: str, metric: str, x_range, bins) -> None:
                full_path = os.path.join(output_dir, filename)
                if self._render_distribution_plot(metric, x_range, bins, section_title, full_path, show):
                    combined_sections.append((section_title, filename))

            add_distribution(
                "Npe Distribution (all gamma events)",
                "distribution_npe.png", "n_pe", n_pe_x_range, n_pe_bins,
            )
            add_distribution(
                "Energy Distribution (all gamma events)",
                "distribution_energy.png", "energy", energy_x_range, energy_bins,
            )

            # ROC curve (gamma vs NSB separability) for every config.
            roc_path = os.path.join(output_dir, "roc_curves.png")
            if self._render_roc_curves(report_plot_items, roc_path, show):
                combined_sections.append(("ROC Curve (gamma vs NSB)", "roc_curves.png"))

            # Cross-validation (per-fold leakage) -- only when a config carries a
            # /folds group. One row per config: gamma efficiency + NSB rate vs
            # fold with Wilson bars; flat = no leak. Embedded as a report section
            # (no standalone file).
            if cross_validation:
                cv_configs = [cfg for cfg, _l, _s, _n in config_descriptions]
                cv_labels = [short_name for _c, _l, _s, short_name in config_descriptions]
                cv_path = os.path.join(output_dir, "cross_validation.png")
                if self._render_cross_validation(
                    cv_configs, cv_labels, cv_path, show, n_sigma=cross_validation_n_sigma,
                ):
                    combined_sections.append(
                        ("Cross-Validation (per-fold gamma efficiency & NSB rate)", "cross_validation.png"))

            # n_pe vs energy correlation (whole gamma sample).
            npe_energy_path = os.path.join(output_dir, "npe_vs_energy.png")
            if self._render_npe_vs_energy(energy_x_range, n_pe_x_range, npe_energy_path, show):
                combined_sections.append(("n_pe vs Energy (all gamma events)", "npe_vs_energy.png"))

            # Trigger efficiency vs impact distance (overlay of all configs).
            if impact_distance_range is None:
                impact_distance_range = self._auto_impact_distance_range()
            impact_path = os.path.join(output_dir, "efficiency_vs_impact_distance.png")
            if self._render_efficiency_vs_impact_distance(
                report_plot_items, target_rate_hz, impact_distance_range, impact_distance_bins,
                impact_path, show,
            ):
                combined_sections.append(("Gamma Efficiency vs Impact Distance", "efficiency_vs_impact_distance.png"))

            # Per-config: pre-threshold score distribution (gamma vs NSB) + 2D efficiency map.
            for _idx, (cfg, _label, slug, short_name) in enumerate(config_descriptions):
                quantizer_fn = f"score_quantizer_{slug}.png"
                if self._render_score_quantizer(
                    cfg, short_name, os.path.join(output_dir, quantizer_fn), show,
                    presentation_svg=presentation_svg,
                ):
                    combined_sections.append((f"Score Quantizer — {short_name}", quantizer_fn))

                score_fn = f"score_distribution_{slug}.png"
                if self._render_score_distribution(
                    cfg, short_name, target_rate_hz, os.path.join(output_dir, score_fn), show,
                ):
                    combined_sections.append((f"Score Distribution — {short_name}", score_fn))

                map_fn = f"efficiency_map_{slug}.png"
                if self._render_efficiency_map_energy_impact(
                    cfg, short_name, target_rate_hz, energy_x_range, impact_distance_range,
                    os.path.join(output_dir, map_fn), show,
                ):
                    combined_sections.append((f"Efficiency Map (E × impact distance) — {short_name}", map_fn))

            individual_sections: List[Tuple[str, str]] = []
            for idx, (cfg, label, slug, short_name) in enumerate(config_descriptions):
                result = self.get_results(cfg)
                if result is None:
                    continue
                if not result.get("has_pre_threshold_score", False):
                    skipped_configs.append(f"{label}: no '{PRE_THRESHOLD_SCORE_DATASET}' dataset, skipped individual trigger-rate plot")
                    continue

                filename = f"trigger_rate_vs_threshold_{slug}.png"
                full_path = os.path.join(output_dir, filename)
                with self._report_style_context():
                    plt.close("all")
                    self.initPlotTriggerRateVsThreshold(
                        title=f"NSB Trigger Rate vs Threshold\n{short_name}",
                    )
                    self.addPlotTriggerRateVsThreshold(
                        to_compare_config=cfg,
                        label=short_name,
                        target_rate_hz=target_rate_hz,
                        max_points=individual_rate_scan_max_points,
                        draw_target_rate_line=True,
                        draw_target_threshold_line=True,
                    )
                    self._report_style_current_figure()
                    self._report_stash_vector_figure(full_path)
                    plt.gcf().savefig(full_path, bbox_inches="tight", dpi=320)
                    if show:
                        plt.show()
                    plt.close("all")
                if os.path.exists(full_path):
                    individual_sections.append((label, filename))
                else:
                    skipped_configs.append(f"{label}: failed to generate individual trigger-rate plot")

            report_path = os.path.join(output_dir, report_name)
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(f"# {title}\n\n")
                fh.write(f"- Stat folder: `{self.stat_folder}`\n")
                fh.write(f"- Base reference config: `{self.base_reference_config}`\n")
                if target_rate_hz is not None:
                    fh.write(f"- Target rate: `{float(target_rate_hz):.1f} Hz`\n")
                fh.write("\n")

                fh.write("## Configurations\n\n")
                for idx, (_cfg, label, slug, short_name) in enumerate(config_descriptions, start=1):
                    fname = os.path.basename(str((self.get_results(_cfg) or {}).get("_filename") or slug))
                    fh.write(f"{idx}. `{short_name}` — file: `{fname}`\n")
                    for si, (stage, detail) in enumerate(self._report_chain_lines(_cfg), start=1):
                        fh.write(f"   - Stage {si} — `{stage}`: {detail}\n")
                fh.write("\n")

                fh.write("## Combined Plots\n\n")
                for section_title, filename in combined_sections:
                    fh.write(f"### {section_title}\n\n")
                    fh.write(f"![{section_title}]({filename})\n\n")

                fh.write("## Individual Trigger Rate Scans\n\n")
                for label, filename in individual_sections:
                    fh.write(f"### {label}\n\n")
                    fh.write(f"![{label}]({filename})\n\n")

                if skipped_configs:
                    fh.write("## Skipped Items\n\n")
                    for item in skipped_configs:
                        fh.write(f"- {item}\n")
                    fh.write("\n")

            # HTML must be built before the PDF: the PDF renderer pops figures
            # out of the stash as it embeds them.
            if generate_html:
                html_path = os.path.join(output_dir, html_name)
                try:
                    self._render_report_html(
                        html_path=html_path,
                        output_dir=output_dir,
                        title=title,
                        target_rate_hz=target_rate_hz,
                        config_descriptions=config_descriptions,
                        combined_sections=combined_sections,
                        individual_sections=individual_sections,
                        skipped_configs=skipped_configs,
                        vector_figs=self._report_vector_figs,
                    )
                except Exception as exc:  # pragma: no cover - HTML is best-effort
                    self._warn_once(
                        "report-html-failed",
                        f"Failed to render HTML report ({exc!r}); other formats still available.",
                    )

            if generate_pdf:
                pdf_path = os.path.join(output_dir, pdf_name)
                try:
                    self._render_report_pdf(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        title=title,
                        target_rate_hz=target_rate_hz,
                        config_descriptions=config_descriptions,
                        combined_sections=combined_sections,
                        individual_sections=individual_sections,
                        skipped_configs=skipped_configs,
                        vector_figs=self._report_vector_figs,
                    )
                    return pdf_path
                except Exception as exc:  # pragma: no cover - PDF is best-effort
                    self._warn_once(
                        "report-pdf-failed",
                        f"Failed to render PDF report ({exc!r}); markdown report still available.",
                    )

            return report_path
        finally:
            self._plot_init_kwargs = saved_plot_init_kwargs
            self._queued_plots = saved_queue
            # Drop any figures still held by the vector-plot stash so they don't
            # linger in memory after the report is built.
            stale_figs = getattr(self, "_report_vector_figs", None)
            if stale_figs:
                for _fig in stale_figs.values():
                    try:
                        plt.close(_fig)
                    except Exception:
                        pass
            self._report_vector_figs = None
            self._report_save_svg = False

    def generate_report(self, *args, **kwargs):
        return self.generateReport(*args, **kwargs)

    def _render_report_pdf(
        self,
        *,
        pdf_path: str,
        output_dir: str,
        title: str,
        target_rate_hz: Optional[float],
        config_descriptions: List[Tuple[ConfigType, str, str, str]],
        combined_sections: List[Tuple[str, str]],
        individual_sections: List[Tuple[str, str]],
        skipped_configs: List[str],
        vector_figs: Optional[Dict[str, "plt.Figure"]] = None,
    ) -> None:
        """Render a clean, self-contained PDF bundling the report metadata and plots.

        Plot pages are embedded as vector graphics whenever the live Matplotlib
        figure is available in ``vector_figs`` (keyed by absolute output path) —
        sharp at any zoom, like the score-quantizer transfer-function inset. When
        a figure is not stashed (vector mode off, or the plot was produced
        elsewhere), the page falls back to the rasterized PNG written by
        generateReport, so styling stays identical either way.
        """
        vector_figs = vector_figs or {}
        import datetime
        import textwrap
        from matplotlib.backends.backend_pdf import PdfPages

        # A4 landscape (inches) - plots are wide, so landscape reads best.
        page_w, page_h = 11.69, 8.27
        ink = "#2f2f2f"
        muted = "#6b6b6b"
        accent = "#2E91E5"
        paper = "#fbfbf8"

        def _new_page() -> "plt.Figure":
            fig = plt.figure(figsize=(page_w, page_h), facecolor=paper)
            return fig

        with PdfPages(pdf_path) as pdf:
            # ---------- Cover / metadata page ----------
            fig = _new_page()
            fig.text(0.06, 0.93, title, fontsize=26, fontweight="bold", color=ink, va="top")
            fig.text(
                0.06, 0.875,
                datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
                fontsize=11, color=muted, va="top",
            )
            fig.add_artist(
                plt.Line2D([0.06, 0.94], [0.855, 0.855], color=accent, linewidth=2.0, transform=fig.transFigure)
            )

            meta_lines = [
                ("Stat folder", str(self.stat_folder)),
                ("Base reference config", self._report_config_short_name(self.base_reference_config, index=-1)),
                ("Configurations", str(len(config_descriptions))),
            ]
            if target_rate_hz is not None:
                meta_lines.append(("Target NSB rate", f"{float(target_rate_hz):,.0f} Hz"))
            counts_line = self._format_counts_line()
            if counts_line:
                meta_lines.append(("Event counts", counts_line))

            y = 0.81
            for key, value in meta_lines:
                fig.text(0.06, y, f"{key}", fontsize=11, fontweight="bold", color=ink, va="top")
                fig.text(0.30, y, value, fontsize=11, color=ink, va="top")
                y -= 0.038

            # Configurations table
            y -= 0.02
            fig.text(0.06, y, "Configurations", fontsize=14, fontweight="bold", color=ink, va="top")
            y -= 0.015

            col_x = [0.06, 0.11, 0.42]
            headers = ["#", "Legend", "File"]
            header_y = y - 0.025
            for hx, htext in zip(col_x, headers):
                fig.text(hx, header_y, htext, fontsize=10, fontweight="bold", color=muted, va="top")
            fig.add_artist(
                plt.Line2D([0.06, 0.94], [header_y - 0.012, header_y - 0.012],
                           color="#d9d3c7", linewidth=1.0, transform=fig.transFigure)
            )

            row_y = header_y - 0.028
            row_step = 0.030
            for idx, (_cfg, label, slug, short_name) in enumerate(config_descriptions, start=1):
                if row_y < 0.06:
                    pdf.savefig(fig, facecolor=paper)
                    plt.close(fig)
                    fig = _new_page()
                    fig.text(0.06, 0.93, "Configurations (cont.)", fontsize=16, fontweight="bold", color=ink, va="top")
                    row_y = 0.88
                color = self._get_config_plot_color(_cfg)
                fname = os.path.basename(str((self.get_results(_cfg) or {}).get("_filename") or slug))
                fname_disp = fname if len(fname) <= 95 else fname[:94] + "…"
                fig.text(col_x[0], row_y, str(idx), fontsize=9, color=ink, va="top")
                fig.text(col_x[1], row_y, str(short_name)[:34], fontsize=8, fontweight="bold", color=color, va="top")
                fig.text(col_x[2], row_y, fname_disp, fontsize=7, color=muted, va="top", family="monospace")
                row_y -= row_step

            pdf.savefig(fig, facecolor=paper)
            plt.close(fig)

            # ---------- Trigger chain details (incl. tdscan weights) ----------
            details_title = "Trigger Chain Details"
            fig = _new_page()
            fig.text(0.06, 0.93, details_title, fontsize=18, fontweight="bold", color=ink, va="top")
            dy = 0.86
            legend_step = 0.030
            line_step = 0.022
            plot_w, plot_h = 0.28, 0.13

            def _new_details_page():
                nonlocal fig, dy
                pdf.savefig(fig, facecolor=paper)
                plt.close(fig)
                fig = _new_page()
                fig.text(0.06, 0.93, details_title + " (cont.)", fontsize=16,
                         fontweight="bold", color=ink, va="top")
                dy = 0.86

            for idx, (_cfg, _label, _slug, short_name) in enumerate(config_descriptions, start=1):
                # Resolve the stored chain to find a score_quantizer (for the inset).
                stored_chain = (self.get_results(_cfg) or {}).get("trigger_chain") or _cfg
                sq_edges = None
                for stage, raw in stored_chain:
                    if str(stage).lower() == "score_quantizer":
                        sq_edges = (self._as_mapping(raw) or {}).get("edges")
                        break

                # Keep text clear of the inset when one is drawn.
                wrap_width = 84 if sq_edges else 110
                render_lines = []
                for si, (stage, detail) in enumerate(self._report_chain_lines(_cfg), start=1):
                    wrapped = textwrap.wrap(f"Stage {si}: {stage}: {detail}", width=wrap_width) or [""]
                    for j, line in enumerate(wrapped):
                        render_lines.append(("         " if j else "  ") + line)

                text_height = legend_step + line_step * len(render_lines)
                block_height = max(text_height, (plot_h + 0.03) if sq_edges else 0.0) + 0.014

                # Keep each config block (and its inset) together on one page.
                if dy - block_height < 0.05:
                    _new_details_page()

                block_top = dy
                color = self._get_config_plot_color(_cfg)
                fig.text(0.06, dy, f"{idx}. {short_name}", fontsize=11, fontweight="bold",
                         color=color, va="top")
                dy -= legend_step
                for line in render_lines:
                    fig.text(0.07, dy, line, fontsize=8, color=ink, va="top", family="monospace")
                    dy -= line_step

                if sq_edges:
                    self._draw_score_quantizer_inset(
                        fig, sq_edges,
                        rect=[0.64, block_top - plot_h, plot_w, plot_h],
                        color=color, ink=ink, muted=muted,
                    )

                dy = block_top - block_height
            pdf.savefig(fig, facecolor=paper)
            plt.close(fig)

            # ---------- One page per plot ----------
            def _vector_page(section_title: str, fig: "plt.Figure") -> bool:
                """Embed a live Matplotlib figure as vector graphics (no rasterizing).

                Mirrors the score-quantizer inset path: the figure's artists are
                drawn straight onto the PDF page, so the plot stays sharp at any
                zoom. The figure's axes are nudged down into a reserved band (as
                the rasterized image page did with its [.., 0.88] sub-axes) so the
                section-title banner clears the plot's own title.
                """
                try:
                    fig.set_size_inches(page_w, page_h)
                    fig.set_facecolor(paper)
                    # Reserve a top band for the section banner by scaling every
                    # axes down in y (keeps main axes + colorbar aligned).
                    band = 0.10
                    scale = 1.0 - band
                    for ax in fig.axes:
                        p = ax.get_position()
                        ax.set_position([p.x0, p.y0 * scale, p.width, p.height * scale])
                    fig.suptitle(section_title, fontsize=16, fontweight="bold", color=ink,
                                 y=1.0 - band / 2.0)
                    pdf.savefig(fig, facecolor=paper)
                    return True
                except Exception:
                    return False
                finally:
                    plt.close(fig)

            def _image_page(section_title: str, filename: str) -> None:
                full_path = filename if os.path.isabs(filename) else os.path.join(output_dir, filename)
                # Prefer the stashed live figure (vector); fall back to the PNG.
                vec_fig = vector_figs.pop(os.path.abspath(full_path), None)
                if vec_fig is not None and _vector_page(section_title, vec_fig):
                    return
                if not os.path.exists(full_path):
                    return
                fig = _new_page()
                fig.suptitle(section_title, fontsize=16, fontweight="bold", color=ink, y=0.97)
                ax = fig.add_axes([0.03, 0.03, 0.94, 0.88])
                ax.axis("off")
                try:
                    img = plt.imread(full_path)
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, f"(could not load {os.path.basename(full_path)})",
                            ha="center", va="center", color=muted)
                pdf.savefig(fig, facecolor=paper)
                plt.close(fig)

            for section_title, filename in combined_sections:
                _image_page(section_title, filename)

            for label, filename in individual_sections:
                _image_page(f"Trigger Rate Scan — {label}", filename)

            # ---------- Skipped items ----------
            if skipped_configs:
                fig = _new_page()
                fig.text(0.06, 0.93, "Skipped Items", fontsize=18, fontweight="bold", color=ink, va="top")
                yy = 0.86
                for item in skipped_configs:
                    for line in textwrap.wrap(str(item), width=110) or [""]:
                        fig.text(0.06, yy, f"• {line}" if line else "", fontsize=10, color=ink, va="top")
                        yy -= 0.028
                        if yy < 0.06:
                            pdf.savefig(fig, facecolor=paper)
                            plt.close(fig)
                            fig = _new_page()
                            yy = 0.93
                pdf.savefig(fig, facecolor=paper)
                plt.close(fig)

            info = pdf.infodict()
            info["Title"] = title
            info["Creator"] = "StatPlotter.generateReport"
            info["CreationDate"] = datetime.datetime.now()

    @staticmethod
    def _figure_to_inline_svg(fig: "plt.Figure", uid: str, paper: str = "#fbfbf8") -> Optional[str]:
        """Serialize a Matplotlib figure to a standalone, inline-able SVG string.

        Every internal id (clipPaths, glyph defs, ...) is prefixed with ``uid``
        so many figures can share one HTML document without id collisions, and
        the outer <svg> is made responsive (width:100%, height auto) so the
        browser can scale it crisply — it stays vector, exactly like the PDF.
        """
        import io as _io
        import re as _re

        buf = _io.StringIO()
        try:
            fig.savefig(buf, format="svg", bbox_inches="tight", facecolor=paper)
        except Exception:
            return None
        svg = buf.getvalue()

        # Drop the XML/doctype preamble; keep from the first <svg ...> tag on.
        start = svg.find("<svg")
        if start == -1:
            return None
        svg = svg[start:]

        # Namespace every id and the references to it (url(#id), href="#id").
        ids = set(_re.findall(r'id="([^"]+)"', svg))
        for old in sorted(ids, key=len, reverse=True):
            new = f"{uid}-{old}"
            svg = svg.replace(f'id="{old}"', f'id="{new}"')
            svg = svg.replace(f'#{old}"', f'#{new}"')
            svg = svg.replace(f'url(#{old})', f'url(#{new})')

        # Make the root <svg> responsive: keep the viewBox (set by Matplotlib),
        # but let width/height be driven by the container/zoom transform.
        m = _re.search(r"<svg[^>]*>", svg)
        if m:
            head = m.group(0)
            vb = _re.search(r'viewBox="([^"]+)"', head)
            if vb is None:
                # Synthesize a viewBox from the pt width/height if needed.
                w = _re.search(r'width="([\d.]+)pt"', head)
                h = _re.search(r'height="([\d.]+)pt"', head)
                if w and h:
                    head2 = head[:-1] + f' viewBox="0 0 {w.group(1)} {h.group(1)}">'
                    svg = svg.replace(head, head2, 1)
                    head = head2
            # Strip fixed pt width/height so CSS controls the rendered size.
            head_new = _re.sub(r'\s(width|height)="[\d.]+pt"', "", head)
            head_new = head_new.replace("<svg", '<svg class="plot-svg" preserveAspectRatio="xMidYMid meet"', 1)
            svg = svg.replace(head, head_new, 1)
        return svg

    def _render_report_html(
        self,
        *,
        html_path: str,
        output_dir: str,
        title: str,
        target_rate_hz: Optional[float],
        config_descriptions: List[Tuple[ConfigType, str, str, str]],
        combined_sections: List[Tuple[str, str]],
        individual_sections: List[Tuple[str, str]],
        skipped_configs: List[str],
        vector_figs: Optional[Dict[str, "plt.Figure"]] = None,
    ) -> None:
        """Render a single self-contained, interactive HTML report.

        Each plot is inlined as a zoomable SVG (vector, pixel-for-pixel like the
        PDF — heatmap meshes included). A small dependency-free script adds
        scroll-to-zoom, drag-to-pan and double-click-to-reset per figure, plus a
        sticky table of contents. No server, no external assets, no JS libs.
        """
        import datetime
        import html as _html

        vector_figs = vector_figs or {}
        paper = "#fbfbf8"

        def esc(text: Any) -> str:
            return _html.escape(str(text))

        # ---- Assemble the same metadata the PDF cover shows. ----
        meta_lines: List[Tuple[str, str]] = [
            ("Stat folder", str(self.stat_folder)),
            ("Base reference config", self._report_config_short_name(self.base_reference_config, index=-1)),
            ("Configurations", str(len(config_descriptions))),
        ]
        if target_rate_hz is not None:
            meta_lines.append(("Target NSB rate", f"{float(target_rate_hz):,.0f} Hz"))
        counts_line = self._format_counts_line()
        if counts_line:
            meta_lines.append(("Event counts", counts_line))

        # ---- Collect the sections in the same order as the PDF. ----
        # Each entry: (anchor_id, nav_label, full_title, kind, payload)
        sections: List[Tuple[str, str, str, str, Any]] = []
        idx = 0

        def add_plot_section(full_title: str, filename: str) -> None:
            nonlocal idx
            full_path = filename if os.path.isabs(filename) else os.path.join(output_dir, filename)
            anchor = f"sec{idx}"
            fig = vector_figs.get(os.path.abspath(full_path))
            svg = self._figure_to_inline_svg(fig, anchor, paper=paper) if fig is not None else None
            if svg is None:
                # Fall back to the PNG (relative link) if no live figure exists.
                rel = os.path.relpath(full_path, output_dir)
                sections.append((anchor, full_title, full_title, "img", rel))
            else:
                sections.append((anchor, full_title, full_title, "svg", svg))
            idx += 1

        for section_title, filename in combined_sections:
            add_plot_section(section_title, filename)
        for label, filename in individual_sections:
            add_plot_section(f"Trigger Rate Scan — {label}", filename)

        # ---- Trigger-chain details (text + score-quantizer note). ----
        detail_rows: List[str] = []
        for cfg_i, (_cfg, _label, _slug, short_name) in enumerate(config_descriptions, start=1):
            color = self._get_config_plot_color(_cfg)
            lines_html = "".join(
                f'<div class="chain-line">Stage {si}: <b>{esc(stage)}</b>: {esc(detail)}</div>'
                for si, (stage, detail) in enumerate(self._report_chain_lines(_cfg), start=1)
            )
            detail_rows.append(
                f'<div class="cfg-block">'
                f'<div class="cfg-head" style="color:{esc(color)}">{cfg_i}. {esc(short_name)}</div>'
                f'{lines_html}</div>'
            )
        details_html = "".join(detail_rows)

        # ---- Build the navigation (table of contents). ----
        nav_items = ['<a href="#top">Overview</a>',
                     '<a href="#details">Trigger Chain Details</a>']
        nav_items += [f'<a href="#{anchor}">{esc(nav_label)}</a>'
                      for anchor, nav_label, _t, _k, _p in sections]
        if skipped_configs:
            nav_items.append('<a href="#skipped">Skipped Items</a>')
        nav_html = "\n".join(nav_items)

        # ---- Build the plot sections. ----
        body_parts: List[str] = []
        for anchor, _nav_label, full_title, kind, payload in sections:
            if kind == "svg":
                fig_html = (
                    f'<div class="plot-stage" tabindex="0">'
                    f'<div class="plot-pan">{payload}</div>'
                    f'<div class="plot-hint">scroll = zoom · drag = pan · double-click = reset</div>'
                    f'</div>'
                )
            else:  # img fallback
                fig_html = f'<div class="plot-stage"><img src="{esc(payload)}" alt="{esc(full_title)}"></div>'
            body_parts.append(
                f'<section id="{anchor}" class="plot-section">'
                f'<h2>{esc(full_title)}</h2>{fig_html}</section>'
            )
        sections_html = "\n".join(body_parts)

        meta_html = "".join(
            f'<div class="meta-row"><span class="meta-key">{esc(k)}</span>'
            f'<span class="meta-val">{esc(v)}</span></div>'
            for k, v in meta_lines
        )

        skipped_html = ""
        if skipped_configs:
            items = "".join(f"<li>{esc(it)}</li>" for it in skipped_configs)
            skipped_html = (
                f'<section id="skipped" class="plot-section">'
                f'<h2>Skipped Items</h2><ul class="skipped">{items}</ul></section>'
            )

        generated = datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M")

        html_doc = self._HTML_REPORT_TEMPLATE.format(
            title=esc(title),
            generated=esc(generated),
            nav=nav_html,
            meta=meta_html,
            details=details_html,
            sections=sections_html,
            skipped=skipped_html,
        )
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)

    # Self-contained HTML shell: layout + per-figure pan/zoom, no external assets.
    # Doubled braces survive str.format(); placeholders are {title}, {nav}, etc.
    _HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --paper:#fbfbf8; --ink:#2f2f2f; --muted:#6b6b6b; --accent:#2E91E5; --line:#e3ddd0; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  #layout {{ display:flex; align-items:flex-start; }}
  nav {{ position:sticky; top:0; align-self:flex-start; height:100vh; overflow-y:auto;
         width:280px; min-width:280px; padding:18px 14px; border-right:1px solid var(--line);
         background:#f4f1ea; }}
  nav .nav-title {{ font-weight:700; font-size:15px; margin-bottom:10px; }}
  nav a {{ display:block; padding:4px 8px; margin:1px 0; color:var(--ink); text-decoration:none;
           font-size:12.5px; border-radius:5px; line-height:1.3; }}
  nav a:hover {{ background:#e7e2d6; }}
  nav a.active {{ background:var(--accent); color:#fff; }}
  main {{ flex:1; min-width:0; padding:26px 34px 80px; max-width:1180px; margin:0 auto; }}
  header h1 {{ margin:0 0 4px; font-size:28px; }}
  header .sub {{ color:var(--muted); font-size:13px; margin-bottom:6px; }}
  .rule {{ height:2px; background:var(--accent); border:0; margin:10px 0 22px; }}
  .meta-grid {{ display:grid; grid-template-columns:max-content 1fr; gap:4px 18px; margin-bottom:14px; }}
  .meta-row {{ display:contents; }}
  .meta-key {{ font-weight:700; }}
  .meta-val {{ color:var(--ink); word-break:break-word; }}
  h2 {{ font-size:18px; margin:30px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  .cfg-block {{ margin:0 0 14px; }}
  .cfg-head {{ font-weight:700; font-size:14px; }}
  .chain-line {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
                 font-size:12px; color:var(--ink); margin-left:14px; }}
  .plot-section {{ scroll-margin-top:12px; }}
  .plot-stage {{ position:relative; border:1px solid var(--line); border-radius:8px;
                 background:var(--paper); overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
  .plot-pan {{ width:100%; transform-origin:0 0; cursor:grab; will-change:transform; }}
  .plot-pan:active {{ cursor:grabbing; }}
  .plot-svg {{ display:block; width:100%; height:auto; }}
  .plot-stage img {{ display:block; width:100%; height:auto; }}
  .plot-hint {{ position:absolute; top:8px; right:10px; font-size:11px; color:var(--muted);
                background:rgba(251,251,248,.85); padding:2px 8px; border-radius:10px;
                border:1px solid var(--line); pointer-events:none; opacity:0; transition:opacity .15s; }}
  .plot-stage:hover .plot-hint {{ opacity:1; }}
  ul.skipped {{ font-size:13px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:40px; }}
</style>
</head>
<body>
<div id="layout">
  <nav>
    <div class="nav-title">{title}</div>
    {nav}
  </nav>
  <main>
    <header id="top">
      <h1>{title}</h1>
      <div class="sub">{generated} · interactive · scroll to zoom, drag to pan, double-click to reset</div>
      <hr class="rule">
      <div class="meta-grid">{meta}</div>
    </header>
    <section id="details" class="plot-section">
      <h2>Trigger Chain Details</h2>
      {details}
    </section>
    {sections}
    {skipped}
    <footer>Generated by StatPlotter.generateReport — plots are inline SVG (vector, zoomable).</footer>
  </main>
</div>
<script>
(function() {{
  // Per-figure pan/zoom on the inline SVG. Pure DOM, no dependencies.
  document.querySelectorAll('.plot-stage').forEach(function(stage) {{
    var pan = stage.querySelector('.plot-pan');
    if (!pan) return;
    var scale = 1, tx = 0, ty = 0, dragging = false, sx = 0, sy = 0;
    var MIN = 1, MAX = 40;
    function apply() {{ pan.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')'; }}
    function reset() {{ scale = 1; tx = 0; ty = 0; apply(); }}
    stage.addEventListener('wheel', function(e) {{
      e.preventDefault();
      var rect = pan.getBoundingClientRect();
      var ox = e.clientX - rect.left, oy = e.clientY - rect.top;   // pointer in content space
      var factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      var ns = Math.min(MAX, Math.max(MIN, scale * factor));
      factor = ns / scale;
      // keep the point under the cursor fixed
      tx -= ox * (factor - 1);
      ty -= oy * (factor - 1);
      scale = ns;
      if (scale <= MIN + 1e-3) {{ reset(); }} else {{ apply(); }}
    }}, {{ passive: false }});
    stage.addEventListener('mousedown', function(e) {{
      if (e.button !== 0) return;
      dragging = true; sx = e.clientX - tx; sy = e.clientY - ty; e.preventDefault();
    }});
    window.addEventListener('mousemove', function(e) {{
      if (!dragging) return;
      tx = e.clientX - sx; ty = e.clientY - sy; apply();
    }});
    window.addEventListener('mouseup', function() {{ dragging = false; }});
    stage.addEventListener('dblclick', function(e) {{ e.preventDefault(); reset(); }});
  }});

  // Scroll-spy: highlight the nav link of the section in view.
  var links = Array.prototype.slice.call(document.querySelectorAll('nav a'));
  var byId = {{}};
  links.forEach(function(a) {{ byId[a.getAttribute('href').slice(1)] = a; }});
  var targets = Object.keys(byId)
    .map(function(id) {{ return document.getElementById(id); }})
    .filter(Boolean);
  if ('IntersectionObserver' in window && targets.length) {{
    var obs = new IntersectionObserver(function(entries) {{
      entries.forEach(function(en) {{
        if (en.isIntersecting) {{
          links.forEach(function(a) {{ a.classList.remove('active'); }});
          var a = byId[en.target.id];
          if (a) a.classList.add('active');
        }}
      }});
    }}, {{ rootMargin: '-10% 0px -80% 0px', threshold: 0 }});
    targets.forEach(function(t) {{ obs.observe(t); }});
  }}
}})();
</script>
</body>
</html>
"""

    def initPlotAbsolute(self, title: str = "", metrics: str = "n_pe", mode: str = "efficiency",
                        x_range=None, y_range=None, bins: int = 50,
                        target_rate_hz: Optional[float] = None,
                        score_threshold: Optional[float] = None):
        self._reset_custom_legend_state()
        self.x_range = x_range
        self.metrics = self._normalize_metric_name(metrics)
        self.nbins = bins
        self.mode = mode
        self.target_rate_hz = target_rate_hz
        self.score_threshold = score_threshold

        plt.figure(figsize=(17, 7), constrained_layout=True)
        counts_line = self._format_counts_line()
        base_title = self._generate_title_config_text(
            self.base_reference_config,
            result=self.base_config_result,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
            include_threshold=True,
        )
        if mode == "efficiency":
            title_parts = [title, "Efficiency Absolute", counts_line]
        else:
            title_parts = [title, f"Metric: {mode}  Base: {base_title}", counts_line]
        plt.title("\n".join([p for p in title_parts if p]))

        # base trigger rate
        if isinstance(self.base_config_result.get("trigger_rate"), tuple):
            self.base_ref_trigger_rate = self.base_config_result["trigger_rate"][0]
            self.base_ref_trigger_rate_std = self.base_config_result["trigger_rate"][1]
        else:
            self.base_ref_trigger_rate = self.base_config_result.get("trigger_rate", 0.0)
            self.base_ref_trigger_rate_std = -1.0

        plt.xlabel(self._metric_xlabel(self.metrics))
        plt.ylabel("Efficiency")
        plt.grid()
        # plt.xscale("log")
        # add more gride lines for x and and more labels for easy reading
        plt.xscale("log")
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

        if x_range is not None:
            plt.xlim(left=max(x_range[0], 1e-3), right=x_range[1])
        if y_range is not None:
            plt.ylim(y_range)

    def initPlotRatio(self, title: str = "", metrics: str = "n_pe",
                      x_range=None, y_range=None, nbins: int = 50,
                      target_rate_hz: Optional[float] = None,
                      score_threshold: Optional[float] = None):
        self._reset_custom_legend_state()
        self.metrics = self._normalize_metric_name(metrics)
        self.x_range = x_range
        self.nbins = nbins
        self.target_rate_hz = target_rate_hz
        self.score_threshold = score_threshold

        plt.figure(figsize=(17, 7), constrained_layout=True)
        counts_line = self._format_counts_line()
        base_title = self._generate_title_config_text(
            self.base_reference_config,
            result=self.base_config_result,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
            include_threshold=True,
        )
        title_parts = [title, f"Efficiency Ratio  Base: {base_title}", counts_line]
        plt.title("\n".join([p for p in title_parts if p]))

        if isinstance(self.base_config_result.get("trigger_rate"), tuple):
            self.base_ref_trigger_rate = self.base_config_result["trigger_rate"][0]
            self.base_ref_trigger_rate_std = self.base_config_result["trigger_rate"][1]
        else:
            self.base_ref_trigger_rate = self.base_config_result.get("trigger_rate", 0.0)
            self.base_ref_trigger_rate_std = -1.0

        plt.xlabel(self._metric_xlabel(self.metrics))
        plt.ylabel("Efficiency Ratio")
        plt.axhline(1.0, color="gray", linestyle="--")
        plt.grid(True, which="both")
        plt.xscale("log")
        if x_range is not None:
            plt.xlim(left=max(x_range[0], 1e-3), right=x_range[1])
        if y_range is not None:
            plt.ylim(y_range)

    def _config_roc_auc(self, result):
        """Gamma-vs-NSB ROC AUC (Mann-Whitney) of a config's pre-threshold score.

        This is the same quantity the training loss optimizes and the live
        simulation prints: P(score_gamma > score_nsb), threshold- and
        scale-independent. Cached per file. Returns NaN if scores are missing.
        """
        if result is None or not result.get("has_pre_threshold_score", False):
            return float("nan")
        key = result.get("_path")
        if key in self._roc_auc_cache:
            return self._roc_auc_cache[key]
        gamma = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="gamma")
        nsb = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
        auc = roc_auc_mann_whitney(gamma, nsb)
        if key is not None:
            self._roc_auc_cache[key] = auc
        return auc

    def addPlotAbsolute(
        self,
        to_compare_config: ConfigType,
        label: Optional[str] = None,
        mul_trig_rate: float = 1.0,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        to_compare_result = self.get_results(to_compare_config)
        if to_compare_result is None:
            print("Configuration to compare not found.")
            return

        # only efficiency mode kept here (precision removed from focus)
        stats = self._compute_efficiency_absolute_stats(
            to_compare_result,
            target_rate_hz=self.target_rate_hz if target_rate_hz is None else target_rate_hz,
            score_threshold=self.score_threshold if score_threshold is None else score_threshold,
        )
        bin_centers = stats["bin_centers"]
        binning = stats["binning"]
        eff = stats["efficiency"]
        eff_err = np.vstack([stats["err_low"], stats["err_high"]])
        all_h = stats["all_h"]
        trigger_strategy = stats["trigger_strategy"]

        auc = self._config_roc_auc(to_compare_result)
        auc_txt = f"AUC={auc * 100.0:.1f}%" if np.isfinite(auc) else "AUC=NA"

        rate = float(trigger_strategy.get("trigger_rate_hz", 0.0))
        if label is None:
            label = self.generate_label_text(
                to_compare_config,
                rate * mul_trig_rate,
                threshold_override=trigger_strategy.get("reference_threshold"),
            )
        else:
            label = f"{label}; {self._rate_suffix(rate)}"

        label = f"{label}; {auc_txt}"
        curve_color = self._get_config_plot_color(to_compare_config)
        plt.errorbar(
            bin_centers,
            eff,
            xerr=[bin_centers - binning[:-1], binning[1:] - bin_centers],
            yerr=eff_err,
            fmt="o",
            capsize=3,
            label=label,
            color=curve_color,
        )

    def addPlotRatio(
        self,
        to_compare_config: ConfigType,
        label: Optional[str] = None,
        mul_trig_rate: float = 1000.0,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        to_compare_result = self.get_results(to_compare_config)
        if to_compare_result is None:
            print("Configuration to compare not found.")
            return

        compare_target_rate_hz = self.target_rate_hz if target_rate_hz is None else target_rate_hz
        compare_score_threshold = self.score_threshold if score_threshold is None else score_threshold
        ratio_model = self.compute_efficiency_ratio(
            to_compare_result,
            base_target_rate_hz=self.target_rate_hz,
            base_score_threshold=self.score_threshold,
            compare_target_rate_hz=compare_target_rate_hz,
            compare_score_threshold=compare_score_threshold,
        )
        bin_centers = np.asarray(ratio_model["bin_centers"], dtype=np.float64)
        binning = np.asarray(ratio_model["binning"], dtype=np.float64)
        ratio = np.asarray(ratio_model["efficiency"], dtype=np.float64)
        ratio_err = np.vstack([
            np.asarray(ratio_model["err_low"], dtype=np.float64),
            np.asarray(ratio_model["err_high"], dtype=np.float64),
        ])
        xerr = np.vstack([
            np.asarray(bin_centers - binning[:-1], dtype=np.float64),
            np.asarray(binning[1:] - bin_centers, dtype=np.float64),
        ])

        strategy = self._resolve_trigger_strategy(
            to_compare_result,
            target_rate_hz=compare_target_rate_hz,
            score_threshold=compare_score_threshold,
        )
        rate = float(strategy.get("trigger_rate_hz", 0.0))
        if label is None:
            label = self.generate_label_text(
                to_compare_config,
                rate * mul_trig_rate,
                threshold_override=strategy.get("reference_threshold"),
            )
        else:
            label = f"{label}; {self._rate_suffix(rate)}"

        curve_color = self._get_config_plot_color(to_compare_config)
        plt.errorbar(
            bin_centers,
            ratio,
            xerr=xerr,
            yerr=ratio_err,
            fmt="o",
            capsize=3,
            label=label,
            color=curve_color,
        )

    def addPlotRatioPerfect(self, label: str = "Perfect Efficiency"):
        if self.x_range is None:
            lo, hi = self._get_default_range(self.base_config_result, self.metrics, kind="gamma")
        else:
            lo, hi = self.x_range
        bins = self._make_log_bins(lo, hi, self.nbins)
        centers = (bins[:-1] + bins[1:]) / 2.0
        ratio = np.ones_like(centers)
        plt.plot(centers, ratio, marker="o", label=label)

    def addPlotAbsolutePerfect(self, label: str = "Perfect Efficiency"):
        if self.x_range is None:
            lo, hi = self._get_default_range(self.base_config_result, self.metrics, kind="gamma")
        else:
            lo, hi = self.x_range
        bins = self._make_log_bins(lo, hi, self.nbins)
        centers = (bins[:-1] + bins[1:]) / 2.0
        eff = np.ones_like(centers)
        eff_err = np.zeros_like(centers)
        plt.errorbar(centers, eff, yerr=eff_err, fmt="-o", capsize=3, label=label)

    def _generate_config_label_text(
        self,
        to_compare_config: ConfigType,
        threshold_override: Optional[float] = None,
        include_threshold: bool = True,
    ) -> str:
        label = ""
        saw_threshold_stage = False
        threshold_stage_indices = [
            idx for idx, (stage_type, _raw_params) in enumerate(to_compare_config)
            if str(stage_type).lower() == "threshold"
        ]
        last_threshold_stage_index = threshold_stage_indices[-1] if threshold_stage_indices else None

        for idx, (stage_type, raw_params) in enumerate(to_compare_config):
            params = self._as_mapping(raw_params) or {}
            st = str(stage_type).lower()
            if st == "tdscan":
                eps_xy = params.get("eps_xy", params.get("epx_xy"))
                eps_t = params.get("eps_t")
                tdscan_id_short = self._format_hash_short(params.get("id"))
                pad_value = params.get("pad_value")
                quantize = params.get("quantize")
                quantize_txt = f" qz={quantize}" if quantize is not None else ""
                quantize_step_txt = self._format_quantize_step_label(params.get("quantize_step"))
                id_txt = f" h{tdscan_id_short}" if tdscan_id_short is not None else ""
                pad_txt = f" p={self._format_label_value(pad_value)}" if pad_value is not None else ""
                label += f"td e{eps_xy}/{eps_t}{id_txt}{pad_txt}{quantize_txt}{quantize_step_txt}; "
            elif st == "dbscan":
                label += f"db eps={params.get('eps'):.2f} min={params.get('min_points')} t={params.get('time_norm'):.2f}; "
            elif st == "baselinesubstractor" or st == "fadc":
                label += "fadc; "
            elif st == "digital_sum":
                # label += f"DS{params.get('threshold_flower')} {params.get('mode')}; "
                # sometime threhsold is missing in params
                if "threshold_flower" in params:
                    threshold_flower = self._format_label_value(params.get("threshold_flower"))
                    label += f"ds {params.get('mode','')} {threshold_flower}; "
                else:
                    label += f"ds {params.get('mode','')}; "
            elif st == "shift":
                shift_value = params.get("value")
                quantize_step_txt = self._format_quantize_step_label(params.get("quantize_step"))
                label += f"sub {self._format_label_value(shift_value)}{quantize_step_txt}; "
            elif st == "score_quantizer":
                edges = params.get("edges") or []
                label += f"qscore {len(edges) + 1}lvl; "
            elif st == "threshold":
                saw_threshold_stage = True
                if include_threshold:
                    if threshold_override is not None and idx == last_threshold_stage_index:
                        threshold_value = threshold_override
                    else:
                        threshold_value = params.get("threshold")
                    comparison = self._normalize_threshold_comparison(params.get("comparison", "gt"))
                    comparison_txt = ">=" if comparison == "ge" else ">"
                    threshold_mode = " bin" if params.get("binary") or params.get("binary_output") else ""
                    label += f"thr {comparison_txt} {self._format_label_value(threshold_value)}{threshold_mode}; "
            elif st == "cnn":
                label += f"cnn {params.get('name','')}; "
            else:
                label += f"{stage_type}; "

        if include_threshold and threshold_override is not None and not saw_threshold_stage:
            label += f"thr {self._format_label_value(threshold_override)}; "
        return label.strip("; ")

    @staticmethod
    def _rate_suffix(rate_hz) -> str:
        """Format the realized NSB trigger rate to append next to a custom label."""
        try:
            return f"rate={float(rate_hz):.1f}Hz"
        except (TypeError, ValueError):
            return ""

    def generate_label_text(
        self,
        to_compare_config: ConfigType,
        to_compare_trigger_rate: Union[float, Tuple[float, float]],
        threshold_override: Optional[float] = None,
    ) -> str:
        if isinstance(to_compare_trigger_rate, tuple):
            to_compare_trigger_rate = to_compare_trigger_rate[0]

        trigger_rate = float(to_compare_trigger_rate)
        prefix = self._generate_config_label_text(
            to_compare_config,
            threshold_override=threshold_override,
            include_threshold=True,
        )
        if prefix:
            return f"{prefix}; rate={trigger_rate:.1f}Hz"
        return f"rate={trigger_rate:.1f}Hz"

    def _generate_title_config_text(
        self,
        to_compare_config: ConfigType,
        result: Optional[Dict[str, Any]] = None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
        include_threshold: bool = True,
    ) -> str:
        if result is None:
            result = self.get_results(to_compare_config)
        threshold_override = None
        if result is not None:
            try:
                strategy = self._resolve_trigger_strategy(
                    result,
                    target_rate_hz=target_rate_hz,
                    score_threshold=score_threshold,
                )
                threshold_override = strategy.get("reference_threshold")
            except Exception:
                threshold_override = None
        return self._generate_config_label_text(
            to_compare_config,
            threshold_override=threshold_override,
            include_threshold=include_threshold,
        )

    def add_all_plots(self, metrics: str = "n_pe", x_range: Tuple[float, float] = (1, 250)):
        for _, result in self.all_results:
            cfg = result.get("trigger_chain")
            if cfg:
                self.addPlotRatio(to_compare_config=cfg, label=None)

    def resetPlot(self):
        plt.clf()

    def get_event_to_compare(self, config_trig: ConfigType, config_notrig: ConfigType, n_pe_range: Tuple[float, float] = (50, 150)):
        raise NotImplementedError("get_event_to_compare required legacy PKL stats; PKL support has been removed.")

    # sanitycheck funciton
    # will check if all configs contains the same number of events simulated and the same distribution of n_pe
    def sanity_check_configs(self, configs: List[ConfigType], n_pe_bins: int = 50):
        n_events_list = []
        n_pe_hist_list = []
        if not configs:
            return True

        ref_result = self.get_results(configs[0])
        if ref_result is None:
            raise ValueError("Reference configuration not found.")

        # Build common bins from reference (HDF5 streaming)
        lo, hi = self._get_default_range(ref_result, "n_pe", kind="gamma")

        bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi + 1e-3), n_pe_bins)

        def _hist_and_count(res: Dict[str, Any]) -> Tuple[int, np.ndarray]:
            all_h, _ = self._histogram_stream(res, "n_pe", kind="gamma", bins=bins)
            return int(all_h.sum()), all_h.astype(np.float64)

        ok = True
        n_ref, hist_ref = _hist_and_count(ref_result)
        ref_norm = hist_ref / hist_ref.sum() if hist_ref.sum() > 0 else hist_ref

        for cfg in configs[1:]:
            res = self.get_results(cfg)
            if res is None:
                print(f"Config not found: {cfg}")
                ok = False
                continue

            n, hist = _hist_and_count(res)
            n_events_list.append(n)
            n_pe_hist_list.append(hist)

            if n != n_ref:
                print(f"Different number of events for {cfg}: {n} vs {n_ref}")
                ok = False

            norm = hist / hist.sum() if hist.sum() > 0 else hist
            if not np.allclose(norm, ref_norm, rtol=5e-2, atol=1e-6):
                print(f"Different n_pe distribution for {cfg}")
                ok = False

        return ok


    def showPlot(
        self,
        filename: Optional[str] = None,
        show: bool = True,
        location: str = "lower right",
        plot_type: Optional[str] = None,
        **plot_kwargs,
    ):
        """
        Finalize + save/show the plot.

        - Legacy behavior: if plot_type is None, assumes you already called initPlotXXX/addPlotXXX.
        - New behavior: if plot_type is provided, it will render the queued configs (from init_plot/add_plot)
          as the requested plot_type before saving/showing.

        Examples:
            plotter.init_plot(title="Absolute Efficiency", metrics="energy", x_range=(0.2, 50), bins=50)
            plotter.add_plot(cfg1)
            plotter.add_plot(cfg2)
            plotter.showPlot(plot_type="absolute", filename="abs.png")

            plotter.showPlot(plot_type="effective_area", filename="aeff.png", emin_tev=0.01, emax_tev=50, nbins=25)
            plotter.showPlot(plot_type="effective_area_counts", filename="aeff_counts.png", emin_tev=0.01, emax_tev=50, nbins=25)
        """
        # apply the sanity check before plotting
        if not self.sanity_check_configs([item["config"] for item in getattr(self, "_queued_plots", [])]):
            print("Sanity check failed: configurations have different number of events or n_pe distributions.")
            return
        else:
            print("Sanity check passed: all configurations have consistent number of events and n_pe distributions.")
        if plot_type is not None:
            # Render from the queue (init_plot/add_plot) into a fresh figure
            self._render_queued_plot(plot_type, **(plot_kwargs or {}))

        fig = plt.gcf()
        axes = fig.get_axes()
        if not axes:
            axes = [plt.gca()]

        legend_layout_mode = None
        if getattr(self, "_use_custom_eff_rate_legend", False):
            for ax in axes:
                try:
                    self._apply_efficiency_vs_rate_legends(ax=ax)
                except Exception:
                    pass
        else:
            # Apply legend on every axis that actually has labeled artists
            for ax in axes:
                try:
                    _, labels = ax.get_legend_handles_labels()
                    if labels:
                        current_mode = self._apply_standard_legend(ax=ax, location=location)
                        if current_mode:
                            legend_layout_mode = current_mode
                except Exception:
                    pass

        # Only call tight_layout if constrained_layout is not enabled
        try:
            if not (getattr(fig, "get_constrained_layout", lambda: False)()):
                if getattr(self, "_use_custom_eff_rate_legend", False):
                    fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
                elif legend_layout_mode == "outside_right":
                    width, height = fig.get_size_inches()
                    if width < 16.0:
                        fig.set_size_inches(16.0, height, forward=True)
                    fig.subplots_adjust(left=0.10, right=0.55, bottom=0.12, top=0.90)
                elif legend_layout_mode == "outside_bottom":
                    width, height = fig.get_size_inches()
                    if height < 8.5:
                        fig.set_size_inches(width, 8.5, forward=True)
                    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.42, top=0.90)
                else:
                    fig.tight_layout()
        except Exception:
            pass

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
            
    def plotNpeDistribution(
        self,
        range: Tuple[float, float] = (0, 2000),
        to_compare_config: Optional[ConfigType] = None,
        bins: int = 100,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        if to_compare_config is None:
            r = self.base_config_result
            title_config = self.base_reference_config
        else:
            r = self.get_results(to_compare_config)
            title_config = to_compare_config
        if r is None:
            print("Config not found")
            return

        edges = np.linspace(range[0], range[1], bins)

        plt.figure(figsize=(10, 6))
        plt.title(f"Npe Distribution - Config: {title_config}")
        plt.xlabel("Npe")
        plt.ylabel("Number of Events")
        plt.grid()
        plt.yscale("log")

        # stream gamma only
        strategy = self._resolve_trigger_strategy(r, target_rate_hz=target_rate_hz, score_threshold=score_threshold)
        all_h, trig_h = self._histogram_stream(
            r,
            "n_pe",
            kind="gamma",
            bins=edges,
            score_threshold=strategy.get("score_threshold"),
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        plt.step(centers, all_h, where="mid", label="All Events")
        plt.step(centers, trig_h, where="mid", label="Triggered Events")

        plt.legend()
        plt.show()

    def plotNpeDistributionBase(self, bins: int = 100, x_range: Optional[Tuple[float, float]] = None):
        """
        Convenience helper to quickly inspect the n_pe distribution of the base configuration (all events only).
        """
        r = self.base_config_result
        if r is None:
            print("Base configuration not found.")
            return

        if x_range is None:
            lo, hi = self._get_default_range(r, "n_pe", kind="gamma")
        else:
            lo, hi = x_range

        edges = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi + 1e-3), bins)

        plt.figure(figsize=(10, 6))
        plt.title(f"Npe Distribution - Base Config: {self.base_reference_config}")
        plt.xlabel("Npe")
        plt.ylabel("Number of Events")
        plt.grid()
        plt.xscale("log")
        plt.yscale("log")

        all_h, _ = self._histogram_stream(r, "n_pe", kind="gamma", bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        plt.step(centers, all_h, where="mid", label="All Events (base)")

        plt.legend()
        plt.show()

    def plotEnergyDistribution(
        self,
        to_compare_config: Optional[ConfigType] = None,
        bins: int = 50,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        if to_compare_config is None:
            r = self.base_config_result
            title_config = self.base_reference_config
        else:
            r = self.get_results(to_compare_config)
            title_config = to_compare_config
        if r is None:
            print("Config not found")
            return

        edges = np.logspace(np.log10(0.005), np.log10(50), bins)

        plt.figure(figsize=(10, 6))
        plt.title(f"Energy Distribution - Config: {title_config}")
        plt.xlabel("Energy (TeV)")
        plt.ylabel("Number of Events")
        plt.grid()
        plt.xscale("log")
        plt.yscale("log")

        strategy = self._resolve_trigger_strategy(r, target_rate_hz=target_rate_hz, score_threshold=score_threshold)
        all_h, trig_h = self._histogram_stream(
            r,
            "energy",
            kind="gamma",
            bins=edges,
            score_threshold=strategy.get("score_threshold"),
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        plt.step(centers, all_h, where="mid", label="All Events")
        plt.step(centers, trig_h, where="mid", label="Triggered Events")

        plt.legend()
        plt.show()

    def plotEnergyDistributionBase(self, bins: int = 50, x_range: Optional[Tuple[float, float]] = None):
        """
        Convenience helper to inspect the energy distribution of the base configuration (all events only).
        """
        r = self.base_config_result
        if r is None:
            print("Base configuration not found.")
            return

        if x_range is None:
            lo, hi = self._get_default_range(r, "energy", kind="gamma")
        else:
            lo, hi = x_range

        edges = np.logspace(np.log10(max(lo, 1e-4)), np.log10(hi + 1e-4), bins)

        plt.figure(figsize=(10, 6))
        plt.title(f"Energy Distribution - Base Config: {self.base_reference_config}")
        plt.xlabel("Energy (TeV)")
        plt.ylabel("Number of Events")
        plt.grid()
        plt.xscale("log")
        plt.yscale("log")

        all_h, _ = self._histogram_stream(r, "energy", kind="gamma", bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        plt.step(centers, all_h, where="mid", label="All Events (base)")

        plt.legend()
        plt.show()

    def find_score_threshold_for_target_rate(
        self,
        target_rate_hz: float,
        to_compare_config: Optional[ConfigType] = None,
    ) -> Tuple[Optional[float], float]:
        if to_compare_config is None:
            result = self.base_config_result
        else:
            result = self.get_results(to_compare_config)
        if result is None:
            raise ValueError("Configuration not found.")

        strategy = self._resolve_trigger_strategy(result, target_rate_hz=target_rate_hz)
        return strategy.get("score_threshold"), float(strategy.get("trigger_rate_hz", float("nan")))

    @classmethod
    def _build_trigger_rate_curve_from_scores(
        cls,
        scores: np.ndarray,
        window_sec: float,
        comparison: Optional[str] = "gt",
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float64)

        comparison = cls._normalize_threshold_comparison(comparison)
        unique_scores, counts = np.unique(scores, return_counts=True)
        counts = counts.astype(np.int64)
        total = int(scores.size)

        cumulative = np.cumsum(counts, dtype=np.int64)
        counts_gt = total - cumulative
        counts_ge = counts_gt + counts

        if comparison == "ge":
            tau_strict = np.nextafter(unique_scores.astype(np.float32), np.float32(np.inf))
            tau_include_ties = unique_scores.astype(np.float32)
        else:
            tau_strict = unique_scores.astype(np.float32)
            tau_include_ties = np.nextafter(unique_scores.astype(np.float32), np.float32(-np.inf))

        thresholds = np.concatenate([tau_include_ties, tau_strict])
        rates_hz = np.concatenate([
            counts_ge.astype(np.float64) / total / float(window_sec),
            counts_gt.astype(np.float64) / total / float(window_sec),
        ])

        order = np.argsort(thresholds, kind="mergesort")
        thresholds = thresholds[order]
        rates_hz = rates_hz[order]
        return thresholds, rates_hz

    def _get_efficiency_vs_rate_metric_bins(
        self,
        result: Dict[str, Any],
        metric: str,
        metric_bins: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        if metric_bins is not None:
            edges = np.asarray(list(metric_bins), dtype=np.float64).reshape(-1)
        elif metric == "energy":
            edges = np.asarray([0.1, 0.3, 1.0, 3.0, 10.0], dtype=np.float64)
        elif metric == "n_pe":
            edges = np.asarray([20.0, 50.0, 100.0, 200.0, 400.0], dtype=np.float64)
        else:
            lo, hi = self._get_default_range(result, metric, kind="gamma")
            edges = self._make_log_bins(lo, hi, 5).astype(np.float64)

        edges = edges[np.isfinite(edges)]
        edges = np.unique(edges)
        if edges.size < 2:
            raise ValueError("metric_bins must contain at least two finite values.")
        return edges

    @staticmethod
    def _format_efficiency_vs_rate_bin_label(metric: str, lo: float, hi: float) -> str:
        if metric == "energy":
            return f"E in [{lo:g}, {hi:g}] TeV"
        if metric == "n_pe":
            return f"n_pe in [{lo:g}, {hi:g}]"
        return f"{metric} in [{lo:g}, {hi:g}]"

    def getEfficiencyVsTriggerRateCurves(
        self,
        to_compare_config: Optional[ConfigType] = None,
        metric: str = "energy",
        metric_bins: Optional[Iterable[float]] = None,
        max_points: Optional[int] = 4_000,
    ) -> Dict[str, Any]:
        if to_compare_config is None:
            result = self.base_config_result
        else:
            result = self.get_results(to_compare_config)
        if result is None:
            raise ValueError("Configuration not found.")
        if not result.get("has_pre_threshold_score", False):
            raise ValueError(f"Configuration has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset.")

        metric = self._normalize_metric_name(metric)
        gamma_scores = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="gamma")
        thresholds, rates_hz = self.get_trigger_rate_curve(to_compare_config=to_compare_config)

        if max_points is not None and thresholds.size > max_points:
            idx = np.linspace(0, thresholds.size - 1, int(max_points), dtype=int)
            idx = np.unique(idx)
            thresholds = thresholds[idx]
            rates_hz = rates_hz[idx]

        total = int(gamma_scores.size)
        efficiency = np.full(thresholds.shape, np.nan, dtype=np.float64)
        if total > 0:
            gamma_scores = np.sort(gamma_scores.astype(np.float32, copy=False))
            passed = total - np.searchsorted(gamma_scores, thresholds, side="right")
            efficiency = passed.astype(np.float64) / float(total)

        curves: List[Dict[str, Any]] = [{
            "bin_range": None,
            "bin_label": "All gamma events",
            "total": total,
            "efficiency": efficiency,
        }]

        return {
            "metric": metric,
            "bin_edges": None,
            "thresholds": thresholds,
            "rates_hz": rates_hz,
            "curves": curves,
            "total": total,
            "efficiency": efficiency,
        }

    def get_efficiency_vs_trigger_rate_curves(self, *args, **kwargs):
        return self.getEfficiencyVsTriggerRateCurves(*args, **kwargs)

    def _get_result_trigger_rate_curve(self, result: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if result is None:
            raise ValueError("Configuration not found.")
        if not result.get("has_pre_threshold_score", False):
            raise ValueError(f"Configuration has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset.")

        nsb_scores = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
        return self._build_trigger_rate_curve_from_scores(
            nsb_scores,
            window_sec=float(result.get("window_sec", 75e-9)),
            comparison=result.get("threshold_comparison", "gt"),
        )

    def get_trigger_rate_curve(
        self,
        to_compare_config: Optional[ConfigType] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if to_compare_config is None:
            result = self.base_config_result
        else:
            result = self.get_results(to_compare_config)
        return self._get_result_trigger_rate_curve(result)

    def _reset_custom_legend_state(self):
        self._use_custom_eff_rate_legend = False
        self._eff_rate_config_styles = {}
        self._eff_rate_config_handles = []
        self._eff_rate_bin_colors = {}
        self._eff_rate_bin_handles = []
        self._eff_rate_target_handle = None
        self._active_plot_colors = {}

    def _apply_efficiency_vs_rate_legends(self, ax=None):
        if not getattr(self, "_use_custom_eff_rate_legend", False):
            return

        if ax is None:
            ax = plt.gca()

        fig = ax.figure
        try:
            fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
        except Exception:
            pass

        config_handles = list(getattr(self, "_eff_rate_config_handles", []) or [])
        bin_handles = list(getattr(self, "_eff_rate_bin_handles", []) or [])
        target_handle = getattr(self, "_eff_rate_target_handle", None)
        if target_handle is not None:
            config_handles = config_handles + [target_handle]

        if config_handles:
            config_legend = ax.legend(
                handles=config_handles,
                title="Configs",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
                fontsize="small",
                title_fontsize="small",
            )
            ax.add_artist(config_legend)

        if bin_handles:
            bin_title = "Energy Bins" if getattr(self, "metrics", "energy") == "energy" else f"{self.metrics} bins"
            ax.legend(
                handles=bin_handles,
                title=bin_title,
                loc="lower left",
                bbox_to_anchor=(1.02, 0.0),
                borderaxespad=0.0,
                frameon=True,
                fontsize="small",
                title_fontsize="small",
            )

    def initPlotEfficiencyVsTriggerRate(
        self,
        title: str = "",
        metric: str = "energy",
        metric_bins: Optional[Iterable[float]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        target_rate_hz: Optional[float] = None,
        max_points: Optional[int] = 4_000,
    ):
        self._reset_custom_legend_state()
        self.metrics = self._normalize_metric_name(metric)
        self.target_rate_hz = target_rate_hz
        self._efficiency_vs_rate_metric_bins = metric_bins
        self._efficiency_vs_rate_max_points = max_points

        plt.figure(figsize=(11, 6))
        counts_line = self._format_counts_line()
        plot_title = title or "Gamma Efficiency vs NSB Trigger Rate"
        plt.title("\n".join([p for p in (plot_title, counts_line) if p]))
        plt.xlabel("NSB Trigger Rate (Hz)")
        plt.ylabel("Gamma Efficiency")
        plt.grid(True, which="both", alpha=0.3)
        plt.xscale("log")
        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)
        else:
            plt.ylim(0.0, 1.05)

    def addPlotEfficiencyVsTriggerRate(
        self,
        to_compare_config: ConfigType,
        label: Optional[str] = None,
        metric: str = "energy",
        metric_bins: Optional[Iterable[float]] = None,
        target_rate_hz: Optional[float] = None,
        max_points: Optional[int] = 4_000,
        draw_target_rate_line: bool = True,
    ):
        result = self.get_results(to_compare_config)
        if result is None:
            print("Config not found")
            return
        if not result.get("has_pre_threshold_score", False):
            print(f"Config {to_compare_config} has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset.")
            return

        resolved_metric = self._normalize_metric_name(metric if metric is not None else getattr(self, "metrics", "energy"))
        resolved_metric_bins = metric_bins if metric_bins is not None else getattr(self, "_efficiency_vs_rate_metric_bins", None)
        resolved_max_points = max_points if max_points is not None else getattr(self, "_efficiency_vs_rate_max_points", 4_000)
        resolved_target_rate_hz = self.target_rate_hz if target_rate_hz is None else target_rate_hz

        curve_data = self.getEfficiencyVsTriggerRateCurves(
            to_compare_config=to_compare_config,
            metric=resolved_metric,
            metric_bins=resolved_metric_bins,
            max_points=resolved_max_points,
        )
        strategy = self._resolve_trigger_strategy(result, target_rate_hz=resolved_target_rate_hz)
        target_threshold = strategy.get("score_threshold")
        marker_rate_hz = float(strategy.get("trigger_rate_hz", float("nan")))

        curve_label = label
        if curve_label is None:
            curve_label = self._generate_config_label_text(
                to_compare_config,
                include_threshold=False,
            )
        elif np.isfinite(marker_rate_hz):
            curve_label = f"{curve_label}; {self._rate_suffix(marker_rate_hz)}"
        curve_label = str(curve_label or "config")
        curve_color = self._get_config_plot_color(to_compare_config)

        thresholds = np.asarray(curve_data["thresholds"], dtype=np.float64)
        rates_hz = np.asarray(curve_data["rates_hz"], dtype=np.float64)
        target_idx: Optional[int] = None
        if resolved_target_rate_hz is not None and target_threshold is not None and thresholds.size > 0:
            close_idx = np.flatnonzero(np.isclose(thresholds, float(target_threshold), rtol=0.0, atol=1e-9))
            if close_idx.size > 0:
                target_idx = int(close_idx[0])
            else:
                target_idx = int(np.argmin(np.abs(thresholds - float(target_threshold))))

        efficiency = np.asarray(curve_data["efficiency"], dtype=np.float64)
        mask = np.isfinite(rates_hz) & (rates_hz > 0.0) & np.isfinite(efficiency)
        if not np.any(mask):
            return

        line = plt.plot(
            rates_hz[mask][::-1],
            efficiency[mask][::-1],
            linewidth=2.0,
            label=curve_label,
            color=curve_color,
        )
        curve_color = line[0].get_color() if line else None

        if (
            target_idx is not None
            and curve_color is not None
            and np.isfinite(marker_rate_hz)
            and marker_rate_hz > 0.0
            and 0 <= target_idx < efficiency.size
            and np.isfinite(efficiency[target_idx])
        ):
            plt.scatter(
                [marker_rate_hz],
                [efficiency[target_idx]],
                color=curve_color,
                s=24,
                zorder=3,
                label="_nolegend_",
            )

        if resolved_target_rate_hz is not None and draw_target_rate_line and float(resolved_target_rate_hz) > 0.0:
            plt.axvline(
                float(resolved_target_rate_hz),
                color="black",
                linestyle="--",
                label=f"target rate={float(resolved_target_rate_hz):.1f} Hz",
            )

    def initPlotTriggerRateVsThreshold(
        self,
        title: str = "",
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
    ):
        self._reset_custom_legend_state()
        plt.figure(figsize=(10, 6))
        plt.title(title or "NSB Trigger Rate vs Threshold")
        plt.xlabel("Threshold on pre-threshold score")
        plt.ylabel("NSB Trigger Rate (Hz)")
        plt.grid(True, which="both", alpha=0.3)
        plt.yscale("log")
        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)

    def addPlotTriggerRateVsThreshold(
        self,
        to_compare_config: ConfigType,
        label: Optional[str] = None,
        target_rate_hz: Optional[float] = None,
        max_points: Optional[int] = 20_000,
        draw_target_rate_line: bool = True,
        draw_target_threshold_line: bool = True,
    ):
        result = self.get_results(to_compare_config)
        if result is None:
            print("Config not found")
            return
        if not result.get("has_pre_threshold_score", False):
            print(f"Config {to_compare_config} has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset.")
            return

        thresholds, rates_hz = self.get_trigger_rate_curve(to_compare_config=to_compare_config)
        if thresholds.size == 0:
            print("No NSB pre-threshold scores found.")
            return

        if max_points is not None and thresholds.size > max_points:
            idx = np.linspace(0, thresholds.size - 1, int(max_points), dtype=int)
            idx = np.unique(idx)
            thresholds = thresholds[idx]
            rates_hz = rates_hz[idx]

        strategy = self._resolve_trigger_strategy(result, target_rate_hz=target_rate_hz)
        rate = float(strategy.get("trigger_rate_hz", 0.0))
        if label is None:
            label = self.generate_label_text(
                to_compare_config,
                rate,
                threshold_override=strategy.get("reference_threshold"),
            )
        else:
            label = f"{label}; {self._rate_suffix(rate)}"

        curve_color = self._get_config_plot_color(to_compare_config)
        line = plt.step(thresholds, rates_hz, where="post", label=label, color=curve_color)
        curve_color = line[0].get_color() if line else None

        if target_rate_hz is not None:
            if draw_target_rate_line:
                plt.axhline(
                    float(target_rate_hz),
                    color="black",
                    linestyle="--",
                    label=f"target rate={float(target_rate_hz):.1f} Hz",
                )

            target_threshold = strategy.get("score_threshold")
            if draw_target_threshold_line and target_threshold is not None and curve_color is not None:
                plt.axvline(
                    float(target_threshold),
                    color=curve_color,
                    linestyle=":",
                    alpha=0.8,
                )

    def _is_outside_right_legend_location(self, location: Optional[str]) -> bool:
        normalized = str(location or "").strip().lower().replace("-", " ").replace("_", " ")
        return normalized in {
            "outside right",
            "right outside",
            "outside",
            "outside right center",
            "outside center right",
        }

    def _is_outside_bottom_legend_location(self, location: Optional[str]) -> bool:
        normalized = str(location or "").strip().lower().replace("-", " ").replace("_", " ")
        return normalized in {
            "outside bottom",
            "bottom outside",
            "below",
            "below plot",
            "under plot",
            "outside bottom center",
            "outside center bottom",
        }

    def _apply_standard_legend(self, ax, location: str = "best") -> Optional[str]:
        if self._is_outside_right_legend_location(location):
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
            return "outside_right"
        if self._is_outside_bottom_legend_location(location):
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                borderaxespad=0.0,
                ncol=1,
            )
            return "outside_bottom"
        ax.legend(loc=location)
        return None

    def plotEfficiencyVsTriggerRate(
        self,
        to_compare_config: Optional[ConfigType] = None,
        metric: str = "energy",
        metric_bins: Optional[Iterable[float]] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        target_rate_hz: Optional[float] = None,
        max_points: Optional[int] = 4_000,
        filename: Optional[str] = None,
        show: bool = True,
    ):
        if to_compare_config is None:
            result = self.base_config_result
            title_config = self.base_reference_config
        else:
            result = self.get_results(to_compare_config)
            title_config = to_compare_config
        if result is None:
            print("Config not found")
            return
        if not result.get("has_pre_threshold_score", False):
            print(f"Config {title_config} has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset.")
            return

        resolved_metric = self._normalize_metric_name(metric)
        title_config_text = self._generate_title_config_text(
            title_config,
            result=result,
            target_rate_hz=target_rate_hz,
            include_threshold=True,
        )
        self.initPlotEfficiencyVsTriggerRate(
            title=f"Gamma Efficiency vs NSB Trigger Rate - Config: {title_config_text}",
            metric=resolved_metric,
            metric_bins=metric_bins,
            x_range=x_range,
            y_range=y_range,
            target_rate_hz=target_rate_hz,
            max_points=max_points,
        )
        self.addPlotEfficiencyVsTriggerRate(
            to_compare_config=title_config,
            label=None,
            metric=resolved_metric,
            metric_bins=metric_bins,
            target_rate_hz=target_rate_hz,
            max_points=max_points,
            draw_target_rate_line=target_rate_hz is not None,
        )

        plt.legend()
        fig = plt.gcf()
        try:
            fig.tight_layout()
        except Exception:
            pass
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()

    def plot_efficiency_vs_trigger_rate(self, *args, **kwargs):
        return self.plotEfficiencyVsTriggerRate(*args, **kwargs)

    def plotTriggerRateVsThreshold(
        self,
        to_compare_config: Optional[ConfigType] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        target_rate_hz: Optional[float] = None,
        max_points: Optional[int] = 20_000,
        filename: Optional[str] = None,
        show: bool = True,
        location: str = "outside bottom",
    ):
        if to_compare_config is None:
            result = self.base_config_result
            title_config = self.base_reference_config
        else:
            result = self.get_results(to_compare_config)
            title_config = to_compare_config
        if result is None:
            print("Config not found")
            return
        if not result.get("has_pre_threshold_score", False):
            print(f"Config {title_config} has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset.")
            return

        title_config_text = self._generate_title_config_text(
            title_config,
            result=result,
            target_rate_hz=target_rate_hz,
            include_threshold=True,
        )
        self.initPlotTriggerRateVsThreshold(
            title=f"NSB Trigger Rate vs Threshold - Config: {title_config_text}",
            x_range=x_range,
            y_range=y_range,
        )
        self.addPlotTriggerRateVsThreshold(
            to_compare_config=title_config,
            label="Empirical NSB rate",
            target_rate_hz=target_rate_hz,
            max_points=max_points,
            draw_target_rate_line=target_rate_hz is not None,
            draw_target_threshold_line=False,
        )

        if target_rate_hz is not None:
            target_threshold, predicted_rate_hz = self.find_score_threshold_for_target_rate(
                float(target_rate_hz),
                to_compare_config=to_compare_config,
            )
            if target_threshold is not None:
                plt.axvline(
                    float(target_threshold),
                    color="tab:red",
                    linestyle=":",
                    label=(
                        f"target threshold={float(target_threshold):.6g} "
                        f"({float(predicted_rate_hz):.1f} Hz)"
                    ),
                )

        ax = plt.gca()
        legend_layout_mode = self._apply_standard_legend(ax=ax, location=location)
        fig = plt.gcf()
        try:
            if legend_layout_mode == "outside_right":
                width, height = fig.get_size_inches()
                if width < 16.0:
                    fig.set_size_inches(16.0, height, forward=True)
                fig.subplots_adjust(left=0.10, right=0.55, bottom=0.12, top=0.90)
            elif legend_layout_mode == "outside_bottom":
                width, height = fig.get_size_inches()
                if height < 8.5:
                    fig.set_size_inches(width, 8.5, forward=True)
                fig.subplots_adjust(left=0.10, right=0.97, bottom=0.42, top=0.90)
            else:
                fig.tight_layout()
        except Exception:
            pass
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()

    def plot_trigger_rate_vs_threshold(self, *args, **kwargs):
        return self.plotTriggerRateVsThreshold(*args, **kwargs)

    def plotScoreDistribution(
        self,
        to_compare_config: Optional[ConfigType] = None,
        bins: int = 200,
        x_range: Optional[Tuple[float, float]] = None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        if to_compare_config is None:
            result = self.base_config_result
            title_config = self.base_reference_config
        else:
            result = self.get_results(to_compare_config)
            title_config = to_compare_config
        if result is None:
            print("Config not found")
            return
        if not result.get("has_pre_threshold_score", False):
            print(f"Config {title_config} has no '{PRE_THRESHOLD_SCORE_DATASET}' dataset.")
            return

        gamma_scores = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="gamma")
        nsb_scores = self._collect_metric_values(result, PRE_THRESHOLD_SCORE_DATASET, kind="nsb")
        if gamma_scores.size == 0 and nsb_scores.size == 0:
            print("No pre-threshold scores found.")
            return

        all_scores = np.concatenate([arr for arr in (gamma_scores, nsb_scores) if arr.size > 0])
        if x_range is None:
            lo = float(np.min(all_scores))
            hi = float(np.max(all_scores))
            if np.isclose(lo, hi):
                hi = lo + 1.0
        else:
            lo, hi = x_range

        edges = np.linspace(lo, hi, bins + 1)
        gamma_hist = np.histogram(gamma_scores, bins=edges)[0] if gamma_scores.size > 0 else np.zeros(bins, dtype=np.int64)
        nsb_hist = np.histogram(nsb_scores, bins=edges)[0] if nsb_scores.size > 0 else np.zeros(bins, dtype=np.int64)
        centers = 0.5 * (edges[:-1] + edges[1:])

        title_config_text = self._generate_title_config_text(
            title_config,
            result=result,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
            include_threshold=True,
        )
        plt.figure(figsize=(10, 6))
        plt.title(f"Pre-threshold score distribution - Config: {title_config_text}")
        plt.xlabel("Pre-threshold score")
        plt.ylabel("Events / bin")
        plt.grid()
        plt.yscale("log")
        if gamma_scores.size > 0:
            plt.step(centers, gamma_hist, where="mid", label="Gamma")
        if nsb_scores.size > 0:
            plt.step(centers, nsb_hist, where="mid", label="NSB")

        strategy = self._resolve_trigger_strategy(
            result,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
        )
        threshold_value = strategy.get("score_threshold")
        if threshold_value is not None:
            rate_hz = float(strategy.get("trigger_rate_hz", float("nan")))
            plt.axvline(
                threshold_value,
                color="black",
                linestyle="--",
                label=f"threshold={threshold_value:.6g}, rate={rate_hz:.1f} Hz",
            )

        plt.legend()
        plt.show()

    # show the difference relation between the energy distribution and the n_pe distribution for the base configuration (av envent can be high energy but low n_pe or vice versa, because it can be far from the telescope or close to the edge of the camera, etc)
    def plotEnergyVsNpeBase(self, energy_bins: int = 50, n_pe_bins: int = 50, energy_range: Optional[Tuple[float, float]] = None, n_pe_range: Optional[Tuple[float, float]] = None):
        r = self.base_config_result
        if r is None:
            print("Base configuration not found.")
            return
        if energy_range is None:
            energy_lo, energy_hi = self._get_default_range(r, "energy", kind="gamma")
        else:
            energy_lo, energy_hi = energy_range
        if n_pe_range is None:
            n_pe_lo, n_pe_hi = self._get_default_range(r, "n_pe", kind="gamma")
        else:
            n_pe_lo, n_pe_hi = n_pe_range
        energy_edges = np.logspace(np.log10(max(energy_lo, 1e-4)), np.log10(energy_hi + 1e-4), energy_bins)
        n_pe_edges = np.logspace(np.log10(max(n_pe_lo, 1e-3)), np.log10(n_pe_hi + 1e-3), n_pe_bins)
        plt.figure(figsize=(10, 6))
        plt.title(f"Energy vs Npe - Base Config: {self.base_reference_config}")
        plt.xlabel("Energy (TeV)")
        plt.ylabel("Npe")
        plt.grid()
        plt.xscale("log") # energy
        plt.yscale("log") # n_pe
        # set the limits for x and y axis to have a better visualization
        plt.xlim(left=max(energy_lo, 1e-4), right=energy_hi + 1e-4)
        plt.ylim(10, n_pe_hi + 1e-3)
        if self._is_h5(r):
            counts = self._histogram2d_stream(r, "energy", "n_pe", kind="gamma", x_bins=energy_edges, y_bins=n_pe_edges)
        else:
            all_energy = np.asarray(r.get("energy", []))
            all_n_pe = np.asarray(r.get("n_pe", []))
            if all_energy.size != all_n_pe.size:
                raise ValueError("Energy and n_pe arrays have different sizes.")
            mfinite = np.isfinite(all_energy) & np.isfinite(all_n_pe)
            all_energy = all_energy[mfinite]
            all_n_pe = all_n_pe[mfinite]
            counts, _, _ = np.histogram2d(all_energy, all_n_pe, bins=[energy_edges, n_pe_edges])

        max_count = float(np.nanmax(counts)) if counts.size else 0.0
        use_log = max_count > 0
        norm = colors.LogNorm(vmin=1, vmax=max_count) if use_log else None
        mesh = plt.pcolormesh(energy_edges, n_pe_edges, counts.T, norm=norm, cmap="viridis", shading="auto")
        plt.colorbar(mesh, label="Number of Events")
        if max_count <= 0:
            print("No events found in the selected energy/n_pe ranges.")
        plt.show()


    def _powerlaw_expected_counts(self, bins: np.ndarray, N: int, slope: float) -> np.ndarray:
        """
        Expected bin counts for dN/dE ~ E^-slope over the given bin edges.
        No randomness (unlike np.random sampling).
        """
        bins = np.asarray(bins, dtype=np.float64)
        if slope == 1.0:
            # integral E^-1 dE = ln(E)
            w = np.log(bins[1:] / bins[:-1])
        else:
            a = 1.0 - slope
            w = (bins[1:] ** a - bins[:-1] ** a) / a
        w = np.clip(w, 0.0, np.inf)
        if w.sum() <= 0:
            return np.zeros(len(bins) - 1, dtype=np.float64)
        w = w / w.sum()
        return (N * w).astype(np.float64)

    def _effective_area_denominator_hist(
        self,
        result: Dict[str, Any],
        use_base_thrown: bool,
        use_theoretical_thrown: bool,
    ) -> np.ndarray:
        if use_theoretical_thrown:
            thrown = getattr(self, "_ea_expected_thrown_hist", None)
            if thrown is None:
                raise RuntimeError("Internal error: missing cached expected thrown histogram. Re-run initPlotEffectiveArea.")
            return np.asarray(thrown, dtype=np.float64)

        if use_base_thrown:
            thrown = getattr(self, "_ea_base_all_hist", None)
            if thrown is None:
                raise RuntimeError("Internal error: missing cached base thrown histogram. Re-run initPlotEffectiveArea.")
            return np.asarray(thrown, dtype=np.float64)

        thrown, _ = self._histogram_stream(result, "energy", kind="gamma", bins=self._ea_bins)
        return thrown.astype(np.float64)

    def _prepare_effective_area_state(
        self,
        emin_tev: float = EA_DEFAULT_EMIN_TEV,
        emax_tev: float = EA_DEFAULT_EMAX_TEV,
        nbins: int = 25,
        A_gen_m2: float = EA_DEFAULT_A_GEN_M2,
        use_base_thrown: bool = True,
        use_theoretical_thrown: bool = True,
        plot_errors: bool = True,
        expected_N: int = EA_DEFAULT_TOTAL_THROWN,
        expected_slope: float = EA_DEFAULT_SLOPE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ea_bins = np.logspace(np.log10(emin_tev), np.log10(emax_tev), nbins)
        self._ea_A_gen_m2 = float(A_gen_m2)
        self._ea_use_base_thrown = bool(use_base_thrown)
        self._ea_use_theoretical_thrown = bool(use_theoretical_thrown)
        self._ea_plot_errors = bool(plot_errors)

        bins = self._ea_bins
        x_values = bins[:-1]

        expected_thrown = self._powerlaw_expected_counts(
            bins,
            N=int(expected_N),
            slope=float(expected_slope),
        )
        base_strategy = self._resolve_trigger_strategy(
            self.base_config_result,
            target_rate_hz=self.target_rate_hz,
            score_threshold=self.score_threshold,
        )
        base_all, base_trig = self._histogram_stream(
            self.base_config_result,
            "energy",
            kind="gamma",
            bins=bins,
            score_threshold=base_strategy.get("score_threshold"),
        )
        base_all = base_all.astype(np.float64)
        base_trig = base_trig.astype(np.float64)

        self._ea_expected_thrown_hist = expected_thrown
        self._ea_base_all_hist = base_all
        return x_values, expected_thrown, base_all, base_trig

    def initPlotEffectiveAreaCounts(
        self,
        title: str = "",
        emin_tev: float = EA_DEFAULT_EMIN_TEV,
        emax_tev: float = EA_DEFAULT_EMAX_TEV,
        nbins: int = 25,
        show_expected_powerlaw: bool = True,
        expected_N: int = EA_DEFAULT_TOTAL_THROWN,
        expected_slope: float = EA_DEFAULT_SLOPE,
        x_range=None,
        y_range_counts=None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        self._reset_custom_legend_state()
        self.target_rate_hz = target_rate_hz
        self.score_threshold = score_threshold
        x_values, expected_thrown, base_all, base_trig = self._prepare_effective_area_state(
            emin_tev=emin_tev,
            emax_tev=emax_tev,
            nbins=nbins,
            expected_N=expected_N,
            expected_slope=expected_slope,
        )

        plt.figure(figsize=(17, 7), constrained_layout=True)
        counts_line = self._format_counts_line()
        title_parts = [title, "Number of simulated and triggered events", counts_line]
        plt.title("\n".join([p for p in title_parts if p]))

        if show_expected_powerlaw:
            plt.plot(x_values, expected_thrown, color="blue", linestyle="--", label="Expected thrown (power law)")

        plt.plot(x_values, base_all, color="green", linestyle="--", label="All gamma events (base)")
        plt.plot(x_values, base_trig, color="red", linestyle="--", label="Triggered events (base)")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (TeV)")
        plt.ylabel("Events / bin")
        plt.grid(True, which="both", alpha=0.3)

        if x_range is not None:
            plt.xlim(x_range)
        if y_range_counts is not None:
            plt.ylim(y_range_counts)

    def initPlotEffectiveArea(
        self,
        title: str = "",
        emin_tev: float = EA_DEFAULT_EMIN_TEV,
        emax_tev: float = EA_DEFAULT_EMAX_TEV,
        nbins: int = 25,
        A_gen_m2: float = EA_DEFAULT_A_GEN_M2,
        use_base_thrown: bool = True,
        use_theoretical_thrown: bool = True,
        plot_errors: bool = True,
        expected_N: int = EA_DEFAULT_TOTAL_THROWN,
        expected_slope: float = EA_DEFAULT_SLOPE,
        x_range=None,
        y_range_aeff=None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Single-panel effective-area plot.

        - Aeff(E) = A_gen_m2 * N_trig(E) / N_thrown(E)
        - By default N_thrown(E) is the analytic power-law expectation for the
          CORSIKA generation setup, which matches the older plotting code more closely.
        - If use_theoretical_thrown=False and use_base_thrown=True, N_thrown(E) is
          taken from the base configuration gamma histogram.
        """
        self._reset_custom_legend_state()
        self.target_rate_hz = target_rate_hz
        self.score_threshold = score_threshold
        x_values, _, _, base_trig = self._prepare_effective_area_state(
            emin_tev=emin_tev,
            emax_tev=emax_tev,
            nbins=nbins,
            A_gen_m2=A_gen_m2,
            use_base_thrown=use_base_thrown,
            use_theoretical_thrown=use_theoretical_thrown,
            plot_errors=plot_errors,
            expected_N=expected_N,
            expected_slope=expected_slope,
        )

        plt.figure(figsize=(17, 7), constrained_layout=True)
        counts_line = self._format_counts_line()
        title_parts = [title, "Trigger effective collection area", counts_line]
        plt.title("\n".join([p for p in title_parts if p]))

        thrown_for_base = self._effective_area_denominator_hist(
            result=self.base_config_result,
            use_base_thrown=bool(use_base_thrown),
            use_theoretical_thrown=bool(use_theoretical_thrown),
        )
        aeff_base, sigma_base = self._aeff_from_hist(base_trig, thrown_for_base, A_gen_m2)

        if plot_errors:
            plt.errorbar(x_values, aeff_base, yerr=sigma_base, fmt="-o", capsize=3, label="Base")
        else:
            plt.plot(x_values, aeff_base, "-o", label="Base")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (TeV)")
        plt.ylabel("Effective Area (m²)" if A_gen_m2 != 1.0 else "Aeff (A_gen units)")
        plt.grid(True, which="both", alpha=0.3)

        if x_range is not None:
            plt.xlim(x_range)
        if y_range_aeff is not None:
            plt.ylim(y_range_aeff)

    def _aeff_from_hist(self, trig_hist: np.ndarray, thrown_hist: np.ndarray, A_gen_m2: float):
        trig_hist = np.asarray(trig_hist, dtype=np.float64)
        thrown_hist = np.asarray(thrown_hist, dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            aeff = (trig_hist / thrown_hist) * float(A_gen_m2)
            sigma = (np.sqrt(trig_hist) / thrown_hist) * float(A_gen_m2)

        aeff = np.nan_to_num(aeff, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        return aeff, sigma


    def _gamma_trigger_hist(
        self,
        result: dict,
        bins: np.ndarray,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Return only the triggered gamma histogram for a config (1 pass for that config)."""
        strategy = self._resolve_trigger_strategy(
            result,
            target_rate_hz=target_rate_hz,
            score_threshold=score_threshold,
        )
        _, trig = self._histogram_stream(
            result,
            "energy",
            kind="gamma",
            bins=bins,
            score_threshold=strategy.get("score_threshold"),
        )
        return trig.astype(np.float64)

    def addPlotEffectiveAreaCounts(
        self,
        to_compare_config,
        label: str = None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        to_compare_result = self.get_results(to_compare_config)
        if to_compare_result is None:
            print("Configuration to compare not found.")
            return

        if not hasattr(self, "_ea_bins"):
            raise RuntimeError("Call initPlotEffectiveAreaCounts(...) first.")

        if label is None and self._is_base_config(result=to_compare_result):
            return

        compare_target_rate_hz = self.target_rate_hz if target_rate_hz is None else target_rate_hz
        compare_score_threshold = self.score_threshold if score_threshold is None else score_threshold
        trig = self._gamma_trigger_hist(
            to_compare_result,
            bins=self._ea_bins,
            target_rate_hz=compare_target_rate_hz,
            score_threshold=compare_score_threshold,
        )
        x_values = self._ea_bins[:-1]

        strategy = self._resolve_trigger_strategy(
            to_compare_result,
            target_rate_hz=compare_target_rate_hz,
            score_threshold=compare_score_threshold,
        )
        rate = float(strategy.get("trigger_rate_hz", 0.0))
        if label is None:
            label = self.generate_label_text(
                to_compare_config,
                rate,
                threshold_override=strategy.get("reference_threshold"),
            )
        else:
            label = f"{label}; {self._rate_suffix(rate)}"

        curve_color = self._get_config_plot_color(to_compare_config)
        plt.plot(x_values, trig, linestyle="--", label=f"Triggered ({label})", color=curve_color)

    def addPlotEffectiveArea(
        self,
        to_compare_config,
        label: str = None,
        A_gen_m2: float = None,
        use_base_thrown: bool = None,
        use_theoretical_thrown: bool = None,
        plot_errors: bool = None,
        target_rate_hz: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Adds a config on the effective-area plot.
        """
        to_compare_result = self.get_results(to_compare_config)
        if to_compare_result is None:
            print("Configuration to compare not found.")
            return

        if not hasattr(self, "_ea_bins"):
            raise RuntimeError("Call initPlotEffectiveArea(...) first.")

        bins = self._ea_bins

        if label is None and self._is_base_config(result=to_compare_result):
            return

        if A_gen_m2 is None:
            A_gen_m2 = getattr(self, "_ea_A_gen_m2", 1.0)
        if use_base_thrown is None:
            use_base_thrown = getattr(self, "_ea_use_base_thrown", True)
        if use_theoretical_thrown is None:
            use_theoretical_thrown = getattr(self, "_ea_use_theoretical_thrown", True)
        if plot_errors is None:
            plot_errors = getattr(self, "_ea_plot_errors", True)

        thrown = self._effective_area_denominator_hist(
            result=to_compare_result,
            use_base_thrown=bool(use_base_thrown),
            use_theoretical_thrown=bool(use_theoretical_thrown),
        )

        compare_target_rate_hz = self.target_rate_hz if target_rate_hz is None else target_rate_hz
        compare_score_threshold = self.score_threshold if score_threshold is None else score_threshold
        trig = self._gamma_trigger_hist(
            to_compare_result,
            bins=bins,
            target_rate_hz=compare_target_rate_hz,
            score_threshold=compare_score_threshold,
        )
        x_values = bins[:-1]

        strategy = self._resolve_trigger_strategy(
            to_compare_result,
            target_rate_hz=compare_target_rate_hz,
            score_threshold=compare_score_threshold,
        )
        rate = float(strategy.get("trigger_rate_hz", 0.0))
        if label is None:
            label = self.generate_label_text(
                to_compare_config,
                rate,
                threshold_override=strategy.get("reference_threshold"),
            )
        else:
            label = f"{label}; {self._rate_suffix(rate)}"

        aeff, sigma = self._aeff_from_hist(trig, thrown, A_gen_m2)
        curve_color = self._get_config_plot_color(to_compare_config)
        if plot_errors:
            plt.errorbar(x_values, aeff, yerr=sigma, fmt="-o", capsize=3, label=label, color=curve_color)
        else:
            plt.plot(x_values, aeff, "-o", label=label, color=curve_color)



# # -----------------------------
# if __name__ == "__main__":
#     plotter = StatPlotter(
#         base_reference_config=[("baselinesubstractor", {}), ("digital_sum", {'threshold_flower': 322, 'mode': 'triplet'})],
#         stat_folder="sample/",
#     )
#     # add a absolute plot
#     plotter.initPlotAbsolute(title="Absolute Efficiency Ratio", x_range=(55, 375), y_range=(0, 1.2), metrics="n_pe", bins=35)
#     plotter.addPlotAbsolute(
#         to_compare_config=[("baselinesubstractor", {}), ("digital_sum", {'threshold_flower': 322, 'mode': 'triplet'})],
#         label="Base Config",
#     )


#     # show plot
#     plotter.showPlot(filename="absolute_efficiency.png", show=True)
