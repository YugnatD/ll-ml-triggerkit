"""Per-epoch training loggers.

Concise per-filter logging for the multi-filter restart training (one line per
epoch, one line per TDSCAN filter). Split out of the sandbox's ``train_utils.py``;
behaviour unchanged. Note ``WeightStatsLogger`` reads TDSCAN-specific attributes
(``kernel_rings`` / ``kernel``, penalty targets), so it only applies to a TDSCAN
body, not the generic CNN.
"""

import numpy as np
import tensorflow as tf


class PerFilterAUCLogger(tf.keras.callbacks.Callback):
    """One concise line per epoch: loss + best AUC + the per-filter AUC array.

    Replaces the wide live progress bar (which redraws all auc_fk columns and is
    unreadable with many filters). Use with ``model.fit(..., verbose=0)``. Reads
    the per-filter values straight from ``logs`` (auc_fk / val_auc_fk), which
    Keras populates from the registered metrics.
    """

    def __init__(self, n_filters):
        super().__init__()
        self.n_filters = int(n_filters)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train = [logs.get(f"auc_f{k}") for k in range(self.n_filters)]
        val = [logs.get(f"val_auc_f{k}") for k in range(self.n_filters)]

        def fmt(arr):
            return "[" + ", ".join("--" if v is None else f"{v:.4f}" for v in arr) + "]"

        parts = [f"Epoch {epoch + 1:4d}"]
        if logs.get("loss") is not None:
            parts.append(f"loss={logs['loss']:.4f}")
        if logs.get("val_auc_best") is not None:
            parts.append(f"val_auc_best={logs['val_auc_best']:.4f}")
        elif logs.get("auc_best") is not None:
            parts.append(f"auc_best={logs['auc_best']:.4f}")
        if logs.get("lr") is not None:
            parts.append(f"lr={logs['lr']:.2e}")
        parts.append(f"val_auc_filters={fmt(val)}")
        print("  ".join(parts), flush=True)


class WeightStatsLogger(tf.keras.callbacks.Callback):
    """Per-epoch, one line per TDSCAN filter: weight mean, std, and val AUC.

    Lets you watch the mean/std penalties (PerFilterMeanRegularizer) act during
    training, side by side with each filter's AUC. Weight mean/std are read from
    the layer each epoch (reduced over every axis except the filter axis, last);
    the AUC is taken from ``logs`` (val_auc_fk, falling back to auc_fk). One line
    per filter::

        [w] targets mean->0 std->2  (* = best on val AUC)
          Filter 00: mean=+0.133, stddev=0.992, auc=0.6745
          Filter 01: mean=+0.107, stddev=1.130, auc=0.7012 *
          ...

    The ``*`` marks the filter currently winning on val AUC (the one that will be
    kept). The header shows the active penalty targets; each is omitted when its
    penalty is off, and ``auc=`` is omitted when the AUC isn't in ``logs``.

    Pass the TDSCAN layer (or a list of them, for a multi-branch OR). Pairs with
    ``model.fit(..., verbose=0)`` like PerFilterAUCLogger.
    """

    def __init__(self, tdscan_layers, precision=3):
        super().__init__()
        if not isinstance(tdscan_layers, (list, tuple)):
            tdscan_layers = [tdscan_layers]
        self.tdscan_layers = list(tdscan_layers)
        self.precision = int(precision)

    def _weights(self, layer):
        w = layer.kernel_rings if layer.share_neighbors else layer.kernel
        return None if w is None else w.numpy()

    @staticmethod
    def _per_filter_auc(logs, n_filters):
        """Per-filter (val) AUC list this epoch, or None if not in logs."""
        for prefix in ("val_auc_f", "auc_f"):
            vals = [logs.get(f"{prefix}{k}") for k in range(n_filters)]
            if all(v is not None for v in vals):
                return [float(v) for v in vals]
        return None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        p = self.precision
        for i, layer in enumerate(self.tdscan_layers):
            w = self._weights(layer)
            if w is None:
                continue
            # Filter axis is last; reduce over all the others -> one value per filter.
            axes = tuple(range(w.ndim - 1))
            means = np.mean(w, axis=axes)
            stds = np.std(w, axis=axes)
            n_f = means.shape[0]
            aucs = self._per_filter_auc(logs, n_f)
            best = int(np.argmax(aucs)) if aucs is not None else None

            tag = f"[w{i}]" if len(self.tdscan_layers) > 1 else "[w]"
            targets = []
            if layer.mean_penalty_lambda > 0:
                targets.append(f"mean->{layer.mean_penalty_target:.{p}g}")
            if layer.std_penalty_lambda > 0:
                targets.append(f"std->{layer.std_penalty_target:.{p}g}")
            header = f"  {tag} per-filter mean/std/auc"
            if targets:
                header += "  targets " + " ".join(targets)
            if best is not None:
                header += "  (* = best on val AUC)"
            print(header, flush=True)

            for f in range(n_f):
                line = f"    Filter {f:02d}: mean={means[f]:+.{p}f}, stddev={stds[f]:.{p}f}"
                if aucs is not None:
                    line += f", auc={aucs[f]:.4f}"
                if f == best:
                    line += " *"
                print(line, flush=True)
