"""Pairwise-AUC training metrics (single- and multi-filter).

Hard AUC = fraction of (gamma, NSB) pairs correctly ranked, computed on raw
pre-threshold scores at any scale (ties counted as 0.5). Split out of the
sandbox's ``train_utils.py``; behaviour unchanged.
"""

import tensorflow as tf


class PairwiseAUCMetric(tf.keras.metrics.Metric):
    """Hard AUC: fraction of (gamma, NSB) pairs with score_gamma > score_NSB.

    Works on raw pre-threshold scores at any scale. Treats ties as 0.5 (standard
    Wilcoxon-Mann-Whitney convention).
    """

    def __init__(self, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        pos = tf.boolean_mask(y_pred, y_true > 0.5)
        neg = tf.boolean_mask(y_pred, y_true < 0.5)
        n_pos = tf.size(pos)
        n_neg = tf.size(neg)

        def _update():
            diff = pos[:, None] - neg[None, :]
            above = tf.reduce_sum(tf.cast(diff > 0, tf.float32))
            tied = tf.reduce_sum(tf.cast(diff == 0, tf.float32)) * 0.5
            pairs = tf.cast(n_pos * n_neg, tf.float32)
            return above + tied, pairs

        c, p = tf.cond(
            tf.logical_and(n_pos > 0, n_neg > 0),
            _update,
            lambda: (tf.constant(0.0), tf.constant(0.0)),
        )
        self.correct.assign_add(c)
        self.total.assign_add(p)

    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class PerFilterPairwiseAUCMetric(tf.keras.metrics.Metric):
    """Per-filter (or best-of) hard AUC for a multi-filter pooled score ``(B, F)``.

    ``column=k`` reports filter ``k``'s AUC; ``column=None`` reports the max over
    filters (the score that matters for the run -- the best restart so far).
    Pair counts are shared across filters; only ``correct`` differs per column.
    """

    def __init__(self, n_filters, column=None, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_filters = int(n_filters)
        self.column = column
        self.correct = self.add_weight(
            name="correct", shape=(self.n_filters,), initializer="zeros"
        )
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        if y_pred.shape.rank == 1:
            y_pred = y_pred[:, None]
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        pos = tf.boolean_mask(y_pred, y_true > 0.5)  # (n_pos, F)
        neg = tf.boolean_mask(y_pred, y_true < 0.5)  # (n_neg, F)
        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[0]

        def _update():
            diff = pos[:, None, :] - neg[None, :, :]                       # (n_pos,n_neg,F)
            above = tf.reduce_sum(tf.cast(diff > 0, tf.float32), axis=[0, 1])  # (F,)
            tied = tf.reduce_sum(tf.cast(tf.equal(diff, 0), tf.float32), axis=[0, 1]) * 0.5
            pairs = tf.cast(n_pos * n_neg, tf.float32)
            return above + tied, pairs

        c, p = tf.cond(
            tf.logical_and(n_pos > 0, n_neg > 0),
            _update,
            lambda: (tf.zeros((self.n_filters,)), tf.constant(0.0)),
        )
        self.correct.assign_add(c)
        self.total.assign_add(p)

    def result(self):
        per_col = self.correct / tf.maximum(self.total, 1.0)
        if self.column is None:
            return tf.reduce_max(per_col)
        return per_col[self.column]

    def reset_state(self):
        self.correct.assign(tf.zeros((self.n_filters,)))
        self.total.assign(0.0)


def make_per_filter_auc_metrics(n_filters):
    """One ``auc_fk`` metric per filter plus ``auc_best`` (max over filters).

    Keras shows each as its own column and stores a ``val_`` twin in the history,
    so you see all restarts diverge during training and can monitor ``val_auc_best``.
    """
    metrics = [
        PerFilterPairwiseAUCMetric(n_filters, column=k, name=f"auc_f{k}")
        for k in range(n_filters)
    ]
    metrics.append(PerFilterPairwiseAUCMetric(n_filters, column=None, name="auc_best"))
    return metrics
