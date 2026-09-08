"""Training losses for the trigger chain.

Threshold-invariant pairwise-AUC (Wilcoxon-Mann-Whitney) losses, single- and
multi-filter, plus the differentiable soft-OR that lets several branches train
jointly. Split out of the sandbox's ``train_utils.py``; behaviour unchanged.
"""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="triggerkit")
class PairwiseAUCLoss(tf.keras.losses.Loss):
    """Serializable form of the pairwise-AUC loss.

    A closure returned by a factory cannot be saved: Keras stores the *name*
    ``loss`` and has nothing to rebuild it from, so a compiled model fails to
    reload with "Could not locate function 'loss'". Holding ``sharpness`` on a
    registered class instead makes the compiled model round-trip.

    ``reduction="sum"`` is a no-op on the scalar ``call`` returns; it is set so
    no reduction can rescale the value away from the closure's.
    """

    def __init__(self, sharpness=1.0, name="pairwise_auc", reduction="sum", **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.sharpness = float(sharpness)

    def call(self, y_true, y_pred):
        s = self.sharpness
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        pos = tf.boolean_mask(y_pred, y_true > 0.5)
        neg = tf.boolean_mask(y_pred, y_true < 0.5)
        n_pos = tf.size(pos)
        n_neg = tf.size(neg)

        def _pairwise():
            diff = pos[:, None] - neg[None, :]
            return tf.reduce_mean(tf.nn.softplus(-s * diff))

        return tf.cond(
            tf.logical_and(n_pos > 0, n_neg > 0),
            _pairwise,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sharpness=self.sharpness)
        return cfg


@keras.utils.register_keras_serializable(package="triggerkit")
class PairwiseAUCLossMulti(tf.keras.losses.Loss):
    """Serializable multi-filter pairwise-AUC loss. See ``PairwiseAUCLoss``."""

    def __init__(self, sharpness=1.0, name="pairwise_auc_multi", reduction="sum", **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.sharpness = float(sharpness)

    def call(self, y_true, y_pred):
        s = self.sharpness
        y_pred = tf.cast(y_pred, tf.float32)
        if y_pred.shape.rank == 1:
            y_pred = y_pred[:, None]
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        pos = tf.boolean_mask(y_pred, y_true > 0.5)
        neg = tf.boolean_mask(y_pred, y_true < 0.5)
        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[0]

        def _pairwise():
            diff = pos[:, None, :] - neg[None, :, :]
            per_col = tf.reduce_mean(tf.nn.softplus(-s * diff), axis=[0, 1])
            return tf.reduce_sum(per_col)

        return tf.cond(
            tf.logical_and(n_pos > 0, n_neg > 0),
            _pairwise,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sharpness=self.sharpness)
        return cfg


@keras.utils.register_keras_serializable(package="triggerkit")
class SoftOr(tf.keras.layers.Layer):
    """Probabilistic OR of several branches' pre-threshold scores.

    Replaces a ``Lambda`` wrapping a closure: Keras refuses to deserialize a
    Lambda holding a Python lambda (arbitrary-code-execution risk), which made
    every soft-OR model unloadable. The gates live in the config instead.
    """

    def __init__(self, taus, temps, **kwargs):
        super().__init__(**kwargs)
        self.taus = [float(t) for t in taus]
        self.temps = [float(t) for t in temps]

    def call(self, scores):
        not_fired = None
        for s, tau, temp in zip(scores, self.taus, self.temps):
            s = tf.cast(s, tf.float32)
            if s.shape.rank == 1:
                s = s[:, None]
            p = tf.sigmoid((s - tau) * temp)
            term = 1.0 - p
            not_fired = term if not_fired is None else not_fired * term
        return 1.0 - not_fired

    def get_config(self):
        cfg = super().get_config()
        cfg.update(taus=self.taus, temps=self.temps)
        return cfg



def make_pairwise_auc_loss(sharpness=1.0):
    """Wilcoxon-Mann-Whitney soft AUC loss (= 1 - soft_AUC).

    For every (gamma, NSB) pair in a batch, penalize when score_NSB >= score_gamma.
    Threshold-invariant: only the *ranking* of gamma vs NSB matters, so the loss
    doesn't fight ``tau`` and isn't sensitive to absolute score scale.

    ``sharpness`` controls how steeply the softplus saturates. Set so
    ``sharpness * typical_gamma_minus_NSB_gap`` is O(1).
    """
    return PairwiseAUCLoss(sharpness=sharpness)


def make_pairwise_auc_loss_multi(sharpness=1.0):
    """Multi-filter version of ``make_pairwise_auc_loss``.

    ``y_pred`` is the pooled score with one column per TDSCAN filter, shape
    ``(B, F)``. Each filter is an independent random-restart of the same tiny
    model. We compute the pairwise-AUC loss per column and SUM over columns: the
    sum keeps the filters' gradients independent (column ``f``'s loss only depends
    on column ``f``'s weights), so the F filters train as F parallel restarts in a
    single forward/backward pass. Pick the best one after training.
    """
    return PairwiseAUCLossMulti(sharpness=sharpness)


def soft_or_scores(branch_scores, taus, temps, name="soft_or"):
    """Differentiable soft-OR of several branches' pre-threshold scores.

    Each ``branch_scores[i]`` is a pooled pre-threshold score of shape ``(B, F)``
    (F = parallel TDSCAN filters). We gate each branch through its calibrated
    threshold with a sigmoid -- ``p_i = sigmoid((score_i - tau_i) * temp_i)`` --
    and combine the branches with the probabilistic OR ``1 - prod_i (1 - p_i)``,
    returning a single ``(B, F)`` tensor: filter column ``f`` of the OR couples
    column ``f`` of *every* branch.

    Feed the result to ``make_pairwise_auc_loss_multi`` / the per-filter metrics
    exactly as you would a single chain's pooled score. Because the loss for
    column ``f`` then depends on filter ``f`` of all branches at once, the
    branches train *jointly*: a branch is only rewarded for separating gamma from
    NSB on events the others do not already separate, which is what pushes the
    branches to be complementary rather than redundant.

    ``taus``/``temps`` are per-branch python floats (the calibrated gate of each
    branch). They are gating constants here, not trained: the branches' TDSCAN
    weights move, the gates that define "fire" stay put, so the OR keeps a fixed
    meaning while the score distributions are reshaped under it.
    """
    if not isinstance(branch_scores, (list, tuple)) or len(branch_scores) < 2:
        raise ValueError("soft_or_scores expects a list of >= 2 branch score tensors.")
    if not (len(branch_scores) == len(taus) == len(temps)):
        raise ValueError("branch_scores, taus and temps must have the same length.")

    return SoftOr(taus=taus, temps=temps, name=name)(list(branch_scores))
