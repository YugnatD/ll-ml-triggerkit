"""TDSCAN weight helpers: ring initializer and score-preserving remap.

Split out of the sandbox's ``train_utils.py``; behaviour unchanged.
"""

import numpy as np
import tensorflow as tf


def make_decreasing_ring_init(jitter_std=0.05, floor=1e-4):
    """Positive, ring-decreasing initializer for TDSCAN kernel_rings.

    Shape: (L, num_rings, 1, Cin, Cout). ring0 ~ 1.0, ring1 ~ 0.5, ring2 ~ 0.33...
    Small jitter keeps weights distinct without breaking the ordering.
    """
    def init(shape, dtype=None):
        dtype = dtype or tf.keras.backend.floatx()
        _, R, _, _, _ = shape
        base = 1.0 / (tf.range(R, dtype=dtype) + 1.0)
        scale = tf.reshape(base, (1, R, 1, 1, 1))
        noise = tf.random.truncated_normal(shape, mean=0.0, stddev=jitter_std, dtype=dtype)
        w = scale * (1.0 + noise)
        return tf.maximum(w, tf.constant(floor, dtype=dtype))
    return init


def remap_weights(weights, new_min, new_max):
    """Scale weights by a single factor to fill a signed range, no offset.

    The TDSCAN score is S = sum_i w_i * x_i with NO bias term, so a pure scale
    w -> s*w just rescales the score (S -> s*S): the gamma/NSB ranking, and thus
    the AUC, is unchanged and only tau scales with it. An *offset* (w -> s*w + b),
    as a min-max remap applies, instead adds b*sum_i(x_i) to every score -- an
    input-dependent term that depends on the event's total charge and destroys
    the ranking (this is what tanked evaluate_perf). So we only ever scale.

    We pick the largest positive ``s`` such that every scaled weight still lies
    inside ``[new_min, new_max]``. The binding side fills its bound exactly; the
    other side uses fewer codes (unavoidable without an offset, and correct).

    Note: scaling preserves the sign of the mean -- it cannot produce a negative
    target mean from positive-mean weights. A shifted operating point (mean != 0)
    must be trained in via a penalty on mean(weights), not bolted on here.
    """
    weights = np.asarray(weights, dtype=np.float32)
    if new_min >= new_max:
        raise ValueError("new_min must be < new_max.")

    w_max = float(weights.max())
    w_min = float(weights.min())

    # Largest scale that keeps the positive side <= new_max AND the negative
    # side >= new_min. Sides with no weight (w_max<=0 or w_min>=0) don't bind.
    bounds = []
    if w_max > 0:
        bounds.append(new_max / w_max)
    if w_min < 0:
        bounds.append(new_min / w_min)  # both negative -> positive ratio
    scale = min(bounds) if bounds else 1.0

    return (weights * scale).astype(np.float32)
