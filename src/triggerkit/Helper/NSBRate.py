import tensorflow as tf

def compute_trig_rate_from_pred(y_true, y_pred, window_size_sec, hard=True, from_logits=False):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    # Convert logits -> probabilities if needed
    if from_logits:
        y_pred = tf.sigmoid(y_pred)

    bg_mask = tf.equal(y_true, 0.0)               # background events
    y_bg = tf.boolean_mask(y_pred, bg_mask)       # predictions for background

    if hard:
        trig = tf.cast(y_bg > 0.5, tf.float32)    # non-differentiable
    else:
        trig = y_bg                               # differentiable proxy (soft count)

    total_trig_bg_events = tf.reduce_sum(trig)
    total_bg_events = tf.cast(tf.size(trig), tf.float32)

    rate_hz = (total_trig_bg_events / (total_bg_events + 1e-6)) / tf.cast(window_size_sec, tf.float32)
    return rate_hz, total_trig_bg_events, total_bg_events

    