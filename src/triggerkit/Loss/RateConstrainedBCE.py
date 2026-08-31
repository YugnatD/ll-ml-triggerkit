import tensorflow as tf
from keras.saving import register_keras_serializable
from triggerkit.Helper.NSBRate import compute_trig_rate_from_pred

@register_keras_serializable(package="Trigger")
class RateConstrainedBCE(tf.keras.losses.Loss):
    """
    Total loss: BCE + rate penalty on background events.
    Uses compute_trig_rate_from_pred() as the single source of truth.
    """

    def __init__(self, window_size,
                 target_rate_hz=7000.0,
                 alpha=10.0,
                 from_logits=False,  # With trainable thresholds, a sigmoid alredy exists before this loss.
                 label_smoothing=0.0,
                 axis=-1,
                 reduction='sum_over_batch_size',
                 rate_loss="mse",     # "mse" | "mae" | "huber"
                 huber_delta=100.0,
                 pos_weight=1.0,
                 neg_weight=1.0,
                 name="rate_constrained_bce"):
        super().__init__(reduction=reduction, name=name)

        self.window_size = float(window_size)
        self.target_rate_hz = float(target_rate_hz)
        if isinstance(alpha, tf.Variable):
            self.alpha = alpha
        else:
            self.alpha = tf.Variable(alpha, trainable=False, dtype=tf.float32, name="alpha")

        self.from_logits = bool(from_logits)
        self.label_smoothing = float(label_smoothing)
        self.axis = axis

        self.rate_loss = str(rate_loss).lower()
        self.huber_delta = float(huber_delta)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

        # For logging/inspection without breaking graph mode
        self.last_rate_hz = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="last_rate_hz")

        # We implement BCE manually to support per-class weights.
        # Reduction is handled by the base Loss class.

    def call(self, y_true, y_pred):
        # 1) classification term (weighted BCE)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_raw = y_true

        if self.label_smoothing:
            # Match Keras BinaryCrossentropy behavior.
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.from_logits:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        else:
            eps = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
            bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        weights = y_true_raw * self.pos_weight + (1.0 - y_true_raw) * self.neg_weight
        cls = bce * weights

        # Reduce over the feature axis to get per-sample loss.
        if self.axis is not None:
            cls = tf.reduce_mean(cls, axis=self.axis)

        # 2) rate term (DIFFERENTIABLE): use hard=False
        rate_hz, total_trig_bg_events, total_bg_events = compute_trig_rate_from_pred(
            y_true_raw, y_pred,
            window_size_sec=self.window_size,
            hard=False,
            from_logits=self.from_logits
        )
        self.last_rate_hz.assign(rate_hz)

        # diff = rate_hz - tf.cast(self.target_rate_hz, tf.float32)
        rel = rate_hz / tf.cast(self.target_rate_hz, tf.float32) - 1.0
        diff = rel
        if self.rate_loss == "mse":
            rate_pen = tf.square(diff)
        elif self.rate_loss == "mae":
            rate_pen = tf.abs(diff)
        elif self.rate_loss == "huber":
            abs_diff = tf.abs(diff)
            delta = tf.cast(self.huber_delta, tf.float32)
            # Manual Huber on scalar to avoid invalid axis reduction
            rate_pen = tf.where(
                abs_diff <= delta,
                0.5 * tf.square(diff),
                delta * (abs_diff - 0.5 * delta),
            )
        else:
            raise ValueError(f"Unknown rate_loss='{self.rate_loss}'. Use 'mse', 'mae', or 'huber'.")

        return cls + tf.cast(self.alpha, tf.float32) * rate_pen

    def get_config(self):
        config = super().get_config()
        config.update({
            "window_size": self.window_size,
            "target_rate_hz": self.target_rate_hz,
            "alpha": float(tf.keras.backend.get_value(self.alpha)),
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
            "rate_loss": self.rate_loss,
            "huber_delta": self.huber_delta,
            "pos_weight": self.pos_weight,
            "neg_weight": self.neg_weight,
            "name": self.name,
        })
        return config
