import tensorflow as tf

class TrainSoftMaxPool2D(tf.keras.layers.Layer):
    def __init__(self, beta=5.0, reduce_channels=False, **kwargs):
        super().__init__(**kwargs)
        # beta controls the smoothness: larger beta ≈ hard max, smaller beta ≈ smoother average
        self.beta = beta
        # when True, also collapse the channel/filter axis so the layer outputs (B,) instead of (B, C)
        self.reduce_channels = reduce_channels

    def call(self, x, training=None):
        """Soft (training) or hard (inference) max-pooling.

        Input shape: (B, H, W, C) i.e. (B, N, T, filters).
        If reduce_channels is True, the reduction also spans the channel axis to
        yield a single score per sample (B,).
        """
        x = tf.cast(x, tf.float32)

        # axes to pool over (space/time) and optionally channels
        axes = [1, 2]
        if self.reduce_channels and x.shape.rank == 4:
            axes.append(3)

        if training:
            # Soft max-pooling using log-sum-exp with temperature 1/beta to approximate max
            return (1.0 / self.beta) * tf.reduce_logsumexp(self.beta * x, axis=axes)
        return tf.reduce_max(x, axis=axes)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"beta": self.beta, "reduce_channels": self.reduce_channels})
        return cfg
