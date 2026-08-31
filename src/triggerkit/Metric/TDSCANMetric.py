import tensorflow as tf
from triggerkit.Stages.TDSCAN import TDSCAN

class TDSCANMetric(tf.keras.metrics.Metric):
    def __init__(self, tdscan_layer: TDSCAN, name="kernel", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tdscan_layer = tdscan_layer



    def update_state(self, y_true=None, y_pred=None, sample_weight=None):
        # No internal state: we read weights directly in result()
        return

    def result(self):
        """
        Keras expects a single tensor. We return the layer weights directly:
          - share_neighbors=True  -> shape (L, R, Cin, Cout) squeezed to (L, R, Cout) when Cin==1
          - share_neighbors=False -> shape (L, K, Cin, Cout) squeezed to (L, K, Cout) when Cin==1
        """
        if self.tdscan_layer.share_neighbors:
            kr = self.tdscan_layer.kernel_rings  # (L, R, 1, Cin, Cout)
            kr = tf.squeeze(kr, axis=2)          # (L, R, Cin, Cout)
            if kr.shape.rank == 4 and kr.shape[2] == 1:
                kr = tf.squeeze(kr, axis=2)      # (L, R, Cout)
            return kr

        k = self.tdscan_layer.kernel  # (L, K, Cin, Cout)
        # if Cin==1 we drop that axis for readability
        if k.shape.rank == 4 and k.shape[2] == 1:
            k = tf.squeeze(k, axis=2)
        return k

    def reset_state(self):
        # Do nothing: tau is not a running statistic
        pass
