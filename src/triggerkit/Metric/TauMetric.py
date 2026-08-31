import tensorflow as tf
from triggerkit.Stages.TrainableThreshold import TrainableThreshold

class TauMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold_layer: TrainableThreshold, name="tau", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold_layer = threshold_layer

        # scalar state
        self._tau = self.add_weight(
            name="tau",
            shape=(),
            initializer="zeros",
        )

    def update_state(self, y_true=None, y_pred=None, sample_weight=None):
        # tau exists after the layer is built
        if hasattr(self.threshold_layer, "tau"):
            self._tau.assign(self.threshold_layer.tau)

    def result(self):
        return self._tau

    def reset_state(self):
        # Do nothing: tau is not a running statistic
        pass
