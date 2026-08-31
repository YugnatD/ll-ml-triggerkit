import tensorflow as tf
from triggerkit.Stages.TrainableThreshold import TrainableThreshold

class TauRateController(tf.keras.callbacks.Callback):
    def __init__(self, threshold_layer: TrainableThreshold,
                 model: tf.keras.Model,
                 target_hz=50000.0,
                 k=1e-5,
                 integer_step=False,
                 metric_name="nsb_hz"):
        
        super().__init__()
        self.layer = threshold_layer
        # Keep a weak reference to the model; Keras will inject the live model
        # into `self.model` via `set_model` when training starts. Avoid direct
        # assignment because `Callback.model` can be read-only in newer Keras.
        self._model_ref = model
        self.target = float(target_hz)
        self.k = float(k)
        self.metric = metric_name
        self.integer_step = integer_step

    def _apply_trainable(self, trainable: bool, epoch: int):
        """Toggle layer trainability and force retrace of train_function."""
        self.layer.set_trainable(trainable)
        # Force Keras to rebuild the train_function with new trainable vars
        model = self.model or self._model_ref
        if hasattr(model, "make_train_function"):
            model.train_function = model.make_train_function()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        r = logs.get(self.metric, None)
        if r is None:
            return
    
        # rate too high -> increase tau (harder to fire)
        # rate too low  -> decrease tau (easier to fire)
        # if self.integer_step:
        #     error=r - self.target
        #     if error > 0:
        #         self.layer.tau.assign_add(self.k)
        #     elif error < 0:
        #         self.layer.tau.assign_sub(self.k)
        # else:
        #     self.layer.tau.assign_add(self.k * (r - self.target))
