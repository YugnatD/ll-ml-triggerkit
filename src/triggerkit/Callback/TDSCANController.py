import tensorflow as tf
from triggerkit.Stages.TDSCAN import TDSCAN
from triggerkit.Stages.TrainableThreshold import TrainableThreshold
# imp
import sys

class TDSCANController(tf.keras.callbacks.Callback):
    def __init__(self, tdscan_layer: TDSCAN, model: tf.keras.Model, print_weights: bool = False):
        super().__init__() # initialize the base class
        self.layer = tdscan_layer
        # Store reference; Keras will attach the real model via set_model.
        self._model_ref = model
        # Per-epoch weight dump is debug noise (and unreadable with many filters);
        # off by default. Set print_weights=True to re-enable.
        self.print_weights = print_weights

    def _apply_trainable(self, trainable: bool, epoch: int):
        """Toggle layer trainability and force retrace of train_function."""
        self.layer.set_trainable(trainable)
        # Force Keras to rebuild the train_function with new trainable vars
        model = self.model or self._model_ref
        if hasattr(model, "make_train_function"):
            model.train_function = model.make_train_function()

    def on_epoch_end(self, epoch, logs=None):
        if not self.print_weights:
            return
        if self.layer.share_neighbors:
            ring_weights = list(self.layer.kernel_rings.numpy().flatten())
            # Plain print + flush to ensure visibility in tqdm/keras progress bars.
            # convert to list of floats for better readability
            ring_weights = [float(w) for w in ring_weights]
            print(f" TD_W= {ring_weights}", flush=True)
        else:
            kernel_weights = list(self.layer.kernel.numpy().flatten())
            # convert to list of floats for better readability
            kernel_weights = [float(w) for w in kernel_weights]
            print(f" TD_W= {kernel_weights}", flush=True)
