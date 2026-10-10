import tensorflow as tf


class GradNormLogger(tf.keras.callbacks.Callback):
    """Print gradient-norm statistics on a few batches at the end of each epoch.

    The iterator over ``sample_ds`` is created lazily at the first epoch end (and
    released in ``on_train_end``): building it in ``__init__`` started the
    dataset's background readers before training even began, in parallel with
    the training pipeline, and never stopped them. When ``sample_ds`` is
    exhausted it is simply restarted at the next epoch.
    """

    def __init__(self, sample_ds, steps=2, verbose=1):
        super().__init__()
        self.verbose = int(verbose)   # 0 -> silent (like Keras' fit(verbose=0))
        self.sample_ds = sample_ds
        self.steps = steps  # small number to keep it cheap
        self._it = None

    def on_train_end(self, logs=None):
        self._it = None  # drop the iterator (and its prefetch/reader threads)

    def on_epoch_end(self, epoch, logs=None):
        if self._it is None:
            self._it = iter(self.sample_ds)
        norms = []
        for _ in range(self.steps):
            try:
                x_batch, y_batch = next(self._it)
            except StopIteration:
                self._it = None   # restart from the beginning next epoch
                break
            with tf.GradientTape() as tape:
                y_pred = self.model(x_batch, training=True)
                if hasattr(self.model, "compute_loss"):          # Keras 3 (compiled_loss is deprecated)
                    loss = self.model.compute_loss(x=x_batch, y=y_batch, y_pred=y_pred, training=True)
                else:
                    loss = self.model.compiled_loss(y_batch, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
            batch_norms = [tf.norm(g) for g in grads if g is not None]
            norms.extend(batch_norms)
        if norms and self.verbose:
            norms_tensor = tf.stack(norms)
            print(
                f"[grad] mean={tf.reduce_mean(norms_tensor):.3f} "
                f"std={tf.math.reduce_std(norms_tensor):.3f} "
                f"max={tf.reduce_max(norms_tensor):.3f}"
            )
