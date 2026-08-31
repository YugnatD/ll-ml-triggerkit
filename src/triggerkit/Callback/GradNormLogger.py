
import tensorflow as tf

class GradNormLogger(tf.keras.callbacks.Callback):
    def __init__(self, sample_ds, steps=2):
        super().__init__()
        self.sample_iter = iter(sample_ds)
        self.steps = steps  # small number to keep it cheap

    def on_epoch_end(self, epoch, logs=None):
        norms = []
        for _ in range(self.steps):
            try:
                x_batch, y_batch = next(self.sample_iter)
            except StopIteration:
                break
            with tf.GradientTape() as tape:
                y_pred = self.model(x_batch, training=True)
                loss = self.model.compiled_loss(y_batch, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
            batch_norms = [tf.norm(g) for g in grads if g is not None]
            norms.extend(batch_norms)
        if norms:
            norms_tensor = tf.stack(norms)
            print(
                f"[grad] mean={tf.reduce_mean(norms_tensor):.3f} "
                f"std={tf.math.reduce_std(norms_tensor):.3f} "
                f"max={tf.reduce_max(norms_tensor):.3f}"
            )
