import tensorflow as tf


class ScheduleCallback(tf.keras.callbacks.Callback):
    """
    Update training hyperparameters at specific epochs.

    Schedule format:
      {epoch: {"alpha": 10.0, "lr": 1e-4}, ...}
    """
    def __init__(self, schedule, alpha=None, optimizer=None, verbose=True, apply_at="begin"):
        super().__init__()
        self.schedule = self._normalize_schedule(schedule)
        self.alpha = alpha
        self.optimizer = optimizer
        self.verbose = bool(verbose)
        self.apply_at = str(apply_at).lower()

    def on_epoch_begin(self, epoch, logs=None):
        if self.apply_at == "begin":
            self._maybe_apply(epoch)

    def on_epoch_end(self, epoch, logs=None):
        if self.apply_at == "end":
            self._maybe_apply(epoch)

    def _maybe_apply(self, epoch):
        updates = self.schedule.get(int(epoch))
        if updates is None:
            return
        if not isinstance(updates, dict):
            raise ValueError("Schedule updates must be a dict.")
        self._apply_updates(updates, epoch)

    def _apply_updates(self, updates, epoch):
        messages = []

        if "alpha" in updates and self.alpha is not None:
            self._assign(self.alpha, updates["alpha"])
            messages.append(f"alpha={self._as_float(self.alpha)}")

        if "lr" in updates and self.optimizer is not None:
            self._assign_lr(updates["lr"])
            messages.append(f"lr={self._as_float(self.optimizer.learning_rate)}")

        if self.verbose and messages:
            print(f"ScheduleCallback epoch {epoch}: " + ", ".join(messages), flush=True)

    def _normalize_schedule(self, schedule):
        if schedule is None:
            return {}
        if isinstance(schedule, dict):
            return {int(k): v for k, v in schedule.items()}
        normalized = {}
        for item in schedule:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Schedule entries must be (epoch, updates) pairs.")
            epoch, updates = item
            normalized[int(epoch)] = updates
        return normalized

    def _assign(self, var, value):
        if hasattr(var, "assign"):
            var.assign(value)
            return
        raise TypeError("Target does not support assign().")

    def _assign_lr(self, value):
        lr = self.optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(value)
        else:
            self.optimizer.learning_rate = value

    def _as_float(self, value):
        try:
            if hasattr(value, "numpy"):
                return float(value.numpy())
            if tf.is_tensor(value):
                return float(tf.keras.backend.get_value(value))
            return float(value)
        except Exception:
            return value
