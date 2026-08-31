import tensorflow as tf
from keras.saving import register_keras_serializable
from triggerkit.Helper.NSBRate import compute_trig_rate_from_pred

@register_keras_serializable(package="Trigger")
class NSBRateHz(tf.keras.metrics.Metric):
    def __init__(self, window_size_ns, name="nsb_rate_hz", hard=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.window_sec = float(window_size_ns)
        self.hard = hard
        self.trig_sum = self.add_weight(name="trig_sum", initializer="zeros")
        self.bg_count = self.add_weight(name="bg_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        rate_hz, total_trig_bg_events, total_bg_events = compute_trig_rate_from_pred(y_true, y_pred, window_size_sec=self.window_sec, hard=self.hard)

        self.trig_sum.assign_add(total_trig_bg_events)
        self.bg_count.assign_add(total_bg_events)

    def result(self):
        p_bg = self.trig_sum / (self.bg_count + 1e-6)
        return p_bg / self.window_sec

    def reset_state(self):
        self.trig_sum.assign(0.0)
        self.bg_count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'window_size_ns': self.window_sec,
            'hard': self.hard
        })
        return config