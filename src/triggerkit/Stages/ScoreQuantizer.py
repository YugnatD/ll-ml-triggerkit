import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Trigger")
class ScoreQuantizer(tf.keras.layers.Layer):
    """
    Quantize a score tensor into integer-like codes using explicit edges.

    The convention is fixed:
        code = count(score >= edge for edge in edges)

    Example with edges [32, 98, 164]:
        score < 32          -> 0
        32 <= score < 98    -> 1
        98 <= score < 164   -> 2
        score >= 164        -> 3
    """

    def __init__(self, edges, input_geometry=None, **kwargs):
        super().__init__(**kwargs)
        self.input_geometry = input_geometry
        self.output_geometry = input_geometry
        self.edges = self._normalize_edges(edges)

    @staticmethod
    def _normalize_edges(edges):
        if edges is None:
            raise ValueError("score_quantizer requires a non-empty 'edges' list.")
        if isinstance(edges, (str, bytes)):
            raise ValueError("score_quantizer edges must be a list of numbers, not a string.")

        try:
            normalized = [float(edge) for edge in edges]
        except TypeError as exc:
            raise ValueError("score_quantizer edges must be an iterable of numbers.") from exc
        except ValueError as exc:
            raise ValueError("score_quantizer edges must contain only numbers.") from exc

        if not normalized:
            raise ValueError("score_quantizer requires at least one edge.")

        for previous, current in zip(normalized, normalized[1:]):
            if not current > previous:
                raise ValueError("score_quantizer edges must be strictly increasing.")

        return normalized

    @staticmethod
    def _format_edge(edge):
        return f"{edge:g}".replace("-", "m").replace(".", "p")

    def stage_name(self):
        return "scorequantizer" + "_".join(self._format_edge(edge) for edge in self.edges)

    def stage_type(self):
        return "score_quantizer"

    def get_params(self):
        return {
            "edges": list(self.edges),
        }

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def call(self, inputs):
        score = tf.cast(inputs, tf.float32)
        edges = tf.constant(self.edges, dtype=tf.float32)
        crossed_edges = tf.cast(tf.expand_dims(score, axis=-1) >= edges, tf.float32)
        return tf.reduce_sum(crossed_edges, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"edges": list(self.edges)})
        return cfg
