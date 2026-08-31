
import tensorflow as tf
"""
Trainable differentiable threshold layer.
This layer learns a scalar threshold ``tau`` and converts an input score into a
soft trigger probability using a sigmoid gate:
    output = sigmoid((score - tau_used) * temp)
Temperature (`temp`) tuning:
- `temp` MULTIPLIES `(score - tau)` (not divides it).
- Larger `temp` -> steeper transition, more step-like gating.
- Smaller `temp` -> smoother probabilities and gentler gradients.
"""
from keras.saving import register_keras_serializable
from ctapipe.instrument import CameraGeometry
import astropy.units as u


@register_keras_serializable(package="Trigger")
class TrainableThreshold(tf.keras.layers.Layer):
    _COMPARISON_ALIASES = {
        ">": "gt",
        "gt": "gt",
        "strict": "gt",
        "score > tau": "gt",
        ">=": "ge",
        "ge": "ge",
        "inclusive": "ge",
        "score >= tau": "ge",
    }

    def __init__(
        self,
        input_geometry: CameraGeometry,
        init_tau=0.0,
        temp=0.1,
        quantize=None,
        binary_output=False,
        comparison="gt",
        inclusive=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_geometry = input_geometry
        self.output_geometry = input_geometry
        self.init_tau = init_tau
        self.temp = temp
        self.binary_output = bool(binary_output)
        self.comparison = self.normalize_comparison(comparison, inclusive=inclusive)
        # ``quantize`` is ignored and accepted only for backward compatibility
        # with older configs and call sites.

    @classmethod
    def normalize_comparison(cls, comparison="gt", inclusive=None):
        if inclusive is not None:
            return "ge" if bool(inclusive) else "gt"
        if comparison is None:
            return "gt"
        if not isinstance(comparison, str):
            raise ValueError("threshold comparison must be a string or None.")
        canonical = cls._COMPARISON_ALIASES.get(comparison.strip().lower())
        if canonical is None:
            raise ValueError("threshold comparison must be 'gt'/'>' or 'ge'/'>='.")
        return canonical
    
    def stage_name(self):
        # get actual tau value
        tau_value = float(self.tau.numpy().flatten()[0])
        suffix = ""
        if self.comparison == "ge":
            suffix += "ge"
        if self.binary_output:
            suffix += "bin"
        return f"threshold{tau_value}{suffix}"
    
    def stage_type(self):
        return "threshold"
    
    def get_params(self):
        tau_value = float(self.tau.numpy().flatten()[0])
        return {
            'threshold': tau_value,
            'binary': bool(self.binary_output),
            'comparison': self.comparison,
        }

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def set_trainable(self, trainable: bool):
        self.trainable = trainable
        self.tau._trainable = trainable

    def build(self, input_shape):
        self.tau = self.add_weight(
            name="tau",
            shape=(),
            trainable=True,
            initializer=tf.keras.initializers.Constant(self.init_tau),
        )
        super().build(input_shape)

    def call(self, score, training=None):
        score = tf.cast(score, tf.float32)
        self.last_score = score  # cache for potential use in loss/metrics
        pre_sigmoid = (score - self.tau) * self.temp
        soft_output = tf.sigmoid(pre_sigmoid)
        self.last_probability = soft_output
        if not self.binary_output:
            return soft_output

        hard_output = tf.cast(self.hard_decision(score), tf.float32)
        if training:
            # Straight-through estimator: hard values in the forward pass,
            # sigmoid gradients during backpropagation.
            return soft_output + tf.stop_gradient(hard_output - soft_output)
        return hard_output

    def hard_decision(self, score):
        score = tf.cast(score, tf.float32)
        if self.comparison == "ge":
            return score >= self.tau
        return score > self.tau
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "init_tau": self.init_tau,
            "temp": self.temp,
            "binary_output": self.binary_output,
            "comparison": self.comparison,
        })
        # cfg.update({"input_geometry": self.input_geometry.to_dict()}) # not serializable
        pix_x = self.input_geometry.pix_x.to_value(u.m).tolist()
        pix_y = self.input_geometry.pix_y.to_value(u.m).tolist()
        pix_area = self.input_geometry.pix_area.to_value(u.m**2).tolist()
        pix_id = self.input_geometry.pix_id.tolist()
        camera_name = self.input_geometry.name
        pix_type = self.input_geometry.pix_type.value
        cfg.update({"input_geometry": {
            "name": camera_name,
            "pix_id": pix_id,
            "pix_x": pix_x,
            "pix_y": pix_y,
            "pix_area": pix_area,
            "pix_type": pix_type
        }})
        return cfg
    
    @classmethod
    def from_config(cls, config):
        config.pop("center_idx", None)
        config.pop("quantize", None)
        if "binary" in config and "binary_output" not in config:
            config["binary_output"] = config.pop("binary")
        inclusive = config.pop("inclusive", None)
        if inclusive is not None and "comparison" not in config:
            config["comparison"] = cls.normalize_comparison(inclusive=inclusive)
        input_geometry_dict = config.pop("input_geometry")
        input_geometry = CameraGeometry(
            name=input_geometry_dict["name"],
            pix_id=input_geometry_dict["pix_id"],
            pix_x=input_geometry_dict["pix_x"] * u.m,
            pix_y=input_geometry_dict["pix_y"] * u.m,
            pix_area=input_geometry_dict["pix_area"] * u.m**2,
            pix_type=input_geometry_dict["pix_type"]
        )
        return cls(input_geometry=input_geometry, **config)
