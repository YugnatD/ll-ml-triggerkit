import tensorflow as tf
from ctapipe.instrument import CameraGeometry
import astropy.units as u

from triggerkit.Helper.Quantize import (
    fixed_point_quantize,
    parse_overflow_mode,
    parse_qspec,
    parse_quantization_mode,
)


@tf.keras.utils.register_keras_serializable(package="Trigger")
class Shift(tf.keras.layers.Layer):
    """
    Subtract a scalar value from the input tensor.

    Optional quantization can be enabled with:
        quantize_step = {
            "input": "UQ8.0",
            "shift_value": "UQ8.0",   # optional, defaults to input
            "output": "SQ8.0",        # optional, defaults to input
        }
    """

    _QUANTIZE_STEP_ORDER = (
        "input",
        "shift_value",
        "output",
    )
    _QUANTIZE_STEP_ALIASES = {
        "input": "input",
        "inputs": "input",
        "shift_value": "shift_value",
        "value": "shift_value",
        "shift": "shift_value",
        "constant": "shift_value",
        "output": "output",
        "out": "output",
        "result": "output",
    }

    def __init__(
        self,
        input_geometry: CameraGeometry,
        value: float = 0.0,
        quantize_step=None,
        overflow_mode="AP_WRAP",
        quantization_mode="AP_TRN",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_geometry = input_geometry
        self.output_geometry = input_geometry
        self.value = float(value)
        self.quantize_step_request = self._normalize_quantize_step(quantize_step)
        self.overflow_mode = parse_overflow_mode(overflow_mode)
        self.fixed_point_quantization_mode = parse_quantization_mode(quantization_mode)
        self.quantize_step = self._resolve_quantize_step()

    @staticmethod
    def _canonicalize_qspec(qspec):
        if qspec is None:
            return None
        return parse_qspec(qspec)["canonical_qspec"]

    @classmethod
    def _normalize_quantize_step(cls, quantize_step):
        if quantize_step is None:
            return None
        if not isinstance(quantize_step, dict):
            raise ValueError(
                "shift quantize_step must be a dict with keys 'input', "
                "'shift_value', and/or 'output'."
            )
        if len(quantize_step) == 0:
            return None

        normalized = {key: None for key in cls._QUANTIZE_STEP_ORDER}
        for raw_key, raw_value in quantize_step.items():
            canonical_key = cls._QUANTIZE_STEP_ALIASES.get(raw_key)
            if canonical_key is None:
                allowed = ", ".join(cls._QUANTIZE_STEP_ORDER)
                raise ValueError(
                    f"Unsupported shift quantize_step key '{raw_key}'. "
                    f"Allowed keys: {allowed}."
                )
            normalized[canonical_key] = cls._canonicalize_qspec(raw_value)
        return normalized

    def _resolve_quantize_step(self):
        requested = self.quantize_step_request or {}
        if not requested:
            return None

        input_qspec = requested.get("input")
        if input_qspec is None:
            raise ValueError("shift quantize_step['input'] must be set when quantization is enabled.")

        shift_value_qspec = requested.get("shift_value") or input_qspec
        output_qspec = requested.get("output") or input_qspec

        return {
            "input": input_qspec,
            "shift_value": shift_value_qspec,
            "output": output_qspec,
        }

    def stage_name(self):
        if self.quantize_step is None:
            return f"shift{self.value}"
        return (
            f"shift{self.value}"
            f"_in{self.quantize_step['input']}"
            f"_val{self.quantize_step['shift_value']}"
            f"_out{self.quantize_step['output']}"
        )

    def stage_type(self):
        return "shift"

    def get_params(self):
        params = {
            "value": self.value,
        }
        if self.quantize_step is not None:
            params["quantize_step"] = self.quantize_step
            params["overflow_mode"] = self.overflow_mode
            params["quantization_mode"] = self.fixed_point_quantization_mode
        return params

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        shift_value = tf.cast(self.value, tf.float32)

        if self.quantize_step is None:
            return x - shift_value

        x_q = fixed_point_quantize(
            x,
            qspec=self.quantize_step["input"],
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )
        shift_q = fixed_point_quantize(
            shift_value,
            qspec=self.quantize_step["shift_value"],
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )
        out = x_q - shift_q
        return fixed_point_quantize(
            out,
            qspec=self.quantize_step["output"],
            overflow_mode=self.overflow_mode,
            quantization_mode=self.fixed_point_quantization_mode,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "value": self.value,
            "quantize_step": self.quantize_step_request,
            "overflow_mode": self.overflow_mode,
            "quantization_mode": self.fixed_point_quantization_mode,
        })

        pix_x = self.input_geometry.pix_x.to_value(u.m).tolist()
        pix_y = self.input_geometry.pix_y.to_value(u.m).tolist()
        pix_area = self.input_geometry.pix_area.to_value(u.m**2).tolist()
        pix_id = self.input_geometry.pix_id.tolist()
        camera_name = self.input_geometry.name
        pix_type = self.input_geometry.pix_type.value
        cfg.update({
            "input_geometry": {
                "name": camera_name,
                "pix_id": pix_id,
                "pix_x": pix_x,
                "pix_y": pix_y,
                "pix_area": pix_area,
                "pix_type": pix_type,
            }
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        input_geometry_dict = config.pop("input_geometry")
        input_geometry = CameraGeometry(
            name=input_geometry_dict["name"],
            pix_id=input_geometry_dict["pix_id"],
            pix_x=input_geometry_dict["pix_x"] * u.m,
            pix_y=input_geometry_dict["pix_y"] * u.m,
            pix_area=input_geometry_dict["pix_area"] * u.m**2,
            pix_type=input_geometry_dict["pix_type"],
        )
        return cls(input_geometry=input_geometry, **config)
