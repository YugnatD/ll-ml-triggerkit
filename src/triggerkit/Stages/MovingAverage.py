import numpy as np
import tensorflow as tf
from ctapipe.instrument import CameraGeometry
import astropy.units as u


@tf.keras.utils.register_keras_serializable(package="Trigger")
class TemporalMovingAverage(tf.keras.layers.Layer):
    """
    Temporal moving-average over the waveform axis.

    - Operates independently on each pixel (and channel).
    - Keeps the spatial layout intact; only smooths along time.
    - Output geometry is identical to the input geometry.
    """

    def __init__(self, input_geometry: CameraGeometry, window_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = int(window_size)
        self.input_geometry = input_geometry
        self.output_geometry = input_geometry
        # Kernel is built once input channel count is known.
        self._kernel = None

    def stage_name(self):
        return f"moving_average{self.window_size}"

    def stage_type(self):
        return "moving_average"

    def get_params(self):
        return {"window_size": self.window_size}

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def build(self, input_shape):
        # input_shape: (B, N, T) or (B, N, T, C)
        if len(input_shape) == 3:
            channels = 1
        elif len(input_shape) == 4:
            channels = int(input_shape[-1])
        else:
            raise ValueError("TemporalMovingAverage expects rank 3 or 4 inputs")

        dtype = tf.as_dtype(self.compute_dtype or tf.keras.backend.floatx())
        # Conv1D kernel: [time, in_ch, out_ch]; we keep channel_multiplier=1 (depthwise per channel)
        kernel = np.ones((self.window_size, channels, 1), dtype=dtype.as_numpy_dtype)
        kernel /= float(self.window_size)
        self._kernel = tf.constant(kernel, dtype=dtype)
        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        rank = x.shape.rank
        if rank == 3:
            x = x[..., tf.newaxis]  # (B, N, T, 1)
        elif rank != 4:
            raise ValueError("TemporalMovingAverage expects rank 3 or 4 inputs")

        x = tf.cast(x, self._kernel.dtype)

        # Collapse pixel axis into batch so we convolve only along time.
        b, n, t, c = tf.unstack(tf.shape(x))
        x_flat = tf.reshape(x, (b * n, t, c))  # (B*N, T, C)

        y_flat = tf.nn.conv1d(
            x_flat,
            self._kernel,
            stride=1,
            padding="SAME",
        )  # (B*N, T, C)

        y = tf.reshape(y_flat, (b, n, t, c))  # (B, N, T, C)

        if rank == 3:
            return tf.squeeze(y, axis=-1)
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "window_size": self.window_size,
        })

        # Serialize geometry explicitly for reloads
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
