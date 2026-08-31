import numpy as np
from ctapipe.instrument import CameraGeometry
import astropy.units as u
import tensorflow as tf
from keras.saving import register_keras_serializable

@tf.keras.utils.register_keras_serializable(package="Trigger")
class FADC(tf.keras.layers.Layer):
    """
    Inputs:
      - wf:       (B, N, T) or (B, N, T, 1)   waveforms
      - baseline: (B, N) or (N,)              pedestal per channel (known in advance)
    Output:
      - (B, M, T) where M = number of patches in digi_sum_channel_list
        (one output per patch: the sum of the FIRST triplet in that patch)
    """
    def __init__(self, input_geometry: CameraGeometry, digi_sum_channel_list, **kwargs):
        super().__init__(**kwargs)
        self.input_geometry = input_geometry

        # Keep python copy for serialization
        self.digi_sum_channel_list = [
            [list(trip) for trip in patch] for patch in digi_sum_channel_list
        ]

        self.output_geometry = self.generate_output_geometry()

        # Only FIRST triplet per patch, as in your numpy code: triplet_channels[0]
        first_triplets = [patch[0] for patch in self.digi_sum_channel_list]  # shape (M, 3)
        self.triplet_idx = tf.constant(first_triplets, dtype=tf.int32)
    
    def stage_name(self):
        return f"fadc"
    
    def stage_type(self):
        return "fadc"

    def get_params(self):
        return {}

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def generate_output_geometry(self):
        if self.input_geometry.name != "DigiCam": # SST-1M equivalent name
            raise ValueError("FADC only supports DigiCam geometry.")
        data = np.loadtxt(
            "ConfigFile_SST1M/CTA_SST1M_Pixels_info_shrink.csv",
            delimiter=",",
            skiprows=1,
            usecols=(1, 2),
            dtype=float,
        )
        pixel_x, pixel_y = data[:, 0], data[:, 1]
        output_geometry = CameraGeometry(
            name=self.input_geometry.name,
            pix_id=np.arange(len(pixel_x)),
            pix_x=pixel_x * u.m,
            pix_y=pixel_y * u.m,
            pix_area= np.full(len(pixel_x), fill_value=15) * u.m**2,
            pix_type='hexagonal'
        )
        return output_geometry

    def call(self, inputs):
        # Keras will pass a list/tuple of inputs
        wf, baseline = inputs

        # wf: (B,N,T) or (B,N,T,1)
        if wf.shape.rank == 3:
            wf = wf[..., tf.newaxis]  # (B,N,T,1)

        # Cast to int32 for safe integer arithmetic
        wf_i = tf.cast(wf, tf.int32)  # (B,N,T,1)

        # baseline: (B,N) or (N,)
        b = tf.cast(baseline, tf.int32)
        if b.shape.rank == 1:
            # (N,) -> (1,N,1,1) broadcast over batch and time
            b = b[tf.newaxis, :, tf.newaxis, tf.newaxis]
        else:
            # (B,N) -> (B,N,1,1)
            b = b[:, :, tf.newaxis, tf.newaxis]

        # Integer baseline subtract + clip to [0, 4095]
        wf_sub = wf_i - b
        wf_sub = tf.clip_by_value(wf_sub, 0, 4095)  # (B,N,T,1)

        # Gather triplet channels for each patch: (B, M, 3, T, 1)
        gathered = tf.gather(wf_sub, self.triplet_idx, axis=1)

        # Sum the 3 channels: (B, M, T, 1) -> (B, M, T)
        triplet_sum = tf.reduce_sum(gathered, axis=2)
        triplet_sum = tf.squeeze(triplet_sum, axis=-1)

        # Clip sum to [0, 255]
        triplet_sum = tf.clip_by_value(triplet_sum, 0, 255)

        # Keep integer output (close to FPGA behavior). If you want float, cast here.
        return tf.cast(triplet_sum, tf.int32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"digi_sum_channel_list": self.digi_sum_channel_list})
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
    
def FADCList():
    digi_sum_channel_list = []
    with open("ConfigFile_SST1M/CTA_SST1M_Pixels_info_trigger.csv") as f:
        for line in f:
            line = line.strip()
            if line:
                sum_channel = np.fromstring(line, sep=',', dtype=int)
                sum_patch = sum_channel.reshape(-1, 3).tolist()
                digi_sum_channel_list.append(sum_patch)
    return digi_sum_channel_list
