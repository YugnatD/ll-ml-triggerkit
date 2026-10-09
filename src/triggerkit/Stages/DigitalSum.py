import os
import tensorflow as tf
from ctapipe.instrument import CameraGeometry
import astropy.units as u

from keras.saving import register_keras_serializable

from triggerkit.camera import sst1m

@tf.keras.utils.register_keras_serializable(package="Trigger")
class DigitalSum(tf.keras.layers.Layer):
    def __init__(self, input_geometry: CameraGeometry, neighbors, mode="patch7", threshold_flower=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.input_geometry = input_geometry
        # Optional threshold applied in this same stage (sum > threshold_flower).
        # The name comes from the former LST "flower" mode; it is kept because
        # it is part of the saved model configs and of the stats file names.
        self.threshold_flower = threshold_flower

        # Keep a pure-Python structure for serialization
        if isinstance(neighbors, (list, tuple)):
            self.neighbors = [list(n) for n in neighbors]
        else:
            # e.g., numpy array
            self.neighbors = neighbors.tolist()
        # make sure every list in neighbors has the same length by padding with -1
        max_length = max(len(n) for n in self.neighbors)
        for n in self.neighbors:
            while len(n) < max_length:
                n.append(-1)

        self.output_geometry = self.generate_output_geometry()

        # Works if all rows have same length (dense)
        self.neigh = tf.constant(self.neighbors, dtype=tf.int32)

    def generate_output_geometry(self):
        if self.input_geometry.name == "DigiCam" or self.input_geometry.name == "DigiCam_R0Alpha":
            if self.mode != "patch7":
                raise ValueError(f"DigitalSum: mode {self.mode} not recognized for camera {self.input_geometry.name}")
            return self.input_geometry
        else:
            raise ValueError(f"DigitalSum: camera {self.input_geometry.name} not recognized")
        
    def stage_name(self): #
        if self.threshold_flower is None:
            return f"{self.stage_type()}{self.mode}"
        return f"{self.stage_type()}{self.mode}{self.threshold_flower}"
    
    def stage_type(self): # *
        return "digital_sum"
    
    def get_params(self): # *
        return {
            'mode': self.mode,
            'threshold_flower': self.threshold_flower,
        }
    
    def get_stages(self): # *
        return (self.stage_type(), self.get_params())
        

    def call(self, inputs):
        x = inputs
        if x.shape.rank == 3:
            x = x[..., tf.newaxis]  # (B, N, T, 1)

        # digital_sum_result = np.array([np.sum(wf[self.digi_sum_channel_list[i]], axis=0) for i in np.arange(0, len(self.digi_sum_channel_list))])

        # neigh: (M, K) with -1 used as padding
        valid = tf.not_equal(self.neigh, -1)                  # (M, K) bool
        safe  = tf.where(valid, self.neigh, 0)                # (M, K) int, replace -1 -> 0

        gathered = tf.gather(x, safe, axis=1)                 # (B, M, K, T, C) where C=1

        # expand mask to (1, M, K, 1, 1) so it broadcasts over B, T, C
        mask = valid[tf.newaxis, :, :, tf.newaxis, tf.newaxis]

        # zero out invalid entries
        gathered = tf.where(mask, gathered, tf.zeros_like(gathered))

        summed = tf.reduce_sum(gathered, axis=2)              # (B, M, T, C)
        if self.threshold_flower is not None:
            return tf.cast(summed > self.threshold_flower, summed.dtype)
        return summed
        # return tf.squeeze(summed, axis=-1)            # (B, M, T)

    def get_config(self):
        config = super().get_config()
        config.update({
            "neighbors": self.neighbors,
            "mode": self.mode,
            "threshold_flower": self.threshold_flower,
        })
        # config.update({"input_geometry": self.input_geometry.to_dict()})
        pix_x = self.input_geometry.pix_x.to_value(u.m).tolist()
        pix_y = self.input_geometry.pix_y.to_value(u.m).tolist()
        pix_area = self.input_geometry.pix_area.to_value(u.m**2).tolist()
        pix_id = self.input_geometry.pix_id.tolist()
        camera_name = self.input_geometry.name
        pix_type = self.input_geometry.pix_type.value
        config.update({"input_geometry": {
            "name": camera_name,
            "pix_id": pix_id,
            "pix_x": pix_x,
            "pix_y": pix_y,
            "pix_area": pix_area,
            "pix_type": pix_type
        }})
        return config

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


    
def DigitalSumChannelList(camera_name="DigiCam"):
    if camera_name == "DigiCam" or camera_name == "DigiCam_R0Alpha":
        # The patches of the patch7 cluster of each patch, the patch itself first
        # (camera_config.cfg, see triggerkit.camera.sst1m).
        return sst1m.patch7_clusters()
    else:
        raise ValueError(f"DigitalSumChannelList: camera_name {camera_name} not recognized")