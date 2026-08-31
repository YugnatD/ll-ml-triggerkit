import os
import numpy as np
import tensorflow as tf
from ctapipe.instrument import CameraGeometry
import astropy.units as u

from keras.saving import register_keras_serializable

@tf.keras.utils.register_keras_serializable(package="Trigger")
class DigitalSum(tf.keras.layers.Layer):
    def __init__(self, input_geometry: CameraGeometry, neighbors, mode="flower", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.input_geometry = input_geometry
        self.output_geometry = self.generate_output_geometry()

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
        return f"{self.stage_type()}{self.mode}"
    
    def stage_type(self): # *
        return "digital_sum"
    
    def get_params(self): # *
        return {
            'mode': self.mode
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
        return summed
        # return tf.squeeze(summed, axis=-1)            # (B, M, T)

    def get_config(self):
        config = super().get_config()
        config.update({
            "neighbors": self.neighbors,
            "mode": self.mode,
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
        with open("ConfigFile_SST1M/CTA_SST1M_Pixels_info_trigger.csv") as f:
            digi_sum_channel_list = [
                np.fromstring(line.strip(), sep=',', dtype=int)
                for line in f
                if line.strip()
            ]
            # now we have : [[0,2,3,1,6,7,4,9,10,8,16,17], [1,6,7,0,2,3,5,13,14,8,16,17,15,26,27], ...]
            # we need to recreate the patches : [[[0,2,3],[1,6,7],[4,9,10],[8,16,17], [1,6,7],[0,2,3],...], ...]
            digi_sum_channel_list = [
                np.array(sum_channel).reshape(-1, 3).tolist()
                for sum_channel in digi_sum_channel_list
            ]
            # we need to generate a list to compute the sum for each patch (summ of the 7 triplet sums)
            # so we need to assign each list to a patch so [0,2,3] -> 0, [1,6,7] -> 1, [4,9,10] -> 2, [8,16,17] -> 3
            # then we know that for the first sum we need to use triplet 0,1,2,3, for the second sum we need to use triplet 4,5,6,7, etc
            # start by making a list of the first channel of each triplet
            list_first_triplet_channel = []
            for patch_list in digi_sum_channel_list:
                list_first_triplet_channel.append(patch_list[0]) # ex [1,6,7]
            # print(list_first_triplet_channel)
            # now create a mapping from triplet index to patch index
            converted_digi_sum_list = []
            for patch_sum_channel in digi_sum_channel_list:
                triplet_indices = []
                for triplet in patch_sum_channel:
                    # search the index of triplet in list_first_triplet_channel
                    index = list_first_triplet_channel.index(triplet)
                    # print(f"Triplet: {triplet}, Index: {index}")
                    triplet_indices.append(index)
                converted_digi_sum_list.append(triplet_indices)
            digi_sum_channel_list = converted_digi_sum_list
            return digi_sum_channel_list
    else:
        raise ValueError(f"DigitalSumChannelList: camera_name {camera_name} not recognized")