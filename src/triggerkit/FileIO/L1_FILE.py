import numpy as np
from dataclasses import dataclass
import enum
import struct
import os
import sys
from ctapipe.instrument import CameraGeometry
from astropy import units as u
    

class L1CameraGeometry:
    num_pixels: np.uint32
    pix_id: np.ndarray  # shape (num_pixels,)  np.int32
    pix_x: np.ndarray  # shape (num_pixels,)  np.float32
    pix_y: np.ndarray  # shape (num_pixels,)  np.float32
    pix_area: np.ndarray  # shape (num_pixels,)  np.float32
    pix_type: np.uint8 # 0-> rectangular, 1-> hexagonal
    pix_rotation: np.float32 # in degree
    cam_rotation: np.float32 # in degree

@dataclass
class L1Header:
    magic: np.uint64
    version: np.uint32
    num_frames_per_datacube: np.uint32
    total_num_events: np.uint64
    pre_computed_sum: np.bool_
    pixel_resolution: np.uint8 # if 1: 8 bit, if 2: 16 bit, if 3: 32 bit, if 4: 64 bit
    camera_geometry: L1CameraGeometry


def generate_l1_header_bytes(magic:np.uint64, version:np.uint32, num_frames_per_datacube:np.uint32, total_num_events:np.uint64, pre_computed_sum:np.bool_, pixel_resolution:np.uint8, camera_geometry: CameraGeometry) -> bytes:
    header_bytes = struct.pack(
        '<Q I I Q ? B',
        magic,
        version,
        num_frames_per_datacube,
        total_num_events,
        pre_computed_sum,
        pixel_resolution
    )
    geometry_bytes = camera_geometry_to_bytes(camera_geometry)
    return header_bytes + geometry_bytes

def camera_geometry_to_bytes(geometry: CameraGeometry) -> bytes:
    num_pixels = geometry.n_pixels
    pix_id = geometry.pix_id.astype(np.int32)
    pix_x = geometry.pix_x.value.astype(np.float32)
    pix_y = geometry.pix_y.value.astype(np.float32)
    pix_area = geometry.pix_area.value.astype(np.float32)
    pix_type = np.uint8(1) if geometry.pix_type == 'hexagonal' else np.uint8(0)
    pix_rotation = np.float32(geometry.pix_rotation.value if hasattr(geometry.pix_rotation, 'value') else geometry.pix_rotation)
    cam_rotation = np.float32(geometry.cam_rotation.value if hasattr(geometry.cam_rotation, 'value') else geometry.cam_rotation)

    geometry_bytes = struct.pack('<I', num_pixels)
    geometry_bytes += pix_id.tobytes()
    geometry_bytes += pix_x.tobytes()
    geometry_bytes += pix_y.tobytes()
    geometry_bytes += pix_area.tobytes()
    geometry_bytes += struct.pack('<B f f', pix_type, pix_rotation, cam_rotation)

    return geometry_bytes


def event_stat_dict_to_bytes(event_stat: dict) -> bytes:
    return struct.pack(
        '<f i i i f f f f f f f f f f',
        np.float32(event_stat['energy']),
        np.int32(event_stat['n_pe']),
        np.int32(event_stat['n_pixels']),
        np.int32(event_stat['nphotons']),
        np.float32(event_stat['ev_time']),
        np.float32(event_stat['azimuth']),
        np.float32(event_stat['altitude']),
        np.float32(event_stat['h_first_int']),
        np.float32(event_stat['xmax']),
        np.float32(event_stat['hmax']),
        np.float32(event_stat['emax']),
        np.float32(event_stat['cmax']),
        np.float32(event_stat['xcore']),
        np.float32(event_stat['ycore'])
    )


class L1_FILE:
    def __init__(self, filepath,):
        self.filepath = filepath
        self.file = open(self.filepath, 'rb')

        # read header
        self.header = self._read_header()

        # generate a CameraGeometry from the L1 header
        pix_x = self.header.camera_geometry.pix_x
        pix_y = self.header.camera_geometry.pix_y
        pix_id = self.header.camera_geometry.pix_id
        pix_area = self.header.camera_geometry.pix_area
        pix_type = 'hexagonal' if self.header.camera_geometry.pix_type == 0 else 'rectangular'
        cam_rotation = self.header.camera_geometry.cam_rotation
        pix_rotation = self.header.camera_geometry.pix_rotation
        self.geom = CameraGeometry(
            name="L1 Camera",
            pix_x=pix_x,
            pix_y=pix_y,
            pix_id=pix_id,
            pix_area=pix_area,
            cam_rotation=cam_rotation,
            pix_rotation=pix_rotation,
            pix_type=pix_type
        )
        self.camera_name = "Digicam"
        self.tel_id = 1
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.file:
            try:
                event = self.read_event()
                return event
            except Exception as e:
                self.file.close()
                raise StopIteration
        else:
            raise StopIteration
    
    def _read_header(self) -> L1Header:
        # read magic number
        magic_bytes = self.file.read(8)
        magic = struct.unpack('<Q', magic_bytes)[0]

        # read version
        version_bytes = self.file.read(4)
        version = struct.unpack('<I', version_bytes)[0]

        # read num_frames_per_datacube
        num_frames_per_datacube_bytes = self.file.read(4)
        num_frames_per_datacube = struct.unpack('<I', num_frames_per_datacube_bytes)[0]

        # read total_num_events
        total_num_events_bytes = self.file.read(8)
        total_num_events = struct.unpack('<Q', total_num_events_bytes)[0]

        # read pre_computed_sum
        pre_computed_sum_bytes = self.file.read(1)
        pre_computed_sum = struct.unpack('<?', pre_computed_sum_bytes)[0]

        # read pixel_resolution
        pixel_resolution_bytes = self.file.read(1)
        pixel_resolution = struct.unpack('<B', pixel_resolution_bytes)[0]

        # read camera geometry
        camera_geometry = self._read_camera_geometry()
        

        return L1Header(
            magic=magic,
            version=version,
            num_frames_per_datacube=num_frames_per_datacube,
            total_num_events=total_num_events,
            pre_computed_sum=pre_computed_sum,
            pixel_resolution=pixel_resolution,
            camera_geometry=camera_geometry
        )
    
    def _read_camera_geometry(self) -> L1CameraGeometry:
        # read num_pixels
        num_pixels_bytes = self.file.read(4)
        num_pixels = struct.unpack('<I', num_pixels_bytes)[0]

        # read pix_id
        pix_id_bytes = self.file.read(num_pixels * 4)
        pix_id = np.frombuffer(pix_id_bytes, dtype=np.int32)

        # read pix_x
        pix_x_bytes = self.file.read(num_pixels * 4)
        pix_x = np.frombuffer(pix_x_bytes, dtype=np.float32)
        # convert to meters
        pix_x = pix_x * u.meter

        # read pix_y
        pix_y_bytes = self.file.read(num_pixels * 4)
        pix_y = np.frombuffer(pix_y_bytes, dtype=np.float32)
        # convert to meters
        pix_y = pix_y * u.meter

        # read pix_area
        pix_area_bytes = self.file.read(num_pixels * 4)
        pix_area = np.frombuffer(pix_area_bytes, dtype=np.float32)
        # convert to m2
        pix_area = pix_area * u.meter**2

        # read pix_type
        pix_type_bytes = self.file.read(1)
        pix_type = struct.unpack('<B', pix_type_bytes)[0]

        # read pix_rotation
        pix_rotation_bytes = self.file.read(4)
        pix_rotation = struct.unpack('<f', pix_rotation_bytes)[0]
        # convert to degree
        pix_rotation = pix_rotation * u.degree

        # read cam_rotation
        cam_rotation_bytes = self.file.read(4)
        cam_rotation = struct.unpack('<f', cam_rotation_bytes)[0]
        cam_rotation = cam_rotation * u.degree

        geometry = L1CameraGeometry()
        geometry.num_pixels = num_pixels
        geometry.pix_id = pix_id
        geometry.pix_x = pix_x
        geometry.pix_y = pix_y
        geometry.pix_area = pix_area
        geometry.pix_type = pix_type
        geometry.pix_rotation = pix_rotation
        geometry.cam_rotation = cam_rotation
        return geometry
    
    def read_event(self):
        num_pixels = self.header.camera_geometry.num_pixels
        num_frames = self.header.num_frames_per_datacube

        # read datacube
        if self.header.pixel_resolution == 2:
            datacube_bytes = self.file.read(num_pixels * num_frames * 2)
            datacube = np.frombuffer(datacube_bytes, dtype=np.uint16).reshape((num_pixels, num_frames))
        else:
            raise ValueError("Unsupported pixel resolution. Only 16 bit (2) is supported.")

        # read event stats
        event_stats_bytes = self.file.read(56)
        event_stats = struct.unpack('<f i i i f f f f f f f f f f', event_stats_bytes)
        event_stat_dict = {
            'energy': event_stats[0],
            'n_pe': event_stats[1],
            'n_pixels': event_stats[2],
            'nphotons': event_stats[3],
            'ev_time': event_stats[4],
            'azimuth': event_stats[5],
            'altitude': event_stats[6],
            'h_first_int': event_stats[7],
            'xmax': event_stats[8],
            'hmax': event_stats[9],
            'emax': event_stats[10],
            'cmax': event_stats[11],
            'xcore': event_stats[12],
            'ycore': event_stats[13],
        }
        return datacube, event_stat_dict

    def close(self):
        self.file.close()

# try iterating over the file NSB/biascurve/biascurve_run46_tel1_nsb0.270_dsum260.l1
if __name__ == "__main__":
    filepath = "../simtelFileData/NSB/biascurve/biascurve_run46_tel1_nsb0.270_dsum260.l1"
    l1_file = L1_FILE(filepath)
    print("L1 Header:")
    print(l1_file.header)
    print("Reading first 5 events:")
    for i in range(5):
        event = l1_file.read_event()
        print(f"Event {i}: energy={event.energy}, n_pe={event.n_pe}, n_pixels={event.n_pixels}, nphotons={event.nphotons}")
        if i == 0:
            geom = l1_file.geom
            datacube = event.datacube
            from ctapipe.visualization import CameraDisplay
            display = CameraDisplay(geom)
            display.image = np.sum(datacube, axis=1)
            display.show()
            import matplotlib.pyplot as plt
            plt.title(f"Event 0 integrated signal")
            plt.show()
