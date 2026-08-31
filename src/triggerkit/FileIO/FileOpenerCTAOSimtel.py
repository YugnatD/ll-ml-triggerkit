from __future__ import annotations

import numpy as np
from astropy import units as u
from ctapipe.calib import CameraCalibrator
from ctapipe.io import EventSource, SimTelEventSource

TEL_ID = 1


class FileOpenerCTAOSimtel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.i_readed_events = 0
        self._file = None
        self._iterator = None

        self.src = SimTelEventSource(filepath)
        self.camera_name = self.src.subarray.tel[TEL_ID].camera.geometry.name
        self.geom = (
            self.src.subarray.tel[TEL_ID].camera.geometry
        )  # CameraGeometry class
        self.tel_ids = self.src.subarray.tel_ids
        self.tel_positions = {}
        for tel_id in self.tel_ids:
            self.tel_positions[int(tel_id)] = self._extract_tel_position_xyz(self.src.subarray, int(tel_id))

    @staticmethod
    def _extract_tel_position_xyz(subarray, tel_id: int) -> tuple[float, float, float]:
        try:
            pos = subarray.positions[int(tel_id)]
            pos_xyz = np.asarray(u.Quantity(pos, copy=False).to_value(u.m), dtype=np.float32).reshape(-1)
            if pos_xyz.size >= 3:
                return float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])
        except Exception:
            pass
        return -1.0, -1.0, -1.0

    @staticmethod
    def _normalize_waveform_no_channel(waveform: np.ndarray) -> np.ndarray:
        """
        Return waveform as (n_pix, n_samples).
        Accepts input shaped (n_chan, n_pix, n_samples) or (n_pix, n_samples).
        """
        wf = np.asarray(waveform)
        if wf.ndim == 3:
            return wf[0]
        if wf.ndim == 2:
            return wf
        raise ValueError(
            f"Unexpected waveform ndim={wf.ndim}, shape={wf.shape}. "
            "Expected (n_pix, n_samples) or (n_chan, n_pix, n_samples)."
        )

    def __enter__(self):
        """Optional: allow use as a context manager."""
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Optional: clean up resources when exiting context."""
        self._close()

    def _open(self):
        """Open file and prepare iterator."""
        if self._file is None:
            # self._file = SimTelFile(self.filepath)
            # self._iterator = iter(self._file)
            self._file = EventSource(self.filepath)
            self._iterator = iter(self._file)
            self.calibrator = CameraCalibrator(self._file.subarray)

    def _close(self):
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._iterator = None

    def __iter__(self):
        self._open()
        return self

    def __next__(self):
        self.i_readed_events += 1

        # check if the iterator is initialized
        if self._iterator is None:
            self._open()

        try:
            event = next(self._iterator)

            # calibrate once per event (instead of once per telescope)
            self.calibrator(event)

            telescope_list = []
            wf0_list = []
            wf1_list = []
            dl0_list = []
            dl1_list = []
            true_image_list = []
            pedestal_per_sample_list = []
            event_stat_list = []

            # cache references to avoid repeated attribute lookups
            event_r0_tel = event.r0.tel
            event_r1_tel = event.r1.tel
            event_dl0_tel = event.dl0.tel
            event_dl1_tel = event.dl1.tel
            event_mon_tel = event.mon.tel
            event_sim_tel = event.simulation.tel
            shower = event.simulation.shower
            trigger_time_unix = event.trigger.time.to_value("unix")
            event_id = event.index.event_id

            for tel_id in self.tel_ids:
                # waveforms
                if tel_id not in event_r0_tel:
                    continue  # skip if no data for this telescope
                tel_pos_x, tel_pos_y, tel_pos_z = self.tel_positions.get(int(tel_id), (-1.0, -1.0, -1.0))

                r0 = self._normalize_waveform_no_channel(
                    event_r0_tel[tel_id].waveform
                )
                r1 = self._normalize_waveform_no_channel(
                    event_r1_tel[tel_id].waveform
                )
                dl0 = self._normalize_waveform_no_channel(
                    event_dl0_tel[tel_id].waveform
                )
                # dl1 is not waveform but image
                dl1 = event_dl1_tel[tel_id].image

                # true_image 
                sim_image = event_sim_tel[tel_id].true_image

                wf0_list.append(r0)
                wf1_list.append(r1)
                dl0_list.append(dl0)
                dl1_list.append(dl1)
                true_image_list.append(sim_image)
                telescope_list.append(tel_id)

                # get the pedestal data
                pedestal_per_sample = (
                    event_mon_tel[tel_id]
                    .calibration.pedestal_per_sample[0]
                    .astype(np.float32)
                )
                pedestal_per_sample_list.append(pedestal_per_sample)

                # event stats
                stat_event = {
                    "event_id": event_id,
                    # for n_pe use true_image_sum
                    "n_pe": int(event_sim_tel[tel_id].true_image_sum)
                    if tel_id in event_sim_tel
                    else -1,
                    "n_pixels": -1,
                    "nphotons": -1,
                    # absolute trigger time in unix seconds
                    "ev_time": trigger_time_unix,
                    # shower properties (guard with shower is not None)
                    "energy": shower.energy.to_value(u.TeV)
                    if shower is not None
                    else -1.0,
                    "azimuth": shower.az.to_value(u.rad)
                    if shower is not None
                    else -1.0,
                    "altitude": shower.alt.to_value(u.rad)
                    if shower is not None
                    else -1.0,
                    "h_first_int": shower.h_first_int.to_value(u.m)
                    if shower is not None
                    else -1.0,
                    "xmax": shower.x_max.to_value(u.g / u.cm**2)
                    if shower is not None
                    else -1.0,
                    "hmax": -1.0,
                    "emax": -1.0,
                    "cmax": -1.0,
                    "xcore": shower.core_x.to_value(u.m)
                    if shower is not None
                    else -1.0,
                    "ycore": shower.core_y.to_value(u.m)
                    if shower is not None
                    else -1.0,
                    "telescope": tel_id,
                    "tel_pos_x": tel_pos_x,
                    "tel_pos_y": tel_pos_y,
                    "tel_pos_z": tel_pos_z,
                }

                event_stat_list.append(stat_event)

            return (
                list(telescope_list),
                wf0_list,
                wf1_list,
                dl0_list,
                dl1_list,
                true_image_list,
                pedestal_per_sample_list,
                event_stat_list,
                self.i_readed_events - 1,
            )
        except StopIteration:
            self._close()
            raise
