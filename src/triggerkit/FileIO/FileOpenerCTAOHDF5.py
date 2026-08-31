from __future__ import annotations

import os
import numpy as np
import h5py
import hdf5plugin  # do not remove, needed to read some h5 files compressed with specific plugins
from astropy import units as u
from ctapipe.instrument import CameraGeometry

# Ensure HDF5 can find filter plugins bundled with hdf5plugin.
_plugin_path = os.environ.get("HDF5_PLUGIN_PATH")
if not _plugin_path or not os.path.isdir(_plugin_path):
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
try:
    hdf5plugin.register()
except Exception:
    # If registration fails, we'll attempt to read anyway; errors will surface on file access.
    pass


class FileOpenerCTAOHDF5:
    def __init__(self, filepath):
        self.filepath = filepath
        self.i_readed_events = 0
        self._file = None
        self._iterator = None

        self.src = h5py.File(filepath, "r")
        # for debug : print the keys of the h5 file recursively
        # def print_h5_keys(name, obj):
        #     print(name, "->", obj)
        #     # if a dataset, print its shape and type
        #     if isinstance(obj, h5py.Dataset):
        #         print(f"    shape: {obj.shape}, type: {obj.dtype}")
        # self.src.visititems(print_h5_keys)
        geom_group = self.src["configuration/instrument/telescope/camera/geometry_0"]
        subarray = self.src["configuration/instrument/subarray"]
        self.camera_name = subarray["layout"]["camera_name"][0].decode("utf-8")
        self.tel_ids = subarray["layout"]["tel_id"]
        pix_id = geom_group["pix_id"]
        pix_x = geom_group["pix_x"]
        pix_y = geom_group["pix_y"]
        pix_area = geom_group["pix_area"]
        # pix_rotation = geom_group["pix_rotation"]
        self.geom = CameraGeometry(
            name=self.camera_name,
            pix_x=pix_x * u.m,
            pix_y=pix_y * u.m,
            pix_id=pix_id,
            pix_area=pix_area * u.m**2,
            cam_rotation=0.0 * u.deg,
            # pix_rotation=8.213 * u.deg,  # from the .dat camera file (usefulle for LST)
            # pix_rotation=pix_rotation * u.deg,  # from the h5 file
            pix_rotation = 0.0 * u.deg,  # for now, set to 0 to be able to use the same weights for LST and SST-1M, will need to be updated later when we will have the correct pix_rotation for each telescope
            pix_type="hexagonal",
        )
        # configuration/instrument/telescope/optics
        try:
            self.lst_tel = [
                key for key in self.src["r0/event/telescope"].keys()
            ]  # [tel_001, tel_002, tel_003, tel_004]
        except KeyError:
            self.lst_tel = [
                key for key in self.src["r1/event/telescope"].keys()
            ]  # [tel_001, tel_002, tel_003, tel_004]
        self.lst_tel_ids = subarray["layout"]["tel_id"].tolist()  # [1, 2, 3, 4]
        self.tel_positions = {}
        try:
            pos_x = subarray["layout"]["pos_x"]
            pos_y = subarray["layout"]["pos_y"]
            pos_z = subarray["layout"]["pos_z"]
            for tel_id, x, y, z in zip(self.lst_tel_ids, pos_x, pos_y, pos_z):
                self.tel_positions[int(tel_id)] = (float(x), float(y), float(z))
        except Exception:
            for tel_id in self.lst_tel_ids:
                self.tel_positions[int(tel_id)] = (-1.0, -1.0, -1.0)
        # access the members of the subarray
        # for key in subarray.keys():
        #     print(key, "->", subarray[key])

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
            # Reuse the handle opened in __init__ (self.src) instead of opening
            # a second h5py.File on the same path -- everything below (calib
            # tables, h5_event_iterator) already reads through self.src; the
            # separate self._file was only ever used to close, so this fixes a
            # dangling-handle leak (self.src was never closed by _close()).
            self._file = self.src
            self.calib_tables = {}
            for tel_name in self.lst_tel:
                try:
                    self.calib_tables[tel_name] = self.src[
                        f"calibration/event/telescope/{tel_name}"
                    ]
                except KeyError:
                    self.calib_tables[tel_name] = None

            def h5_event_iterator():
                """
                Iterator over events in a CTA HDF5 file.

                Synchronizes telescopes by event_id and yields per-event
                lists of waveforms/images plus a list of event stats dicts.
                """
                # lazy-build an index from event_id -> shower row
                if not hasattr(self, "_shower_index"):
                    shower_ds = self.src["simulation/event/subarray/shower"]
                    # create a dict: event_id -> full row
                    self._shower_index = {
                        int(row["event_id"]): row for row in shower_ds
                    }

                # reset event index for each telescope
                self.current_event_index = [0] * len(self.lst_tel)

                while True:
                    tel_ids = []
                    wf0_list = []
                    wf1_list = []
                    dl0_list = []
                    dl1_list = []
                    true_image_list = []
                    pedestal_per_sample_list = []
                    event_stat_list = []
                    event_ids = []

                    # for each telescope, get the current event id
                    for tel_idx, tel_name in enumerate(self.lst_tel):
                        # HDF5 LST dataset contains only R0, and DL1
                        # HDF5 SST-1M dataset contains only R1, DL1
                        try:
                            tel_group_r0 = self.src[
                                f"r0/event/telescope/{tel_name}"
                            ]
                        except KeyError:
                            tel_group_r0 = None
                        try:
                            tel_group_r1 = self.src[
                                f"r1/event/telescope/{tel_name}"
                            ]
                        except KeyError:
                            tel_group_r1 = None

                        if tel_group_r0 is None and tel_group_r1 is None:
                            continue  # no data for this telescope

                        if tel_group_r0 is not None:
                            tel_group = tel_group_r0
                        else:
                            if tel_group_r1 is not None:
                                tel_group = tel_group_r1
                            else:
                                raise ValueError(
                                    f"No R0 or R1 data found for telescope {tel_name}"
                                )

                        if self.current_event_index[tel_idx] >= len(tel_group):
                            continue  # no more events in this telescope

                        event = tel_group[self.current_event_index[tel_idx]]
                        event_id = int(event["event_id"])
                        event_ids.append(event_id)

                    if not event_ids:
                        break  # no more events in any telescope

                    # find the minimum event id to synchronize telescopes
                    min_event_id = min(event_ids)

                    # look up shower-level (truth) information for this event
                    shower_row = self._shower_index.get(int(min_event_id), None)

                    # default values if no shower info is found
                    true_energy = -1.0
                    true_alt = -1.0
                    true_az = -1.0
                    true_h_first_int = -1.0
                    true_xmax = -1.0
                    true_core_x = -1.0
                    true_core_y = -1.0

                    if shower_row is not None:
                        true_energy = float(shower_row["true_energy"])
                        true_alt = float(shower_row["true_alt"])
                        true_az = float(shower_row["true_az"])
                        true_h_first_int = float(shower_row["true_h_first_int"])
                        true_xmax = float(shower_row["true_x_max"])
                        true_core_x = float(shower_row["true_core_x"])
                        true_core_y = float(shower_row["true_core_y"])

                    for tel_idx, tel_name in enumerate(self.lst_tel):
                        tel_id = self.lst_tel_ids[tel_idx]
                        tel_pos_x, tel_pos_y, tel_pos_z = self.tel_positions.get(int(tel_id), (-1.0, -1.0, -1.0))
                        # get the real telescope id
                        tel_ids.append(tel_id)

                        # try getting r0 group first
                        tel_group_r0 = None
                        tel_group_r1 = None
                        try:
                            tel_group_r0 = self.src[
                                f"r0/event/telescope/{tel_name}"
                            ]
                        except KeyError:
                            # it's probably a SST-1M file with r1 instead of r0
                            pass
                        try:
                            tel_group_r1 = self.src[
                                f"r1/event/telescope/{tel_name}"
                            ]
                        except KeyError:
                            pass

                        # try getting dl0 and dl1 groups
                        try:
                            tel_group_dl0 = self.src[
                                f"dl0/event/telescope/{tel_name}"
                            ]
                        except KeyError:
                            tel_group_dl0 = None
                        try:
                            tel_group_dl1 = self.src[
                                f"dl1/event/telescope/images/{tel_name}"
                            ]
                        except KeyError:
                            tel_group_dl1 = None

                        # get the right tel_group
                        if tel_group_r0 is not None:
                            tel_group = tel_group_r0
                        else:
                            if tel_group_r1 is not None:
                                tel_group = tel_group_r1
                            else:
                                raise ValueError(
                                    f"No R0 or R1 data found for telescope {tel_name}"
                                )

                        sim_image_group = self.src[
                            f"simulation/event/telescope/images/{tel_name}"
                        ]
                        # n_pe is true_image_sum
                        if self.current_event_index[tel_idx] >= len(tel_group):
                            continue  # no more events in this telescope

                        event = tel_group[self.current_event_index[tel_idx]]
                        event_id = int(event["event_id"])

                        if event_id == min_event_id:
                            # try to get the different waveforms
                            if tel_group_r0 is not None:
                                waveform_r0 = self._normalize_waveform_no_channel(
                                    event["waveform"]
                                ).astype(np.float32)  # (n_pix, n_samples)
                                wf0_list.append(waveform_r0)
                            if tel_group_r1 is not None:
                                waveform_r1 = self._normalize_waveform_no_channel(
                                    event["waveform"]
                                ).astype(np.float32)  # (n_pix, n_samples)
                                wf1_list.append(waveform_r1)
                            if tel_group_dl0 is not None:
                                event_dl0 = tel_group_dl0[
                                    self.current_event_index[tel_idx]
                                ]
                                waveform_dl0 = self._normalize_waveform_no_channel(
                                    event_dl0["waveform"]
                                ).astype(np.float32)
                                dl0_list.append(waveform_dl0)
                            event_dl1 = None
                            if tel_group_dl1 is not None:
                                event_dl1 = tel_group_dl1[
                                    self.current_event_index[tel_idx]
                                ]
                                image_dl1 = event_dl1["image"].astype(np.float32)
                                dl1_list.append(image_dl1)
                            calib_ds = self.calib_tables.get(tel_name, None)
                            # get the true_image for this event and telescope
                            true_image_list.append(sim_image_group[self.current_event_index[tel_idx]]["true_image"].astype(np.float32))


                            if calib_ds is not None:
                                # assuming same ordering as r0/r1: one row per event
                                calib_row = calib_ds[self.current_event_index[tel_idx]]

                                # optional sanity check:
                                if int(calib_row["event_id"]) != event_id:
                                    raise RuntimeError("Calibration/event mismatch")

                                # WaveformCalibrationContainer fields (no prefix in default ctapipe):
                                #   pedestal_per_sample: (n_chan, n_pix) or (n_chan, n_pix, n_samples)
                                #   dc_to_pe: (n_chan, n_pix)
                                # ped_all_chan = calib_row["pedestal_per_sample"]
                                ped_all_chan = calib_row["waveformcalibration_pedestal_per_sample"]

                                # keep the same convention you had with simtel: channel 0 only
                                pedestal = ped_all_chan[0].astype(np.float32)
                                pedestal_per_sample_list.append(pedestal)

                                # dc_to_pe is optional but should be there in your merged script
                                # try:
                                #     dc_all_chan = calib_row["dc_to_pe"]
                                #     dc2pe = dc_all_chan[0].astype(np.float32)
                                # except (ValueError, KeyError):  # field missing
                                #     dc2pe = None
                                # dc_to_pe_list.append(dc2pe)

                            else:
                                # no calibration table for this telescope
                                pedestal_per_sample_list.append(None)
                                # dc_to_pe_list.append(None)
                                true_image_list.append(None)

                            # event stats, now with energy (and other truth info)
                            stat_event = {
                                "event_id": min_event_id,
                                "n_pe": int(
                                    sim_image_group[
                                        self.current_event_index[tel_idx]
                                    ]["true_image_sum"]
                                ),
                                "n_pixels": -1,
                                "nphotons": -1,
                                "ev_time": -1.0,
                                "energy": true_energy,
                                "azimuth": true_az,
                                "altitude": true_alt,
                                "h_first_int": true_h_first_int,
                                "xmax": true_xmax,
                                "hmax": -1.0,
                                "emax": -1.0,
                                "cmax": -1.0,
                                "xcore": true_core_x,
                                "ycore": true_core_y,
                                "telescope": tel_id,
                                "tel_pos_x": tel_pos_x,
                                "tel_pos_y": tel_pos_y,
                                "tel_pos_z": tel_pos_z,
                            }

                            event_stat_list.append(stat_event)

                            # increment only if this telescope had the min event id
                            self.current_event_index[tel_idx] += 1

                    yield (
                        tel_ids,
                        wf0_list,
                        wf1_list,
                        dl0_list,
                        dl1_list,
                        true_image_list,
                        pedestal_per_sample_list,
                        event_stat_list,
                    )

            self.current_event_index = [0] * len(self.lst_tel)
            self._iterator = h5_event_iterator()

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
            (
                list_tel_ids,
                wf_list,
                wf1_list,
                dl0_list,
                dl1_list,
                true_image_list,
                pedestal_per_sample_list,
                event_stat_list,
            ) = next(self._iterator)
            return (
                list_tel_ids,
                wf_list,
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
