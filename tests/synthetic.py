"""Tiny synthetic ctapipe-style HDF5 files (SST-1M R0Alpha: 432 patches x 50 samples).

Lets the tests drive the REAL reader / chain / statistics code without any
simtel data. Waveforms are baseline noise plus, for 'gamma' files, a bright
compact pulse of a known size, so a trigger chain can separate the classes.
"""
import h5py
import numpy as np

from triggerkit.camera import sst1m

N_PIX, N_SAMPLES = 432, 50


def write_ctapipe_h5(path, *, n_events=20, tels=(1,), kind="gamma", seed=0,
                     tel_present=None, fill_with_tel_id=False, with_calibration=True,
                     layout_tels=None):
    """Write one file. ``tel_present(event_index) -> iterable of tel ids`` controls which
    telescopes saw each event (default: all of ``tels``). ``fill_with_tel_id`` makes every
    waveform sample equal to its telescope id, to check tel_id <-> waveform alignment."""
    rng = np.random.default_rng(seed)
    pos = sst1m.patch_positions()
    tel_present = tel_present or (lambda i: tels)
    with h5py.File(path, "w") as f:
        cam = f.create_group("configuration/instrument/telescope/camera/geometry_0")
        cam["pix_id"] = np.arange(N_PIX)
        cam["pix_x"] = pos[:, 0] / 100.0        # cm -> m
        cam["pix_y"] = pos[:, 1] / 100.0
        cam["pix_area"] = np.full(N_PIX, 4e-4)

        layout_tels = tuple(layout_tels) if layout_tels is not None else tuple(tels)
        layout = np.zeros(len(layout_tels), dtype=[("camera_name", "S20"), ("tel_id", "i4"),
                                            ("pos_x", "f4"), ("pos_y", "f4"), ("pos_z", "f4")])
        for k, t in enumerate(layout_tels):
            layout[k] = (b"DigiCam_R0Alpha", t, 100.0 * t, -50.0 * t, 5.0)
        f.create_group("configuration/instrument/subarray")["layout"] = layout

        shower = np.zeros(n_events, dtype=[("event_id", "i8"), ("true_energy", "f4"), ("true_alt", "f4"),
                                           ("true_az", "f4"), ("true_h_first_int", "f4"),
                                           ("true_x_max", "f4"), ("true_core_x", "f4"), ("true_core_y", "f4")])
        shower["event_id"] = np.arange(n_events)
        shower["true_energy"] = rng.uniform(0.1, 10, n_events)
        shower["true_core_x"] = rng.uniform(-200, 200, n_events)
        shower["true_core_y"] = rng.uniform(-200, 200, n_events)
        f.create_group("simulation/event/subarray")["shower"] = shower

        for t in tels:
            events = [i for i in range(n_events) if t in tel_present(i)]
            n = len(events)
            name = f"tel_{t:03d}"
            r0 = np.zeros(n, dtype=[("event_id", "i8"), ("waveform", "u2", (1, N_PIX, N_SAMPLES))])
            img = np.zeros(n, dtype=[("true_image", "f4", (N_PIX,)), ("true_image_sum", "f4")])
            cal = np.zeros(n, dtype=[("event_id", "i8"),
                                     ("waveformcalibration_pedestal_per_sample", "f4", (1, N_PIX))])
            for row, ev in enumerate(events):
                wf = rng.normal(100, 3, (N_PIX, N_SAMPLES))
                if kind == "gamma":                   # compact pulse of ~30 patches, samples 20..25
                    centre = int(rng.integers(40, 392))
                    wf[centre - 15:centre + 15, 20:25] += rng.uniform(40, 80)
                    img["true_image"][row, centre - 15:centre + 15] = 5.0
                    img["true_image_sum"][row] = 150.0
                if fill_with_tel_id:
                    wf[:] = t
                r0["event_id"][row] = ev
                r0["waveform"][row, 0] = np.clip(wf, 0, 4095).astype(np.uint16)
                cal["event_id"][row] = ev
                cal["waveformcalibration_pedestal_per_sample"][row, 0] = 100.0
            f.require_group("r0/event/telescope")[name] = r0
            f.require_group("simulation/event/telescope/images")[name] = img
            if with_calibration:
                f.require_group("calibration/event/telescope")[name] = cal
    return str(path)
