"""The ctapipe-HDF5 reader on synthetic multi-telescope files."""
import numpy as np
import pytest

from synthetic import N_PIX, N_SAMPLES, write_ctapipe_h5
from triggerkit.FileIO.AsyncFileOpener import FileOpenerCTAO


def read(path):
    with FileOpenerCTAO(str(path)) as fo:
        return fo, list(fo)


def test_single_telescope_file(tmp_path):
    p = write_ctapipe_h5(tmp_path / "g.h5", n_events=5, tels=(1,))
    fo, events = read(p)
    assert fo.camera_name == "DigiCam_R0Alpha" and fo.geom.n_pixels == N_PIX
    assert len(events) == 5
    tel_ids, wf0, *_rest, stats, i = events[0]
    assert tel_ids == [1] and wf0[0].shape == (N_PIX, N_SAMPLES) and stats[0]["n_pe"] == 150
    assert [e[-1] for e in events] == list(range(5))                 # running event index


def test_tel_ids_stay_aligned_with_waveforms_when_a_telescope_misses_an_event(tmp_path):
    """Regression: tel_ids used to list EVERY telescope, so a missing one shifted the pairing."""
    present = lambda i: {0: (1, 2), 1: (2,), 2: (1,), 3: (1, 2)}[i]      # event 1 has no tel 1
    p = write_ctapipe_h5(tmp_path / "g.h5", n_events=4, tels=(1, 2), fill_with_tel_id=True, tel_present=present)
    _, events = read(p)
    assert len(events) == 4
    for (tel_ids, wf0, _w1, _d0, _d1, true_img, _pt, ped, stats, i) in events:
        assert tel_ids == list(present(i))
        assert [int(w.mean()) for w in wf0] == tel_ids                   # waveform of tel k holds the value k
        assert [s["telescope"] for s in stats] == tel_ids
        assert len(true_img) == len(ped) == len(tel_ids)


def test_file_holding_only_some_telescopes_of_the_layout(tmp_path):
    """Regression: lst_tel was paired with the layout list by position (tel_002 -> id 1)."""
    p = write_ctapipe_h5(tmp_path / "g.h5", n_events=3, tels=(2,), layout_tels=(1, 2), fill_with_tel_id=True)
    _, events = read(p)
    for tel_ids, wf0, *_ in events:
        assert tel_ids == [2] and int(wf0[0].mean()) == 2


def test_missing_calibration_table_keeps_lists_aligned(tmp_path):
    """Regression: true_image used to get two entries per telescope without a calibration table."""
    p = write_ctapipe_h5(tmp_path / "g.h5", n_events=3, tels=(1, 2), with_calibration=False)
    _, events = read(p)
    for tel_ids, _w0, _w1, _d0, _d1, true_img, _pt, ped, stats, _i in events:
        assert len(true_img) == len(tel_ids) == len(stats) == len(ped)
        assert all(x is None for x in ped)


def test_event_stats_come_from_the_shower_table(tmp_path):
    p = write_ctapipe_h5(tmp_path / "g.h5", n_events=3, tels=(1,))
    _, events = read(p)
    energies = [e[8][0]["energy"] for e in events]
    assert all(0.1 <= x <= 10 for x in energies) and len(set(energies)) == 3
    assert events[0][8][0]["tel_pos_x"] == pytest.approx(100.0)


def test_nsb_file_has_zero_true_image(tmp_path):
    p = write_ctapipe_h5(tmp_path / "n.h5", n_events=3, tels=(1,), kind="nsb")
    _, events = read(p)
    assert all(float(np.sum(e[5][0])) == 0.0 and e[8][0]["n_pe"] == 0 for e in events)
