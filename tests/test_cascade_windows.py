"""TriggerCascade time windows: Full / Fixed / AlignedClamped / AlignedStrict, trigger times,
ref_stage, and TriggerChain(num_samples=...)."""
import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.TriggerChain import TriggerChain
from triggerkit.cascade import (AlignedClamped, AlignedStrict, Fixed, Full, TriggerCascade,
                                _crop, trigger_times)

TAU = 730.0     # patch7 sum of 7 baseline-100 patches = 700; the synthetic gamma pulse adds ~400


@pytest.fixture(scope="module")
def files50(tmp_path_factory):
    d = tmp_path_factory.mktemp("w50")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=20, seed=i) for i in range(2)]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=20, kind="nsb", seed=10)]
    return gamma, nsb


@pytest.fixture(scope="module")
def files82(tmp_path_factory):
    """82-sample datacube, gamma pulse on samples 30..34 (inside the central 16..65)."""
    d = tmp_path_factory.mktemp("w82")
    gamma = [write_ctapipe_h5(d / "g.h5", n_events=20, seed=1, n_samples=82, pulse_samples=(30, 35))]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=20, kind="nsb", seed=11, n_samples=82)]
    return gamma, nsb


def patch7(files, num_samples=None, pool=True):
    gamma, nsb = files
    chain = TriggerChain(gamma, simtel_nsb_path=nsb, num_samples=num_samples)
    chain.add_stage("digital_sum", mode="patch7")
    if pool:
        chain.add_stage("global_max_pooling_2d")
    chain.add_stage("threshold", init_tau=TAU, temp=1.0, binary_output=True)
    chain.compile_chain()
    return chain


def batch(n_samples, pulse=None):
    """One event: baseline 100 everywhere, optional pulse (first, last) on 30 patches."""
    wf = np.full((1, N_PIX, n_samples), 100, dtype=np.uint16)
    if pulse is not None:
        wf[0, 200:230, pulse[0]:pulse[1] + 1] += 80
    return wf, np.zeros((1, N_PIX), dtype=np.int32)


# --------------------------------------------------------------------- pure helpers
def test_trigger_times_use_the_first_pulse():
    above = np.zeros((3, 10), bool)
    above[0, 2:5] = True; above[0, 7:9] = True      # two pulses: only 2..4 counts
    above[1, 6:10] = True                            # pulse running to the end of the window
    above[2, 3] = True                               # single sample
    np.testing.assert_array_equal(trigger_times(above, "rising"), [2, 6, 3])
    np.testing.assert_array_equal(trigger_times(above, "falling"), [4, 9, 3])
    np.testing.assert_array_equal(trigger_times(above, "middle"), [3, 7, 3])
    with pytest.raises(ValueError):
        trigger_times(above, "peak")
    with pytest.raises(ValueError):
        trigger_times(np.zeros((1, 10), bool))


def test_crop_takes_a_different_window_per_event():
    wf = np.arange(2 * 3 * 10).reshape(2, 3, 10)
    out = _crop(wf, np.array([0, 4]), 5)
    np.testing.assert_array_equal(out[0], wf[0, :, 0:5])
    np.testing.assert_array_equal(out[1], wf[1, :, 4:9])


# --------------------------------------------------------------------- num_samples
def test_num_samples_is_checked_and_short_chains_cannot_read_the_files(files50):
    gamma, nsb = files50
    with pytest.raises(ValueError, match="between 1 and"):
        TriggerChain(gamma, simtel_nsb_path=nsb, num_samples=60)
    short = patch7(files50, num_samples=32)
    assert short.num_samples == 32 and short.file_num_samples == 50
    assert short.window_size == pytest.approx(32 * 4e-9)
    with pytest.raises(ValueError, match="cannot read the files itself"):
        short.find_threshold_for_target_rate(target_rate_hz=1000, N_event_esimate_threshold=10,
                                             batch_size=8, nsb_roll_copies=0,
                                             nsb_skip_original_events=False)


# --------------------------------------------------------------------- construction checks
def test_window_length_must_match_the_stage(files50):
    with pytest.raises(ValueError, match="num_samples=32"):
        TriggerCascade([patch7(files50), patch7(files50)], windows=[Full(), Fixed(0, 32)])
    with pytest.raises(ValueError, match="leaves the datacube"):
        TriggerCascade([patch7(files50), patch7(files50, 32)], windows=[Full(), Fixed(30, 32)])
    with pytest.raises(ValueError, match="stage 0"):
        TriggerCascade([patch7(files50, 32), patch7(files50, 32)],
                       windows=[AlignedClamped(32, 16), Full()])


def test_ref_stage_skips_stages_without_a_time_axis(files50):
    timed, untimed, last = patch7(files50), patch7(files50, pool=False), patch7(files50, 32)
    casc = TriggerCascade([timed, untimed, last], windows=[Full(), Full(), AlignedClamped(32, 16)])
    assert casc.ref_stage == [None, None, 0]           # the closest stage WITH a time axis
    with pytest.raises(ValueError, match="no per-sample score"):
        TriggerCascade([timed, untimed, last], windows=[Full(), Full(), AlignedClamped(32, 16, ref_stage=1)])
    with pytest.raises(ValueError, match="no stage above it has a time axis"):
        TriggerCascade([untimed, last], windows=[Full(), AlignedClamped(32, 16)])


def test_strict_window_that_could_leave_the_datacube_is_refused(files82):
    s0, s1 = patch7(files82, 50), patch7(files82, 32)
    casc = TriggerCascade([s0, s1], windows=[Fixed(16, 50), AlignedStrict(32, 16)])
    assert casc.sample_ranges == [(16, 65), (0, 80)]   # worst cases: trigger at 16 (0..31) and at 65 (49..80)
    with pytest.raises(ValueError, match="AlignedStrict"):
        TriggerCascade([s0, s1], windows=[Fixed(16, 50), AlignedStrict(32, 20)])   # would start at -4
    with pytest.raises(ValueError, match="AlignedStrict"):
        TriggerCascade([patch7(files82), s1], windows=[Full(), AlignedStrict(32, 16)])


# --------------------------------------------------------------------- per-event behaviour
def test_trigger_time_modes_on_a_known_pulse(files50):
    for mode, expected in (("rising", 40), ("falling", 43), ("middle", 41)):
        casc = TriggerCascade([patch7(files50), patch7(files50, 32)],
                              windows=[Full(), AlignedClamped(32, 16)], trig_time=mode)
        wf, ped = batch(50, pulse=(40, 43))
        fired, t = casc._fire_and_times(0, tf.convert_to_tensor(wf))
        assert fired.tolist() == [True] and t.tolist() == [expected]


def test_clamped_window_is_shifted_back_inside_the_datacube(files50):
    casc = TriggerCascade([patch7(files50), patch7(files50, 32)], windows=[Full(), AlignedClamped(32, 16)])
    t_abs = {0: np.array([40, 5, 20])}
    # 40-16=24 would end at 55 > 49 -> 18 (samples 18..49); 5-16 < 0 -> 0; 20-16=4 fits
    np.testing.assert_array_equal(casc._window_starts(1, np.arange(3), t_abs), [18, 0, 4])
    wf, ped = batch(50, pulse=(40, 43))
    assert casc._propagate(wf, ped)[0] == [1, 1]                            # the late pulse is still seen by stage 1


def test_fixed_first_samples_lose_a_late_pulse_that_an_aligned_window_keeps(files50):
    wf, ped = batch(50, pulse=(40, 43))
    fixed = TriggerCascade([patch7(files50), patch7(files50, 32)], windows=[Full(), Fixed(0, 32)])
    assert fixed._propagate(wf, ped)[0] == [1, 0]      # same trigger, cut by the fixed window
    aligned = TriggerCascade([patch7(files50), patch7(files50, 32)], windows=[Full(), AlignedClamped(32, 8)])
    assert aligned._propagate(wf, ped)[0] == [1, 1]


def test_strict_window_uses_the_real_samples_of_a_longer_datacube(files82):
    casc = TriggerCascade([patch7(files82, 50), patch7(files82, 32)],
                          windows=[Fixed(16, 50), AlignedStrict(32, 16)])
    wf, ped = batch(82, pulse=(70, 72))                # outside the first stage's window 16..65
    assert casc._propagate(wf, ped)[0] == [0, 0]
    wf, ped = batch(82, pulse=(60, 62))                # rising 60 -> stage 1 reads samples 44..75
    assert casc._propagate(wf, ped)[0] == [1, 1]
    stats = casc.compute_statistics(batch_size=8, tel_id_only=1, nsb_roll_copies=0,
                                    nsb_skip_original_events=False)
    g = stats["gamma"]["stages"]
    assert g[0]["passed"] == stats["gamma"]["n_total"] == 20 and g[1]["passed"] == 20
    assert stats["window_size_s"] == pytest.approx(50 * 4e-9)    # rates per first-stage window


def test_full_windows_keep_the_previous_behaviour(files50):
    casc = TriggerCascade([patch7(files50), patch7(files50)])
    stats = casc.compute_statistics(batch_size=8, tel_id_only=1, nsb_roll_copies=0,
                                    nsb_skip_original_events=False)
    g = stats["gamma"]["stages"]
    assert g[0]["passed"] == 40 and g[1]["conditional_fraction"] == 1.0
    assert stats["window_size_s"] == pytest.approx(50 * 4e-9)


# --------------------------------------------------------------------- defaults + per-event output
def test_defaults_read_telescope_1_and_every_original_nsb_event(files50):
    casc = TriggerCascade([patch7(files50), patch7(files50)])
    stats = casc.compute_statistics(batch_size=8)                  # no telescope / NSB arguments
    assert stats["gamma"]["n_total"] == 40 and stats["nsb"]["n_total"] == 20
    with pytest.raises(ValueError, match="NO NSB"):
        casc.compute_statistics(nsb_skip_original_events=True, nsb_roll_copies=0)
    with pytest.raises(ValueError, match="NO NSB"):
        casc.calibrate([1e6, 1e6], nsb_skip_original_events=True, nsb_roll_copies=0)


def test_per_stage_stats_files_are_readable_by_statplotter(files50, tmp_path):
    import json
    import h5py
    from triggerkit.Statistics.StatPlotter import StatPlotter

    casc = TriggerCascade([patch7(files50), patch7(files50, 32)],
                          names=["L1", "L2 first 32"], windows=[Full(), Fixed(0, 32)])
    stats = casc.compute_statistics(batch_size=8, folder=str(tmp_path), base_name="demo")
    assert len(stats["files"]) == 2 and all(os.path.exists(p) for p in stats["files"])

    with h5py.File(stats["files"][1], "r") as f:
        ev = {k: f["events"][k][()] for k in f["events"]}
        desc = json.loads(f.attrs["cascade_json"])
        assert f.attrs["gamma_trig"] == stats["gamma"]["stages"][1]["passed"]
    gam = ev["label"] == 1
    # the synthetic gamma pulse starts on sample 20 (a noise fluctuation can fire a bit earlier)
    t0 = ev["t_trig_stage0"][gam & (ev["fired_stage0"] == 1)]
    assert (t0 >= 0).all() and np.mean(t0 == 20) > 0.9
    assert (ev["t_trig_stage0"][ev["fired_stage0"] == 0] == -1).all()
    np.testing.assert_array_equal(ev["triggered"], ev["fired_stage0"] & ev["fired_stage1"])
    assert np.isnan(ev["score_stage1"][ev["reached_stage1"] == 0]).all()
    assert (ev["window_start_stage1"][ev["reached_stage1"] == 1] == 0).all()
    assert desc["stages"][1]["window"] == {"type": "Fixed", "start": 0, "length": 32}

    plotter = StatPlotter(base_reference_config=casc.stats_config(0, "demo"), stat_folder=str(tmp_path))
    res = plotter.get_results(casc.stats_config(1, "demo"))
    assert res is not None and res["num_events_gamma"] == 40 and res["num_events_nsb"] == 20

    with pytest.raises(FileExistsError):
        casc.compute_statistics(batch_size=8, folder=str(tmp_path), base_name="demo")
    casc.compute_statistics(batch_size=8, folder=str(tmp_path), base_name="demo", overwrite=True)


# --------------------------------------------------------------------- audit guards
def test_stage_needs_the_threshold_fed_by_its_pooling_to_have_a_time_axis(files50):
    gamma, nsb = files50
    rescaled = TriggerChain(gamma, simtel_nsb_path=nsb)
    rescaled.add_stage("digital_sum", mode="patch7")
    rescaled.add_stage("global_max_pooling_2d")
    rescaled.add_stage("rescaling", scale=1.0)          # a layer between the pooling and the threshold
    rescaled.add_stage("threshold", init_tau=TAU, temp=1.0, binary_output=True)
    rescaled.compile_chain()
    with pytest.raises(ValueError, match="no stage above it has a time axis"):
        TriggerCascade([rescaled, patch7(files50, 32)], windows=[Full(), AlignedClamped(32, 16)])


def test_uncompiled_stage_and_model_length_mismatch_are_refused(files50):
    gamma, nsb = files50
    raw = TriggerChain(gamma, simtel_nsb_path=nsb)
    with pytest.raises(ValueError, match="compile_chain"):
        TriggerCascade([patch7(files50), raw])
    wrong = patch7(files50)
    wrong.num_samples = 32                               # e.g. a model reloaded on the wrong length
    with pytest.raises(ValueError, match="its model takes 50 samples"):
        TriggerCascade([patch7(files50), wrong], windows=[Full(), Fixed(0, 32)])
