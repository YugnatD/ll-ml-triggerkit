"""StatPlotter on small synthetic stats files written by H5StatsWriter."""
import shutil

import numpy as np
import pytest

from triggerkit.Statistics.H5StatsWriter import H5StatsWriter
from triggerkit.Statistics.StatPlotter import StatPlotter, _FOLD_UNSET

CHAIN = [("threshold", {"threshold": 2.0, "binary": True, "comparison": "gt"})]


def write_stats(path, comparison="gt", folds=("a", "b"), n=400, seed=0, with_score=True, p_trig_only=False, chain=None):
    rng = np.random.default_rng(seed)
    w = H5StatsWriter(str(path), trigger_chain=chain or CHAIN, camera_name="DigiCam")
    if with_score:
        w.f.attrs["has_pre_threshold_score"] = True
        w.f.attrs["pre_threshold_reference_threshold"] = 2.0
        w.f.attrs["pre_threshold_comparison"] = comparison
    for i, name in enumerate(folds):
        idx = w.begin_fold(name)
        label = np.r_[np.ones(n), np.zeros(n)].astype(np.uint8)
        # integer scores -> lots of ties; fold i has a different NSB level
        score = np.r_[rng.integers(0, 8, n), rng.integers(0, 4 + 2 * i, n)].astype(np.float32)
        cols = {
            "label": label,
            "n_pe": np.r_[rng.uniform(20, 300, n), np.zeros(n)].astype(np.float32),
            "energy": np.r_[rng.uniform(0.1, 50, n), np.zeros(n)].astype(np.float32),
        }
        if p_trig_only:
            cols["p_trig"] = np.where(score > 3, 0.9, 0.2).astype(np.float32)
        else:
            cols["triggered"] = (score >= 2 if comparison == "ge" else score > 2).astype(np.uint8)
            if with_score:
                cols["pre_threshold_score"] = score
        w.append(cols, fold_idx=idx)
    w.close(window_sec=2e-7)


def plotter(folder, **kw):
    return StatPlotter(base_reference_config=CHAIN, stat_folder=str(folder), **kw)


def test_rate_of_selected_fold_is_its_own_not_fold_zero(tmp_path):
    write_stats(tmp_path / "s.h5")
    sp_all = plotter(tmp_path)
    r0 = sp_all._get_result_trigger_rate_hz(sp_all.base_config_result)
    rates = {}
    for fold in ("a", "b"):
        sp = plotter(tmp_path, fold=fold)
        rates[fold] = sp._get_result_trigger_rate_hz(sp.base_config_result)
    assert rates["a"] == pytest.approx(r0)          # fold 0 = the file attribute
    assert rates["b"] != pytest.approx(rates["a"])  # fold b has a different NSB level
    sp = plotter(tmp_path, fold="b")
    assert sp._get_result_trigger_rate_hz(sp.base_config_result) == pytest.approx(rates["b"])  # cached path


def test_multiple_matching_files_are_reported(tmp_path, capsys):
    write_stats(tmp_path / "one.h5")
    shutil.copy(tmp_path / "one.h5", tmp_path / "two.h5")
    sp = plotter(tmp_path)
    assert sp.get_results(CHAIN)["_filename"] == "one.h5"
    assert "matches 2 stats files" in capsys.readouterr().out


@pytest.mark.parametrize("comparison", ["gt", "ge"])
def test_efficiency_vs_rate_is_consistent_with_the_comparison(tmp_path, comparison):
    write_stats(tmp_path / "s.h5", comparison=comparison, folds=("all",))
    sp = plotter(tmp_path)
    curves = sp.getEfficiencyVsTriggerRateCurves()
    # recover the gamma scores to recompute the efficiency with the real rule
    gamma = sp._collect_metric_values(sp.base_config_result, "pre_threshold_score", kind="gamma")
    rule = (lambda t: (gamma >= t).mean()) if comparison == "ge" else (lambda t: (gamma > t).mean())
    for t, eff in zip(curves["thresholds"][::7], curves["efficiency"][::7]):
        assert eff == pytest.approx(rule(t))


def test_histogram_uses_stored_decision_column(tmp_path):
    write_stats(tmp_path / "s.h5", folds=("all",), with_score=False)       # chain with no score (OR-like)
    sp = plotter(tmp_path)
    bins = np.linspace(0, 400, 9)
    all_h, trig_h = sp._histogram_stream(sp.base_config_result, "n_pe", kind="gamma", bins=bins)
    assert all_h.sum() == 400 and 0 < trig_h.sum() < 400


def test_p_trig_is_thresholded_at_half(tmp_path):
    """Legacy files only have p_trig; p>0 would call (almost) everything a trigger."""
    write_stats(tmp_path / "s.h5", folds=("all",), with_score=False, p_trig_only=True)
    sp = plotter(tmp_path)
    bins = np.linspace(0, 400, 9)
    all_h, trig_h = sp._histogram_stream(sp.base_config_result, "n_pe", kind="gamma", bins=bins)
    assert 0 < trig_h.sum() < all_h.sum()


def test_effective_area_points_sit_at_bin_centres_and_warn_on_sample_size(tmp_path, capsys):
    write_stats(tmp_path / "s.h5", folds=("all",), n=400)
    sp = plotter(tmp_path)
    x, expected, base_all, base_trig = sp._prepare_effective_area_state(
        emin_tev=0.1, emax_tev=100.0, nbins=10, expected_N=60_000)
    edges = sp._ea_bins
    assert len(edges) == 11 and len(x) == 10                              # nbins bins
    np.testing.assert_allclose(x, np.sqrt(edges[:-1] * edges[1:]))
    out = capsys.readouterr().out
    assert "expected_N=60000" in out and "400 gamma events" in out        # sample != thrown -> warned


def test_default_base_config_matches_current_digitalsum_modes():
    import inspect
    default = inspect.signature(StatPlotter.__init__).parameters["base_reference_config"].default
    assert default[0][1]["mode"] == "patch7"


def test_threshold_helpers_agree_across_modules():
    from triggerkit.TriggerChain import TriggerChain
    scores = np.random.default_rng(4).integers(0, 9, 500).astype(np.float32)
    for comparison in ("gt", "ge"):
        for want in (0.02, 0.2, 0.7):
            a = StatPlotter._pick_threshold_from_empirical_scores(scores, want, comparison)
            b = TriggerChain._pick_tau_from_empirical_scores(None, scores, want, comparison)
            assert a == b


def test_compute_statistics_refuses_the_no_nsb_combination():
    from triggerkit.TriggerChain import TriggerChain
    chain = object.__new__(TriggerChain)       # the guard fires before anything else is touched
    with pytest.raises(ValueError, match="NO NSB"):
        chain.compute_statistics(nsb_skip_original_events=True, nsb_roll_copies=0)


CHAIN_B = [("threshold", {"threshold": 3.0, "binary": True, "comparison": "gt"})]


def test_active_fold_is_reset_after_a_plot(tmp_path):
    write_stats(tmp_path / "s.h5")
    sp = plotter(tmp_path)
    sp._apply_fold_override("b")                  # what a queued item with a fold does
    assert sp._fold_filter == "b"
    sp.showPlot(filename=None, show=False)        # nothing queued: just finalizes the figure
    assert sp._fold_filter == sp._default_fold is None
    sp2 = plotter(tmp_path, fold="a")
    sp2._apply_fold_override("b")
    sp2.showPlot(filename=None, show=False)
    assert sp2._fold_filter == "a"                # back to the constructor's fold


def test_non_comparable_configs_do_not_produce_a_blank_report_figure(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=400)
    write_stats(tmp_path / "b.h5", folds=("all",), n=300, chain=CHAIN_B)     # different sample size
    sp = plotter(tmp_path)
    out = tmp_path / "plot.png"
    items = [(CHAIN, "A", _FOLD_UNSET), (CHAIN_B, "B", _FOLD_UNSET)]
    assert sp._report_render_queued_plot(items, plot_type="absolute", filename=str(out), show=False) is False
    assert not out.exists()


def test_comparable_configs_do_render(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=400)
    write_stats(tmp_path / "b.h5", folds=("all",), n=400, chain=CHAIN_B)     # same events, other chain
    sp = plotter(tmp_path)
    out = tmp_path / "plot.png"
    items = [(CHAIN, "A", _FOLD_UNSET), (CHAIN_B, "B", _FOLD_UNSET)]
    assert sp._report_render_queued_plot(items, plot_type="absolute", filename=str(out), show=False,
                                         metrics="n_pe", target_rate_hz=1e5) is True
    assert out.stat().st_size > 0


def test_sanity_check_scans_each_file_once(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=300)
    write_stats(tmp_path / "b.h5", folds=("all",), n=300, chain=CHAIN_B)
    sp = plotter(tmp_path)
    calls = []
    original = sp._histogram_stream
    sp._histogram_stream = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    assert sp.sanity_check_configs([CHAIN, CHAIN_B]) is True
    first = len(calls)
    assert first == 2
    assert sp.sanity_check_configs([CHAIN, CHAIN_B]) is True
    assert len(calls) == first                    # served from the cache
    sp._apply_fold_override("all")                # another fold -> different key, rescanned
    sp.sanity_check_configs([CHAIN, CHAIN_B])
    assert len(calls) == 2 * first


def test_generate_report_end_to_end(tmp_path):
    """Smoke test of the whole report (all plots, markdown, HTML and PDF) on synthetic stats."""
    folds = ("rot0_original_medium", "rot120_original_medium")
    write_stats(tmp_path / "a.h5", folds=folds, n=400)
    write_stats(tmp_path / "b.h5", folds=folds, n=400, chain=CHAIN_B)
    sp = plotter(tmp_path, fold=folds[0])
    out = tmp_path / "report"
    result = sp.generateReport(configs=[CHAIN, CHAIN_B], output_dir=str(out), target_rate_hz=1e5,
                               legend_overrides=["A", "B"], effective_area_expected_N=400,
                               generate_pdf=True, generate_html=True, vector_plots=True, save_svg=False)
    assert result.endswith("report.pdf") and (out / "report.pdf").stat().st_size > 0   # PDF when it succeeded
    md = (out / "report.md").read_text()
    for section in ("Absolute Efficiency vs Npe", "Effective Area", "ROC Curve", "Cross-Validation",
                    "Individual Trigger Rate Scans"):
        assert section in md
    assert "Skipped Items" not in md
    assert (out / "report.html").stat().st_size > 0
    for png in ("roc_curves.png", "cross_validation.png", "effective_area.png"):
        assert (out / png).stat().st_size > 0
    assert sp._fold_filter == folds[0]                  # constructor's fold restored


def test_report_lists_unplottable_sections_instead_of_blank_figures(tmp_path):
    write_stats(tmp_path / "a.h5", folds=("all",), n=400)
    write_stats(tmp_path / "b.h5", folds=("all",), n=300, chain=CHAIN_B)       # not comparable
    sp = plotter(tmp_path)
    out = tmp_path / "report"
    sp.generateReport(configs=[CHAIN, CHAIN_B], output_dir=str(out), target_rate_hz=1e5,
                      generate_pdf=False, generate_html=False, save_svg=False, vector_plots=False)
    md = (out / "report.md").read_text()
    assert "Skipped Items" in md and "not drawn" in md
    assert not (out / "absolute_efficiency_npe.png").exists()
