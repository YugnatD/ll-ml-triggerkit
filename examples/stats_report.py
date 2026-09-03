"""Generate an SST-1M trigger statistics report -- on triggerkit.

Port of the sandbox's ``main_stat_sst1m_tf_v3.py``. It reproduces the report in
``simu_sst1m_tel2_nsb_med_report/`` (report.md + combined plots): rate-vs-threshold,
gamma efficiency vs NSB rate, effective area, per-npe / per-energy efficiency, etc.

The report step reads the per-event statistics HDF5 files produced by the stat
scripts -- it does NOT touch raw simtel data or TensorFlow. So the pipeline is:

    1. examples/stats_tdscan.py  GAMMA NSB simu_sst1m_tel2_tdscan     # writes .h5
    2. examples/stats_hexcnn.py  GAMMA NSB [W] simu_sst1m_tel2_hexcnn # writes .h5
    3. examples/stats_report.py                                       # this script

A "config" is an ordered list of ``(stage_name, params)`` tuples describing a
trigger chain; StatPlotter matches each config to the .h5 file whose stored chain
equals it. The ``params`` must match what produced the file (weights, thresholds,
edges, ...), so the configs below are EXAMPLES -- edit them (especially the
tdscan ``ring_weights``/``id`` and the ``threshold`` values printed by the stat
scripts) to match your own runs, or every config lands under "Skipped Items".

Run it:

    python examples/stats_report.py
"""

from triggerkit.Statistics.StatPlotter import StatPlotter

TARGET_RATE_HZ = 50_000.0

# Folders holding the .h5 stat files. These are the outputs of the cross-
# validation run (stats_patch7.py + stats_tdscan.py, 10 folds each) copied into
# examples/results/. Point this at your own OUTPUT_FOLDER(s) if you re-run.
STAT_FOLDERS = ["results"]
OUTPUT_DIR = "trigger_report"

# --- Configs (match the actual cross-validation runs in results/) ------------
# These mirror the two .h5 files in examples/results/. The threshold values are
# the frozen tau each stat script printed (tuned once to ~50 kHz NSB).
#
# The real patch7 telescope trigger (digital_sum patch7 + threshold, no TDSCAN).
CONFIG_PATCH7 = [
    ("digital_sum", {"mode": "patch7"}),
    ("threshold", {"threshold": 241.99998474121094, "binary": True, "comparison": "gt"}),
]

# The deployed TDSCAN chain: eps_xy=1, eps_t=2 (10 shared ring weights), float
# (no score_quantizer). StatPlotter ignores id / weights when matching and
# compares weights with np.allclose, so the flat ring_weights below match the
# nested stored weights; the threshold is what stats_tdscan.py printed.
CONFIG_TDSCAN = [
    ("tdscan", {
        "eps_xy": 1, "eps_t": 2, "filters": 1, "share_neighbors": True,
        "ring_weights": [0.5, 0.0625, -0.5, -0.0039, -1.0, -0.25, 1.0, 0.125, 0.5, 0.25],
    }),
    ("threshold", {"threshold": 97.9535903930664, "binary": True, "comparison": "gt"}),
]

# Base reference for ratio plots (must match one of the .h5 files).
BASE_CONFIG = CONFIG_PATCH7

# The configs to plot, and 1:1 legend labels.
PLOT_CONFIGS = [CONFIG_PATCH7, CONFIG_TDSCAN]
LEGENDS = ["PATCH7", "TDSCAN ml xy1,t2"]


def main():
    plotter = StatPlotter(base_reference_config=BASE_CONFIG, stat_folder=STAT_FOLDERS)

    score_threshold, predicted_rate_hz = plotter.find_score_threshold_for_target_rate(
        TARGET_RATE_HZ)
    print(f"Base-config score threshold for {TARGET_RATE_HZ:.0f} Hz: "
          f"{score_threshold} (predicted {predicted_rate_hz:.1f} Hz)")

    plotter.init_plot(target_rate_hz=TARGET_RATE_HZ)
    for config in PLOT_CONFIGS:
        plotter.add_plot(config)

    # generateReport embeds the cross-validation (per-fold leakage) graph as its
    # own section whenever a config carries a /folds group -- no separate file.
    report_path = plotter.generateReport(
        configs=PLOT_CONFIGS,
        output_dir=OUTPUT_DIR,
        target_rate_hz=TARGET_RATE_HZ,
        legend_overrides=LEGENDS,
        generate_pdf=True,
        presentation_svg=True,
        generate_html=False,
        title="SST-1M Trigger Statistics Report",
    )
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
