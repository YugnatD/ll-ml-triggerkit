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

# Folders holding the .h5 stat files (what the stat scripts wrote to).
STAT_FOLDERS = [
    "simu_sst1m_tel2_tdscan",
    "simu_sst1m_tel2_hexcnn",
]
OUTPUT_DIR = "trigger_report"

# --- Example configs ---------------------------------------------------------
# The classic 7-pixel analog-sum baseline (digital_sum patch7 + threshold).
CONFIG_PATCH7 = [
    ("digital_sum", {"mode": "patch7"}),
    ("threshold", {"threshold": 242.0, "binary": True, "comparison": "gt"}),
]

# A trained TDSCAN chain (score_quantizer + tdscan + threshold). Replace the
# ring_weights / id / threshold with the values your stats_tdscan.py run printed.
CONFIG_TDSCAN = [
    ("score_quantizer", {"edges": [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]}),
    ("tdscan", {
        "eps_xy": 1, "eps_t": 1, "filters": 1, "share_neighbors": True,
        "ring_weights": [0.5, 0.0625, -0.5, -0.0039, -1.0, -0.25, 1.0, 0.125, 0.5, 0.25],
    }),
    ("threshold", {"threshold": 66.5, "binary": True, "comparison": "gt"}),
]

# The hex 3D-CNN chain. Its stored chain is just the threshold-terminated CNN;
# StatPlotter matches on the recorded chain json, so keep this minimal and let
# the file's own metadata carry the details.
CONFIG_HEXCNN = [
    ("threshold", {"threshold": 9.2, "binary": True, "comparison": "gt"}),
]

# Base reference for ratio plots (must match one of the .h5 files).
BASE_CONFIG = CONFIG_PATCH7

# The configs to plot, and 1:1 legend labels.
PLOT_CONFIGS = [CONFIG_PATCH7, CONFIG_TDSCAN, CONFIG_HEXCNN]
LEGENDS = ["PATCH7", "TDSCAN ml xy1,t1", "hex 3D-CNN"]


def main():
    plotter = StatPlotter(base_reference_config=BASE_CONFIG, stat_folder=STAT_FOLDERS)

    score_threshold, predicted_rate_hz = plotter.find_score_threshold_for_target_rate(
        TARGET_RATE_HZ)
    print(f"Base-config score threshold for {TARGET_RATE_HZ:.0f} Hz: "
          f"{score_threshold} (predicted {predicted_rate_hz:.1f} Hz)")

    plotter.init_plot(target_rate_hz=TARGET_RATE_HZ)
    for config in PLOT_CONFIGS:
        plotter.add_plot(config)

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
