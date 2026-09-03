"""Render the deployed TDSCAN trigger chain stage by stage, on ONE gamma event.

Builds the exact deployed (filters=1) TDSCAN chain of ``stats_tdscan.py`` (same
eps_xy/eps_t, pinned ring weights, frozen tau), then draws the labelled
stage-by-stage figure for a single ~300 npe gamma -- WITH the temporal
augmentation of the CV fold ``{"gamma_time_shift": 2}`` applied: the raw
waveform is circularly rolled +2 samples on the time axis before the chain, so
every displayed stage sees the shifted pulse (see TriggerChain.show_trigger_chain
``time_roll`` and augment.make_rotation_folds ``gamma_time_shift``).

Run it (needs a display -- it opens an interactive Qt window):

    python examples/show_tdscan_chain.py GAMMA_GLOB [NSB_GLOB]
"""

import os
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

import glob
import sys

import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np

from triggerkit.TriggerChain import TriggerChain
from triggerkit.models import TDSCANBody

# --- Deployed TDSCAN chain (mirrors examples/stats_tdscan.py) -----------------
EPS_XY = 1
EPS_T = 2
RING_WEIGHTS = np.array(
    [0.5000, 0.0625, -0.5000, -0.0039, -1.0000, -0.2500, 1.0000, 0.1250, 0.5000, 0.2500])
TAU = 97.9535903930664  # frozen tau from the 12-fold run (50 kHz NSB)

# --- What to show -------------------------------------------------------------
RANGE_NPE = (250, 350)   # pick a ~300 npe gamma
TIME_ROLL = 5            # the ((0, 2), "original") fold augmentation: +2 samples
SKIP_FIRST_N_EVENTS = 1  # bump to step past earlier matches


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB [NSB_GLOB]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2])) if len(sys.argv) > 2 else None
    if not gamma_files:
        sys.exit("no gamma files matched the given glob.")
    print(f"Found {len(gamma_files)} gamma files"
          + (f", {len(nsb_files)} NSB files." if nsb_files else "."))

    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    handles = TDSCANBody(filters=1, eps_xy=EPS_XY, eps_t=EPS_T).build(chain)
    tdscan_layer, threshold_layer = handles["tdscan"], handles["threshold"]

    tdscan_layer.set_weights_from_params(share_weights=True, ring_weights=RING_WEIGHTS)
    chain.compile_chain()
    threshold_layer.tau.assign(TAU)
    print(f"Pinned ring weights + tau={TAU}.")

    # cv2 (opencv-python) hijacks QT_QPA_PLATFORM_PLUGIN_PATH at import with its
    # bundled Qt plugins, which clash with the system Qt matplotlib uses. Point
    # it back at the system plugins right before the display. (The real fix is
    # `pip install opencv-python-headless` instead of opencv-python.)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

    chain.show_trigger_chain(
        rangenpe=RANGE_NPE,
        time_roll=TIME_ROLL,
        skip_first_n_events=SKIP_FIRST_N_EVENTS,
        generate_image_gif=False,
        dpi=100,
        show_distrib=True,
        hide_range_axis=True,
    )


if __name__ == "__main__":
    main()
