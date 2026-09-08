"""Load a trained time-coincidence model and score real events with it.

This is the "how a user loads this" companion to train_timecoinc.py. The
architecture (HexOverTime, TimeSummary, ...) lives in ./timecoinc_layers.py,
NOT inside the triggerkit package -- see that file's docstring for why. That
has one practical consequence for loading a saved model: Keras only resolves a
custom class once the module that @register_keras_serializable's it has been
imported, so BOTH of these must happen before tf.keras.models.load_model():

  1. `import timecoinc_layers`            -- registers HexOverTime, TimeSummary,
                                              MaskCells, GlobalHexMeanMax,
                                              SquaredDifference (package
                                              "timecoinc_research")
  2. `triggerkit.models.register_custom_objects()` -- registers triggerkit's
                                              OWN custom objects (PairwiseAUCLoss,
                                              PairwiseAUCMetric, ...), which the
                                              model's compile() config also saved

Skipping either raises "Could not locate class '...'" at load time -- that is
what train_timecoinc.py's own post-save round-trip check catches immediately
(it only knows about triggerkit's classes, so it correctly warns that a model
built from timecoinc_layers needs its own import too -- see that warning in
the training log for exactly this reason).

Usage
-----
    python load_and_use_timecoinc.py [path/to/model.keras]

Defaults to the model train_timecoinc.py just produced.
"""
import glob
import sys

import numpy as np
import tensorflow as tf

# Registers this script's own custom layers -- required before load_model(),
# see the module docstring above. The import's side effect is all that
# matters here; nothing from it is called directly.
import timecoinc_layers  # noqa: F401

import triggerkit.models as M
from triggerkit.data import TriggerDataset

MODEL_PATH = (sys.argv[1] if len(sys.argv) > 1 else
             "trained_models/timecoinc_research_stats__model.keras")

# ---------------------------------------------------------------------------
# 1. Load. triggerkit.models.load_model() already calls
#    register_custom_objects() internally, so this one call covers triggerkit's
#    side; timecoinc_layers's classes are registered by the import above.
# ---------------------------------------------------------------------------
model = M.load_model(MODEL_PATH)
print(f"loaded {MODEL_PATH}")
print(f"  trainable params : {model.count_params()}")
print(f"  input shape      : {model.input_shape}")
print(f"  output shape     : {model.output_shape}")

# ---------------------------------------------------------------------------
# 2. Pull a small batch of REAL events (a few gamma runs + the NSB run) to
#    score, rather than synthetic noise -- this is the shape/dtype contract a
#    caller actually needs to match: (B, 432, 50) float, R0Alpha waveforms.
# ---------------------------------------------------------------------------
ROOT = "/home/tanguy/Bureau/digicam-tdscan-triggering/simtelFileData"
GAMMA = sorted(glob.glob(f"{ROOT}/gammas/NewSimHDF5/medium/*.hdf5"))[:2]
NSB = [f"{ROOT}/NSB/NewSimHDF5/medium/"
       "biascurve_run11110_TEL2_nsb120_tt300.0_dsum260.hdf5"]

dataset = TriggerDataset(
    GAMMA, NSB, batch_size=64, percent_validation=0.2, tel_id_only=1,
    max_gamma_samples_train=256, max_nsb_samples_train=256,
    load_ram=True, seed=0,
)
train_ds, _val_ds = dataset.train_val_datasets()

feats, y = next(iter(train_ds))
# The dataset yields (B, 1, 432, 50); HexOverTime's SequentialBody pack expects
# (B, 432, 50) -- same reshape train_timecoinc.py's own eval block uses.
wf = np.asarray(feats["waveform"]).reshape(-1, 432, 50)
y = np.asarray(y).ravel()

# HexOverTime folds time into the batch (B*T images through the hex conv), so
# score in chunks rather than the whole batch at once -- see HexOverTime's
# docstring for the memory math (T=50 makes a batch of 256 already 12,800
# images).
CHUNK = 32
scores = np.concatenate([
    np.asarray(model(wf[i:i + CHUNK], training=False)).ravel()
    for i in range(0, len(wf), CHUNK)
])

# ---------------------------------------------------------------------------
# 3. Show it actually separates gamma (y=1) from NSB (y=0).
# ---------------------------------------------------------------------------
print(f"\nscored {len(scores)} events "
     f"({int(y.sum())} gamma, {int((1 - y).sum())} nsb)")
print(f"  mean score | gamma : {scores[y == 1].mean():.4f}")
print(f"  mean score | nsb   : {scores[y == 0].mean():.4f}")
print("\nfirst 10 events (true label, score):")
for label, s in list(zip(y, scores))[:10]:
    print(f"  {'gamma' if label else 'nsb  '}  {s:+.4f}")
