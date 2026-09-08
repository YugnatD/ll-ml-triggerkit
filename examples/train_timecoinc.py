"""Train the time-coincidence architecture on triggerkit.

The architecture itself (HexOverTime, TimeSummary, ..., time_coincidence_body)
lives in ./timecoinc_layers.py, NOT inside the triggerkit package: it is one
still-experimental model among many a user could build, and SequentialBody is
exactly the mechanism that lets it be written here, as a plain user script,
with no library PR. triggerkit only supplies the generic pieces (TriggerDataset,
build_chain, training helpers) that any architecture would use.

Reference to reproduce -- the same architecture in the research sandbox, same
data, no auxiliary target: **+0.0179 AUC over TDSCAN** (two seeds: +0.0168,
+0.0190), i.e. an absolute AUC around 0.713 against TDSCAN's 0.696.

    python train_timecoinc.py [SHARE]        SHARE in none|ring|sym
"""
import glob
import sys

import dataclasses

import numpy as np
import tensorflow as tf

from triggerkit import training
from triggerkit.FileIO.FileOpenerCTAO import SimTelTFDatasetConfig  # noqa: F401
from triggerkit.data import TriggerDataset
from triggerkit.models import build_chain

from timecoinc_layers import time_coincidence_body  # local to this script, not part of triggerkit

ROOT = "/home/tanguy/Bureau/digicam-tdscan-triggering/simtelFileData"
GAMMA = sorted(glob.glob(f"{ROOT}/gammas/NewSimHDF5/medium/*.hdf5"))
NSB = [f"{ROOT}/NSB/NewSimHDF5/medium/"
       "biascurve_run11110_TEL2_nsb120_tt300.0_dsum260.hdf5"]

SHARE = (sys.argv[1] if len(sys.argv) > 1 else "none")
SHARE = False if SHARE == "none" else SHARE
import os
SEED, EPOCHS, BATCH, LR = 1337, 20, 64, 5e-4
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else SEED  # for multi-seed runs
# env overrides for a controlled sweep, without disturbing the positional argv API
EPOCHS = int(os.environ.get("TC_EPOCHS", EPOCHS))
LR = float(os.environ.get("TC_LR", LR))
KEEP_ABS_TIME = os.environ.get("TC_KEEP_ABS_TIME", "0") == "1"

for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)
tf.keras.utils.set_random_seed(SEED)
print(f"{len(GAMMA)} runs gamma, {len(NSB)} run nsb, share={SHARE!r}, "
     f"epochs={EPOCHS}, lr={LR:.2g}, keep_abs_time={KEEP_ABS_TIME}")

dataset = TriggerDataset(
    GAMMA, NSB,
    batch_size=BATCH, percent_validation=0.2, tel_id_only=1,
    max_gamma_samples_train=100_000, max_nsb_samples_train=100_000,
    max_gamma_samples_val=20_000, max_nsb_samples_val=20_000,
    load_ram=True, seed=SEED,
)

body = time_coincidence_body(share_neighbors=SHARE, time_window=50,
                             pre_channels=8, spatial_channels=(16, 32),
                             drop_absolute_time=not KEEP_ABS_TIME)
chain, head, threshold = build_chain(dataset, body, filters=1)
score = threshold.input                     # pre-threshold score, (B, 1)
n_w = int(sum(np.prod(w.shape) for w in chain.model.trainable_weights))
print(f"poids entrainables : {n_w}")

# tau/temp from a real forward pass; the pairwise loss only needs the sharpness
# HexOverTime folds time into the batch, so the convolution sees B * T images.
# calibrate_tau's default batch is 256, i.e. 12,800 images with T = 50, which
# exhausts a 24 GB card. Calibration only needs a score distribution, so a small
# batch costs nothing.
calib_cfg = dataclasses.replace(training._default_calib_config(), batch_size=16)
init_tau, temp, sg, sn = training.calibrate_tau(chain, score, GAMMA[:4],
                                                config=calib_cfg)
print(f"calibration : init_tau={init_tau:.3f}, temp={temp:.4g}  "
      f"(gamma {sg.mean():.2f}, nsb {sn.mean():.2f})")
threshold.tau.assign(init_tau)
threshold.set_trainable(False)

inputs = (chain.input_layer if chain.camera_name == "DigiCam_R0Alpha"
          else [chain.input_layer, chain.input_baseline])
chain.model = tf.keras.Model(inputs=inputs, outputs=score)
# steps/epoch = min(gamma, nsb) budget / batch ; a wrong count makes the
# cosine hit its floor halfway and the run trains at ~0 lr for the rest
STEPS = 100_000 * 2 // BATCH
sched = tf.keras.optimizers.schedules.CosineDecay(LR, EPOCHS * STEPS, alpha=0.01)
chain.model.compile(
    optimizer=tf.keras.optimizers.Adam(sched),
    loss=training.make_pairwise_auc_loss(sharpness=float(temp)),
    metrics=[training.PairwiseAUCMetric(name="auc")],
)

# EarlyStopping with patience > epochs never actually stops early; it only
# triggers restore_best_weights at the end. The sandbox reference (0.7126)
# uses this -- without it the delivered model is whatever the LAST epoch
# happened to land on, not the best one (memory: +0.006 for free on the
# baseline model, same trick).
best_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=EPOCHS + 1,
    restore_best_weights=True, verbose=1)
hist = chain.train_chain(epochs=EPOCHS, callbacks=[best_cb], verbose=2,
                         dataset=dataset, base_name="timecoinc_research")
v = np.array(hist.history.get("val_auc", []))
if v.size:
    print(f"\n>>> val_auc (par lot) : max {v.max():.4f} (ep {v.argmax()+1}), "
          f"mediane des 5 dernieres {np.median(v[-5:]):.4f}")

# --- AUC GLOBALE sur la validation ---------------------------------------
# La metrique par lot moyenne des AUC de 64 evenements ; l AUC globale range
# tous les evenements ensemble. C est cette derniere qui se compare au 0.7126
# du bac a sable, et au 0.6957 du TDSCAN sur le meme decoupage.
_, val_ds = dataset.train_val_datasets()
scorer = tf.keras.Model(chain.model.input, score)
S, Y = [], []
CHUNK = 32   # HexOverTime voit B*T images : 32 evenements = 1600 images
for feats, y in val_ds:
    # meme mise en forme que le pack de train_chain : (B,1,P,S) -> (B,P,S)
    wf = np.asarray(feats["waveform"]).reshape(-1, chain.num_pixels,
                                               chain.num_samples)
    for i in range(0, len(wf), CHUNK):
        S.append(np.asarray(scorer(wf[i:i + CHUNK], training=False)).ravel())
    Y.append(np.asarray(y).ravel())
    # sum(len(a) for a in S) IS already the event count -- multiplying by CHUNK
    # again cut the loop 32x too early, on a prefix that held gammas only.
    n_seen = sum(len(a) for a in S)
    seen = np.concatenate(Y)
    both = seen.max() > 0 and seen.min() < 1   # never stop on a single class
    if n_seen >= 40_000 and both:
        break
s_all, y_all = np.concatenate(S), np.concatenate(Y).astype(np.float64)
n1_, n0_ = y_all.sum(), (1 - y_all).sum()
if n1_ == 0 or n0_ == 0:
    raise SystemExit(f"validation a une seule classe : {int(n1_)} gammas, "
                     f"{int(n0_)} nsb -- AUC indefinie")
o = np.argsort(s_all); r = np.empty(len(s_all)); r[o] = np.arange(1, len(s_all) + 1)
n1, n0 = y_all.sum(), (1 - y_all).sum()
auc = (r[y_all == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print(f">>> AUC GLOBALE sur {int(n1)} gammas + {int(n0)} nsb : {max(auc, 1-auc):.4f}")
print("    reference : 0.7126 (bac a sable, meme archi sans tache auxiliaire)")
print("    TDSCAN sur le meme decoupage : 0.6957")
