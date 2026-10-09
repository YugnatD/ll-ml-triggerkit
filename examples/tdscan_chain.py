"""Define the deployed TDSCAN trigger chain ONCE, here -- everything else imports it.

This mirrors the sandbox's own ``tdscan_chain.py``: one ``build_chain(...)``
that every other script (show / train / stats) calls, so the architecture and
its pinned weights live in exactly one place instead of being copy-pasted (and
silently drifting) across ``show_tdscan_chain.py``, ``train_tdscan.py`` and
``stats_tdscan.py``.

    sandbox                            triggerkit (here)
    ----------------------------------------------------------------
    tdscan_chain.build_chain(...)   -> TDSCANBody(...).build(chain)

The defaults below are the currently deployed chain: eps_xy=1, eps_t=2, float
TDSCAN math (no inner quantize_step), no score-quantizer front-end, power-of-2
ring weights, tau frozen from the 12-fold run. Override any keyword for a
one-off experiment without touching this file -- e.g. a script comparing the
4-bit-input variant passes ``edges_range=(16, 128), edges_num_bit=4,
edges_func=generate_lin_space_edges`` explicitly (see stats_report.py's
``CONFIG_TDSCAN_INQ4_P2`` for that variant's weights).

Run this file directly to just build the chain and print it, exactly like the
sandbox's own ``if __name__ == "__main__"`` block:

    python examples/tdscan_chain.py GAMMA_GLOB [NSB_GLOB]
"""

import glob
import sys

import numpy as np

from triggerkit.TriggerChain import TriggerChain
from triggerkit.models import TDSCANBody, generate_lin_space_edges  # noqa: F401 (re-exported for callers)

# --- The deployed chain's knobs -----------------------------------------------
EPS_XY = 1
EPS_T = 2
FILTERS = 1
SHARE_NEIGHBORS = True

# Score-quantizer front-end: off by default (the currently deployed/converged
# chain has no input quantizer). Pass edges_func=generate_lin_space_edges
# explicitly to turn it on for an experiment.
EDGES_RANGE = (16, 128)
EDGES_NUM_BIT = 4
EDGES_FUNC = None

# Inner TDSCAN accumulator quantization: off by default (float weights/math).
QUANTIZE_STEP = None
FAKE_QUANT_ACCUMULATORS = False
OVERFLOW_MODE = "AP_SAT"
QUANTIZATION_MODE = "AP_TRN"
RESCALE_SHIFT = 0

# Optional front-/back-end stages, all off by default.
SUBTRACT_VALUE = None
SUBTRACT_QUANTIZE_STEP = None
SUBTRACT_OVERFLOW_MODE = "AP_WRAP"
SUBTRACT_QUANTIZATION_MODE = "AP_TRN"
DIGITAL_SUM_MODE = None
FADC = False

# Pinned deployed weights (power-of-2, eps_t=2 -> 10 ring weights) + frozen tau
# (12-fold run, 50 kHz NSB target). Set RING_WEIGHTS=None to keep build-time init.
RING_WEIGHTS = np.array(
    [0.5000, 0.0625, -0.5000, -0.0039, -1.0000, -0.2500, 1.0000, 0.1250, 0.5000, 0.2500])
TAU = 97.9535903930664


def build_chain(
    gamma_files, nsb_files=None, *,
    eps_xy=EPS_XY, eps_t=EPS_T, filters=FILTERS, share_neighbors=SHARE_NEIGHBORS,
    edges_range=EDGES_RANGE, edges_num_bit=EDGES_NUM_BIT, edges_func=EDGES_FUNC,
    quantize_step=QUANTIZE_STEP, fake_quant_accumulators=FAKE_QUANT_ACCUMULATORS,
    overflow_mode=OVERFLOW_MODE, quantization_mode=QUANTIZATION_MODE,
    rescale_shift=RESCALE_SHIFT,
    subtract_value=SUBTRACT_VALUE, subtract_quantize_step=SUBTRACT_QUANTIZE_STEP,
    subtract_overflow_mode=SUBTRACT_OVERFLOW_MODE,
    subtract_quantization_mode=SUBTRACT_QUANTIZATION_MODE,
    digital_sum_mode=DIGITAL_SUM_MODE, fadc=FADC,
    ring_weights=RING_WEIGHTS, tau=TAU,
):
    """Build the TDSCAN chain. Returns ``(chain, tdscan_layer, threshold_layer)``.

    Every keyword defaults to the deployed chain's value (the module constants
    above); pass any of them to change ONE knob for an experiment without
    touching this file. ``ring_weights=None`` keeps the build-time init instead
    of pinning; ``tau=None`` keeps the initial tau (e.g. before re-tuning it).
    """
    chain = TriggerChain(gamma_files, simtel_nsb_path=nsb_files)
    handles = TDSCANBody(
        filters=filters, eps_xy=eps_xy, eps_t=eps_t, share_neighbors=share_neighbors,
        edges_range=edges_range, edges_num_bit=edges_num_bit, edges_func=edges_func,
        quantize_step=quantize_step, fake_quant_accumulators=fake_quant_accumulators,
        overflow_mode=overflow_mode, quantization_mode=quantization_mode,
        rescale_shift=rescale_shift,
        subtract_value=subtract_value, subtract_quantize_step=subtract_quantize_step,
        subtract_overflow_mode=subtract_overflow_mode,
        subtract_quantization_mode=subtract_quantization_mode,
        digital_sum_mode=digital_sum_mode, fadc=fadc,
    ).build(chain)
    tdscan_layer, threshold_layer = handles["tdscan"], handles["threshold"]

    if ring_weights is not None:
        if tdscan_layer.share_neighbors:
            tdscan_layer.set_weights_from_params(share_weights=True, ring_weights=ring_weights)
        else:
            tdscan_layer.set_weights_from_params(share_weights=False, kernel_weights=ring_weights)
    # Compiled in every case: find_threshold_for_target_rate (used when tau is
    # re-tuned, tau=None) needs chain.model to locate the threshold layer.
    chain.compile_chain()
    if tau is not None:
        threshold_layer.tau.assign(tau)

    return chain, tdscan_layer, threshold_layer


if __name__ == "__main__":
    # Just build the chain and print it, to check it's what we expect.
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} GAMMA_GLOB [NSB_GLOB]")
    gamma_files = sorted(glob.glob(sys.argv[1]))
    nsb_files = sorted(glob.glob(sys.argv[2])) if len(sys.argv) > 2 else None
    chain, tdscan_layer, threshold_layer = build_chain(gamma_files, nsb_files)
    chain.model.summary()
    print(f"ring weights: {tdscan_layer.get_flat_weights()}")
    print(f"tau: {threshold_layer.tau.numpy()}")
