"""Command line and cross-validation folds shared by the stats_*.py examples.

Every stats script takes the same arguments::

    --condition NAME GAMMA_GLOB NSB_GLOB   (repeat for several NSB conditions)
    --output FOLDER                         where the statistics HDF5 is written
    --quick                                 a few events per fold, to check the script runs

The first --condition is the reference: the chain and its geometry are built
on it and tau is tuned on its NSB. The reference fold is run under every
condition; the other folds only under the reference condition.

Folds (see triggerkit.augment.make_rotation_folds) are written as rows of
optional keys; ``fold_specs`` adds the event budget and the condition(s) each
row runs under, which make_rotation_folds requires on every row.
"""

import argparse
import glob

# Event budget per fold: (gamma_events, nsb_events). None = every event.
REFERENCE_EVENTS = (200_000, 100_000)
FOLD_EVENTS = (10_000, 10_000)
QUICK_EVENTS = (500, 500)

# NSB events used to tune tau on the reference condition.
THRESHOLD_EVENTS = 25_000
QUICK_THRESHOLD_EVENTS = 2_000

# The folds of stats_tdscan.py and stats_patch7.py, kept identical so the two
# runs compare fold by fold. Row 0 is the untouched reference fold. Keys:
#   gamma_deg        camera rotation of the gamma events, multiple of 120 (exact symmetry)
#   gamma_time_shift circular roll of the gamma waveforms, in samples
#   nsb_kind         "original" / "rolled" / "shuffle" (NSB pixel reindexing)
#   nsb_param        roll shift or shuffle seed
#   nsb_time_shift   circular roll of the NSB waveforms, in samples
#   name             explicit fold name
# patch7 learns nothing, so for it the folds are a consistency check (any drift
# is a geometry bug); for TDSCAN a drift that patch7 does not show is leakage.
STANDARD_FOLDS = [
    {},                                                       # reference fold
    # --- rotation symmetry ---------------------------------------------------
    {"gamma_deg": 120},
    {"gamma_deg": 240},
    # --- NSB pixel transforms ------------------------------------------------
    {"nsb_kind": "rolled",  "nsb_param": 1},
    {"nsb_kind": "rolled",  "nsb_param": 42},
    {"nsb_kind": "shuffle", "nsb_param": 2024},
    # --- mixes ---------------------------------------------------------------
    {"gamma_deg": 120, "nsb_kind": "rolled",  "nsb_param": 1},
    {"gamma_deg": 240, "nsb_kind": "rolled",  "nsb_param": 1},
    {"gamma_deg": 120, "nsb_kind": "shuffle", "nsb_param": 2024},
    {"gamma_deg": 240, "nsb_kind": "shuffle", "nsb_param": 2024},
    # --- temporal position: gammas only (unfair: tau stays tuned on unrolled NSB)
    {"gamma_time_shift": 2},
    {"gamma_time_shift": 5},
    # --- temporal position: both classes rolled (the fair test) ---------------
    {"gamma_time_shift": 2, "nsb_time_shift": 2},
    {"gamma_time_shift": 5, "nsb_time_shift": 5},
    # --- temporal position: NSB only -----------------------------------------
    {"nsb_time_shift": 5, "name": "nsb_only_troll5"},
    {"nsb_time_shift": 2, "name": "nsb_only_troll2"},
]


def argument_parser(description, default_output):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--condition", nargs=3, action="append", required=True,
        metavar=("NAME", "GAMMA_GLOB", "NSB_GLOB"),
        help="an NSB condition and its files (quote the globs); repeat for several, the first is the reference")
    parser.add_argument("--output", default=default_output, help=f"output folder (default: {default_output})")
    parser.add_argument("--quick", action="store_true", help="a few events per fold, to check that the script runs")
    return parser


def condition_files(args):
    """``{name: (gamma_files, nsb_files)}`` from the --condition arguments, reference first."""
    conditions = {}
    for name, gamma_glob, nsb_glob in args.condition:
        gamma_files, nsb_files = sorted(glob.glob(gamma_glob)), sorted(glob.glob(nsb_glob))
        if name in conditions:
            raise SystemExit(f"--condition {name} given twice")
        if not gamma_files or not nsb_files:
            raise SystemExit(f"--condition {name}: no file matches {gamma_glob!r} or {nsb_glob!r}")
        conditions[name] = (gamma_files, nsb_files)
        print(f"Condition '{name}': {len(gamma_files)} gamma files, {len(nsb_files)} NSB files.")
    return conditions


def threshold_events(args):
    """NSB events used to tune tau."""
    return QUICK_THRESHOLD_EVENTS if args.quick else THRESHOLD_EVENTS


def fold_specs(rows, condition_names, quick, reference_events=REFERENCE_EVENTS, fold_events=FOLD_EVENTS):
    """Fold rows -> make_rotation_folds specs.

    rows[0] (the reference fold) runs under every condition with
    ``reference_events``; the other rows under the reference condition
    (condition_names[0]) with ``fold_events``. ``quick`` caps every fold to
    QUICK_EVENTS.
    """
    specs = []
    for i, row in enumerate(rows):
        gamma_events, nsb_events = reference_events if i == 0 else fold_events
        if quick:
            gamma_events, nsb_events = QUICK_EVENTS
        conditions = list(condition_names) if i == 0 else [condition_names[0]]
        specs.append({"gamma_events": gamma_events, "nsb_events": nsb_events, "conditions": conditions, **row})
    return specs
