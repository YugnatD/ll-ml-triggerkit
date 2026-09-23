"""Pixel-index transforms + cross-validation folds for the statistics phase.

These build *representation-agnostic* pixel-index reindexings ``(P,)`` that are
applied to the raw ``(B, P, S)`` waveform (and the ``(B, P)`` pedestal) BEFORE
any grid remap / hexagdly, so a single transform serves both the TDSCAN
(pixel-list) and CNN (grid) chains.

Why this exists
---------------
During ``compute_statistics`` we want to detect a model that learned something
we did *not* intend -- a fixed camera orientation, specific "hot" pixels, an NSB
pedestal artefact, etc. A physically-honest trigger is invariant under a camera
rotation symmetry and under reshuffling spatially-iid NSB, so its efficiency and
rate stay flat across such transforms. If they *shift* across folds, the model
is leaking. Each :class:`Fold` pairs a gamma transform with an NSB transform;
the report compares metrics across folds and the spread is the leakage signal.

Transforms
----------
All current transforms are pure pixel-index permutations, expressed as an int
``(P,)`` vector ``idx`` with ``out[..., p, :] = in[..., idx[p], :]``:

* :func:`identity_index`   -- no-op.
* :func:`rotation_index`   -- exact rotation symmetry of the camera (asserted).
* :func:`roll_index`       -- circular roll of the pixel list (a decorrelated
  NSB draw; on a flat pixel list this is a structured permutation, not a spatial
  shift, which is fine for spatially-iid NSB).
* :func:`shuffle_index`    -- fixed-seed random permutation (your NSB shuffle).

Value transforms (e.g. perturbing the pedestal) do not fit the index-vector
model; add them later as a separate fold kind. The plumbing in
``TriggerChain.compute_statistics`` applies ``idx`` to waveform *and* pedestal
together (the baseline rotates with the camera).

Temporal roll (a second fold axis)
----------------------------------
A :class:`Fold` may ALSO carry a ``gamma_time_shift`` and an ``nsb_time_shift``
-- integer circular rolls of the waveform along the time-sample axis ``S``,
applied per class (``compute_statistics`` uses ``tf.roll`` on the ``(B, P, S)``
waveform, NOT the ``(B, P)`` pedestal, which has no time axis). This is a
*value* transform, orthogonal to the pixel index above: it moves the waveform in
time to test whether the model leaked the absolute temporal position of the
pulse. A physically-honest trigger's efficiency is flat under this roll; a drop
means it learned "the pulse always peaks near sample k".

Roll the GAMMAS ONLY and you are testing the signal side against a threshold
tuned on un-rolled NSB -- the two sides are no longer treated alike, so a change
in efficiency mixes real temporal leakage with the model's own response to the
window edges. Roll BOTH classes by the same amount for the fair test: a
time-translation-invariant trigger returns identical gamma efficiency AND
identical NSB rate. (This matters in practice: a filter whose temporal kernel is
zero-padded at the window boundary under-scores pulses sitting on the first and
last samples, so rolling moves them out of that blind spot and shifts the NSB
rate even though NSB values are perfectly stationary in time.)
"""

import numpy as np


def _pixel_pitch(x, y):
    """Median nearest-neighbour distance (numpy-only, no scipy)."""
    # Pairwise is fine for camera-scale P (~1300); avoids a scipy dependency.
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def rotation_permutation(geometry, deg, *, tol_frac=0.1):
    """Pixel-index permutation for an exact camera rotation symmetry.

    Rotates every pixel position by ``deg`` about the camera centroid and maps it
    to the nearest original pixel. Returns an int ``(P,)`` array ``perm`` with
    ``perm[p]`` = the pixel that pixel ``p`` lands on.

    Raises ``ValueError`` if ``deg`` is not an exact symmetry of *this* camera --
    i.e. the nearest-pixel map is not a bijection, or any pixel lands farther than
    ``tol_frac`` of a pixel pitch from its target. This is deliberate: a
    non-symmetric angle (e.g. 60 deg for a 3-fold-symmetric camera) must fail
    loudly rather than silently misplace pixels.
    """
    x = np.asarray(geometry.pix_x.value, dtype=float)
    y = np.asarray(geometry.pix_y.value, dtype=float)
    P = len(x)
    cx, cy = x.mean(), y.mean()
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = np.c_[x - cx, y - cy] @ R.T
    rx, ry = rot[:, 0] + cx, rot[:, 1] + cy

    # Nearest original pixel for each rotated position (numpy argmin).
    d2 = (rx[:, None] - x[None, :]) ** 2 + (ry[:, None] - y[None, :]) ** 2
    perm = d2.argmin(axis=1)
    resid = np.sqrt(d2[np.arange(P), perm])
    pitch = _pixel_pitch(x, y)

    if len(np.unique(perm)) != P:
        raise ValueError(
            f"rotation of {deg} deg is not a symmetry of camera "
            f"'{getattr(geometry, 'name', '?')}': the nearest-pixel map is not a "
            f"bijection ({P - len(np.unique(perm))} collisions).")
    if resid.max() > tol_frac * pitch:
        raise ValueError(
            f"rotation of {deg} deg is not an exact symmetry: max residual "
            f"{resid.max() / pitch:.3f} pitch > tol {tol_frac}. For a 3-fold "
            f"camera use multiples of 120 deg.")
    return perm.astype(np.int64)


def identity_index(num_pixels):
    """No-op reindexing ``arange(P)``."""
    return np.arange(num_pixels, dtype=np.int64)


def rotation_index(geometry, deg, *, tol_frac=0.1):
    """Alias of :func:`rotation_permutation` (index-vector naming)."""
    return rotation_permutation(geometry, deg, tol_frac=tol_frac)


def roll_index(num_pixels, shift):
    """Circular roll of the pixel list by ``shift``."""
    return np.roll(np.arange(num_pixels), int(shift)).astype(np.int64)


def shuffle_index(num_pixels, seed):
    """Fixed-seed random pixel permutation (reproducible)."""
    return np.random.default_rng(int(seed)).permutation(num_pixels).astype(np.int64)


class Fold:
    """One cross-validation fold: a gamma reindexing paired with an NSB one.

    ``gamma_index`` / ``nsb_index`` are int ``(P,)`` vectors (see the transform
    builders above). ``name`` labels the fold in the statistics output. ``config``
    is a small human-readable dict describing how the fold was built (gamma
    degrees, NSB kind + param, seed); it is stored per fold in the statistics
    HDF5 so the file is self-describing and reproducible.

    ``gamma_files`` / ``nsb_files`` optionally override the SOURCE files this
    fold reads from (default ``None`` = the chain's own files). This is what
    makes a fold able to express a real dataset swap -- e.g. a different NSB
    condition (low/medium/... simulated separately, not a permutation of the
    same data) -- rather than only a pixel reindex of one fixed dataset. See
    :func:`make_condition_folds`.

    ``max_gamma_events`` / ``max_nsb_events`` are this fold's OWN event budget
    (``None`` = no fold-specific cap; falls back to whatever the caller passes
    ``compute_statistics`` directly). ``make_rotation_folds`` and
    ``make_condition_folds`` both REQUIRE every spec to set these explicitly --
    see their docstrings -- so a script can give one fold (e.g. the one it
    cares about most) a bigger budget than its siblings without that being
    hidden behind an implicit "total / n_folds" split. Folds that share a read
    pass in ``compute_statistics`` (same source files) still stream together;
    the shared read runs as long as the LARGEST budget among them requires,
    and each fold simply stops accumulating once its own smaller budget is
    met -- no separate pass needed just because budgets differ.
    """

    def __init__(self, name, gamma_index, nsb_index, config=None,
                 gamma_time_shift=0, nsb_time_shift=0,
                 gamma_files=None, nsb_files=None,
                 max_gamma_events=None, max_nsb_events=None):
        self.name = str(name)
        self.gamma_index = np.asarray(gamma_index, dtype=np.int64)
        self.nsb_index = np.asarray(nsb_index, dtype=np.int64)
        self.config = dict(config or {})
        # Circular roll of the waveform along the time-sample axis (samples).
        # Applied waveform-only, per class; 0 = no temporal shift.
        self.gamma_time_shift = int(gamma_time_shift)
        self.nsb_time_shift = int(nsb_time_shift)
        # Per-fold data-source override (None = use the chain's own files).
        self.gamma_files = gamma_files
        self.nsb_files = nsb_files
        # Per-fold event budget (None = no fold-specific cap).
        self.max_gamma_events = None if max_gamma_events is None else int(max_gamma_events)
        self.max_nsb_events = None if max_nsb_events is None else int(max_nsb_events)
        if self.gamma_index.shape != self.nsb_index.shape:
            raise ValueError(
                f"fold {name!r}: gamma_index {self.gamma_index.shape} and "
                f"nsb_index {self.nsb_index.shape} must have the same length.")

    def __repr__(self):
        return (f"Fold(name={self.name!r}, P={self.gamma_index.size}, "
                f"gamma_time_shift={self.gamma_time_shift}, "
                f"nsb_time_shift={self.nsb_time_shift}, "
                f"gamma_files={'override' if self.gamma_files else 'chain'}, "
                f"nsb_files={'override' if self.nsb_files else 'chain'}, "
                f"max_gamma_events={self.max_gamma_events}, "
                f"max_nsb_events={self.max_nsb_events})")


def make_condition_folds(geometry, conditions):
    """Build one no-reindex :class:`Fold` per named data-source condition.

    Unlike :func:`make_rotation_folds` (a synthetic pixel permutation of ONE
    fixed dataset, used to detect leakage), each fold here swaps in a REAL,
    independently-simulated dataset -- e.g. a different NSB brightness level
    (low/medium/high) that was simulated separately, with its OWN gamma files
    too (a lower/higher NSB regime is re-simulated for gammas as well, not just
    for the bias-curve file). ``gamma_index``/``nsb_index`` stay identity (no
    pixel reindex); only the source files change.

    ``conditions`` is a ``{name: (gamma_files, nsb_files, gamma_events,
    nsb_events)}`` mapping -- every condition sets its OWN event budget
    explicitly (no implicit "total / n_conditions" split), e.g.::

        {n: (*dataset_config.get_condition(n), 1_000_000, 400_000)
         for n in dataset_config.CONDITIONS}

    Add a condition to ``dataset_config.CONDITIONS`` (e.g. "high",
    "real_data") and it flows through here unchanged.

    Every resulting fold lands in the SAME statistics HDF5 as any other fold
    (rotation or condition) passed to ``compute_statistics(folds=...)`` -- one
    file, one ``fold`` column, one ``/folds`` summary group -- so
    ``StatPlotter(fold=<condition name>)`` (or a per-plot ``fold=`` override)
    selects a condition exactly like it selects a rotation fold.
    """
    P = int(geometry.n_pixels)
    identity = identity_index(P)
    folds = []
    for name, spec in conditions.items():
        gamma_files, nsb_files, gamma_events, nsb_events = spec
        folds.append(Fold(
            name, gamma_index=identity, nsb_index=identity,
            config={"condition": name},
            gamma_files=list(gamma_files), nsb_files=list(nsb_files),
            max_gamma_events=gamma_events, max_nsb_events=nsb_events,
        ))
    return folds


#: Keys accepted by a dict-form fold spec, with their defaults.
FOLD_SPEC_KEYS = {
    "name": None,                # optional explicit fold name (auto-derived if None)
    "gamma_deg": 0,              # camera rotation applied to gamma rows (degrees)
    "gamma_time_shift": 0,       # circular roll of the gamma waveform, in samples
    "nsb_kind": "original",      # pixel transform for NSB: original / rolled / shuffle
    "nsb_param": None,           # roll shift or shuffle seed (kind-dependent)
    "nsb_time_shift": 0,         # circular roll of the NSB waveform, in samples
}

#: Keys every dict-form fold spec MUST set itself -- no default, deliberately.
#: This fold's own gamma / NSB event budget: e.g. give "rot0_original" a
#: bigger budget than its siblings for more precise stats on the fold you
#: care about most, instead of an implicit total-events / n_folds split.
_REQUIRED_SPEC_KEYS = ("gamma_events", "nsb_events")


def _normalize_spec(spec, i):
    """Return ``(gamma_deg, gamma_time_shift, kind, param, nsb_time_shift,
    name, gamma_events, nsb_events)``.

    Dict form only (self-documenting; see make_rotation_folds). The legacy
    positional ``(gamma_aug, nsb_aug)`` tuple form predates per-fold event
    budgets and isn't extended to support them -- convert those rows to the
    dict form instead.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"fold spec #{i}: the legacy (gamma_aug, nsb_aug) tuple form is no "
            "longer accepted (it predates per-fold gamma_events/nsb_events, "
            "which every fold must now set explicitly) -- use the dict form.")

    missing = [k for k in _REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise ValueError(
            f"fold spec #{i}: missing required key(s) {missing} -- every fold "
            "spec must set its own gamma_events/nsb_events explicitly (no "
            "implicit total-events / n_folds split).")

    allowed = set(FOLD_SPEC_KEYS) | set(_REQUIRED_SPEC_KEYS)
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(
            f"fold spec #{i}: unknown key(s) {sorted(unknown)}; "
            f"allowed keys are {sorted(allowed)}.")
    g = dict(FOLD_SPEC_KEYS, **spec)
    # gamma_events/nsb_events must be PRESENT (checked above) but their value
    # may explicitly be None -- "this fold is uncapped, use every available
    # event" is a legitimate, deliberate choice, not a forgotten budget.
    gamma_events = None if g["gamma_events"] is None else int(g["gamma_events"])
    nsb_events = None if g["nsb_events"] is None else int(g["nsb_events"])
    return (g["gamma_deg"], int(g["gamma_time_shift"]), g["nsb_kind"],
            g["nsb_param"], int(g["nsb_time_shift"]), g["name"],
            gamma_events, nsb_events)


def make_rotation_folds(geometry, specs, *, seed=1337, tol_frac=0.1):
    """Build a list of :class:`Fold` from compact fold specs.

    Each spec row is a flat dict; every key is optional EXCEPT ``gamma_events``
    /``nsb_events``, which every row must set explicitly -- this fold's own
    event budget (no implicit total-events / n_folds split, so e.g. the one
    fold you care most about can get a bigger budget than its siblings). The
    other key names are exactly those stored in the per-fold ``config`` of the
    statistics HDF5, so the spec and the output file read the same::

        make_rotation_folds(chain.geom, [
            {"gamma_events": 100_000, "nsb_events": 40_000},  # reference fold
            {"gamma_events": 100_000, "nsb_events": 40_000, "gamma_deg": 120},
            {"gamma_events": 100_000, "nsb_events": 40_000,
             "nsb_kind": "shuffle", "nsb_param": 2024},        # reshuffle NSB pixels
            {"gamma_events": 500_000, "nsb_events": 40_000,
             "gamma_time_shift": 5, "nsb_time_shift": 5},      # this one gets more stats
            {"gamma_events": 100_000, "nsb_events": 40_000,
             "nsb_time_shift": 5, "name": "nsb_only_troll5"},  # roll NSB only
        ])

    Keys and defaults: ``name`` (auto), ``gamma_deg`` (0),
    ``gamma_time_shift`` (0), ``nsb_kind`` ("original"), ``nsb_param`` (None),
    ``nsb_time_shift`` (0); ``gamma_events``/``nsb_events`` have NO default and
    are required. An unknown or missing key raises, so typos (and a forgotten
    budget) surface immediately.
    """
    P = int(geometry.n_pixels)
    folds = []
    used_names = {}
    for i, spec in enumerate(specs):
        (gamma_deg, gamma_time_shift, kind, param,
         nsb_time_shift, explicit_name, gamma_events, nsb_events) = _normalize_spec(spec, i)
        gamma_idx = rotation_index(geometry, gamma_deg, tol_frac=tol_frac)
        if kind == "original":
            nsb_idx = identity_index(P)
            nsb_param = None
        elif kind == "rolled":
            nsb_param = param if param is not None else P // 2
            nsb_idx = roll_index(P, nsb_param)
        elif kind == "shuffle":
            nsb_param = param if param is not None else seed + i
            nsb_idx = shuffle_index(P, nsb_param)
        else:
            raise ValueError(f"unknown nsb_kind {kind!r}; use original/rolled/shuffle.")
        config = {
            "gamma_deg": gamma_deg,
            "nsb_kind": kind,
            "nsb_param": nsb_param,
            "gamma_time_shift": gamma_time_shift,
            "nsb_time_shift": nsb_time_shift,
            "seed": seed,
        }
        # Fold name must be unique (it keys the /folds table and StatPlotter's
        # fold lookup). Base name is rot<deg>_<kind>; append the param when set,
        # a _troll<k> suffix for a temporal roll, and a _<n> suffix only if that
        # still collides (e.g. two identical rows).
        name = explicit_name or f"rot{gamma_deg}_{kind}"
        if explicit_name is None:
            if nsb_param is not None:
                name = f"{name}{nsb_param}"
            if gamma_time_shift:
                name = f"{name}_troll{gamma_time_shift}"
            if nsb_time_shift:
                name = f"{name}_ntroll{nsb_time_shift}"
        if name in used_names:
            used_names[name] += 1
            name = f"{name}_{used_names[name]}"
        else:
            used_names[name] = 0
        folds.append(Fold(name, gamma_idx, nsb_idx, config=config,
                          gamma_time_shift=gamma_time_shift,
                          nsb_time_shift=nsb_time_shift,
                          max_gamma_events=gamma_events,
                          max_nsb_events=nsb_events))
    return folds
