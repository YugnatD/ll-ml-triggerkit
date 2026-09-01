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
    """

    def __init__(self, name, gamma_index, nsb_index, config=None):
        self.name = str(name)
        self.gamma_index = np.asarray(gamma_index, dtype=np.int64)
        self.nsb_index = np.asarray(nsb_index, dtype=np.int64)
        self.config = dict(config or {})
        if self.gamma_index.shape != self.nsb_index.shape:
            raise ValueError(
                f"fold {name!r}: gamma_index {self.gamma_index.shape} and "
                f"nsb_index {self.nsb_index.shape} must have the same length.")

    def __repr__(self):
        return f"Fold(name={self.name!r}, P={self.gamma_index.size})"


def make_rotation_folds(geometry, specs, *, seed=1337, tol_frac=0.1):
    """Build a list of :class:`Fold` from compact ``(gamma_deg, nsb_kind)`` specs.

    ``specs`` is any-length list of ``(gamma_deg, nsb_kind)`` where ``nsb_kind`` is
    one of ``"original"``, ``"rolled"``, ``"shuffle"`` -- or a tuple
    ``("rolled", shift)`` / ``("shuffle", seed)`` to override the parameter. The
    NSB seed defaults to ``seed + fold_position`` so different shuffle folds get
    distinct-but-reproducible permutations.

    Example (your draft)::

        make_rotation_folds(chain.geom, [(0, "original"),
                                         (120, "rolled"),
                                         (240, "shuffle")])
    """
    P = int(geometry.n_pixels)
    folds = []
    used_names = {}
    for i, (gamma_deg, nsb_kind) in enumerate(specs):
        gamma_idx = rotation_index(geometry, gamma_deg, tol_frac=tol_frac)
        kind, param = (nsb_kind, None) if isinstance(nsb_kind, str) else nsb_kind
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
            "seed": seed,
        }
        # Fold name must be unique (it keys the /folds table and StatPlotter's
        # fold lookup). Base name is rot<deg>_<kind>; append the param when set,
        # and a _<n> suffix only if that still collides (e.g. two identical rows).
        name = f"rot{gamma_deg}_{kind}"
        if nsb_param is not None:
            name = f"{name}{nsb_param}"
        if name in used_names:
            used_names[name] += 1
            name = f"{name}_{used_names[name]}"
        else:
            used_names[name] = 0
        folds.append(Fold(name, gamma_idx, nsb_idx, config=config))
    return folds
