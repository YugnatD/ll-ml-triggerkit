"""SST-1M (DigiCam) trigger tables, built from the official camera_config.cfg.

camera_config.cfg (shipped next to this module, same file as in sst1mpipe,
digicam and CTS) gives, for each of the 1296 pixels, its position, its patch
(patch_sw_id) and the 7 patches of the patch7 cluster of that patch. All the
tables used by the trigger stages are derived from it, in the same order as
the ConfigFile_SST1M/*.csv files they replace:

* pixel_positions():   (1296, 2) pixel positions, CSV units (cm, camera rotated by 90 deg)
* patch_triplets():    (432, 3) pixels of each patch, increasing pixel id
* patch7_clusters():   patches of the patch7 cluster of each patch, itself first
* patch_positions():   (432, 2) position of each patch = its first pixel
* tdscan_neighbors(e): TDSCAN neighbourhood of radius e, one slot per hexagon cell
"""
from functools import lru_cache
from importlib.resources import files

import numpy as np

N_PIXELS = 1296
N_PATCHES = 432

# Columns of camera_config.cfg (see its header).
_PIXEL_SW_ID, _X_PIXEL, _Y_PIXEL, _PATCH_SW_ID = 8, 9, 10, 11
_CLUSTER7 = slice(16, 23)


@lru_cache(maxsize=None)
def _camera_config():
    """camera_config.cfg rows sorted by pixel_sw_id."""
    with files("triggerkit.camera").joinpath("sst1m_camera_config.cfg").open() as f:
        rows = np.array([line.split() for line in f if line.strip() and not line.startswith("#")], dtype=float)
    rows = rows[np.argsort(rows[:, _PIXEL_SW_ID])]
    if not np.array_equal(rows[:, _PIXEL_SW_ID], np.arange(N_PIXELS)):
        raise ValueError("camera_config.cfg must list each of the 1296 pixels once")
    return rows


def pixel_positions():
    """(1296, 2) pixel positions. camera_config.cfg gives (x, y) in mm; the
    historical CSVs (and every geometry built from them) use (-y, -x) in cm."""
    rows = _camera_config()
    return np.round(np.c_[-rows[:, _Y_PIXEL], -rows[:, _X_PIXEL]] / 10, 3)  # CSV precision: 0.01 mm


def patch_triplets():
    """(432, 3) pixel ids of each patch, in increasing order."""
    patch_of_pixel = _camera_config()[:, _PATCH_SW_ID].astype(int)
    return np.array([np.flatnonzero(patch_of_pixel == patch) for patch in range(N_PATCHES)])


def patch7_clusters():
    """List of 432 lists: the patch7 cluster of each patch (the patch itself
    first, then its neighbours in camera_config.cfg order; fewer than 7 at the edge)."""
    rows = _camera_config()
    clusters = {}
    for row in rows:
        clusters[int(row[_PATCH_SW_ID])] = [int(p) for p in row[_CLUSTER7] if p >= 0]
    return [clusters[patch] for patch in range(N_PATCHES)]


def patch_positions():
    """(432, 2) position of each patch, taken as the position of its first pixel."""
    return pixel_positions()[patch_triplets()[:, 0]]


def _patch_hex_coordinates():
    """Integer coordinates (u, v) of the patches on their hexagonal grid.

    Patches sit on rows of constant y; along a row they are 2 steps of u apart
    and consecutive rows are shifted by 1 step (u and v have the same parity).
    v grows with decreasing y, u with increasing x.
    """
    x, y = patch_positions().T
    row_y = np.unique(np.round(y, 3))
    row_step = np.diff(row_y).min()
    v = np.round((row_y.max() - y) / row_step).astype(int)
    half_step = min(np.diff(np.sort(x[v == row])).min() for row in np.unique(v) if np.sum(v == row) > 1) / 2
    u = np.round((x - x.min()) / half_step).astype(int)
    if np.any((u - v) % 2):
        raise ValueError("Patch positions are not on a hexagonal grid")
    return u, v


def tdscan_neighbors(eps_xy):
    """TDSCAN neighbour table (432, n_slots) for a hexagon of radius eps_xy.

    Slot s always holds the same offset from the patch: the cells of the
    hexagon row by row (top to bottom), left to right in each row; the patch
    itself is the middle slot. -1 where the cell is outside the camera.
    """
    u, v = _patch_hex_coordinates()
    patch_at = {(ui, vi): patch for patch, (ui, vi) in enumerate(zip(u, v))}
    offsets = [
        (du, dv)
        for dv in range(-eps_xy, eps_xy + 1)
        for du in range(-2 * eps_xy, 2 * eps_xy + 1)
        if (du + dv) % 2 == 0 and abs(dv) + abs(du) <= 2 * eps_xy
    ]
    return np.array([[patch_at.get((u[p] + du, v[p] + dv), -1) for du, dv in offsets] for p in range(N_PATCHES)])
