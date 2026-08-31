"""``GridTransform``: map the camera's 1D pixel list onto a dense 2D hex grid.

Ported verbatim from the research sandbox (``train_hexagdly.py``). It is the
piece the hex CNN needs to remap ``(B, P, ...)`` waveforms/images to the
``(B, H, W, ...)`` grid ``keras_hexagdly`` convolves over. Pure numpy plus a
``ctapipe`` ``CameraGeometry``; no TensorFlow, so it is safe to import without
the optional hex-CNN dependencies.
"""

import numpy as np


class GridTransform:
    """Convert 1D camera images into a 2D grid using CameraGeometry."""

    def __init__(self, geometry):
        x = geometry.pix_x.value
        y = geometry.pix_y.value

        neighbor_vectors = []
        for index, neighbors in enumerate(geometry.neighbors):
            for neighbor in neighbors:
                if index < neighbor:
                    neighbor_vectors.append(
                        (x[neighbor] - x[index], y[neighbor] - y[index]))
        neighbor_vectors = np.asarray(neighbor_vectors)
        neighbor_distance = np.linalg.norm(neighbor_vectors, axis=1)
        horizontal_mask = (
            np.abs(neighbor_vectors[:, 1]) < np.median(neighbor_distance) * 1e-3)
        if not np.any(horizontal_mask):
            raise ValueError("Could not find horizontal neighbor vectors")

        horizontal_pitch = np.median(np.abs(neighbor_vectors[horizontal_mask, 0]))
        vertical_pitch = np.median(np.abs(neighbor_vectors[~horizontal_mask, 1]))

        best = None
        for y_origin in (0.0, y.max(), y.min()):
            axial_r = np.rint((y - y_origin) / vertical_pitch).astype(np.int64)
            axial_q = np.rint((x - x.min()) / horizontal_pitch - axial_r / 2).astype(np.int64)
            row_idx, col_idx = self._offset_coordinates_from_axial(axial_q, axial_r)
            row_idx = row_idx - row_idx.min()
            col_idx = col_idx - col_idx.min()
            mismatch_count = self._neighbor_mismatch_count(geometry, row_idx, col_idx)
            if best is None or mismatch_count < best[0]:
                best = (mismatch_count, axial_q, axial_r, row_idx, col_idx)

        mismatch_count, q_idx, r_idx, row_idx, col_idx = best
        if mismatch_count != 0:
            raise ValueError(
                f"Hexagdly grid does not match camera neighbors: "
                f"{mismatch_count} mismatched pixels")

        self.neighbor_mismatch_count = mismatch_count
        self.x_idx = q_idx
        self.y_idx = r_idx
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.H = int(self.row_idx.max().item()) + 1
        self.W = int(self.col_idx.max().item()) + 1

        assert len(self.x_idx) == len(self.y_idx) == geometry.n_pixels
        assert len(set(zip(self.row_idx, self.col_idx))) == geometry.n_pixels

    @staticmethod
    def _offset_coordinates_from_axial(q_idx, r_idx):
        col_idx = q_idx.astype(np.int64)
        row_idx = r_idx + np.floor_divide(q_idx - (q_idx & 1), 2)
        return row_idx.astype(np.int64), col_idx.astype(np.int64)

    @staticmethod
    def _hexagdly_neighbor_indexes_from_grid(index_grid, row, col):
        if col % 2 == 0:
            diagonal_offsets = [(-1, -1), (-1, 1), (0, -1), (0, 1)]
        else:
            diagonal_offsets = [(0, -1), (0, 1), (1, -1), (1, 1)]
        neighbor_indexes = []
        for row_offset, col_offset in [(-1, 0), (1, 0), *diagonal_offsets]:
            nr, nc = row + row_offset, col + col_offset
            if 0 <= nr < index_grid.shape[0] and 0 <= nc < index_grid.shape[1]:
                ni = index_grid[nr, nc]
                if ni >= 0:
                    neighbor_indexes.append(int(ni))
        return sorted(neighbor_indexes)

    @classmethod
    def _neighbor_mismatch_count(cls, geometry, row_idx, col_idx):
        index_grid = np.full((int(row_idx.max()) + 1, int(col_idx.max()) + 1), -1, dtype=int)
        for index, (row, col) in enumerate(zip(row_idx, col_idx)):
            if index_grid[row, col] >= 0:
                return geometry.n_pixels
            index_grid[row, col] = index
        mismatch_count = 0
        for index, camera_neighbors in enumerate(geometry.neighbors):
            grid_neighbors = cls._hexagdly_neighbor_indexes_from_grid(
                index_grid, int(row_idx[index]), int(col_idx[index]))
            if grid_neighbors != sorted(map(int, camera_neighbors)):
                mismatch_count += 1
        return mismatch_count

    def scatter_matrix(self):
        """(P, H*W) matrix M with M[p, row_p*W + col_p] = 1, for image = values @ M."""
        P = len(self.row_idx)
        M = np.zeros((P, self.H * self.W), dtype=np.float32)
        flat = self.row_idx * self.W + self.col_idx
        M[np.arange(P), flat] = 1.0
        return M

    def __repr__(self):
        return (f"GridTransform(H={self.H}, W={self.W}, "
                f"neighbor_mismatches={self.neighbor_mismatch_count})")
