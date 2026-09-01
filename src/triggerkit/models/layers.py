"""Adapter layers bridging the 1D pixel-list waveform and the 2D hex grid.

These are plain Keras layers so they can sit *as explicit entries in a declared
layer list* (see :class:`triggerkit.models.sequential.SequentialBody`). Only
:class:`ScatterToGrid` carries state (the scatter matrix); the two pooling
layers are thin, serializable wrappers over ``keras.ops`` reductions.

None of these need ``keras_hexagdly`` -- only the hex convolutions themselves
do, so the grid remap and pooling stay import-safe.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from triggerkit.models.grid import GridTransform


@keras.utils.register_keras_serializable(package="triggerkit")
class ScatterToGrid(keras.layers.Layer):
    """Scatter a ``(B, P, S)`` waveform onto a ``(B, T, H, W, 1)`` grid volume.

    ``T`` is the kept time window ``[time_skip, time_skip + time_window)``; when
    ``time_window`` is ``None`` the full remaining span is kept. The scatter is a
    constant ``(P, H*W)`` matmul (one-hot per pixel), exactly the trick the hex
    CNN sandbox used, kept inside the graph so adapter + CNN train/save as one.

    Two ways to build it:

    * ``ScatterToGrid(time_window=32)`` -- *unbound*: you only give the time knobs
      and the camera geometry is bound later (``SequentialBody`` / ``build_chain``
      call :meth:`bind_geometry` with ``chain.geom`` before applying it). This is
      the declarative form used in a layer list.
    * :meth:`from_geometry` or ``ScatterToGrid(matrix, H, W)`` -- *bound* up front.
    """

    def __init__(self, scatter_matrix=None, H=None, W=None, *, time_skip=0, time_window=None, **kwargs):
        super().__init__(**kwargs)
        self.time_skip = int(time_skip)
        self.time_window = None if time_window is None else int(time_window)
        if scatter_matrix is None:
            # Unbound: geometry supplied later via bind_geometry().
            self.scatter_matrix = None
            self.H = None if H is None else int(H)
            self.W = None if W is None else int(W)
            self._M = None
        else:
            self.scatter_matrix = np.asarray(scatter_matrix, dtype=np.float32)
            self.H = int(H)
            self.W = int(W)
            self._M = tf.constant(self.scatter_matrix)  # (P, H*W)

    @property
    def bound(self):
        return self._M is not None

    def bind_geometry(self, geometry):
        """Fit the grid from a camera geometry and fill the scatter matrix in place.

        No-op if already bound. Called by the body builder before the layer is
        applied, so an unbound ``ScatterToGrid(time_window=...)`` in a layer list
        picks up ``chain.geom`` automatically.
        """
        if self.bound:
            return self
        grid = GridTransform(geometry)
        self.scatter_matrix = np.asarray(grid.scatter_matrix(), dtype=np.float32)
        self.H, self.W = int(grid.H), int(grid.W)
        self._M = tf.constant(self.scatter_matrix)
        return self

    @classmethod
    def from_geometry(cls, geometry, *, time_skip=0, time_window=None, **kwargs):
        """Fit a :class:`~triggerkit.models.grid.GridTransform` and build the layer."""
        grid = GridTransform(geometry)
        return cls(
            grid.scatter_matrix(), grid.H, grid.W,
            time_skip=time_skip, time_window=time_window, **kwargs,
        )

    def call(self, x):
        if self._M is None:
            raise RuntimeError(
                "ScatterToGrid is unbound: no camera geometry. Put it in a "
                "SequentialBody / build_chain (which calls bind_geometry with "
                "chain.geom), or construct it via ScatterToGrid.from_geometry().")
        x = tf.cast(x, tf.float32)                       # (B, P, S)
        t0 = self.time_skip
        t1 = None if self.time_window is None else t0 + self.time_window
        x = x[:, :, t0:t1]                               # (B, P, T)
        x = tf.transpose(x, [0, 2, 1])                   # (B, T, P)
        g = tf.linalg.matmul(x, self._M)                 # (B, T, H*W)
        return tf.reshape(g, [-1, tf.shape(g)[1], self.H, self.W, 1])

    def compute_output_shape(self, input_shape):
        T = self.time_window
        if T is None and input_shape[-1] is not None:
            T = input_shape[-1] - self.time_skip
        return (input_shape[0], T, self.H, self.W, 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "scatter_matrix": None if self.scatter_matrix is None else self.scatter_matrix.tolist(),
            "H": self.H, "W": self.W,
            "time_skip": self.time_skip, "time_window": self.time_window,
        })
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class TimeMean(keras.layers.Layer):
    """Mean over the time axis of a ``(B, T, H, W, C)`` volume -> ``(B, H, W, C)``.

    Matches ``nn.AdaptiveAvgPool3d((1, None, None))`` in the hex CNN backbone.
    """

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return keras.ops.mean(x, axis=self.axis, keepdims=False)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.utils.register_keras_serializable(package="triggerkit")
class GlobalHexMean(keras.layers.Layer):
    """Global mean over the spatial (H, W) axes -> ``(B, C)`` feature vector.

    Matches ``nn.AdaptiveAvgPool2d((1, 1))`` in the hex CNN backbone.
    """

    def __init__(self, axes=(1, 2), **kwargs):
        super().__init__(**kwargs)
        self.axes = tuple(axes)

    def call(self, x):
        return keras.ops.mean(x, axis=self.axes, keepdims=False)

    def get_config(self):
        config = super().get_config()
        config.update({"axes": list(self.axes)})
        return config
