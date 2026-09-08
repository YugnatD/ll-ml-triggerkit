"""The time-coincidence CNN body: a custom SequentialBody architecture.

Lives OUTSIDE triggerkit on purpose. SequentialBody's whole point is that a
user builds a new architecture in their own file -- existing library
primitives (ScatterToGrid, GlobalHexMean, ...) plus their own Keras layers and
the documented `fn(chain)` escape hatch -- with no library PR required. This
module is the proof: everything below is specific to ONE still-experimental
architecture (arrival-time coincidence, see train_timecoinc.py) and has no
business in the shared package's public API.

Registered under package="timecoinc_research", not "triggerkit": these classes
are not part of the library's namespace. Loading a saved model built from this
module needs `import timecoinc_layers` first (exactly as any external user's
custom layers would), the same way triggerkit's own custom objects need
`triggerkit.models.register_custom_objects()` first.

Shape
-----
    scatter_to_grid                      (B, P, S) -> (B, T, H, W, 1)
    hex conv per time slice + Conv3D(taps,1,1) + ReLU
    TimeSummary                          -> (B, H, W, 3C): peak, charge, t-bar
    ring conv on t-bar, (t-bar - ring)^2 -> local time dispersion
    MaskCells                            (432 real pixels of 576 cells)
    [hex conv k=2 s=2 + ReLU] x2
    GlobalHexMeanMax                     (feature vector; head added by build_chain)

Measured on SST-1M R0Alpha: **+0.019 AUC** over a 10-weight TDSCAN (roughly +6%
gamma efficiency at 50 kHz), see train_timecoinc.py for the reproduction and
its current status (ring-sharing variant has an unresolved training-stability
question as of the last sweep).

``drop_absolute_time`` (default True) removes the raw t-bar channel: the
absolute arrival time is a leak in simulated data (readout window positioned
by the trigger), the dispersion feature is shift-invariant and immune.

``share_neighbors="ring"`` takes the body from ~17.9k weights to ~3.5k for a
cost inside the seed-to-seed spread in the sandbox -- NOT yet confirmed stable
in this port, see the ring-sweep results.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from triggerkit.models import GridTransform, ScatterToGrid, SequentialBody


@keras.utils.register_keras_serializable(package="timecoinc_research")
class TimeSummary(keras.layers.Layer):
    """Collapse ``(B, T, H, W, C)`` to ``(B, H, W, 3C)`` with THREE per-pixel summaries.

    ``TimeMean`` keeps one number per pixel -- the average over the window. That
    throws away *when* the light arrived, and with it the question that separates
    a shower from night-sky background best: do neighbouring pixels peak at the
    SAME instant? Shower light is coherent in time, NSB peaks are independent.

    So we keep three channels per input channel, in this order:

    ``[0:C]``    peak height, ``max`` over time
    ``[C:2C]``   total charge, ``mean`` over time
    ``[2C:3C]``  charge-weighted mean arrival time, normalised to ``[0, 1]``

    The third one is the interesting one: compared against its own neighbourhood
    it gives a local time dispersion -- small inside a shower, large in noise.
    Measured on SST-1M, using it is worth about +0.019 AUC over an
    amplitude-only network.

    Note it needs one division per pixel, which is not free in HLS. Propagating
    numerator and denominator separately, or using an argmax instead of the
    centroid, is the way out if that matters.
    """

    def __init__(self, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def call(self, x):
        t = tf.shape(x)[1]
        tvec = tf.reshape(
            tf.range(t, dtype=x.dtype) / tf.cast(t, x.dtype), (1, -1, 1, 1, 1))
        w = tf.nn.relu(x)
        den = tf.reduce_sum(w, axis=1) + self.eps
        peak = tf.reduce_max(x, axis=1)
        charge = tf.reduce_mean(x, axis=1)
        tbar = tf.reduce_sum(w * tvec, axis=1) / den
        return tf.concat([peak, charge, tbar], axis=-1)

    def compute_output_shape(self, input_shape):
        b, _t, h, w, c = input_shape
        return (b, h, w, None if c is None else 3 * c)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg


@keras.utils.register_keras_serializable(package="timecoinc_research")
class MaskCells(keras.layers.Layer):
    """Zero the grid cells that are not real pixels.

    The hex grid is a rectangle: DigiCam's 432 pixels sit in 24x24 = 576 cells,
    so a quarter of them hold nothing. A ``mean`` merely dilutes them, but a
    ``max`` -- or a count -- would happily fire on an empty cell carrying only a
    bias. Anything that reduces by anything other than a mean needs this first.

    Declared unbound (``MaskCells()``) and bound to the camera geometry by the
    body builder, like :class:`ScatterToGrid`.
    """

    def __init__(self, mask=None, **kwargs):
        super().__init__(**kwargs)
        self.mask = None if mask is None else np.asarray(mask, dtype=np.float32)
        self._m = None if mask is None else tf.constant(
            self.mask[None, :, :, None])

    @property
    def bound(self):
        return self._m is not None

    def bind_geometry(self, geometry):
        if self.bound:
            return
        grid = GridTransform(geometry)
        occupied = (np.asarray(grid.scatter_matrix()).sum(axis=0) > 0)
        self.mask = occupied.reshape(grid.H, grid.W).astype(np.float32)
        self._m = tf.constant(self.mask[None, :, :, None])

    def call(self, x):
        if self._m is None:
            raise RuntimeError("MaskCells is unbound: call bind_geometry() first.")
        return x * self._m

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mask": None if self.mask is None else self.mask.tolist()})
        return cfg


@keras.utils.register_keras_serializable(package="timecoinc_research")
class GlobalHexMeanMax(keras.layers.Layer):
    """``(B, H, W, C)`` -> ``(B, 2C)``: the global mean AND the global max.

    The mean says how much light there is in total, the max how concentrated it
    is. Neither alone tells a faint wide shower from a bright speck of noise,
    which is why both are kept and the classifier decides how to weigh them.
    """

    def __init__(self, axes=(1, 2), **kwargs):
        super().__init__(**kwargs)
        self.axes = tuple(axes)

    def call(self, x):
        return tf.concat([tf.reduce_mean(x, axis=self.axes),
                          tf.reduce_max(x, axis=self.axes)], axis=-1)

    def compute_output_shape(self, input_shape):
        b, _h, _w, c = input_shape
        return (b, None if c is None else 2 * c)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axes": list(self.axes)})
        return cfg


@keras.utils.register_keras_serializable(package="timecoinc_research")
class SquaredDifference(keras.layers.Layer):
    """``(a, b) -> (a - b)**2``, elementwise.

    A registered layer rather than a ``Lambda``: Keras refuses to deserialize a
    ``Lambda`` wrapping a Python lambda unless the caller passes
    ``safe_mode=False``, so a model containing one cannot be reloaded normally.
    Used for the local time dispersion, ``(t_bar - ring_mean(t_bar))**2``.
    """

    def call(self, inputs):
        a, b = inputs
        return tf.square(a - b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.utils.register_keras_serializable(package="timecoinc_research")
class HexOverTime(keras.layers.Layer):
    """Apply one hex convolution to every time slice of a ``(B, T, H, W, C)`` volume.

    hexagdly's ``Conv2d`` is 2D, so the time axis is folded into the batch, the
    convolution runs once over ``B*T`` images, and the result is unfolded. That
    keeps the spatial filter shared across time -- which is what we want -- while
    leaving the temporal mixing to a separate ``Conv3D(taps, 1, 1)``.

    Memory note: the convolution sees ``B * T`` images, so activation memory
    scales with the batch times the time window, not with the batch alone. With
    T = 50 a batch of 256 events is 12,800 images and will exhaust a 24 GB card;
    score in sub-batches at inference.
    """

    def __init__(self, filters, kernel_size=1, share_neighbors=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.share_neighbors = share_neighbors
        self._conv = None

    def build(self, input_shape):
        import keras_hexagdly as hgly
        self._conv = hgly.Conv2d(
            self.filters, kernel_size=self.kernel_size, strides=1,
            use_bias=True, share_neighbors=self.share_neighbors,
            name=f"{self.name}_conv")
        # A parent's build() must create ALL of its children's state, otherwise
        # the saved model cannot be restored: Keras reloads the child unbuilt
        # and refuses. The child sees the time-folded shape (B*T, H, W, C).
        _b, _t, h, w, c = input_shape
        self._conv.build((None, h, w, c))
        super().build(input_shape)

    def call(self, x):
        shape = tf.shape(x)
        h, w, c = x.shape[2], x.shape[3], x.shape[4]
        y = tf.reshape(x, (-1, h, w, c))
        y = self._conv(y)
        return tf.reshape(y, (shape[0], shape[1], h, w, self.filters))

    def compute_output_shape(self, input_shape):
        b, t, h, w, _c = input_shape
        return (b, t, h, w, self.filters)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "kernel_size": self.kernel_size,
                    "share_neighbors": self.share_neighbors})
        return cfg


def _time_dispersion(kernel_size, drop_absolute_time):
    """Escape-hatch step: local time dispersion, appended to the summary.

    A branch, so it cannot be a plain list entry: it slices t-bar out of the
    summary, takes its ring mean, and appends the squared difference.
    """

    def _step(chain):
        import keras_hexagdly as hgly
        x = chain.last_layer                       # (B, H, W, 3C)
        c = int(x.shape[-1]) // 3
        tbar = x[..., 2 * c:3 * c]
        ring = hgly.Conv2d(c, kernel_size=kernel_size, strides=1,
                           use_bias=False, share_neighbors="ring",
                           name="ring_time")(tbar)
        disp = SquaredDifference(name="time_dispersion")([tbar, ring])
        keep = x[..., :2 * c] if drop_absolute_time else x
        chain.last_layer = keras.ops.concatenate([keep, disp], axis=-1)
        return None

    return _step


def time_coincidence_layers(
    *,
    time_skip=0,
    time_window=50,
    pre_channels=8,
    spatial_channels=(16, 32),
    taps=5,
    kernel_size=1,
    share_neighbors=False,
    drop_absolute_time=True,
):
    """Layer list for the time-coincidence body. See module docstring."""
    layers = [
        ScatterToGrid(time_skip=time_skip, time_window=time_window),
        HexOverTime(pre_channels, kernel_size=kernel_size,
                    share_neighbors=share_neighbors, name="spatial_pre0"),
        keras.layers.Conv3D(pre_channels, (taps, 1, 1), padding="same",
                            name="temporal_0"),
        keras.layers.ReLU(),
        TimeSummary(name="time_summary"),
        _time_dispersion(kernel_size, drop_absolute_time),
        MaskCells(name="mask_cells"),
    ]
    import keras_hexagdly as hgly
    for i, out_c in enumerate(spatial_channels):
        layers.append(hgly.Conv2d(out_c, kernel_size=2, strides=2, use_bias=True,
                                  share_neighbors=share_neighbors,
                                  name=f"spatial_{i}"))
        layers.append(keras.layers.ReLU())
    layers.append(GlobalHexMeanMax(name="global_hex_mean_max"))
    return layers


def time_coincidence_body(**kwargs):
    """``SequentialBody`` wrapping :func:`time_coincidence_layers`."""
    return SequentialBody(time_coincidence_layers(**kwargs))
