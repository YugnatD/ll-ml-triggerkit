"""``SequentialBody``: declare a trigger-chain body as a plain list of layers.

The CNN half of the chain changes shape constantly between experiments, so it
should not be hardcoded. This body lets you write it out as a list -- conv,
pool, relu, the scatter adapter, a threshold -- and swap it freely::

    from tensorflow import keras
    from triggerkit.models import SequentialBody

    body = SequentialBody([
        "scatter_to_grid",                         # adapter, resolved against chain.geom
        keras.layers.Conv3D(8, (5, 1, 1), padding="same"),
        keras.layers.ReLU(),
        "time_mean",                               # (B,T,H,W,C) -> (B,H,W,C)
        keras.layers.Conv2D(16, 3, padding="same"),
        keras.layers.ReLU(),
        "global_hex_mean",                         # (B,H,W,C) -> (B,C)
        keras.layers.Dense(1, name="classifier"),
        ("threshold", {"init_tau": 0.0, "temp": 10.0, "binary_output": True}),
    ])

Each list entry is one of:

* a Keras ``Layer`` instance -- applied to the current cursor;
* a string naming a stage -- either a :meth:`TriggerChain.add_stage` stage
  (``"global_max_pooling_2d"``, ``"threshold"``, ``"rescaling"``, ...) or an
  adapter (``"scatter_to_grid"``, ``"time_mean"``, ``"global_hex_mean"``);
* a ``(name, kwargs)`` tuple -- the same, with keyword arguments;
* a callable ``fn(chain)`` -- an escape hatch for steps the list form cannot
  express (e.g. a ``keras.ops.pad`` before a conv). It must update
  ``chain.last_layer`` itself; whatever it returns is stored as a handle.

Named layers (``layer.name``) and the returned handles are collected into the
``handles`` dict so runners can grab the classifier / threshold afterwards.
"""

from tensorflow import keras

from triggerkit.models.base import TriggerBody
from triggerkit.models.layers import GlobalHexMean, ScatterToGrid, TimeMean

# Adapter stage names handled here (everything else falls through to add_stage).
_ADAPTERS = ("scatter_to_grid", "time_mean", "global_hex_mean")


class SequentialBody(TriggerBody):
    """A trigger-chain body declared as a list of layers/stages. See module docstring.

    Parameters
    ----------
    layers : list
        The layer/stage specs, applied top to bottom (see module docstring).
    """

    def __init__(self, layers):
        self.layers = list(layers)

    # ------------------------------------------------------------------ #
    def _apply_adapter(self, chain, name, kwargs):
        if name == "scatter_to_grid":
            layer = ScatterToGrid.from_geometry(chain.geom, **kwargs)
        elif name == "time_mean":
            layer = TimeMean(**kwargs)
        elif name == "global_hex_mean":
            layer = GlobalHexMean(**kwargs)
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"not an adapter: {name!r}")
        chain.last_layer = layer(chain.last_layer)
        return layer

    def _apply_named(self, chain, name, kwargs):
        if name in _ADAPTERS:
            return self._apply_adapter(chain, name, kwargs)
        # Delegate to the chain's stage builder (keeps geometry bookkeeping).
        return chain.add_stage(name, **kwargs)

    def _apply(self, chain, spec):
        if isinstance(spec, str):
            return self._apply_named(chain, spec, {})
        if isinstance(spec, tuple) and spec and isinstance(spec[0], str):
            name = spec[0]
            kwargs = spec[1] if len(spec) > 1 and spec[1] is not None else {}
            return self._apply_named(chain, name, dict(kwargs))
        if isinstance(spec, keras.layers.Layer):
            # An unbound ScatterToGrid(time_window=...) picks up the camera
            # geometry here, so it can be declared without a scatter matrix.
            if isinstance(spec, ScatterToGrid) and not spec.bound:
                spec.bind_geometry(chain.geom)
            chain.last_layer = spec(chain.last_layer)
            return spec
        if callable(spec):
            # Escape hatch: fn(chain) mutates chain.last_layer itself.
            return spec(chain)
        raise TypeError(
            f"unsupported layer spec {spec!r}: expected a Keras Layer, a stage "
            "name string, a (name, kwargs) tuple, or a callable fn(chain).")

    # ------------------------------------------------------------------ #
    def build(self, chain):
        handles = {}
        for spec in self.layers:
            handle = self._apply(chain, spec)
            name = getattr(handle, "name", None)
            if isinstance(name, str) and name:
                handles[name] = handle
        return handles
