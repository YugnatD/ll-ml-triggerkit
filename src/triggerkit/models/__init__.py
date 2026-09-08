"""Pluggable trigger-chain model bodies.

* :class:`TriggerBody` + :func:`build_chain` -- the seam. ``build_chain(source,
  body, filters=)`` builds the chain, runs the body, and (for a feature body)
  appends a ``Dense`` classifier head + threshold; it returns ``(chain, head,
  threshold)``.
* :class:`SequentialBody` -- a CNN (or anything) declared directly as a layer list.
* :class:`TDSCANBody` -- the fixed deployed TDSCAN chain (self-headed).
* :func:`hex3d_hybrid_layers` / :func:`hex3d_hybrid_body` -- Jakub's hex CNN preset
  layer list (needs the ``[hexcnn]`` extra).
* :class:`ScatterToGrid` / :class:`TimeMean` / :class:`GlobalHexMean` -- adapter layers.

There is no registry / ``get_body("name")``: construct a body directly and pass
it to :func:`build_chain`.
"""

from triggerkit.models.base import TriggerBody, build_chain
from triggerkit.models.grid import GridTransform
from triggerkit.models.hexcnn import hex3d_hybrid_body, hex3d_hybrid_layers
from triggerkit.models.layers import (
    GlobalHexMean,
    ScatterToGrid,
    TimeMean,
)
from triggerkit.models.sequential import SequentialBody
from triggerkit.models.tdscan import TDSCANBody, generate_lin_space_edges

__all__ = [
    "load_model",
    "register_custom_objects",
    "TriggerBody",
    "build_chain",
    "TDSCANBody",
    "generate_lin_space_edges",
    "SequentialBody",
    "hex3d_hybrid_layers",
    "hex3d_hybrid_body",
    "ScatterToGrid",
    "TimeMean",
    "GlobalHexMean",
    "GridTransform",
]


# Every module that registers a custom class. A saved model may reference any of
# them -- layers, but also the loss and metrics stored by compile() -- and Keras
# can only resolve a class whose module has been imported. Kept lazy so the
# package __init__ stays light, as its docstring promises.
_REGISTRATION_MODULES = (
    "triggerkit.models.layers",
    "triggerkit.models.tdscan",
    "triggerkit.training.losses",
    "triggerkit.training.metrics",
    "triggerkit.Stages.TDSCAN",
    "triggerkit.Stages.Shift",
    "triggerkit.Stages.DigitalSum",
    "triggerkit.Stages.MovingAverage",
    "triggerkit.Loss.RateConstrainedBCE",
    "triggerkit.Metric.NSBRateHz",
    "triggerkit.Metric.TauMetric",
)


def register_custom_objects():
    """Import every module that registers a custom class. Returns the failures."""
    import importlib
    failed = []
    for name in _REGISTRATION_MODULES:
        try:
            importlib.import_module(name)
        except BaseException as e:      # an optional dep must not block a load
            failed.append((name, str(e)))
    return failed


def load_model(path, **kwargs):
    """Load a saved trigger model, with every custom class registered first.

    ``keras.models.load_model`` resolves a custom class only once the module
    defining it has been imported, so a bare call fails on any model from this
    package -- including on the loss and metrics that ``compile()`` saved, which
    live outside ``models``. This wrapper imports all of them first.
    """
    import tensorflow as tf
    register_custom_objects()
    return tf.keras.models.load_model(path, **kwargs)
