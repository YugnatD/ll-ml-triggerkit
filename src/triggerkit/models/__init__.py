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
from triggerkit.models.layers import GlobalHexMean, ScatterToGrid, TimeMean
from triggerkit.models.sequential import SequentialBody
from triggerkit.models.tdscan import TDSCANBody, generate_lin_space_edges

__all__ = [
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
