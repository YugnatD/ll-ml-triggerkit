"""Pluggable trigger-chain model bodies.

* :class:`TriggerBody` + :func:`build_chain` / :func:`get_body` -- the seam.
* :class:`TDSCANBody` -- the fixed deployed TDSCAN chain.
* :class:`SequentialBody` -- a CNN (or anything) declared as a layer list.
* :class:`Hex3DHybridBody` -- Jakub's hex CNN preset (needs the ``[hexcnn]`` extra).
* :class:`ScatterToGrid` / :class:`TimeMean` / :class:`GlobalHexMean` -- adapter layers.
"""

from triggerkit.models.base import (
    TriggerBody,
    build_chain,
    get_body,
    register_body,
    registered_bodies,
)
from triggerkit.models.grid import GridTransform
from triggerkit.models.hexcnn import Hex3DHybridBody, hex3d_hybrid_layers
from triggerkit.models.layers import GlobalHexMean, ScatterToGrid, TimeMean
from triggerkit.models.sequential import SequentialBody
from triggerkit.models.tdscan import TDSCANBody

__all__ = [
    "TriggerBody",
    "build_chain",
    "get_body",
    "register_body",
    "registered_bodies",
    "TDSCANBody",
    "SequentialBody",
    "Hex3DHybridBody",
    "hex3d_hybrid_layers",
    "ScatterToGrid",
    "TimeMean",
    "GlobalHexMean",
    "GridTransform",
]
