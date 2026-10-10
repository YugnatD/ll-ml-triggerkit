"""Training-history sidecar files (JSON, never pickle).

``np.save`` of a dict is a pickle: loading it back needs ``allow_pickle=True``,
which executes code embedded in the file. A history is only floats and lists, so
it is stored as plain JSON next to the model. No TensorFlow import.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):                 # eager tensors / variables
        return _plain(value.numpy())
    return value


def save_history(path: str, history: Dict[str, Any]) -> None:
    """Write a ``History.history`` dict as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_plain(history), fh)


def load_history(path: str) -> Dict[str, Any]:
    """Read a history written by :func:`save_history`."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_legacy_npy_history(path: str) -> Dict[str, Any]:
    """Read an OLD ``*_history.npy`` (a pickled dict). Executes code from the file:
    only call this on a file you produced yourself."""
    return np.load(path, allow_pickle=True).item()
