"""One place that turns "whatever the caller passed" into a clean list of path strings.

A file argument used to be ``str`` for some call sites and ``list`` for others,
and a bare ``str`` was silently iterated character by character (a chain built on
one gamma file then read zero events, without any error). Everything that takes
files goes through :func:`normalize_files` instead. No TensorFlow import.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Union

PathLike = Union[str, "os.PathLike[str]"]


def normalize_files(files, *, name: str = "files", require_nonempty: bool = False) -> List[str]:
    """Return ``files`` as a list of ``str`` paths.

    Accepts ``None`` (-> ``[]``), a single ``str`` / ``os.PathLike`` (-> one
    element), or any iterable of those (list, tuple, generator, ...). Anything
    else -- bytes, numbers, a dict -- raises a ``TypeError`` that names the
    argument. ``require_nonempty`` turns an empty result into a ``ValueError``.
    """
    if files is None:
        out: List[str] = []
    elif isinstance(files, (str, os.PathLike)):
        out = [os.fspath(files)]
    elif isinstance(files, (bytes, bytearray, dict)) or not isinstance(files, Iterable):
        raise TypeError(
            f"{name} must be a path, or a list/tuple of paths, not {type(files).__name__}.")
    else:
        out = []
        for item in files:
            if not isinstance(item, (str, os.PathLike)):
                raise TypeError(
                    f"{name} must contain paths (str or os.PathLike); got "
                    f"{type(item).__name__}: {item!r}.")
            out.append(os.fspath(item))
    if require_nonempty and not out:
        raise ValueError(f"{name}: at least one file is required, got none.")
    return out
