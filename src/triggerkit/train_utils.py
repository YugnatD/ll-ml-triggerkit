"""Backward-compatible shim for the old ``train_utils`` module.

The training helpers now live in the :mod:`triggerkit.training` subpackage, split
by concern (losses / metrics / callbacks / calibration / selection / weights).
This module re-exports every public name so existing call sites that did
``import train_utils`` / ``train_utils.calibrate_tau(...)`` keep working via
``from triggerkit import train_utils``.
"""

from triggerkit.training import *  # noqa: F401,F403
from triggerkit.training import __all__  # noqa: F401
