"""Training utilities for the trigger chain.

Losses, metrics, per-epoch loggers, tau/temp calibration, held-out best-filter
selection, and TDSCAN weight helpers. This was one ``train_utils.py`` in the
research sandbox; it is split here by concern. Everything is re-exported at this
package level, and ``triggerkit.train_utils`` re-exports it too, so old
``train_utils.<name>`` call sites keep working.
"""

from triggerkit.training.calibration import (
    _default_calib_config,
    _score_distributions,
    calibrate_tau,
)
from triggerkit.training.callbacks import PerFilterAUCLogger, WeightStatsLogger
from triggerkit.training.losses import (
    make_pairwise_auc_loss,
    make_pairwise_auc_loss_multi,
    soft_or_scores,
)
from triggerkit.training.metrics import (
    PairwiseAUCMetric,
    PerFilterPairwiseAUCMetric,
    make_per_filter_auc_metrics,
)
from triggerkit.training.selection import (
    _efficiency_at_rate,
    _paired_efficiency_at_rate,
    _selection_config,
    collect_multi_filter_scores,
    pick_best_filter_from_history,
    select_best_filter,
    select_best_joint_column,
)
from triggerkit.training.weights import make_decreasing_ring_init, remap_weights

__all__ = [
    # losses
    "make_pairwise_auc_loss",
    "make_pairwise_auc_loss_multi",
    "soft_or_scores",
    # metrics
    "PairwiseAUCMetric",
    "PerFilterPairwiseAUCMetric",
    "make_per_filter_auc_metrics",
    # callbacks
    "PerFilterAUCLogger",
    "WeightStatsLogger",
    # calibration
    "calibrate_tau",
    "_default_calib_config",
    "_score_distributions",
    # selection
    "pick_best_filter_from_history",
    "collect_multi_filter_scores",
    "select_best_filter",
    "select_best_joint_column",
    "_selection_config",
    "_efficiency_at_rate",
    "_paired_efficiency_at_rate",
    # weights
    "make_decreasing_ring_init",
    "remap_weights",
]
