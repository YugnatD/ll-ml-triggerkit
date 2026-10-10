import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from ctapipe.instrument import CameraGeometry

from triggerkit.Stages.MovingAverage import TemporalMovingAverage
from triggerkit.Stages.OrMerge import OrMerge
from triggerkit.Stages.ScoreQuantizer import ScoreQuantizer
from triggerkit.Stages.TrainSoftMaxPool2D import TrainSoftMaxPool2D
from triggerkit.Stages.TrainableThreshold import TrainableThreshold
from triggerkit.models.layers import ScatterToGrid
from triggerkit.training.losses import PairwiseAUCLossMulti, SoftOr
from triggerkit.training.metrics import PairwiseAUCMetric, PerFilterPairwiseAUCMetric


@pytest.fixture
def geom():
    return CameraGeometry.make_rectangular(4, 4)


def test_moving_average_channels_are_independent(geom):
    x = np.random.default_rng(0).random((2, 5, 10, 2)).astype("float32")
    both = TemporalMovingAverage(geom, window_size=3)(x).numpy()
    for c in range(2):
        single = TemporalMovingAverage(geom, window_size=3)(x[..., c]).numpy()
        np.testing.assert_allclose(both[..., c], single, atol=1e-6)


def test_moving_average_values(geom):
    x = np.zeros((1, 1, 6), "float32"); x[0, 0, 2] = 3.0
    y = TemporalMovingAverage(geom, window_size=3)(x).numpy()[0, 0]
    np.testing.assert_allclose(y, [0, 1, 1, 1, 0, 0], atol=1e-6)


def test_threshold_name_and_params_before_build(geom):
    t = TrainableThreshold(geom, init_tau=7.0)
    assert t.stage_name() == "threshold7.0"
    assert t.get_params()["threshold"] == 7.0


@pytest.mark.parametrize("comparison,expected", [("gt", [0, 0, 1]), ("ge", [0, 1, 1])])
def test_threshold_hard_decision(geom, comparison, expected):
    t = TrainableThreshold(geom, init_tau=2.0, binary_output=True, comparison=comparison)
    out = t(tf.constant([[1.0], [2.0], [3.0]])).numpy().ravel()
    assert out.tolist() == expected


def test_threshold_config_roundtrip(geom):
    t = TrainableThreshold(geom, init_tau=4.0, temp=3.0, binary_output=True, comparison="ge")
    t2 = TrainableThreshold.from_config(t.get_config())
    assert (t2.init_tau, t2.temp, t2.binary_output, t2.comparison) == (4.0, 3.0, True, "ge")


def test_or_merge_is_logical_or_on_bits():
    out = OrMerge()([tf.constant([[0.0], [1.0], [0.0], [1.0]]),
                     tf.constant([[0.0], [0.0], [1.0], [1.0]])]).numpy().ravel()
    assert out.tolist() == [0, 1, 1, 1]


def test_score_quantizer_codes():
    sq = ScoreQuantizer(edges=[32, 98, 164])
    out = sq(tf.constant([0.0, 31.9, 32.0, 100.0, 164.0, 999.0])).numpy()
    assert out.tolist() == [0, 0, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        ScoreQuantizer(edges=[3, 3])


def test_scatter_to_grid_compact_config_roundtrip():
    M = np.zeros((6, 12), np.float32); M[np.arange(6), [0, 3, 5, 7, 9, 11]] = 1
    layer = ScatterToGrid(M, 3, 4, time_window=4)
    cfg = layer.get_config()
    assert "scatter_matrix" not in cfg and len(cfg["cell_index"]) == 6
    again = ScatterToGrid.from_config(dict(cfg))
    assert np.array_equal(again.scatter_matrix, M)
    # a model saved with the old (full matrix) config must still load
    old = dict(cfg); old.pop("cell_index"); old["scatter_matrix"] = M.tolist()
    assert np.array_equal(ScatterToGrid.from_config(old).scatter_matrix, M)
    x = tf.random.normal((2, 6, 5))
    assert layer(x).shape == (2, 4, 3, 4, 1)


def test_softmax_pool_serializable_and_hard_at_inference():
    p = TrainSoftMaxPool2D(beta=3.0)
    assert TrainSoftMaxPool2D.from_config(p.get_config()).beta == 3.0
    x = tf.random.normal((2, 4, 5, 3))
    np.testing.assert_allclose(p(x, training=False).numpy(), tf.reduce_max(x, axis=[1, 2]).numpy())


def test_pairwise_auc_loss_and_metric_agree_on_perfect_ranking():
    y = tf.constant([1, 1, 0, 0], tf.float32)
    good = tf.constant([[3.0], [2.0], [0.0], [-1.0]])
    bad = -good
    loss = PairwiseAUCLossMulti(sharpness=1.0)
    assert float(loss(y, good)) < float(loss(y, bad))
    m = PerFilterPairwiseAUCMetric(1, column=0); m.update_state(y, good)
    assert float(m.result()) == 1.0
    m1 = PairwiseAUCMetric(); m1.update_state(y, tf.reshape(bad, [-1]))
    assert float(m1.result()) == 0.0


def test_soft_or_probabilistic_or():
    out = SoftOr(taus=[0.0, 0.0], temps=[1000.0, 1000.0])([tf.constant([[1.0], [-1.0], [-1.0]]),
                                                         tf.constant([[-1.0], [1.0], [-1.0]])])
    np.testing.assert_allclose(out.numpy().ravel(), [1, 1, 0], atol=1e-6)
