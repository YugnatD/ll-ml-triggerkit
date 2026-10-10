import warnings

import pytest

pytest.importorskip("tensorflow")
from triggerkit.data.dataset import TriggerDataset


def ds(n_files, **kw):
    return TriggerDataset([f"g{i}.h5" for i in range(n_files)], "n.h5", **kw)


def test_single_split_keeps_tail_for_validation():
    train, val = ds(10, percent_validation=0.2).split_files()
    assert train == [f"g{i}.h5" for i in range(8)] and val == ["g8.h5", "g9.h5"]


def test_no_validation_when_disabled():
    train, val = ds(5, percent_validation=0.0).split_files()
    assert len(train) == 5 and val == []


def test_tiny_file_list_warns_instead_of_silently_dropping_validation():
    with pytest.warns(UserWarning, match="NO validation"):
        train, val = ds(3, percent_validation=0.2).split_files()
    assert len(train) == 3 and val == []


@pytest.mark.parametrize("n_files,n_folds", [(10, 5), (7, 3), (6, 6)])
def test_kfold_partitions_every_file_exactly_once(n_files, n_folds):
    seen_val = []
    for k in range(n_folds):
        train, val = ds(n_files, n_folds=n_folds, fold=k).split_files()
        assert set(train).isdisjoint(val) and len(train) + len(val) == n_files
        seen_val += val
    assert sorted(seen_val) == sorted(f"g{i}.h5" for i in range(n_files))


def test_empty_fold_warns():
    with pytest.warns(UserWarning, match="empty"):
        _, val = ds(2, n_folds=4, fold=0).split_files()
    assert val == []


def test_argument_validation():
    with pytest.raises(ValueError):
        ds(3, n_folds=0)
    with pytest.raises(ValueError):
        ds(3, n_folds=2, fold=2)
    with pytest.raises(ValueError):
        ds(3, targets=("true_image",))               # 'class' is mandatory
    with pytest.raises(NotImplementedError):
        ds(3, targets=("class", "bogus"))
    with pytest.raises(NotImplementedError):
        ds(3, gated_by=object())


def test_supported_aux_targets_and_single_nsb_string():
    d = ds(3, targets=("class", "true_image", "peak_time"))
    assert d.nsb_files == ["n.h5"]
    cfg = d._config(training=True)
    assert cfg.emit_true_image and cfg.emit_peak_time and cfg.shuffle_samples
    assert not d._config(training=False).shuffle_samples
    assert d._config(training=False).nsb_roll_seed != cfg.nsb_roll_seed   # independent rotations


def test_gamma_rotations_warns():
    with pytest.warns(UserWarning, match="not implemented"):
        ds(3, gamma_rotations=(0, 120))
