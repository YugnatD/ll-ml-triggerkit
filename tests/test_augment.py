import numpy as np
import pytest
from ctapipe.instrument import CameraGeometry

from triggerkit.augment import (
    Fold, identity_index, make_rotation_folds, roll_index, rotation_index, shuffle_index,
)


@pytest.fixture
def square_cam():
    return CameraGeometry.make_rectangular(4, 4)


def test_rotation_90_is_a_permutation_of_a_square_grid(square_cam):
    perm = rotation_index(square_cam, 90)
    assert sorted(perm) == list(range(square_cam.n_pixels))
    # four quarter turns bring every pixel home
    out = np.arange(square_cam.n_pixels)
    for _ in range(4):
        out = out[perm]
    assert np.array_equal(out, np.arange(square_cam.n_pixels))


def test_non_symmetry_angle_fails_loudly(square_cam):
    with pytest.raises(ValueError):
        rotation_index(square_cam, 45)


def test_simple_index_builders():
    assert np.array_equal(identity_index(5), np.arange(5))
    assert np.array_equal(roll_index(5, 2), [3, 4, 0, 1, 2])
    a, b = shuffle_index(50, 7), shuffle_index(50, 7)
    assert np.array_equal(a, b) and sorted(a) == list(range(50))


def test_fold_requires_matching_lengths():
    with pytest.raises(ValueError):
        Fold("x", np.arange(3), np.arange(4))


def _spec(**kw):
    base = {"gamma_events": 10, "nsb_events": 5, "conditions": ["medium"]}
    base.update(kw)
    return base


CONDS = {"medium": (["g.h5"], ["n.h5"]), "low": (["gl.h5"], ["nl.h5"])}


def test_make_rotation_folds_names_and_budgets(square_cam):
    folds = make_rotation_folds(square_cam, [
        _spec(conditions=["low", "medium"]),
        _spec(gamma_deg=90, gamma_time_shift=5),
        _spec(nsb_kind="rolled", nsb_param=3),
    ], CONDS)
    names = [f.name for f in folds]
    assert names == ["rot0_original_low", "rot0_original_medium",
                     "rot90_original_troll5_medium", "rot0_rolled3_medium"]
    assert len(set(names)) == len(names)
    assert folds[0].gamma_files == ["gl.h5"] and folds[1].gamma_files == ["g.h5"]
    assert all(f.max_gamma_events == 10 and f.max_nsb_events == 5 for f in folds)
    assert folds[2].gamma_time_shift == 5


@pytest.mark.parametrize("bad", [
    {"gamma_events": 1, "nsb_events": 1},                     # missing conditions
    {"gamma_events": 1, "conditions": ["medium"]},            # missing nsb_events
    _spec(typo_key=1),                                        # unknown key
    _spec(conditions=[]),                                     # empty
])
def test_bad_specs_rejected(square_cam, bad):
    with pytest.raises(ValueError):
        make_rotation_folds(square_cam, [bad], CONDS)


def test_unknown_condition_and_kind(square_cam):
    with pytest.raises(ValueError):
        make_rotation_folds(square_cam, [_spec(conditions=["high"])], CONDS)
    with pytest.raises(ValueError):
        make_rotation_folds(square_cam, [_spec(nsb_kind="bogus")], CONDS)


def test_duplicate_rows_get_unique_names(square_cam):
    folds = make_rotation_folds(square_cam, [_spec(), _spec()], CONDS)
    assert len({f.name for f in folds}) == 2
