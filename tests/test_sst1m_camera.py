import numpy as np

from triggerkit.camera import sst1m


def test_tables_have_the_documented_shapes():
    assert sst1m.pixel_positions().shape == (1296, 2)
    trip = sst1m.patch_triplets()
    assert trip.shape == (432, 3)
    assert sorted(trip.ravel()) == list(range(1296))        # every pixel in exactly one patch
    assert sst1m.patch_positions().shape == (432, 2)


def test_patch7_clusters_start_with_the_patch_itself():
    clusters = sst1m.patch7_clusters()
    assert len(clusters) == 432
    for patch, cl in enumerate(clusters):
        assert cl[0] == patch and 1 <= len(cl) <= 7 and len(set(cl)) == len(cl)


def test_tdscan_neighbors_structure():
    for eps, k in ((1, 7), (2, 19)):
        nb = sst1m.tdscan_neighbors(eps)
        assert nb.shape == (432, k)
        assert np.array_equal(nb[:, k // 2], np.arange(432))     # middle slot = the patch itself
        assert nb.max() < 432 and nb.min() >= -1
    nb = sst1m.tdscan_neighbors(1)
    # neighbourhood is symmetric: if b is a neighbour of a, a is one of b
    for a in range(0, 432, 17):
        for b in nb[a]:
            if b >= 0:
                assert a in nb[b]
    # interior patches have all 6 neighbours, edge ones fewer
    counts = (nb >= 0).sum(axis=1)
    assert counts.max() == 7 and counts.min() < 7
