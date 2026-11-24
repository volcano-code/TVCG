from pathlib import Path
import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure


class Sample:
    origin = np.zeros(3)
    side = 8
    pitch = .125
    identity = 'manufactured-support'

    def iter_chunks(self, chunk_size):
        p = np.array([[.8, .21, .3], [.2, .3, .24], [.7, .15, .45], [.21, .27, .01]])
        w = np.array([3., 2., 4., 1.])
        for i in range(0, len(p), 2):
            yield p[i:i+2], w[i:i+2]


def test_packed_blocks_replay_nodes_weights_and_preserve_cross_terms(tmp_path):
    packed = BlockMeasure.build(Sample(), tmp_path / 'cache', block_voxels=4)
    points = np.concatenate([p for p, _ in Sample().iter_chunks(2)])
    weights = np.concatenate([w for _, w in Sample().iter_chunks(2)])
    expected = np.array([weights.sum(), *(weights @ points), *(weights @ (points**2)),
                         weights @ (points[:, 1]*points[:, 2]), weights @ (points[:, 0]*points[:, 2]),
                         weights @ (points[:, 0]*points[:, 1])])
    np.testing.assert_allclose(packed.moments.sum(axis=0), expected)
    active = packed.centers[:, 0] > .5
    rows = np.concatenate([np.c_[p,w] for p,w in packed.chunks(active, max_nodes=1)])
    assert np.all(rows[:,0] > .5)
    assert sorted(rows[:,3]) == [3., 4.]
    assert len(packed.centers) == 2
    same = BlockMeasure.build(Sample(), tmp_path / 'cache', block_voxels=4)
    np.testing.assert_equal(same.moments, packed.moments)
    with pytest.raises(ValueError, match='identity|mismatch'):
        BlockMeasure.open(tmp_path / 'cache', identity='elsewhere', block_voxels=4)


def test_prune_conservatively_for_both_operations(tmp_path):
    packed = BlockMeasure.build(Sample(), tmp_path / 'cache', block_voxels=4)
    class Distances:
        def signed_distance(self, points):
            return np.array([1., -.4])
    full, band = packed.classify(Distances(), .01, 'intersection')
    assert full.tolist() == [True, False] and band.tolist() == [False, True]
    outside, second = packed.classify(Distances(), .01, 'difference')
    assert not outside.any()
    np.testing.assert_equal(band, second)


def test_coarse_encoder_fraction_integrates_same_mass_and_validates_density(tmp_path):
    packed = BlockMeasure.build(Sample(), tmp_path / 'cache', block_voxels=4)
    density = 100.
    grid = packed.encoder_grid(2, density)
    assert grid.shape == (4, 2, 2, 2)
    assert grid[0].sum() * density * .5**3 == pytest.approx(10.)
    with pytest.raises(ValueError, match='volume'):
        packed.encoder_grid(2, .01)
