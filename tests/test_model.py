import numpy as np
import torch

from countfield.model import LocalLatentNP, NeighbourInputs, Standardization, knn
from countfield.synthetic import sample


def test_knn_excludes_self_and_sorts():
    x = np.random.default_rng(np.random.SeedSequence([0, 0])).normal(size=(50, 3))
    idx, dist = knn(x, x, 5, exclude=np.arange(50), chunk=7)
    assert not np.any(idx == np.arange(50)[:, None])
    assert np.all(np.diff(dist, axis=1) >= 0)
    brute = np.argsort(np.linalg.norm(x[:, None] - x[None], axis=-1), axis=1)[:, 1:6]
    np.testing.assert_array_equal(idx, brute)


def test_model_shapes_and_positive_sigma():
    data = sample(100, 2000, seed=0)
    std = Standardization.fit(data)
    inputs = NeighbourInputs(data, std, neighbours=8)
    query, offset, counts = inputs.build(data.x[:10], exclude=np.arange(10))
    assert offset.shape == (10, 8, 3) and counts.shape == (10, 8, 2)
    torch.manual_seed(0)
    mu, sigma = LocalLatentNP(3, hidden=16, heads=2)(query, offset, counts)
    assert mu.shape == (10,) and torch.all(sigma > 0)
