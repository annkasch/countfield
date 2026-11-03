import numpy as np
import pytest

from countfield import CountSet


def _synthetic(seed: int = 0, points: int = 50) -> CountSet:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    x = rng.uniform(-1, 1, size=(points, 3))
    n = rng.integers(500, 2000, size=points)
    p = 3e-3 * (1 + 0.5 * x[:, 2])
    return CountSet(x, rng.binomial(n, p), n)


def test_valid_count_set():
    data = _synthetic()
    assert len(data) == 50
    assert np.all((data.fraction >= 0) & (data.fraction <= 1))
    assert len(data.subset(data.x[:, 0] > 0)) == int(np.sum(data.x[:, 0] > 0))


@pytest.mark.parametrize(
    ("k", "n", "message"),
    [
        ([1, 5], [2, 3], "0 <= k <= n"),
        ([0, 0], [0, 3], "n must be >= 1"),
        ([0.5, 1], [2, 3], "integer"),
    ],
)
def test_invalid_counts_rejected(k, n, message):
    with pytest.raises(ValueError, match=message):
        CountSet(np.zeros((2, 3)), np.array(k), np.array(n))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape"):
        CountSet(np.zeros((3, 3)), np.zeros(2, dtype=int), np.ones(2, dtype=int))
