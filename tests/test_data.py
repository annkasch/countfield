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


def test_merge_sums_counts_at_shared_positions():
    from countfield import merge_by_position

    x = np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 1e-9], [2, 0, 0]])
    data = CountSet(x, np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40]))
    merged, multiplicity = merge_by_position(data, decimals=6)
    assert len(merged) == 3
    np.testing.assert_array_equal(merged.k, [4, 2, 4])
    np.testing.assert_array_equal(merged.n, [40, 20, 40])
    np.testing.assert_array_equal(multiplicity, [2, 1, 1])


def test_groups_roundtrip(tmp_path):
    from countfield import load_groups, save_groups
    from countfield.data import shared_positions

    a, b = _synthetic(0, 10), _synthetic(1, 5)
    save_groups(tmp_path / "c.npz", {"train/lf": a, "val/hf": b})
    groups = load_groups(tmp_path / "c.npz")
    assert list(groups) == ["train/lf", "val/hf"]
    np.testing.assert_array_equal(groups["val/hf"].k, b.k)
    assert shared_positions(a, b, 6) == 0
    assert shared_positions(a, a.subset(np.arange(3)), 6) == 3
