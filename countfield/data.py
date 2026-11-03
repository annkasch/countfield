"""Count sets: points with binomial counts, file format and position merging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CountSet:
    """Points x with k successes out of n trials each.

    x: [P, D] point coordinates, k: [P] successes, n: [P] trials (n >= 1, 0 <= k <= n).
    """

    x: np.ndarray  # [P, D]
    k: np.ndarray  # [P]
    n: np.ndarray  # [P]

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        k = np.asarray(self.k)
        n = np.asarray(self.n)
        if x.ndim != 2:
            raise ValueError(f"x must have shape [P, D], got {x.shape}")
        if k.shape != (x.shape[0],) or n.shape != (x.shape[0],):
            raise ValueError(f"k and n must have shape [{x.shape[0]}], got {k.shape}, {n.shape}")
        for name, a in (("k", k), ("n", n)):
            if not np.issubdtype(a.dtype, np.integer):
                raise ValueError(f"{name} must be integer counts, got dtype {a.dtype}")
        if np.any(n < 1):
            raise ValueError(f"n must be >= 1; {int(np.sum(n < 1))} points violate this")
        bad = (k < 0) | (k > n)
        if np.any(bad):
            raise ValueError(f"need 0 <= k <= n; {int(np.sum(bad))} points violate this")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "k", k.astype(np.int64))
        object.__setattr__(self, "n", n.astype(np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[1]

    @property
    def fraction(self) -> np.ndarray:  # [P]
        """Observed success fraction k / n."""
        return self.k / self.n

    def subset(self, index: np.ndarray) -> CountSet:
        """Points selected by an index or boolean mask."""
        return CountSet(self.x[index], self.k[index], self.n[index])

    @staticmethod
    def concatenate(sets: list[CountSet]) -> CountSet:
        if not sets:
            raise ValueError("no count sets to concatenate")
        dims = {s.dim for s in sets}
        if len(dims) != 1:
            raise ValueError(f"count sets have different dimensions: {sorted(dims)}")
        return CountSet(
            np.concatenate([s.x for s in sets]),
            np.concatenate([s.k for s in sets]),
            np.concatenate([s.n for s in sets]),
        )


def position_keys(x: np.ndarray, decimals: int) -> np.ndarray:  # [P, D] -> [P, D] int
    """Integer keys of positions rounded to `decimals` (equal keys = same position)."""
    return np.round(np.asarray(x, dtype=float) * 10.0**decimals).astype(np.int64)


def merge_by_position(data: CountSet, decimals: int) -> tuple[CountSet, np.ndarray]:
    """Sum k and n of points at the same (rounded) position.

    Exact when all points at one position share the same success probability: the summed
    counts are sufficient for p. Returns the merged set (first-occurrence order) and the number
    of original points per merged point [P'].
    """
    _, first, inverse, multiplicity = np.unique(
        position_keys(data.x, decimals),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    inverse = inverse.reshape(-1)
    k = np.bincount(inverse, weights=data.k, minlength=len(first)).astype(np.int64)
    n = np.bincount(inverse, weights=data.n, minlength=len(first)).astype(np.int64)
    order = np.argsort(first)
    return CountSet(data.x[first[order]], k[order], n[order]), multiplicity[order]


def shared_positions(a: CountSet, b: CountSet, decimals: int) -> int:
    """Number of points of `b` whose position also occurs in `a`."""
    keys_a = {tuple(row) for row in position_keys(a.x, decimals)}
    return sum(tuple(row) in keys_a for row in position_keys(b.x, decimals))


def save_groups(path: str | Path, groups: dict[str, CountSet], **metadata: str) -> None:
    """Write named count sets to one npz: x, k, n and a `group` label per point."""
    if not groups:
        raise ValueError("no groups to save")
    names = list(groups)
    data = CountSet.concatenate([groups[g] for g in names])
    labels = np.concatenate([np.full(len(groups[g]), g) for g in names])
    np.savez_compressed(path, x=data.x, k=data.k, n=data.n, group=labels.astype(str), **metadata)


def load_groups(path: str | Path) -> dict[str, CountSet]:
    """Read the named count sets written by `save_groups`."""
    with np.load(path, allow_pickle=False) as f:
        x, k, n, group = f["x"], f["k"], f["n"], f["group"].astype(str)
    return {g: CountSet(x[group == g], k[group == g], n[group == g]) for g in dict.fromkeys(group)}
