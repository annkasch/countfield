"""Count-set container: points with binomial counts."""

from __future__ import annotations

from dataclasses import dataclass

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
        if np.any((k < 0) | (k > n)):
            raise ValueError(
                f"need 0 <= k <= n; {int(np.sum((k < 0) | (k > n)))} points violate this"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "k", k.astype(np.int64))
        object.__setattr__(self, "n", n.astype(np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def fraction(self) -> np.ndarray:  # [P]
        """Observed success fraction k / n."""
        return self.k / self.n

    def subset(self, index: np.ndarray) -> CountSet:
        """Points selected by an index or boolean mask."""
        return CountSet(self.x[index], self.k[index], self.n[index])
