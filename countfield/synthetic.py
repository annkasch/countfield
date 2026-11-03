"""Synthetic count fields for tests and demos: smooth trend plus a sharp shadow."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from countfield.data import CountSet


def probability(x: np.ndarray, base: float = 2e-3) -> np.ndarray:  # [P, D] in [-1, 1] -> [P]
    """Logit-linear trend along the last axis, damped inside a cylinder along axis 0."""
    logit = np.log(base / (1 - base)) + 0.8 * x[:, -1]
    radius = np.hypot(x[:, 1] - 0.2, x[:, -1] + 0.1) if x.shape[1] >= 3 else np.abs(x[:, -1])
    return expit(logit - 1.5 * (radius < 0.3))


def sample(points: int, trials: int | np.ndarray, seed: int, dim: int = 3) -> CountSet:
    """Uniform points in [-1, 1]^dim with binomial counts (streams 0 = positions, 1 = counts)."""
    x = np.random.default_rng(np.random.SeedSequence([seed, 0])).uniform(-1, 1, (points, dim))
    n = np.broadcast_to(np.asarray(trials, dtype=np.int64), (points,)).copy()
    k = np.random.default_rng(np.random.SeedSequence([seed, 1])).binomial(n, probability(x))
    return CountSet(x, k, n)
