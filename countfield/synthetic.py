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


def photon_probability(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Per-photon hit probability [P]: forward cone towards +z, damped in the shadow region.

    x: [P, 3] emission points, direction: [P, 3] unit vectors. Mean over isotropic
    directions about 2.5e-3 outside the shadow.
    """
    cone = np.exp(6.0 * (direction[:, 2] - 1.0))
    shadow = np.hypot(x[:, 1] - 0.2, x[:, 2] + 0.1) < 0.3
    return 0.03 * cone * np.where(shadow, 0.25, 1.0)


def sample_photons(points: int, photons: int | np.ndarray, seed: int, half_size: float = 0.05):
    """Voxels uniform in [-1, 1]^3 with isotropic photons uniform in each voxel cube.

    Photon features: offset from the voxel centre (in units of half_size) and direction.
    Streams: 0 positions, 1 photon offsets, 2 directions, 3 outcomes.
    """
    from countfield.photons import PhotonSet

    x = np.random.default_rng(np.random.SeedSequence([seed, 0])).uniform(-1, 1, (points, 3))
    n = np.broadcast_to(np.asarray(photons, dtype=np.int64), (points,)).copy()
    owner = np.repeat(np.arange(points), n)
    offset = np.random.default_rng(np.random.SeedSequence([seed, 1])).uniform(
        -1, 1, (len(owner), 3)
    )
    direction = np.random.default_rng(np.random.SeedSequence([seed, 2])).normal(
        size=(len(owner), 3)
    )
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    p = photon_probability(x[owner] + half_size * offset, direction)
    hit = np.random.default_rng(np.random.SeedSequence([seed, 3])).random(len(owner)) < p
    features = np.column_stack([offset, direction])
    return PhotonSet(x, np.concatenate([[0], np.cumsum(n)]), features, hit)
