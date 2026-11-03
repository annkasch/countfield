"""Per-point photon sets: every trial of a point with its features and outcome (hit or miss).

A photon file extends the count-set file (x, k, n, group) with CSR photon arrays:
photon_offsets [V + 1], photon_features [P, F] and photon_hit [P]; point v owns photon rows
photon_offsets[v]:photon_offsets[v + 1]. Photon features are relative to their own point
(e.g. emission offset from the voxel centre and direction), so they mean the same for every
point and can be pooled by a shared encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from countfield.data import CountSet, position_keys


@dataclass(frozen=True)
class PhotonSet:
    """Points with all their photons.

    x: [V, D] positions, offsets: [V + 1] CSR offsets, features: [P, F], hit: [P] (0/1).
    """

    x: np.ndarray  # [V, D]
    offsets: np.ndarray  # [V + 1]
    features: np.ndarray  # [P, F]
    hit: np.ndarray  # [P]

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        features = np.asarray(self.features, dtype=np.float32)
        hit = np.asarray(self.hit).astype(np.int8)
        if x.ndim != 2 or offsets.shape != (len(x) + 1,):
            raise ValueError(f"need x [V, D] and offsets [V + 1]; got {x.shape}, {offsets.shape}")
        if offsets[0] != 0 or np.any(np.diff(offsets) < 1) or offsets[-1] != len(features):
            raise ValueError("offsets must start at 0, increase strictly, end at len(features)")
        if features.ndim != 2 or hit.shape != (len(features),):
            raise ValueError(f"need features [P, F] and hit [P]; got {features.shape}, {hit.shape}")
        if not np.isin(hit, (0, 1)).all():
            raise ValueError("photon_hit must be 0 or 1")
        if not np.isfinite(features).all() or not np.isfinite(x).all():
            raise ValueError("non-finite positions or photon features")
        for name, value in (("x", x), ("offsets", offsets), ("features", features), ("hit", hit)):
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.x)

    @property
    def n(self) -> np.ndarray:  # [V]
        return np.diff(self.offsets)

    @property
    def k(self) -> np.ndarray:  # [V]
        return np.add.reduceat(self.hit.astype(np.int64), self.offsets[:-1])

    @property
    def point(self) -> np.ndarray:  # [P] owning point of each photon
        return np.repeat(np.arange(len(self)), self.n)

    def counts(self) -> CountSet:
        return CountSet(self.x, self.k, self.n)

    def subset(self, index: np.ndarray) -> PhotonSet:
        index = np.arange(len(self))[index]
        rows = np.concatenate([np.arange(self.offsets[v], self.offsets[v + 1]) for v in index])
        offsets = np.concatenate([[0], np.cumsum(self.n[index])])
        return PhotonSet(self.x[index], offsets, self.features[rows], self.hit[rows])

    @staticmethod
    def concatenate(sets: list[PhotonSet]) -> PhotonSet:
        offsets = np.concatenate([[0], np.cumsum(np.concatenate([s.n for s in sets]))])
        return PhotonSet(
            np.concatenate([s.x for s in sets]),
            offsets,
            np.concatenate([s.features for s in sets]),
            np.concatenate([s.hit for s in sets]),
        )


def merge_photons_by_position(data: PhotonSet, decimals: int) -> tuple[PhotonSet, np.ndarray]:
    """Pool the photons of points at the same (rounded) position; first-occurrence order.

    Exact when the points share one hit probability per photon: their photons are then
    draws from the same process. Returns the merged set and points per merged point.
    """
    _, first, inverse, multiplicity = np.unique(
        position_keys(data.x, decimals),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    inverse = inverse.reshape(-1)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    owner = rank[inverse]  # merged index of every original point
    photon_owner = np.repeat(owner, data.n)
    rows = np.argsort(photon_owner, kind="stable")
    counts = np.bincount(owner, weights=data.n, minlength=len(first)).astype(np.int64)
    merged = PhotonSet(
        data.x[first[order]],
        np.concatenate([[0], np.cumsum(counts)]),
        data.features[rows],
        data.hit[rows],
    )
    return merged, multiplicity[order]


def save_photon_groups(path: str | Path, groups: dict[str, PhotonSet], feature_names=None):
    names = list(groups)
    data = PhotonSet.concatenate([groups[g] for g in names])
    np.savez(
        path,
        x=data.x,
        k=data.k,
        n=data.n,
        group=np.concatenate([np.full(len(groups[g]), g) for g in names]).astype(str),
        photon_offsets=data.offsets,
        photon_features=data.features,
        photon_hit=data.hit,
        feature_names=np.array(feature_names or [f"f{i}" for i in range(data.features.shape[1])]),
    )


def load_photon_groups(path: str | Path) -> tuple[dict[str, PhotonSet], list[str]]:
    """Named photon sets of a photon file and the photon feature names."""
    with np.load(path, allow_pickle=False) as f:
        x, group, offsets = f["x"], f["group"].astype(str), f["photon_offsets"]
        features, hit = f["photon_features"], f["photon_hit"]
        names = [str(s) for s in f["feature_names"]]
        n_file, k_file = f["n"], f["k"]
    data = PhotonSet(x, offsets, features, hit)
    if not (np.array_equal(data.n, n_file) and np.array_equal(data.k, k_file)):
        raise ValueError(f"{path}: n/k do not match the photon arrays")
    return {g: data.subset(group == g) for g in dict.fromkeys(group)}, names
