"""Local-latent neural process over points.

For a query position x_q the model reads its K nearest observed points (relative position,
distance and counts) and returns a Gaussian over the logit of the success probability at x_q:
eta_q ~ N(mu_q, sigma_q^2), k_q ~ Binomial(n_q, sigmoid(eta_q)). The latent is local (one per
query), so sigma_q is the interpolation uncertainty given the neighbours, learned separately
from the binomial noise of the query's own counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from countfield.data import CountSet


@dataclass(frozen=True)
class Standardization:
    """Fitted on the training points only."""

    x_mean: list[float]  # [D]
    x_scale: list[float]  # [D]
    distance_scale: float  # median nearest-neighbour distance (standardized units)
    logit_rate: float  # logit of the pooled training fraction
    log_expected_mean: float  # mean log(n * pooled fraction)
    log_expected_sd: float

    @classmethod
    def fit(cls, data: CountSet) -> Standardization:
        mean, scale = data.x.mean(axis=0), data.x.std(axis=0)
        if np.any(scale <= 0):
            raise ValueError("training positions are constant along a dimension")
        z = (data.x - mean) / scale
        _, distance = knn(z, z, 1, exclude=np.arange(len(data)))
        rate = data.k.sum() / data.n.sum()
        if not 0 < rate < 1:
            raise ValueError(f"pooled training fraction must be in (0, 1), got {rate}")
        log_expected = np.log(data.n * rate)
        return cls(
            x_mean=mean.tolist(),
            x_scale=scale.tolist(),
            distance_scale=float(np.median(distance[:, 0])),
            logit_rate=float(np.log(rate / (1 - rate))),
            log_expected_mean=float(log_expected.mean()),
            log_expected_sd=float(max(log_expected.std(), 1e-6)),
        )

    def positions(self, x: np.ndarray) -> np.ndarray:  # [P, D] standardized
        return (x - np.asarray(self.x_mean)) / np.asarray(self.x_scale)

    def count_features(self, k: np.ndarray, n: np.ndarray) -> np.ndarray:  # [P, 2]
        """Logit of the smoothed fraction relative to the pooled rate, log expected count."""
        p = (k + 0.5) / (n + 1.0)
        logit = np.log(p / (1 - p)) - self.logit_rate
        expected = np.log(n) + self.logit_rate - np.log1p(np.exp(self.logit_rate))
        expected = (expected - self.log_expected_mean) / self.log_expected_sd
        return np.stack([logit, expected], axis=-1)

    def to_dict(self) -> dict:
        return asdict(self)


def knn(
    query: np.ndarray,  # [Q, D]
    pool: np.ndarray,  # [P, D]
    k: int,
    *,
    exclude: np.ndarray | None = None,  # [Q] pool index to skip per query (-1: none)
    chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:  # indices [Q, k], distances [Q, k]
    """Exact k nearest pool points per query, ascending distance."""
    available = len(pool) - (0 if exclude is None else 1)
    if k > available:
        raise ValueError(f"need {k} neighbours but the pool has only {available} usable points")
    pool_t = torch.as_tensor(pool, dtype=torch.float64)
    indices, distances = [], []
    for start in range(0, len(query), chunk):
        q = torch.as_tensor(query[start : start + chunk], dtype=torch.float64)
        d = torch.cdist(q, pool_t)  # [C, P]
        if exclude is not None:
            ex = torch.as_tensor(exclude[start : start + chunk])
            rows = torch.nonzero(ex >= 0).squeeze(-1)
            d[rows, ex[rows]] = torch.inf
        dist, idx = torch.topk(d, k, dim=1, largest=False, sorted=True)
        indices.append(idx.numpy())
        distances.append(dist.numpy())
    return np.concatenate(indices), np.concatenate(distances)


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


class LocalLatentNP(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, heads: int = 4, min_sigma: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.min_sigma = min_sigma
        self.point = _mlp([dim + 1 + 2, hidden, hidden, hidden])  # relative pos, dist, counts
        self.query = _mlp([dim, hidden, hidden])  # absolute query position
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.head = _mlp([2 * hidden, hidden, hidden, 2])

    def forward(
        self,
        query: torch.Tensor,  # [B, D] standardized query positions
        offset: torch.Tensor,  # [B, K, D] neighbour - query, in distance-scale units
        counts: torch.Tensor,  # [B, K, 2] neighbour count features
    ) -> tuple[torch.Tensor, torch.Tensor]:  # mu [B], sigma [B] (logit offset from pooled rate)
        distance = offset.norm(dim=-1, keepdim=True)  # [B, K, 1]
        h = self.point(torch.cat([offset, distance, counts], dim=-1))  # [B, K, H]
        h = self.norm(h + self.self_attention(h, h, h, need_weights=False)[0])  # [B, K, H]
        q = self.query(query).unsqueeze(1)  # [B, 1, H]
        r = self.cross_attention(q, h, h, need_weights=False)[0].squeeze(1)  # [B, H]
        out = self.head(torch.cat([r, q.squeeze(1)], dim=-1))  # [B, 2]
        sigma = F.softplus(out[:, 1] - 1.0) + self.min_sigma  # [B]
        return out[:, 0], sigma


class NeighbourInputs:
    """Builds model inputs for queries from a fixed pool of observed points."""

    def __init__(self, pool: CountSet, standardization: Standardization, neighbours: int):
        self.standardization = standardization
        self.neighbours = neighbours
        self.pool_z = standardization.positions(pool.x)  # [P, D]
        self.pool_counts = standardization.count_features(pool.k, pool.n)  # [P, 2]

    def build(
        self, x: np.ndarray, exclude: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.standardization.positions(x)  # [Q, D]
        idx, _ = knn(z, self.pool_z, self.neighbours, exclude=exclude)  # [Q, K]
        offset = (self.pool_z[idx] - z[:, None, :]) / self.standardization.distance_scale
        return (
            torch.as_tensor(z, dtype=torch.float32),
            torch.as_tensor(offset, dtype=torch.float32),  # [Q, K, D]
            torch.as_tensor(self.pool_counts[idx], dtype=torch.float32),  # [Q, K, 2]
        )
