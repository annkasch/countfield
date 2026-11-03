"""Point Transformer over a point cloud whose point features are photon sets.

Points are voxels (positions only, no absolute coordinates enter the network); each visible
point's feature f_j is a learned summary of all its photons, f_j = post(sum_i w_i g(photon_i,
hit_i)) (conditional MC: the encoder sees which photons were emitted where and which of them
hit). Masked points (training targets, holdout, evaluation and new positions) get a learned
null feature and none of their photons enter any layer. A Point Transformer U-Net (vector
attention over k nearest points, farthest-point downsampling, interpolating upsampling with
skip connections) gives every point a feature, and a head returns a Gaussian over the logit of
its hit probability, as in the local-latent model.

The geometry (neighbour lists, downsampling, interpolation) depends only on the positions and
is built once per point cloud (`Geometry`); masks change features, never the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn


def farthest_point_sample(z: np.ndarray, m: int) -> np.ndarray:  # [N, D] -> [m] indices
    """Deterministic farthest-point sampling, starting from the point nearest the centroid."""
    if m >= len(z):
        return np.arange(len(z))
    chosen = np.empty(m, dtype=np.int64)
    chosen[0] = int(np.argmin(np.linalg.norm(z - z.mean(axis=0), axis=1)))
    distance = np.linalg.norm(z - z[chosen[0]], axis=1)
    for i in range(1, m):
        chosen[i] = int(np.argmax(distance))
        distance = np.minimum(distance, np.linalg.norm(z - z[chosen[i]], axis=1))
    return np.sort(chosen)


@dataclass
class Level:
    z: torch.Tensor  # [N_l, D] positions (standardized)
    scale: float  # median nearest-neighbour distance at this level
    neighbours: torch.Tensor  # [N_l, K] (includes the point itself)
    down: torch.Tensor | None  # [N_l, K] nearest points of the finer level (None at level 0)
    up: torch.Tensor | None  # [N_{l-1}, 3] nearest points of this level, for each finer point
    up_weight: torch.Tensor | None  # [N_{l-1}, 3] inverse-distance weights (rows sum to 1)


class Geometry:
    """Multi-level neighbourhood structure of a fixed point cloud."""

    def __init__(self, z: np.ndarray, levels: int, neighbours: int, ratio: int):
        self.levels: list[Level] = []
        index = np.arange(len(z))
        previous = None
        for level in range(levels):
            if level:
                index = farthest_point_sample(previous, max(1, len(previous) // ratio))
            current = z if level == 0 else previous[index]
            tree = cKDTree(current)
            k = min(neighbours, len(current))
            _, nbr = tree.query(current, k=k)
            nbr = nbr.reshape(len(current), k)
            d1 = tree.query(current, k=min(2, len(current)))[0]
            scale = float(np.median(d1[:, -1])) if len(current) > 1 else 1.0
            down = up = weight = None
            if level:
                k_down = min(neighbours, len(previous))
                down = torch.as_tensor(
                    cKDTree(previous).query(current, k=k_down)[1].reshape(len(current), k_down)
                )
                k_up = min(3, len(current))
                dist, up_idx = tree.query(previous, k=k_up)
                dist, up_idx = (
                    dist.reshape(len(previous), k_up),
                    up_idx.reshape(len(previous), k_up),
                )
                inv = 1.0 / np.maximum(dist, 1e-8)
                up, weight = (
                    torch.as_tensor(up_idx),
                    torch.as_tensor(inv / inv.sum(axis=1, keepdims=True), dtype=torch.float32),
                )
            self.levels.append(
                Level(
                    torch.as_tensor(current, dtype=torch.float32),
                    max(scale, 1e-8),
                    torch.as_tensor(nbr),
                    down,
                    up,
                    weight,
                )
            )
            previous = current


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


class PhotonEncoder(nn.Module):
    """Photon set of one point -> feature: post([mean, sum / n_scale, log(n / n_scale)])."""

    def __init__(self, n_features: int, hidden: int, out: int):
        super().__init__()
        self.g = _mlp([n_features + 1, hidden, hidden])
        self.post = _mlp([2 * hidden + 1, hidden, out])

    def forward(
        self,
        features: torch.Tensor,  # [P, F] standardized photon features
        hit: torch.Tensor,  # [P] 0/1
        weight: torch.Tensor,  # [P] subsampling weight (1 for hits)
        owner: torch.Tensor,  # [P] index of the owning point in 0..V-1
        n: torch.Tensor,  # [V] photons per point (all, not only the subsample)
        n_scale: float,
        chunk: int = 1_000_000,
    ) -> torch.Tensor:  # [V, out]
        total = features.new_zeros(len(n), self.g[-1].out_features)  # [V, H]
        for start in range(0, len(features), chunk):
            sl = slice(start, start + chunk)
            h = self.g(torch.cat([features[sl], (2 * hit[sl] - 1).unsqueeze(-1)], dim=-1))
            total = total.index_add(0, owner[sl], h * weight[sl].unsqueeze(-1))
        return self.post(
            torch.cat(
                [total / n.unsqueeze(-1), total / n_scale, torch.log(n / n_scale).unsqueeze(-1)],
                dim=-1,
            )
        )


class PointTransformerLayer(nn.Module):
    """Vector self-attention over k nearest points with a relative position encoding."""

    def __init__(self, dim: int, pos_dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.position = _mlp([pos_dim, dim, dim])
        self.weight = _mlp([dim, dim, dim])

    def forward(self, f: torch.Tensor, level: Level) -> torch.Tensor:  # [N, C] -> [N, C]
        q, k, v = self.qkv(f).chunk(3, dim=-1)  # [N, C] each
        nbr = level.neighbours  # [N, K]
        delta = self.position((level.z[nbr] - level.z.unsqueeze(1)) / level.scale)  # [N, K, C]
        w = torch.softmax(self.weight(q.unsqueeze(1) - k[nbr] + delta), dim=1)  # [N, K, C]
        return (w * (v[nbr] + delta)).sum(dim=1)


class PointTransformerBlock(nn.Module):
    def __init__(self, dim: int, pos_dim: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = PointTransformerLayer(dim, pos_dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = _mlp([dim, 2 * dim, dim])

    def forward(self, f: torch.Tensor, level: Level) -> torch.Tensor:
        f = f + self.proj(self.attention(self.norm1(f), level))
        return f + self.ffn(self.norm2(f))


class TransitionDown(nn.Module):
    """Coarse point feature = max over its nearest finer points of MLP(rel. position, f)."""

    def __init__(self, dim_in: int, dim_out: int, pos_dim: int):
        super().__init__()
        self.mlp = _mlp([dim_in + pos_dim, dim_out, dim_out])

    def forward(self, f: torch.Tensor, fine: Level, coarse: Level) -> torch.Tensor:
        rel = (fine.z[coarse.down] - coarse.z.unsqueeze(1)) / coarse.scale  # [N_c, K, D]
        return self.mlp(torch.cat([rel, f[coarse.down]], dim=-1)).max(dim=1).values


class TransitionUp(nn.Module):
    """Fine feature = Linear(skip) + inverse-distance interpolation of Linear(coarse)."""

    def __init__(self, dim_coarse: int, dim_fine: int):
        super().__init__()
        self.coarse, self.skip = nn.Linear(dim_coarse, dim_fine), nn.Linear(dim_fine, dim_fine)

    def forward(self, coarse_f: torch.Tensor, skip: torch.Tensor, coarse: Level) -> torch.Tensor:
        interpolated = (self.coarse(coarse_f)[coarse.up] * coarse.up_weight.unsqueeze(-1)).sum(1)
        return self.skip(skip) + interpolated


class PhotonPointTransformer(nn.Module):
    def __init__(
        self,
        n_photon_features: int,
        pos_dim: int,
        dims: list[int],
        blocks: int = 1,
        photon_hidden: int = 64,
        min_sigma: float = 1e-3,
    ):
        super().__init__()
        self.min_sigma = min_sigma
        self.encoder = PhotonEncoder(n_photon_features, photon_hidden, dims[0])
        self.null = nn.Parameter(torch.zeros(dims[0]))
        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, dim in enumerate(dims):
            if i:
                self.downs.append(TransitionDown(dims[i - 1], dim, pos_dim))
            self.down_blocks.append(
                nn.ModuleList(PointTransformerBlock(dim, pos_dim) for _ in range(blocks))
            )
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(TransitionUp(dims[i], dims[i - 1]))
            self.up_blocks.append(
                nn.ModuleList(PointTransformerBlock(dims[i - 1], pos_dim) for _ in range(blocks))
            )
        self.head = _mlp([dims[0], dims[0], 2])

    def forward(
        self,
        geometry: Geometry,
        visible: torch.Tensor,  # [V] indices of points whose photons are used
        photon_args: tuple,  # (features, hit, weight, owner in 0..V-1, n [V], n_scale)
        n_points: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # mu [N], sigma [N] (logit offset)
        f = self.null.expand(n_points, -1).clone()  # [N, C0]
        if len(visible):
            f[visible] = self.encoder(*photon_args)
        levels = geometry.levels
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            if i:
                f = self.downs[i - 1](f, levels[i - 1], levels[i])
            for block in blocks:
                f = block(f, levels[i])
            skips.append(f)
        for j, (up, blocks) in enumerate(zip(self.ups, self.up_blocks, strict=True)):
            i = len(levels) - 1 - j  # coarse level index
            f = up(f, skips[i - 1], levels[i])
            for block in blocks:
                f = block(f, levels[i - 1])
        out = self.head(f)  # [N, 2]
        return out[:, 0], F.softplus(out[:, 1] - 1.0) + self.min_sigma
