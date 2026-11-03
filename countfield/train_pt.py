"""Masked training of the photon Point Transformer.

The point cloud holds all observed points (training + holdout) and the evaluation points;
its geometry is built once. Each step masks a random share of the training points: their
photons are dropped (null feature) and their counts are the targets. Holdout and evaluation
points are always masked and never targets during training. Visible points contribute all
their hits and a weighted with-replacement sample of their misses (unbiased photon sums).
Checkpoints are selected on the holdout count NLL with all training points visible; the
final prediction of the evaluation points uses all observed points as visible.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from countfield.config import PointTransformerConfig, RunConfig
from countfield.likelihood import log_marginal
from countfield.photons import PhotonSet
from countfield.pointcloud import Geometry, PhotonPointTransformer


def _stream(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, k]))


class PhotonCloud:
    """Tensors of the observed photons, the cloud geometry and photon sampling."""

    def __init__(
        self,
        observed: PhotonSet,
        train_idx: np.ndarray,
        extra_x: list[np.ndarray],
        m: PointTransformerConfig,
    ):
        self.n_observed = len(observed)
        x = np.concatenate([observed.x, *extra_x]) if extra_x else observed.x
        train_x = observed.x[train_idx]
        self.x_mean, self.x_scale = train_x.mean(axis=0), train_x.std(axis=0)
        if np.any(self.x_scale <= 0):
            raise ValueError("training positions are constant along a dimension")
        self.geometry = Geometry(
            (x - self.x_mean) / self.x_scale, len(m.dims), m.neighbours, m.downsample_ratio
        )
        self.n_points = len(x)

        owner = observed.point  # [P]
        train_photons = np.isin(owner, train_idx)
        feats = observed.features
        sample = feats[train_photons][:: max(1, int(train_photons.sum()) // 1_000_000)]
        self.f_mean = sample.mean(axis=0)
        self.f_scale = np.where(sample.std(axis=0) > 0, sample.std(axis=0), 1.0)
        self.features = torch.as_tensor((feats - self.f_mean) / self.f_scale, dtype=torch.float32)
        self.hit = torch.as_tensor(observed.hit, dtype=torch.float32)
        self.k = observed.k
        self.n = observed.n
        self.n_scale = float(np.median(observed.n[train_idx]))
        rate = observed.k[train_idx].sum() / observed.n[train_idx].sum()
        self.logit_rate = float(np.log(rate / (1 - rate)))
        # Hits and misses grouped by point (rows into the photon arrays).
        hit_rows = np.flatnonzero(observed.hit == 1)
        self.hit_rows, self.hit_owner = hit_rows, owner[hit_rows]
        miss_rows = np.flatnonzero(observed.hit == 0)  # sorted by owner (CSR order)
        self.miss_rows = miss_rows
        self.miss_count = np.bincount(owner[miss_rows], minlength=self.n_observed)
        self.miss_start = np.concatenate([[0], np.cumsum(self.miss_count)[:-1]])
        self.owner = owner

    def photon_args(self, visible: np.ndarray, misses: int | None, rng: np.random.Generator):
        """Encoder inputs for the visible points: all hits + sampled (or all) misses."""
        local = np.full(self.n_observed, -1, dtype=np.int64)
        local[visible] = np.arange(len(visible))
        hits = self.hit_rows[local[self.hit_owner] >= 0]
        if misses is None:
            miss = self.miss_rows[local[self.owner[self.miss_rows]] >= 0]
            miss_weight = np.ones(len(miss))
        else:
            count = self.miss_count[visible]  # [V]
            u = rng.random((len(visible), misses))
            pick = np.minimum(
                (u * count[:, None]).astype(np.int64), np.maximum(count - 1, 0)[:, None]
            )
            miss = self.miss_rows[(self.miss_start[visible][:, None] + pick)[count > 0].ravel()]
            miss_weight = np.repeat(count[count > 0] / misses, misses)
        rows = np.concatenate([hits, miss])
        weight = np.concatenate([np.ones(len(hits)), miss_weight])
        return (
            self.features[rows],
            self.hit[rows],
            torch.as_tensor(weight, dtype=torch.float32),
            torch.as_tensor(local[self.owner[rows]]),
            torch.as_tensor(self.n[visible], dtype=torch.float32),
            self.n_scale,
        )

    def forward(self, model, visible, misses, rng):
        args = self.photon_args(visible, misses, rng)
        mu, sigma = model(self.geometry, torch.as_tensor(visible), args, self.n_points)
        return mu + self.logit_rate, sigma

    def nll(self, mu, sigma, points, nodes):
        k = torch.as_tensor(self.k[points], dtype=mu.dtype)
        n = torch.as_tensor(self.n[points], dtype=mu.dtype)
        return -log_marginal(k, n, mu[points], sigma[points], nodes).mean()


def fit_point_transformer(
    observed: PhotonSet,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    evaluation_x: list[np.ndarray],
    config: RunConfig,
    log=print,
):
    """Train, select on holdout, return (history, checkpoint dict, latents per eval set)."""
    m, t = config.model, config.training
    cloud = PhotonCloud(observed, train_idx, evaluation_x, m)
    torch.manual_seed(int(_stream(t.seed, 1).integers(2**31)))
    model = PhotonPointTransformer(
        observed.features.shape[1], observed.x.shape[1], m.dims, m.blocks, m.photon_hidden
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay
    )
    masks, photons = _stream(t.seed, 2), _stream(t.seed, 3)
    n_mask = max(1, int(round(m.mask_fraction * len(train_idx))))
    history, best, best_nll, running = [], None, np.inf, []
    for step in range(1, t.steps + 1):
        model.train()
        order = masks.permutation(train_idx)
        target, visible = np.sort(order[:n_mask]), np.sort(order[n_mask:])
        mu, sigma = cloud.forward(model, visible, m.misses_per_point, photons)
        loss = cloud.nll(mu, sigma, target, m.quadrature_nodes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running.append(loss.item())
        if step % t.eval_every == 0 or step == t.steps:
            model.eval()
            with torch.no_grad():
                mu, sigma = cloud.forward(
                    model, np.sort(train_idx), m.eval_misses_per_point, _stream(t.seed, 4)
                )
                nll = cloud.nll(mu.double(), sigma.double(), holdout_idx, m.quadrature_nodes)
            history.append(
                dict(
                    step=step,
                    train_nll=float(np.mean(running)),
                    holdout_nll=nll.item(),
                    holdout_sigma_median=float(sigma[holdout_idx].median()),
                )
            )
            running = []
            if nll.item() < best_nll:
                best_nll, best = (
                    nll.item(),
                    dict(step=step, state=copy.deepcopy(model.state_dict())),
                )
            log(f"step {step}: train {history[-1]['train_nll']:.4f} holdout {nll.item():.4f}")
    model.load_state_dict(best["state"])
    history.append(dict(selected_step=best["step"], selected_holdout_nll=best_nll))

    model.eval()
    with torch.no_grad():
        mu, sigma = cloud.forward(
            model, np.arange(cloud.n_observed), m.eval_misses_per_point, _stream(t.seed, 5)
        )
    latents, start = [], cloud.n_observed
    for x in evaluation_x:
        sl = slice(start, start + len(x))
        latents.append((mu[sl].double().numpy(), sigma[sl].double().numpy()))
        start += len(x)
    checkpoint = dict(
        state=model.state_dict(),
        x_mean=cloud.x_mean.tolist(),
        x_scale=cloud.x_scale.tolist(),
        feature_mean=cloud.f_mean.tolist(),
        feature_scale=cloud.f_scale.tolist(),
        n_scale=cloud.n_scale,
        logit_rate=cloud.logit_rate,
    )
    return history, checkpoint, latents
