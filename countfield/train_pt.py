"""Masked training of the photon Point Transformer.

The point cloud holds all observed points (training + holdout) and the evaluation points.
Each step masks a random share of the training points: their photons are dropped (null
feature) and they are the targets. Holdout and evaluation points are always masked and never
targets during training. Visible points contribute all their hits and a weighted
with-replacement sample of their misses (unbiased photon sums), or only their counts
(`point_features: counts`, ablation).

Losses (`loss`):
- voxel: binomial likelihood of each target's count, k ~ Bin(n, sigmoid(eta)),
  eta ~ N(mu, sigma^2).
- photon: Bernoulli likelihood of every photon of each target (all hits + weighted misses),
  logit p_i = mu + s(h, photon_i) + eps with a photon head s and a latent eps ~ N(0, sigma^2)
  shared by the target's photons. A voxel's hit probability is the mean over a photon sample;
  predictions use donor photons (a fresh emission sample from training points), so nothing of
  the query's own simulation enters.

Regularization: graph thinning (each step keeps a random share of the training points and
rebuilds the neighbourhoods), a random masked share, dropout and weight decay.
Checkpoints are selected on the holdout count NLL with all training points visible; the final
prediction of the evaluation points uses all observed points as visible.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.stats import norm

from countfield.config import PointTransformerConfig, RunConfig
from countfield.likelihood import log_marginal, photon_log_marginal
from countfield.photons import PhotonSet
from countfield.pointcloud import Geometry, PhotonPointTransformer
from countfield.resume import load_state, save_state
from countfield.scoring import grid_log_mass


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
        self.m = m
        self.n_observed = len(observed)
        x = np.concatenate([observed.x, *extra_x]) if extra_x else observed.x
        train_x = observed.x[train_idx]
        self.x_mean, self.x_scale = train_x.mean(axis=0), train_x.std(axis=0)
        if np.any(self.x_scale <= 0):
            raise ValueError("training positions are constant along a dimension")
        self.z = (x - self.x_mean) / self.x_scale  # [N, D] all points
        self.n_points = len(x)
        self._full = None
        self.train_idx = np.sort(train_idx)

        owner = observed.point  # [P]
        train_photons = np.isin(owner, train_idx)
        feats = observed.features
        sample = feats[train_photons][:: max(1, int(train_photons.sum()) // 1_000_000)]
        self.f_mean = sample.mean(axis=0)
        self.f_scale = np.where(sample.std(axis=0) > 0, sample.std(axis=0), 1.0)
        self.features = torch.as_tensor((feats - self.f_mean) / self.f_scale, dtype=torch.float32)
        self.hit = torch.as_tensor(observed.hit, dtype=torch.float32)
        self.k, self.n = observed.k, observed.n
        self.offsets = observed.offsets
        self.n_scale = float(np.median(observed.n[train_idx]))
        rate = observed.k[train_idx].sum() / observed.n[train_idx].sum()
        self.logit_rate = float(np.log(rate / (1 - rate)))
        smoothed = (self.k + 0.5) / (self.n + 1.0)
        self.count_features = torch.as_tensor(
            np.column_stack(
                [np.log(smoothed / (1 - smoothed)) - self.logit_rate, np.log(self.n / self.n_scale)]
            ),
            dtype=torch.float32,
        )
        # Hits and misses grouped by point (rows into the photon arrays).
        hit_rows = np.flatnonzero(observed.hit == 1)
        self.hit_rows, self.hit_owner = hit_rows, owner[hit_rows]
        miss_rows = np.flatnonzero(observed.hit == 0)  # sorted by owner (CSR order)
        self.miss_rows = miss_rows
        self.miss_count = np.bincount(owner[miss_rows], minlength=self.n_observed)
        self.miss_start = np.concatenate([[0], np.cumsum(self.miss_count)[:-1]])
        self.owner = owner

    # --- geometry -------------------------------------------------------------------------
    @property
    def full_geometry(self) -> Geometry:
        if self._full is None:
            self._full = self.geometry(np.arange(self.n_points))
        return self._full

    def geometry(self, points: np.ndarray) -> Geometry:
        m = self.m
        return Geometry(self.z[points], len(m.dims), m.neighbours, m.downsample_ratio)

    # --- photons ----------------------------------------------------------------------------
    def sample_rows(self, points: np.ndarray, misses: int | None, rng: np.random.Generator):
        """All hits + sampled (or all) misses of `points`: rows, weights, owner (0..len-1)."""
        local = np.full(self.n_observed, -1, dtype=np.int64)
        local[points] = np.arange(len(points))
        hits = self.hit_rows[local[self.hit_owner] >= 0]
        if misses is None:
            miss = self.miss_rows[local[self.owner[self.miss_rows]] >= 0]
            miss_weight = np.ones(len(miss))
        else:
            count = self.miss_count[points]  # [V]
            u = rng.random((len(points), misses))
            pick = np.minimum(
                (u * count[:, None]).astype(np.int64), np.maximum(count - 1, 0)[:, None]
            )
            miss = self.miss_rows[(self.miss_start[points][:, None] + pick)[count > 0].ravel()]
            miss_weight = np.repeat(count[count > 0] / misses, misses)
        rows = np.concatenate([hits, miss])
        weight = torch.as_tensor(
            np.concatenate([np.ones(len(hits)), miss_weight]), dtype=torch.float32
        )
        return rows, weight, torch.as_tensor(local[self.owner[rows]])

    def encoder_args(self, visible: np.ndarray, misses: int | None, rng: np.random.Generator):
        if self.m.point_features == "counts":
            return (self.count_features[visible],)
        rows, weight, owner = self.sample_rows(visible, misses, rng)
        return (
            self.features[rows],
            self.hit[rows],
            weight,
            owner,
            torch.as_tensor(self.n[visible], dtype=torch.float32),
            self.n_scale,
        )

    def donor_rows(self, queries: int, photons: int, rng: np.random.Generator) -> np.ndarray:
        """[Q, photons] photon rows of a random training donor point per query (labels unused)."""
        donors = rng.choice(self.train_idx, size=queries)
        start, count = self.offsets[donors], self.n[donors]
        pick = np.minimum(
            (rng.random((queries, photons)) * count[:, None]).astype(np.int64), (count - 1)[:, None]
        )
        return start[:, None] + pick

    # --- model passes -----------------------------------------------------------------------
    def forward(self, model, points: np.ndarray | None, visible: np.ndarray, misses, rng):
        """mu (absolute logit), sigma, h on `points` (None = full cloud) with `visible` used."""
        if points is None:
            geometry, local, size = self.full_geometry, visible, self.n_points
        else:
            index = np.full(self.n_points, -1, dtype=np.int64)
            index[points] = np.arange(len(points))
            geometry, local, size = self.geometry(points), index[visible], len(points)
        mu, sigma, h = model(
            geometry, torch.as_tensor(local), self.encoder_args(visible, misses, rng), size
        )
        return mu + self.logit_rate, sigma, h

    def voxel_nll(self, mu, sigma, points_global, at, nodes):
        k = torch.as_tensor(self.k[points_global], dtype=mu.dtype)
        n = torch.as_tensor(self.n[points_global], dtype=mu.dtype)
        return -log_marginal(k, n, mu[at], sigma[at], nodes)

    def photon_nll(self, model, mu, sigma, h, points_global, at, misses, rng, nodes):
        rows, weight, owner = self.sample_rows(points_global, misses, rng)
        at = torch.as_tensor(at)
        base = mu[at][owner] + model.photon_head(h[at][owner], self.features[rows])
        return -photon_log_marginal(base, self.hit[rows], weight, owner, sigma[at], nodes)

    @torch.no_grad()
    def probability_grid(self, model, mu, sigma, h, at, grid, photons, rng, chunk=16):
        """[Q, grid] equal-weight draws of the voxel hit probability at cloud points `at`
        (photon loss): mean over donor photons of sigmoid(mu + s + eps), eps quantiles."""
        rows = self.donor_rows(len(at), photons, rng)  # [Q, D]
        quantiles = torch.as_tensor(norm.ppf((np.arange(grid) + 0.5) / grid), dtype=mu.dtype)
        out = []
        for start in range(0, len(at), chunk):
            idx = torch.as_tensor(at[start : start + chunk])
            r = rows[start : start + chunk]
            hq = h[idx].unsqueeze(1).expand(-1, r.shape[1], -1).reshape(-1, h.shape[1])
            s = model.photon_head(hq, self.features[r.ravel()]).reshape(r.shape)  # [C, D]
            base = (mu[idx].unsqueeze(1) + s).unsqueeze(-1)  # [C, D, 1]
            eps = sigma[idx].unsqueeze(-1) * quantiles  # [C, G]
            out.append(torch.sigmoid(base + eps.unsqueeze(1)).mean(dim=1))  # [C, G]
        return torch.cat(out).double().numpy()


def fit_point_transformer(
    observed: PhotonSet,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    evaluation_x: list[np.ndarray],
    config: RunConfig,
    log=print,
    resume=None,
):
    """Train, select on holdout; returns (history, checkpoint dict, predictions per eval set).

    ``resume``: checkpoint file written at every evaluation step; an existing one continues
    the run from its step.

    Predictions are (mu, sigma) arrays (voxel loss) or a [Q, G] probability grid (photon loss).
    """
    m, t = config.model, config.training
    photon = m.loss == "photon"
    cloud = PhotonCloud(observed, train_idx, evaluation_x, m)
    torch.manual_seed(int(_stream(t.seed, 1).integers(2**31)))
    model = PhotonPointTransformer(
        observed.features.shape[1],
        observed.x.shape[1],
        m.dims,
        m.blocks,
        m.photon_hidden,
        point_features=m.point_features,
        photon_head=photon,
        dropout=m.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay
    )
    masks, photons = _stream(t.seed, 2), _stream(t.seed, 3)
    thinning = m.keep_fraction_min < 1
    selection_grid = min(256, config.evaluation.latent_grid)

    def holdout_nll():
        model.eval()
        with torch.no_grad():
            mu, sigma, h = cloud.forward(
                model, None, cloud.train_idx, m.eval_misses_per_point, _stream(t.seed, 4)
            )
            if photon:
                grid = cloud.probability_grid(
                    model,
                    mu,
                    sigma,
                    h,
                    holdout_idx,
                    selection_grid,
                    m.prediction_photons,
                    _stream(t.seed, 6),
                )
                value = -grid_log_mass(grid, cloud.k[holdout_idx], cloud.n[holdout_idx]).mean()
                spread = float(np.median(grid.std(axis=1) / grid.mean(axis=1)))
            else:
                value = (
                    cloud.voxel_nll(
                        mu.double(), sigma.double(), holdout_idx, holdout_idx, m.quadrature_nodes
                    )
                    .mean()
                    .item()
                )
                spread = float(sigma[holdout_idx].median())
        return float(value), spread

    history, best, best_nll, running = [], None, np.inf, []
    streams = dict(masks=masks, photons=photons)
    state = load_state(resume, model, optimizer, streams)
    start = 1
    if state is not None:
        history, best, best_nll = state["history"], state["best"], state["best_nll"]
        start = state["step"] + 1
        log(f"resuming after step {state['step']}")
    for step in range(start, t.steps + 1):
        model.train()
        keep = cloud.train_idx
        if thinning:
            share = masks.uniform(m.keep_fraction_min, 1.0)
            keep = np.sort(
                masks.choice(cloud.train_idx, max(2, round(share * len(keep))), replace=False)
            )
        fraction = (
            masks.uniform(m.mask_fraction, m.mask_fraction_max)
            if m.mask_fraction_max
            else m.mask_fraction
        )
        order = masks.permutation(keep)
        n_mask = min(len(keep) - 1, max(1, round(fraction * len(keep))))
        target, visible = np.sort(order[:n_mask]), np.sort(order[n_mask:])
        mu, sigma, h = cloud.forward(
            model, keep if thinning else None, visible, m.misses_per_point, photons
        )
        at = np.searchsorted(keep, target) if thinning else target
        if photon:
            loss = cloud.photon_nll(
                model, mu, sigma, h, target, at, m.misses_per_point, photons, m.quadrature_nodes
            ).mean()
        else:
            loss = cloud.voxel_nll(mu, sigma, target, at, m.quadrature_nodes).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running.append(loss.item())
        if step % t.eval_every == 0 or step == t.steps:
            nll, spread = holdout_nll()
            history.append(
                dict(
                    step=step,
                    train_loss=float(np.mean(running)),
                    holdout_nll=nll,
                    holdout_spread=spread,
                )
            )
            running = []
            if nll < best_nll:
                best_nll, best = nll, dict(step=step, state=copy.deepcopy(model.state_dict()))
            log(f"step {step}: train {history[-1]['train_loss']:.4f} holdout {nll:.4f}")
            save_state(
                resume,
                step,
                model,
                optimizer,
                streams,
                history=history,
                best=best,
                best_nll=best_nll,
            )
    model.load_state_dict(best["state"])
    history.append(dict(selected_step=best["step"], selected_holdout_nll=best_nll))

    model.eval()
    with torch.no_grad():
        mu, sigma, h = cloud.forward(
            model, None, np.arange(cloud.n_observed), m.eval_misses_per_point, _stream(t.seed, 5)
        )
    predictions, start = [], cloud.n_observed
    for i, x in enumerate(evaluation_x):
        at = np.arange(start, start + len(x))
        if photon:
            predictions.append(
                cloud.probability_grid(
                    model,
                    mu,
                    sigma,
                    h,
                    at,
                    config.evaluation.latent_grid,
                    m.prediction_photons,
                    _stream(t.seed, 7 + i),
                )
            )
        else:
            predictions.append((mu[at].double().numpy(), sigma[at].double().numpy()))
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
    return history, checkpoint, predictions
