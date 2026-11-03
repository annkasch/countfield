"""Scores of predictive count distributions on held-out points.

Same quantities as the RESUM_FLEX binomial count GP, so results can be compared at file level:
count NLL (probability mass, not a density), coverage of central predictive intervals of the
counts at Gaussian-equivalent 1/2/3 sigma levels, z-scores and denoised squared error.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import expit

from countfield.data import CountSet
from countfield.likelihood import central_interval, latent_grid, log_marginal, predictive_pmf

LEVELS = {1: 0.682689492, 2: 0.954499736, 3: 0.997300204}


def predictive_summary(
    mu: np.ndarray,  # [Q] latent logit mean
    sigma: np.ndarray,  # [Q] latent logit sd
    n: np.ndarray,  # [Q] trials at the queries
    grid: int = 2048,
) -> dict[str, np.ndarray]:
    """Mean and sd of the predicted fraction k/n and central count intervals (as fractions)."""
    q = len(mu)
    mean, sd = np.empty(q), np.empty(q)
    bounds = {level: (np.empty(q), np.empty(q)) for level in LEVELS}
    probability_mean, probability_sd = np.empty(q), np.empty(q)
    for i in range(q):
        p = expit(latent_grid(mu[i], sigma[i], grid))
        probability_mean[i], probability_sd[i] = p.mean(), p.std()
        # k/n: E = E[p]; Var = E[p(1-p)]/n + Var[p].
        mean[i] = p.mean()
        sd[i] = np.sqrt((p * (1 - p)).mean() / n[i] + p.var())
        pmf = predictive_pmf(mu[i], sigma[i], int(n[i]), grid)
        for level, probability in LEVELS.items():
            lo, hi = central_interval(pmf, probability)
            bounds[level][0][i], bounds[level][1][i] = lo / n[i], hi / n[i]
    out = dict(
        mean=mean, sigma=sd, probability_mean=probability_mean, probability_sd=probability_sd
    )
    for level, (lo, hi) in bounds.items():
        out[f"lower_{level}"], out[f"upper_{level}"] = lo, hi
    return out


def score(
    data: CountSet,
    mu: np.ndarray,  # [Q]
    sigma: np.ndarray,  # [Q]
    summary: dict[str, np.ndarray],
    nodes: int = 20,
) -> dict:
    observed = data.fraction
    with torch.no_grad():
        log_mass = log_marginal(
            torch.as_tensor(data.k, dtype=torch.float64),
            torch.as_tensor(data.n, dtype=torch.float64),
            torch.as_tensor(mu, dtype=torch.float64),
            torch.as_tensor(sigma, dtype=torch.float64),
            nodes,
        ).numpy()
    mean, sd = summary["mean"], summary["sigma"]
    z = (observed - mean) / sd
    noise = observed * (1 - observed) / np.maximum(data.n - 1, 1)
    return dict(
        points=len(data),
        predictive_nll=float(-log_mass.mean()),
        nll_measure="count probability mass; not comparable to a density NLL",
        coverage={
            str(level): float(
                (
                    (observed >= summary[f"lower_{level}"])
                    & (observed <= summary[f"upper_{level}"])
                ).mean()
            )
            for level in LEVELS
        },
        z_mean=float(z.mean()),
        z2_mean=float((z**2).mean()),
        mean_observed=float(observed.mean()),
        mean_predicted=float(mean.mean()),
        rmse=float(np.sqrt(np.mean((observed - mean) ** 2))),
        mse_denoised=float(np.mean((observed - mean) ** 2 - noise)),
        pearson_r=float(np.corrcoef(observed, mean)[0, 1]) if len(data) > 1 else None,
        latent_sigma_median=float(np.median(sigma)),
    )
