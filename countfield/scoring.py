"""Scores of predictive count distributions on held-out points.

The metric set and names match the RESUM_FLEX binomial count GP (core.gp_metrics plus its
count NLL and coverage), reimplemented here so both models' metrics.json compare one-to-one:
point metrics on the observed fraction k/n (bias, MAE, RMSE, Pearson/Spearman, descriptive
calibration slope), CRPS, central-interval coverage / width / interval score at
Gaussian-equivalent 1/2/3 sigma levels, weighted interval score, and the count NLL
(probability mass, not a density). Differences in how they are computed: the GP uses
Monte Carlo draws of k/n; here the predictive count pmf is evaluated on a deterministic
latent grid, so CRPS, median and intervals are exact for that pmf.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata

from countfield.data import CountSet
from countfield.likelihood import central_interval, latent_grid, log_marginal

LEVELS = {1: 0.682689492, 2: 0.954499736, 3: 0.997300204}


def predictive_summary(
    mu: np.ndarray,  # [Q] latent logit mean
    sigma: np.ndarray,  # [Q] latent logit sd
    n: np.ndarray,  # [Q] trials at the queries
    grid: int = 2048,
    k: np.ndarray | None = None,  # [Q] observed counts, for the CRPS
) -> dict[str, np.ndarray]:
    """Moments, median and central intervals of the predicted fraction k/n (and CRPS)."""
    p_grid = np.stack([expit(latent_grid(mu[i], sigma[i], grid)) for i in range(len(mu))])
    return predictive_summary_grid(p_grid, n, k)


def predictive_summary_grid(
    p_grid: np.ndarray,  # [Q, G] equally weighted draws (quantiles) of the hit probability
    n: np.ndarray,  # [Q] trials at the queries
    k: np.ndarray | None = None,  # [Q] observed counts, for the CRPS
) -> dict[str, np.ndarray]:
    """As predictive_summary for a predictive distribution of p given as equal-weight draws."""
    q = len(p_grid)
    out = {
        key: np.empty(q)
        for key in ("mean", "sigma", "median", "probability_mean", "probability_sd")
    }
    for level in LEVELS:
        out[f"lower_{level}"], out[f"upper_{level}"] = np.empty(q), np.empty(q)
    if k is not None:
        out["crps"] = np.empty(q)
    for i in range(q):
        p = p_grid[i]
        out["probability_mean"][i], out["probability_sd"][i] = p.mean(), p.std()
        # k/n: E = E[p]; Var = E[p(1-p)]/n + Var[p].
        out["mean"][i] = p.mean()
        out["sigma"][i] = np.sqrt((p * (1 - p)).mean() / n[i] + p.var())
        pmf = mixture_pmf(p, int(n[i]))
        cdf = np.cumsum(pmf)
        out["median"][i] = int(np.searchsorted(cdf, 0.5)) / n[i]
        for level, probability in LEVELS.items():
            lo, hi = central_interval(pmf, probability)
            out[f"lower_{level}"][i], out[f"upper_{level}"][i] = lo / n[i], hi / n[i]
        if k is not None:
            # Exact CRPS of a pmf on the lattice j/n: sum_j (F(j) - 1[j >= k])^2 / n.
            size = max(len(cdf), int(k[i]) + 1)
            cdf_full = np.concatenate([cdf, np.ones(size - len(cdf))])
            step = (np.arange(size) >= k[i]).astype(float)
            out["crps"][i] = np.sum((cdf_full - step) ** 2) / n[i]
    return out


def mixture_pmf(p: np.ndarray, n: int) -> np.ndarray:
    """P(k), k = 0..K, of Binomial(n, p) mixed over equal-weight draws p [G]."""
    from scipy.stats import binom

    k_max = int(min(n, binom.ppf(1 - 1e-9, n, p.max())))
    return binom.pmf(np.arange(k_max + 1)[None, :], n, p[:, None]).mean(axis=0)


def grid_log_mass(p_grid: np.ndarray, k: np.ndarray, n: np.ndarray) -> np.ndarray:  # [Q]
    """log P(k) of the binomial mixed over the equal-weight draws of each row of p_grid."""
    from scipy.special import logsumexp
    from scipy.stats import binom

    log_pmf = binom.logpmf(k[:, None], n[:, None], p_grid)  # [Q, G]
    return logsumexp(log_pmf, axis=1) - np.log(p_grid.shape[1])


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score(
    data: CountSet,
    mu: np.ndarray,  # [Q]
    sigma: np.ndarray,  # [Q]
    summary: dict[str, np.ndarray],
    nodes: int = 20,
    log_mass: np.ndarray | None = None,  # [Q] if given, replaces the logit-normal count NLL
) -> dict:
    """All held-out metrics of one evaluation group; `summary` from predictive_summary(k=...).

    mu/sigma may be None when `log_mass` is given (predictions as probability draws)."""
    y = data.fraction
    p = summary["mean"]
    residual = p - y
    slope = (
        None
        if np.ptp(p) == 0
        else float(np.dot(p - p.mean(), y - y.mean()) / np.sum((p - p.mean()) ** 2))
    )
    if log_mass is None:
        with torch.no_grad():
            log_mass = log_marginal(
                torch.as_tensor(data.k, dtype=torch.float64),
                torch.as_tensor(data.n, dtype=torch.float64),
                torch.as_tensor(mu, dtype=torch.float64),
                torch.as_tensor(sigma, dtype=torch.float64),
                nodes,
            ).numpy()

    details, weighted = {}, []
    for level in LEVELS:
        lo, hi = summary[f"lower_{level}"], summary[f"upper_{level}"]
        mass = erf(level / sqrt(2))
        alpha = 1 - mass
        width = hi - lo
        interval_score = width + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        inside = int(((y >= lo) & (y <= hi)).sum())
        details[str(level)] = dict(
            nominal_probability=mass,
            inside=inside,
            total=len(y),
            coverage=inside / len(y),
            mean_width=float(width.mean()),
            interval_score=float(interval_score.mean()),
        )
        weighted.append(alpha / 2 * interval_score)
    wis = (0.5 * np.abs(y - summary["median"]) + np.sum(weighted, axis=0)) / (len(weighted) + 0.5)

    z = (y - p) / summary["sigma"]
    noise = y * (1 - y) / np.maximum(data.n - 1, 1)
    result = dict(
        voxels=len(y),
        weighting="equal per validation voxel",
        response_units="original response",
        mean_observed=float(y.mean()),
        mean_predicted=float(p.mean()),
        mean_bias=float(residual.mean()),
        mae=float(np.abs(residual).mean()),
        rmse=float(np.sqrt(np.mean(residual**2))),
        pearson_r=_correlation(p, y),
        spearman_r=_correlation(rankdata(p), rankdata(y)),
        calibration_slope=slope,
        calibration_intercept=None if slope is None else float(y.mean() - slope * p.mean()),
        calibration_note=(
            "Descriptive OLS: observed = intercept + slope * predicted; noisy observations"
        ),
        interval_metrics=details,
        weighted_interval_score=float(wis.mean()),
        predictive_nll=float(-log_mass.mean()),
        predictive_distribution="binomial counts mixed over logistic Gaussian latent p",
        nll_measure="count probability mass; not comparable to a density NLL",
        coverage={level: d["coverage"] for level, d in details.items()},
        # countfield extras (not in the GP metrics)
        z_mean=float(z.mean()),
        z_std=float(z.std(ddof=1)) if len(z) > 1 else 0.0,
        z_outside_2=float(np.mean(np.abs(z) > 2)),
        mse_denoised=float(np.mean(residual**2 - noise)),
        latent_sigma_median=None if sigma is None else float(np.median(sigma)),
    )
    if "crps" in summary:
        result["crps"] = float(summary["crps"].mean())
    return result
