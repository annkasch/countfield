"""Binomial counts with a Gaussian latent logit: k ~ Binomial(n, sigmoid(eta)), eta ~ N(mu, s^2).

`log_marginal` integrates the latent with Gauss-Hermite quadrature centred on a Gaussian
approximation of the posterior of eta (adaptive quadrature), so it stays accurate when the
binomial likelihood is much narrower than the latent prior. `predictive_pmf` gives the full
count distribution at a query point for intervals and moments.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from scipy.stats import binom, norm


def binomial_log_pmf(
    k: torch.Tensor,  # [...]
    n: torch.Tensor,  # [...]
    eta: torch.Tensor,  # [..., Q] logits
) -> torch.Tensor:  # [..., Q]
    k, n = k.unsqueeze(-1), n.unsqueeze(-1)
    log_choose = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return log_choose + k * F.logsigmoid(eta) + (n - k) * F.logsigmoid(-eta)


def log_marginal(
    k: torch.Tensor,  # [B] observed successes
    n: torch.Tensor,  # [B] trials
    mu: torch.Tensor,  # [B] latent mean (logit)
    sigma: torch.Tensor,  # [B] latent sd (> 0)
    nodes: int = 20,
) -> torch.Tensor:  # [B]
    """log ∫ Binomial(k | n, sigmoid(eta)) N(eta; mu, sigma^2) d eta."""
    k, n = k.to(mu.dtype), n.to(mu.dtype)
    x, w = np.polynomial.hermite.hermgauss(nodes)
    x = torch.as_tensor(x, dtype=mu.dtype)  # [Q]
    log_w = torch.as_tensor(np.log(w / math.sqrt(math.pi)), dtype=mu.dtype)  # [Q]
    # Gaussian approximation of the binomial likelihood in eta, combined with the prior.
    p_hat = (k + 0.5) / (n + 1.0)  # [B]
    eta_hat = torch.log(p_hat) - torch.log1p(-p_hat)  # [B]
    tau_lik = n * p_hat * (1.0 - p_hat)  # [B]
    tau_prior = sigma.pow(-2)  # [B]
    tau = tau_prior + tau_lik  # [B]
    m = (mu * tau_prior + eta_hat * tau_lik) / tau  # [B]
    s = tau.rsqrt()  # [B]
    eta = m.unsqueeze(-1) + math.sqrt(2.0) * s.unsqueeze(-1) * x  # [B, Q]
    log_prior = _normal_log_pdf(eta, mu.unsqueeze(-1), sigma.unsqueeze(-1))  # [B, Q]
    log_proposal = _normal_log_pdf(eta, m.unsqueeze(-1), s.unsqueeze(-1))  # [B, Q]
    terms = log_w + binomial_log_pmf(k, n, eta) + log_prior - log_proposal  # [B, Q]
    return torch.logsumexp(terms, dim=-1)


def _normal_log_pdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd).pow(2) - torch.log(sd) - 0.5 * math.log(2 * math.pi)


def latent_grid(mu: float, sigma: float, size: int) -> np.ndarray:  # [size]
    """Equal-probability quantiles of N(mu, sigma^2) (deterministic latent draws)."""
    return mu + sigma * norm.ppf((np.arange(size) + 0.5) / size)


def predictive_pmf(mu: float, sigma: float, n: int, size: int = 2048) -> np.ndarray:
    """P(k) for k = 0..K of the latent-mixed binomial; K covers all but < 1e-9 of the mass."""
    p = expit(latent_grid(mu, sigma, size))  # [size]
    k_max = int(min(n, binom.ppf(1 - 1e-9, n, p.max())))
    ks = np.arange(k_max + 1)
    return binom.pmf(ks[None, :], n, p[:, None]).mean(axis=0)  # [K + 1]


def central_interval(pmf: np.ndarray, probability: float) -> tuple[int, int]:
    """Smallest count with CDF >= (1-q)/2 and smallest with CDF >= (1+q)/2."""
    cdf = np.cumsum(pmf)
    lower = int(np.searchsorted(cdf, (1 - probability) / 2))
    upper = int(np.searchsorted(cdf, (1 + probability) / 2))
    return lower, min(upper, len(pmf) - 1)


def photon_log_marginal(
    base: torch.Tensor,  # [P] per-photon logit without the shared latent (mu_j + s_ij)
    hit: torch.Tensor,  # [P] 0/1
    weight: torch.Tensor,  # [P] subsampling weights (1 for hits)
    owner: torch.Tensor,  # [P] target index 0..T-1 of each photon
    sigma: torch.Tensor,  # [T] sd of the shared latent eps_j
    nodes: int = 20,
    newton: int = 6,
) -> torch.Tensor:  # [T]
    """log ∫ prod_i Bernoulli(hit_i | sigmoid(base_i + eps))^w_i N(eps; 0, sigma^2) d eps.

    Adaptive Gauss-Hermite: the nodes are centred on the posterior mode of eps (a few Newton
    steps, no gradient) with the Laplace width; the estimate stays a quadrature of the same
    integral, so the centring does not bias it. Weighted misses give an unbiased estimate of
    the log-likelihood sum at each node.
    """
    t = len(sigma)
    dtype = base.dtype
    with torch.no_grad():
        m = torch.zeros(t, dtype=dtype)
        for _ in range(newton + 1):
            p = torch.sigmoid(base + m[owner])
            curvature = torch.zeros(t, dtype=dtype).index_add(0, owner, weight * p * (1 - p))
            curvature = curvature + sigma.pow(-2)
            if _ == newton:
                break
            grad = torch.zeros(t, dtype=dtype).index_add(0, owner, weight * (hit - p))
            m = m + (grad - m * sigma.pow(-2)) / curvature
        s = curvature.rsqrt()
    x, w = np.polynomial.hermite.hermgauss(nodes)
    x = torch.as_tensor(x, dtype=dtype)
    log_w = torch.as_tensor(np.log(w / math.sqrt(math.pi)), dtype=dtype)
    eps = m.unsqueeze(-1) + math.sqrt(2.0) * s.unsqueeze(-1) * x  # [T, Q]
    eta = base.unsqueeze(-1) + eps[owner]  # [P, Q]
    ll = weight.unsqueeze(-1) * (
        hit.unsqueeze(-1) * F.logsigmoid(eta) + (1 - hit.unsqueeze(-1)) * F.logsigmoid(-eta)
    )
    total = torch.zeros(t, len(x), dtype=dtype).index_add(0, owner, ll)  # [T, Q]
    log_prior = _normal_log_pdf(eps, torch.zeros_like(eps), sigma.unsqueeze(-1))
    log_proposal = _normal_log_pdf(eps, m.unsqueeze(-1), s.unsqueeze(-1))
    return torch.logsumexp(log_w + total + log_prior - log_proposal, dim=-1)
