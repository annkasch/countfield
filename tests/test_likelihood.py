import numpy as np
import pytest
import torch
from scipy.integrate import trapezoid
from scipy.special import expit
from scipy.stats import binom, norm

from countfield.likelihood import central_interval, log_marginal, predictive_pmf


def _brute(k, n, mu, sigma):
    eta = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 200001)
    integrand = binom.pmf(k, n, expit(eta)) * norm.pdf(eta, mu, sigma)
    return np.log(trapezoid(integrand, eta))


@pytest.mark.parametrize("k", [0, 2, 10, 40])
@pytest.mark.parametrize("n", [1500, 6500])
@pytest.mark.parametrize("sigma", [0.01, 0.3, 2.0])
def test_log_marginal_matches_brute_force(k, n, sigma):
    mu = -6.5
    value = log_marginal(*(torch.tensor([v], dtype=torch.float64) for v in (k, n, mu, sigma)))
    assert value.item() == pytest.approx(_brute(k, n, mu, sigma), abs=1e-4)


def test_predictive_pmf_normalized_and_binomial_limit():
    pmf = predictive_pmf(-6.0, 1e-6, 2000)
    assert pmf.sum() == pytest.approx(1, abs=1e-8)
    np.testing.assert_allclose(pmf, binom.pmf(np.arange(len(pmf)), 2000, expit(-6.0)), atol=1e-6)
    wide = predictive_pmf(-6.0, 1.0, 2000)
    assert wide.sum() == pytest.approx(1, abs=1e-6)
    lo, hi = central_interval(wide, 0.954499736)
    assert 0 <= lo < hi < len(wide)
