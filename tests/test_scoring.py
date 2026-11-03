import numpy as np
import pytest
from scipy.special import expit

from countfield.data import CountSet
from countfield.scoring import predictive_summary, score

# Keys of RESUM_FLEX core.gp_metrics plus the binomial GP's count NLL and coverage.
GP_KEYS = {
    "voxels",
    "weighting",
    "response_units",
    "mean_observed",
    "mean_predicted",
    "mean_bias",
    "mae",
    "rmse",
    "pearson_r",
    "spearman_r",
    "calibration_slope",
    "calibration_intercept",
    "calibration_note",
    "crps",
    "interval_metrics",
    "weighted_interval_score",
    "predictive_nll",
    "predictive_distribution",
    "nll_measure",
    "coverage",
}


def _case(seed=0, q=40):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    mu, sigma = rng.normal(-6.5, 0.3, q), rng.uniform(0.05, 0.6, q)
    n = rng.choice([1500, 5000], q)
    k = rng.binomial(n, expit(mu + sigma * rng.normal(size=q)))
    return CountSet(rng.normal(size=(q, 3)), k, n), mu, sigma


def test_metric_keys_cover_the_gp_set():
    data, mu, sigma = _case()
    summary = predictive_summary(mu, sigma, data.n, 512, k=data.k)
    row = score(data, mu, sigma, summary)
    assert GP_KEYS <= set(row)
    assert set(row["interval_metrics"]) == {"1", "2", "3"}


def test_crps_matches_sample_estimate():
    data, mu, sigma = _case(q=5)
    summary = predictive_summary(mu, sigma, data.n, 2048, k=data.k)
    rng = np.random.default_rng(np.random.SeedSequence([1, 0]))
    for i in range(5):
        p = expit(rng.normal(mu[i], sigma[i], 200000))
        x = rng.binomial(data.n[i], p) / data.n[i]
        y = data.k[i] / data.n[i]
        a, b = x[:100000], x[100000:]
        sample_crps = np.abs(x - y).mean() - 0.5 * np.abs(a - b).mean()
        assert summary["crps"][i] == pytest.approx(sample_crps, rel=0.02, abs=1e-6)
