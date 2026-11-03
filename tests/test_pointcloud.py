import json

import numpy as np
import pytest
import torch
import yaml

from countfield.config import RunConfig, load_config
from countfield.photons import (
    PhotonSet,
    load_photon_groups,
    merge_photons_by_position,
    save_photon_groups,
)
from countfield.pointcloud import Geometry, PhotonPointTransformer, farthest_point_sample
from countfield.synthetic import sample_photons
from countfield.train_pt import PhotonCloud


def _pt_config(**model):
    return RunConfig.model_validate(
        {
            "data": {"path": "x.npz", "observations": ["a"]},
            "model": {
                "kind": "point_transformer",
                "dims": [16, 32],
                "neighbours": 8,
                "photon_hidden": 16,
                **model,
            },
            "output_directory": "out",
        }
    )


def test_photon_set_merge_subset_roundtrip(tmp_path):
    data = sample_photons(20, 50, seed=0)
    twin = data.subset(np.arange(3))
    merged, multiplicity = merge_photons_by_position(PhotonSet.concatenate([data, twin]), 6)
    assert len(merged) == 20 and multiplicity[:3].tolist() == [2, 2, 2]
    np.testing.assert_array_equal(merged.n[:3], 100)
    np.testing.assert_array_equal(merged.k[:3], 2 * data.k[:3])
    save_photon_groups(tmp_path / "p.npz", {"a": data, "b": twin})
    groups, names = load_photon_groups(tmp_path / "p.npz")
    assert list(groups) == ["a", "b"] and len(names) == 6
    np.testing.assert_array_equal(groups["b"].hit, twin.hit)


def test_geometry_levels():
    z = np.random.default_rng(np.random.SeedSequence([0, 0])).normal(size=(200, 3))
    assert len(set(farthest_point_sample(z, 50))) == 50
    geometry = Geometry(z, levels=3, neighbours=8, ratio=4)
    sizes = [len(level.z) for level in geometry.levels]
    assert sizes == [200, 50, 12]
    coarse = geometry.levels[1]
    assert coarse.down.shape == (50, 8) and coarse.up.shape == (200, 3)
    torch.testing.assert_close(coarse.up_weight.sum(1), torch.ones(200))


def test_masked_photons_never_reach_any_output():
    config = _pt_config()
    data = sample_photons(80, 100, seed=1)
    train_idx = np.arange(80)
    cloud = PhotonCloud(data, train_idx, [], config.model)
    torch.manual_seed(0)
    model = PhotonPointTransformer(6, 3, [16, 32], 1, 16).eval()
    visible = np.arange(10, 80)  # points 0..9 masked
    rng = lambda: np.random.default_rng(0)  # noqa: E731
    with torch.no_grad():
        mu, sigma, _ = cloud.forward(model, None, visible, None, rng())
        cloud.features[cloud.owner < 10] += 5.0  # change masked points' photons
        cloud.hit[cloud.owner < 10] = 1 - cloud.hit[cloud.owner < 10]
        mu2, sigma2, _ = cloud.forward(model, None, visible, None, rng())
        torch.testing.assert_close(mu, mu2)
        torch.testing.assert_close(sigma, sigma2)
        cloud.features[cloud.owner == 20] += 5.0  # a visible point does matter
        mu3, _, _ = cloud.forward(model, None, visible, None, rng())
    assert not torch.allclose(mu, mu3)


def test_miss_subsampling_is_unbiased():
    config = _pt_config()
    data = sample_photons(5, 400, seed=2)
    cloud = PhotonCloud(data, np.arange(5), [], config.model)
    visible = np.arange(5)

    def sums(args):
        features, hit, weight, owner, *_ = args
        return torch.zeros(5, features.shape[1]).index_add(0, owner, features * weight[:, None])

    full = sums(cloud.encoder_args(visible, None, np.random.default_rng(0)))
    rng = np.random.default_rng(np.random.SeedSequence([3, 0]))
    draws = torch.stack([sums(cloud.encoder_args(visible, 32, rng)) for _ in range(1000)])
    standard_error = draws.std(dim=0) / np.sqrt(len(draws))
    z = (draws.mean(dim=0) - full) / standard_error
    assert z.abs().max() < 4.5  # 30 sums; unbiased estimator -> |z| of a few at most


def test_point_transformer_run_end_to_end(tmp_path):
    from countfield.likelihood import log_marginal
    from countfield.train import run

    torch.set_num_threads(1)
    observed = sample_photons(300, 1500, seed=4)
    save_photon_groups(
        tmp_path / "photons.npz", {"obs": observed, "val": sample_photons(60, 1500, seed=5)}
    )
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "photons.npz", "observations": ["obs"], "evaluation": ["val"]},
                "model": {
                    "kind": "point_transformer",
                    "dims": [16, 32],
                    "neighbours": 8,
                    "photon_hidden": 16,
                    "misses_per_point": 64,
                    "mask_fraction": 0.2,
                },
                "training": {"steps": 120, "learning_rate": 3e-3, "eval_every": 40},
                "evaluation": {"latent_grid": 256},
                "output_directory": "out",
            }
        )
    )
    out = run(load_config(path), log=lambda *_: None)
    (row,) = json.loads((out / "metrics.json").read_text())
    with np.load(out / "evaluation_val.npz") as f:
        k, n = f["hits"], f["target_events"]
    rate = k.sum() / n.sum()
    baseline = (
        -log_marginal(
            torch.tensor(k, dtype=torch.float64),
            torch.tensor(n, dtype=torch.float64),
            torch.full((len(k),), float(np.log(rate / (1 - rate))), dtype=torch.float64),
            torch.full((len(k),), 1e-4, dtype=torch.float64),
        )
        .mean()
        .item()
    )
    assert row["predictive_nll"] == pytest.approx(row["predictive_nll"])  # finite
    assert row["predictive_nll"] < baseline + 0.05
    assert (out / "evaluation_val_coverage.png").stat().st_size > 0


def test_photon_log_marginal_matches_numerical_integral():
    from scipy.integrate import trapezoid
    from scipy.special import expit
    from scipy.stats import norm

    from countfield.likelihood import photon_log_marginal

    rng = np.random.default_rng(np.random.SeedSequence([5, 0]))
    owner = np.repeat([0, 1], [300, 500])
    base = rng.normal(-5.5, 1.0, len(owner))
    hit = (rng.random(len(owner)) < expit(base + 0.4)).astype(float)
    weight = rng.choice([1.0, 2.0], len(owner))
    sigma = np.array([0.3, 1.5])
    value = photon_log_marginal(
        *(torch.as_tensor(a, dtype=torch.float64) for a in (base, hit, weight)),
        torch.as_tensor(owner),
        torch.as_tensor(sigma, dtype=torch.float64),
    )
    for t in range(2):
        eps = np.linspace(-10 * sigma[t], 10 * sigma[t], 40001)
        sel = owner == t
        eta = base[sel][:, None] + eps
        ll = (
            weight[sel][:, None]
            * (
                hit[sel][:, None] * np.log(expit(eta))
                + (1 - hit[sel][:, None]) * np.log(expit(-eta))
            )
        ).sum(0)
        brute = np.log(trapezoid(np.exp(ll) * norm.pdf(eps, 0, sigma[t]), eps))
        assert value[t].item() == pytest.approx(brute, abs=1e-4)


@pytest.mark.parametrize("features", ["photons", "counts"])
def test_photon_head_and_thinning_keep_masked_points_hidden(features):
    config = _pt_config(loss="photon", point_features=features, keep_fraction_min=0.6, dropout=0.1)
    data = sample_photons(60, 120, seed=6)
    cloud = PhotonCloud(data, np.arange(60), [], config.model)
    torch.manual_seed(0)
    model = PhotonPointTransformer(
        6, 3, [16, 32], 1, 16, point_features=features, photon_head=True, dropout=0.1
    ).eval()
    keep, visible = np.arange(5, 60), np.arange(15, 60)  # 5..14 masked targets
    with torch.no_grad():
        out = cloud.forward(model, keep, visible, None, np.random.default_rng(0))
        cloud.features[cloud.owner < 15] += 3.0
        cloud.count_features[:15] += 3.0
        out2 = cloud.forward(model, keep, visible, None, np.random.default_rng(0))
    for a, b in zip(out, out2, strict=True):
        torch.testing.assert_close(a, b)
    grid = cloud.probability_grid(model, *out, np.arange(5, 10), 64, 32, np.random.default_rng(1))
    assert grid.shape == (5, 64) and ((grid > 0) & (grid < 1)).all()


@pytest.mark.parametrize(
    "model_extra",
    [
        {"loss": "photon"},
        {"point_features": "counts", "keep_fraction_min": 0.7, "mask_fraction_max": 0.3},
    ],
)
def test_regularized_variants_run_end_to_end(tmp_path, model_extra):
    from countfield.train import rescore, run

    torch.set_num_threads(1)
    save_photon_groups(
        tmp_path / "photons.npz",
        {"obs": sample_photons(150, 800, seed=8), "val": sample_photons(30, 800, seed=9)},
    )
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "photons.npz", "observations": ["obs"], "evaluation": ["val"]},
                "model": {
                    "kind": "point_transformer",
                    "dims": [16, 32],
                    "neighbours": 8,
                    "photon_hidden": 16,
                    "misses_per_point": 32,
                    "eval_misses_per_point": 64,
                    "prediction_photons": 64,
                    "dropout": 0.1,
                    **model_extra,
                },
                "training": {"steps": 6, "eval_every": 3, "weight_decay": 1e-4},
                "evaluation": {"latent_grid": 64},
                "output_directory": "out",
            }
        )
    )
    out = run(load_config(path), log=lambda *_: None)
    (row,) = json.loads((out / "metrics.json").read_text())
    assert np.isfinite(row["predictive_nll"]) and np.isfinite(row["crps"])
    (again,) = rescore(out)
    assert again["predictive_nll"] == pytest.approx(row["predictive_nll"])
    assert (out / "history.png").stat().st_size > 0
