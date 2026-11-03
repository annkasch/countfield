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
        mu, sigma = cloud.forward(model, visible, None, rng())
        cloud.features[cloud.owner < 10] += 5.0  # change masked points' photons
        cloud.hit[cloud.owner < 10] = 1 - cloud.hit[cloud.owner < 10]
        mu2, sigma2 = cloud.forward(model, visible, None, rng())
        torch.testing.assert_close(mu, mu2)
        torch.testing.assert_close(sigma, sigma2)
        cloud.features[cloud.owner == 20] += 5.0  # a visible point does matter
        mu3, _ = cloud.forward(model, visible, None, rng())
    assert not torch.allclose(mu, mu3)


def test_miss_subsampling_is_unbiased():
    config = _pt_config()
    data = sample_photons(5, 400, seed=2)
    cloud = PhotonCloud(data, np.arange(5), [], config.model)
    visible = np.arange(5)

    def sums(args):
        features, hit, weight, owner, *_ = args
        return torch.zeros(5, features.shape[1]).index_add(0, owner, features * weight[:, None])

    full = sums(cloud.photon_args(visible, None, np.random.default_rng(0)))
    rng = np.random.default_rng(np.random.SeedSequence([3, 0]))
    draws = torch.stack([sums(cloud.photon_args(visible, 32, rng)) for _ in range(1000)])
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
