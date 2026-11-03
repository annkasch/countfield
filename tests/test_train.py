import json

import numpy as np
import torch
import yaml

from countfield.config import load_config
from countfield.data import save_groups
from countfield.likelihood import log_marginal
from countfield.synthetic import sample
from countfield.train import run


def _config(tmp_path, steps):
    observed = sample(600, 20000, seed=1)
    # Duplicate a few positions with independent counts (an LF/HF pair) to exercise merging.
    twin = observed.subset(np.arange(20))
    save_groups(
        tmp_path / "counts.npz",
        {"obs/lf": observed, "obs/hf": twin, "val/hf": sample(150, 20000, seed=2)},
    )
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "path": "counts.npz",
                    "observations": ["obs/lf", "obs/hf"],
                    "evaluation": ["val/hf"],
                },
                "model": {"neighbours": 12, "hidden": 32, "heads": 2},
                "training": {
                    "steps": steps,
                    "batch_size": 128,
                    "learning_rate": 3e-3,
                    "eval_every": 50,
                },
                "evaluation": {"latent_grid": 256},
                "output_directory": "out",
            }
        )
    )
    return load_config(path)


def test_run_end_to_end_beats_constant_rate(tmp_path):
    torch.set_num_threads(1)
    out = run(_config(tmp_path, steps=300), log=lambda *_: None)
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["observed_points"] == 600 and provenance["merged_positions"] == 20
    (row,) = json.loads((out / "metrics.json").read_text())
    assert set(row["coverage"]) == {"1", "2", "3"}
    with np.load(out / "evaluation_val_hf.npz") as f:
        k, n = f["hits"], f["target_events"]
        assert np.all(f["latent_sigma"] > 0)
    rate = k.sum() / n.sum()  # constant baseline even gets the evaluation rate
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
    assert row["predictive_nll"] < baseline


def test_evaluation_sharing_observed_position_rejected(tmp_path):
    import pytest

    config = _config(tmp_path, steps=1)
    from countfield.data import load_groups

    groups = load_groups(config.data.path)
    groups["val/hf"] = groups["obs/lf"].subset(np.arange(3))
    save_groups(config.data.path, groups)
    with pytest.raises(ValueError, match="share a position"):
        run(config, log=lambda *_: None)
