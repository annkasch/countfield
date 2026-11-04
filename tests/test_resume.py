"""An interrupted run resumes from its last evaluation step to the same result."""

import json
import shutil

import pytest
import torch
import yaml

from countfield.config import load_config
from countfield.data import save_groups
from countfield.photons import save_photon_groups
from countfield.synthetic import sample, sample_photons
from countfield.train import run


class Interrupt(Exception):
    pass


def _stop_at(step):
    def log(message):
        if str(message).startswith(f"step {step}:"):
            raise Interrupt  # like a cancel: the checkpoint of this step is never written

    return log


def _config(tmp_path, model):
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "data.npz", "observations": ["obs"], "evaluation": ["val"]},
                "model": model,
                "training": {"steps": 8, "eval_every": 2},
                "evaluation": {"latent_grid": 64},
                "output_directory": "out",
            }
        )
    )
    return load_config(path)


@pytest.mark.parametrize("kind", ["local_latent", "point_transformer"])
def test_resumed_run_equals_uninterrupted(tmp_path, kind):
    torch.set_num_threads(1)
    if kind == "local_latent":
        model = {"neighbours": 8, "hidden": 16, "heads": 2, "batch_size": 32}
        save_groups(
            tmp_path / "data.npz",
            {"obs": sample(120, 2000, seed=1), "val": sample(30, 2000, seed=2)},
        )
    else:
        model = {
            "kind": "point_transformer",
            "dims": [16, 32],
            "neighbours": 8,
            "photon_hidden": 16,
            "misses_per_point": 32,
            "dropout": 0.1,
            "keep_fraction_min": 0.7,
            "mask_fraction_max": 0.3,
        }
        save_photon_groups(
            tmp_path / "data.npz",
            {"obs": sample_photons(80, 300, seed=3), "val": sample_photons(20, 300, seed=4)},
        )
    config = _config(tmp_path, model)
    reference = run(config, log=lambda *_: None)
    expected = json.loads((reference / "history.json").read_text())
    expected_metrics = json.loads((reference / "metrics.json").read_text())
    assert not (reference / "resume.pt").exists()
    shutil.move(reference, tmp_path / "reference")

    with pytest.raises(Interrupt):
        run(config, log=_stop_at(6))
    assert (config.output_directory / "resume.pt").exists()
    messages = []
    out = run(config, log=messages.append)
    assert "resuming after step 4" in messages
    assert json.loads((out / "history.json").read_text()) == expected
    assert json.loads((out / "metrics.json").read_text()) == expected_metrics
    with pytest.raises(FileExistsError, match="finished"):
        run(config, log=lambda *_: None)


def test_existing_directory_with_other_config_is_refused(tmp_path):
    save_groups(
        tmp_path / "data.npz", {"obs": sample(60, 2000, seed=1), "val": sample(20, 2000, seed=2)}
    )
    config = _config(tmp_path, {"neighbours": 8, "hidden": 16, "heads": 2})
    config.output_directory.mkdir()
    (config.output_directory / "config.json").write_text("{}")
    with pytest.raises(FileExistsError, match="different"):
        run(config, log=lambda *_: None)
