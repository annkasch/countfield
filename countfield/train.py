"""Training, checkpoint selection, prediction and the full run.

Training is leave-one-out: every training point is predicted from its K nearest other
training points, so no target sees its own counts and the context density matches
prediction at a new position. Checkpoints are selected on a held-out share of the observed
points (predicted from the training points only); evaluation groups are never used for
selection. Final predictions use all observed points as context.

Output directory::

    config.json, provenance.json   resolved config, git commit, data file and merge summary
    model.pt                       selected weights + standardization
    history.json                   training loss and holdout NLL per evaluation step
    evaluation_<group>.npz         per evaluation group: counts, latent mu/sigma, predictive
                                   mean/sd and central intervals of k/n
    metrics.json                   one row per evaluation group
    *.png                          history, coverage and z-score plots (countfield.plots)
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from countfield.config import RunConfig
from countfield.data import (
    CountSet,
    load_groups,
    merge_by_position,
    shared_positions,
)
from countfield.likelihood import log_marginal
from countfield.model import LocalLatentNP, NeighbourInputs, Standardization
from countfield.scoring import predictive_summary, score


def _stream(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, k]))


def split_holdout(points: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint training and holdout indices (stream 0)."""
    order = _stream(seed, 0).permutation(points)
    n_holdout = max(1, int(round(fraction * points)))
    return np.sort(order[n_holdout:]), np.sort(order[:n_holdout])


class Predictor:
    """A trained model with its standardization and a context pool."""

    def __init__(self, model: LocalLatentNP, standardization: Standardization, neighbours: int):
        self.model = model
        self.standardization = standardization
        self.neighbours = neighbours

    @torch.no_grad()
    def latent(
        self, pool: CountSet, x: np.ndarray, exclude: np.ndarray | None = None, chunk: int = 4096
    ) -> tuple[np.ndarray, np.ndarray]:  # mu [Q], sigma [Q] (absolute logit)
        self.model.eval()
        inputs = NeighbourInputs(pool, self.standardization, self.neighbours)
        mus, sigmas = [], []
        for start in range(0, len(x), chunk):
            ex = None if exclude is None else exclude[start : start + chunk]
            mu, sigma = self.model(*inputs.build(x[start : start + chunk], ex))
            mus.append(mu.double().numpy() + self.standardization.logit_rate)
            sigmas.append(sigma.double().numpy())
        return np.concatenate(mus), np.concatenate(sigmas)


def fit(
    train: CountSet, holdout: CountSet, config: RunConfig, log=print
) -> tuple[Predictor, list[dict]]:
    """Train on `train` (leave-one-out), select the step with the lowest holdout count NLL."""
    m, t = config.model, config.training
    standardization = Standardization.fit(train)
    torch.manual_seed(int(_stream(t.seed, 1).integers(2**31)))
    model = LocalLatentNP(train.dim, hidden=m.hidden, heads=m.heads)
    predictor = Predictor(model, standardization, m.neighbours)

    inputs = NeighbourInputs(train, standardization, m.neighbours)
    query, offset, counts = inputs.build(train.x, exclude=np.arange(len(train)))
    k = torch.as_tensor(train.k, dtype=torch.float32)
    n = torch.as_tensor(train.n, dtype=torch.float32)
    offset_rate = standardization.logit_rate

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay
    )
    batches = _stream(t.seed, 2)
    history, best, best_nll = [], None, np.inf
    running = []
    for step in range(1, t.steps + 1):
        model.train()
        idx = torch.as_tensor(
            batches.choice(len(train), min(t.batch_size, len(train)), replace=False)
        )
        mu, sigma = model(query[idx], offset[idx], counts[idx])
        loss = -log_marginal(k[idx], n[idx], mu + offset_rate, sigma, m.quadrature_nodes).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running.append(loss.item())
        if step % t.eval_every == 0 or step == t.steps:
            h_mu, h_sigma = predictor.latent(train, holdout.x)
            with torch.no_grad():
                nll = (
                    -log_marginal(
                        torch.as_tensor(holdout.k, dtype=torch.float64),
                        torch.as_tensor(holdout.n, dtype=torch.float64),
                        torch.as_tensor(h_mu),
                        torch.as_tensor(h_sigma),
                        m.quadrature_nodes,
                    )
                    .mean()
                    .item()
                )
            history.append(
                dict(
                    step=step,
                    train_nll=float(np.mean(running)),
                    holdout_nll=nll,
                    holdout_sigma_median=float(np.median(h_sigma)),
                )
            )
            running = []
            if nll < best_nll:
                best_nll, best = nll, dict(step=step, state=copy.deepcopy(model.state_dict()))
            log(f"step {step}: train {history[-1]['train_nll']:.4f} holdout {nll:.4f}")
    model.load_state_dict(best["state"])
    history.append(dict(selected_step=best["step"], selected_holdout_nll=best_nll))
    return predictor, history


def _git_commit() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)
    return dict(commit=commit, dirty=dirty)


def run(config: RunConfig, log=print) -> Path:
    out = config.output_directory
    out.mkdir(parents=True, exist_ok=False)
    groups = load_groups(config.data.path)
    missing = set(config.data.observations + config.data.evaluation) - set(groups)
    if missing:
        raise KeyError(f"groups not in {config.data.path}: {sorted(missing)}; have {list(groups)}")
    decimals = config.data.merge_decimals
    raw = CountSet.concatenate([groups[g] for g in config.data.observations])
    observed, multiplicity = merge_by_position(raw, decimals)
    for g in config.data.evaluation:
        overlap = shared_positions(observed, groups[g], decimals)
        if overlap:
            raise ValueError(
                f"{overlap} points of evaluation group {g!r} share a position with observed points"
            )
    train_idx, holdout_idx = split_holdout(
        len(observed), config.training.holdout_fraction, config.training.seed
    )
    train, holdout = observed.subset(train_idx), observed.subset(holdout_idx)

    provenance = dict(
        git=_git_commit(),
        data_file=str(config.data.path),
        data_sha256=hashlib.sha256(Path(config.data.path).read_bytes()).hexdigest(),
        groups={
            g: dict(points=len(s), k=int(s.k.sum()), n=int(s.n.sum())) for g, s in groups.items()
        },
        observed_points_before_merge=len(raw),
        observed_points=len(observed),
        merged_positions=int(np.sum(multiplicity > 1)),
        training_points=len(train),
        holdout_points=len(holdout),
    )
    (out / "config.json").write_text(config.model_dump_json(indent=2))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log(
        f"observed {len(raw)} points -> {len(observed)} after merging "
        f"({provenance['merged_positions']} merged positions); "
        f"train {len(train)}, holdout {len(holdout)}"
    )

    predictor, history = fit(train, holdout, config, log)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(
        dict(
            state=predictor.model.state_dict(),
            standardization=predictor.standardization.to_dict(),
            model=config.model.model_dump(),
        ),
        out / "model.pt",
    )

    metrics = []
    for g in config.data.evaluation:
        data = groups[g]
        mu, sigma = predictor.latent(observed, data.x)
        summary = predictive_summary(mu, sigma, data.n, config.evaluation.latent_grid)
        row = dict(group=g, **score(data, mu, sigma, summary, config.model.quadrature_nodes))
        metrics.append(row)
        log(f"{g}: NLL {row['predictive_nll']:.4f}, coverage {row['coverage']}")
        np.savez_compressed(
            out / f"evaluation_{g.replace('/', '_')}.npz",
            x=data.x,
            hits=data.k,
            target_events=data.n,
            observed=data.fraction,
            latent_mu=mu,
            latent_sigma=sigma,
            **summary,
        )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    from countfield.plots import plot_run

    plot_run(out)
    return out
