"""Training, checkpoint selection, prediction and the full run.

Model kinds: `local_latent` (here) and `point_transformer` (countfield.train_pt, masked
training on photon sets). For the local-latent model, training is leave-one-out: every
training point is predicted from its K nearest other
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
from countfield.resume import load_state, save_state
from countfield.scoring import grid_log_mass, predictive_summary, predictive_summary_grid, score


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
    train: CountSet, holdout: CountSet, config: RunConfig, log=print, resume: Path | None = None
) -> tuple[Predictor, list[dict]]:
    """Train on `train` (leave-one-out), select the step with the lowest holdout count NLL.

    ``resume``: checkpoint file written at every evaluation step; an existing one continues
    the run from its step.
    """
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
    streams = dict(batches=batches)
    state = load_state(resume, model, optimizer, streams)
    start = 1
    if state is not None:
        history, best, best_nll, start = (
            state["history"],
            state["best"],
            state["best_nll"],
            (state["step"] + 1),
        )
        log(f"resuming after step {state['step']}")
    for step in range(start, t.steps + 1):
        model.train()
        idx = torch.as_tensor(
            batches.choice(len(train), min(m.batch_size, len(train)), replace=False)
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
            save_state(
                resume,
                step,
                model,
                optimizer,
                streams,
                history=history,
                best=best,
                best_nll=best_nll,
            )
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
    """Train, score and plot one configured run into ``config.output_directory``.

    An existing directory is accepted only for the same config and an unfinished run: with a
    training checkpoint (``resume.pt``) the run continues from it, otherwise it restarts.
    """
    out = config.output_directory
    resume = out / "resume.pt"
    if out.exists():
        saved = out / "config.json"
        if not saved.exists() or saved.read_text() != config.model_dump_json(indent=2):
            raise FileExistsError(f"{out} exists and holds a different (or no) run config")
        if (out / "model.pt").exists():
            raise FileExistsError(f"{out} holds a finished run; choose a new output_directory")
        log(f"continuing unfinished run in {out}" + ("" if resume.exists() else " from step 0"))
    else:
        out.mkdir(parents=True)
    point_transformer = config.model.kind == "point_transformer"
    if point_transformer:
        from countfield.photons import load_photon_groups

        photon_groups, _ = load_photon_groups(config.data.path)
        groups = {g: s.counts() for g, s in photon_groups.items()}
    else:
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

    if point_transformer:
        from countfield.photons import PhotonSet, merge_photons_by_position
        from countfield.train_pt import fit_point_transformer

        photons, _ = merge_photons_by_position(
            PhotonSet.concatenate([photon_groups[g] for g in config.data.observations]), decimals
        )
        if not (np.array_equal(photons.k, observed.k) and np.array_equal(photons.n, observed.n)):
            raise RuntimeError("merged photon sets disagree with merged counts")
        history, checkpoint, evaluation_latents = fit_point_transformer(
            photons,
            train_idx,
            holdout_idx,
            [groups[g].x for g in config.data.evaluation],
            config,
            log,
            resume=resume,
        )
        latents = dict(zip(config.data.evaluation, evaluation_latents, strict=True))
    else:
        predictor, history = fit(train, holdout, config, log, resume=resume)
        checkpoint = dict(
            state=predictor.model.state_dict(),
            standardization=predictor.standardization.to_dict(),
        )
        latents = {g: predictor.latent(observed, groups[g].x) for g in config.data.evaluation}
    (out / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(dict(checkpoint, model=config.model.model_dump()), out / "model.pt")

    metrics = []
    for g in config.data.evaluation:
        data = groups[g]
        row, arrays = _evaluate(data, latents[g], config)
        row = dict(group=g, **row)
        metrics.append(row)
        log(f"{g}: NLL {row['predictive_nll']:.4f}, coverage {row['coverage']}")
        np.savez_compressed(
            out / f"evaluation_{g.replace('/', '_')}.npz",
            x=data.x,
            hits=data.k,
            target_events=data.n,
            observed=data.fraction,
            **arrays,
        )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    from countfield.plots import plot_run

    plot_run(out)
    resume.unlink(missing_ok=True)  # finished: the training checkpoint is no longer needed
    return out


def _evaluate(data: CountSet, prediction, config: RunConfig) -> tuple[dict, dict]:
    """Metrics and arrays of one evaluation set from (mu, sigma) or a probability grid."""
    if isinstance(prediction, tuple):
        mu, sigma = prediction
        summary = predictive_summary(mu, sigma, data.n, config.evaluation.latent_grid, k=data.k)
        row = score(data, mu, sigma, summary, config.model.quadrature_nodes)
        return row, dict(latent_mu=mu, latent_sigma=sigma, **summary)
    grid = np.asarray(prediction)  # [Q, G]
    summary = predictive_summary_grid(grid, data.n, k=data.k)
    row = score(data, None, None, summary, log_mass=grid_log_mass(grid, data.k, data.n))
    return row, dict(p_grid=grid, **summary)


def rescore(directory: str | Path) -> list[dict]:
    """Recompute metrics.json and plots of a finished run from its saved predictions."""
    from countfield.config import RunConfig
    from countfield.plots import plot_run

    directory = Path(directory)
    config = RunConfig.model_validate_json((directory / "config.json").read_text())
    metrics = []
    for g in config.data.evaluation:
        path = directory / f"evaluation_{g.replace('/', '_')}.npz"
        with np.load(path) as f:
            arrays = {key: f[key] for key in f.files}
        data = CountSet(arrays["x"], arrays["hits"], arrays["target_events"])
        prediction = (
            arrays["p_grid"]
            if "p_grid" in arrays
            else (arrays["latent_mu"], arrays["latent_sigma"])
        )
        row, new = _evaluate(data, prediction, config)
        metrics.append(dict(group=g, **row))
        np.savez_compressed(path, **{**arrays, **new})
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    plot_run(directory)
    return metrics
