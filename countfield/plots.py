"""Run plots: training history, predictive coverage and z-scores per evaluation group.

python -m countfield plot RUN_DIRECTORY
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1, 2
BANDS = {3: "#b7d3f6", 2: "#86b6ef", 1: "#5598e7"}  # blue ramp 150 / 250 / 350
MEAN = "#104281"  # blue 650
EXPECTED = {1: 68.3, 2: 95.4, 3: 99.7}


def _axes(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=SECONDARY)
    ax.set_ylabel(ylabel, color=SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_history(history: list[dict], path: Path) -> None:
    rows = [h for h in history if "step" in h]
    selected = next((h["selected_step"] for h in history if "selected_step" in h), None)
    steps = [h["step"] for h in rows]
    fig, ax = _axes()
    for (key, label), color in zip(
        (("train_nll", "training (leave-one-out)"), ("holdout_nll", "holdout")),
        SERIES,
        strict=True,
    ):
        values = [h[key] for h in rows]
        ax.plot(steps, values, color=color, linewidth=2, label=label)
        ax.annotate(
            label,
            (steps[-1], values[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=SECONDARY,
            va="center",
            fontsize=8,
        )
    if selected is not None:
        ax.axvline(selected, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"selected step {selected}",
            (selected, ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            color=SECONDARY,
            fontsize=8,
        )
    ax.legend(frameon=False, labelcolor=SECONDARY, fontsize=8)
    ax.margins(x=0.12)
    _finish(fig, ax, path, "Count NLL per point during training", "step", "count NLL")


def plot_coverage(arrays: dict, coverage: dict, path: Path, label: str) -> None:
    order = np.argsort(arrays["mean"])
    index = np.arange(len(order))
    fig, ax = _axes((8, 4.5))
    for level in (3, 2, 1):
        ax.fill_between(
            index,
            arrays[f"lower_{level}"][order],
            arrays[f"upper_{level}"][order],
            step="mid",
            color=BANDS[level],
            linewidth=0,
            label=f"{level}σ count interval: {100 * coverage[str(level)]:.1f} % "
            f"(expected {EXPECTED[level]} %)",
        )
    ax.plot(index, arrays["mean"][order], color=MEAN, linewidth=2, label="predictive mean")
    ax.scatter(
        index,
        arrays["observed"][order],
        s=9,
        color=INK,
        linewidths=0,
        zorder=3,
        label="observed fraction k/n",
    )
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=SECONDARY, fontsize=8, loc="upper left")
    _finish(
        fig,
        ax,
        path,
        f"{label}: predictive count intervals (as fractions)",
        "evaluation point, sorted by predictive mean",
        "hit fraction",
    )


def plot_zscores(arrays: dict, path: Path, label: str) -> None:
    z = (arrays["observed"] - arrays["mean"]) / arrays["sigma"]
    fig, ax = _axes()
    edges = np.linspace(-4, 4, 33)
    ax.hist(
        np.clip(z, -4, 4),
        bins=edges,
        density=True,
        color=SERIES[0],
        edgecolor=SURFACE,
        linewidth=1,
        label=f"z (mean {z.mean():.2f}, sd {z.std():.2f})",
    )
    grid = np.linspace(-4, 4, 400)
    ax.plot(
        grid,
        np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
        color=INK,
        linewidth=1.5,
        label="standard normal",
    )
    ax.legend(frameon=False, labelcolor=SECONDARY, fontsize=8)
    _finish(
        fig,
        ax,
        path,
        f"{label}: z = (observed − predictive mean) / predictive sd",
        "z (clipped to ±4)",
        "density",
    )


def plot_run(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    written = []
    history = directory / "history.json"
    if history.is_file():
        written.append(directory / "history.png")
        plot_history(json.loads(history.read_text()), written[-1])
    metrics_file = directory / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else []
    for row in metrics:
        stem = row["group"].replace("/", "_")
        with np.load(directory / f"evaluation_{stem}.npz") as f:
            arrays = {key: f[key] for key in f.files}
        written.append(directory / f"coverage_{stem}.png")
        plot_coverage(arrays, row["coverage"], written[-1], row["group"])
        written.append(directory / f"zscore_{stem}.png")
        plot_zscores(arrays, written[-1], row["group"])
    return written
