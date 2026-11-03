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
    """Training loss and holdout count NLL; separate panels when they are different measures
    (photon or masked training loss vs holdout count NLL)."""
    rows = [h for h in history if "step" in h]
    selected = next((h["selected_step"] for h in history if "selected_step" in h), None)
    steps = [h["step"] for h in rows]
    same_measure = "train_nll" in rows[0]
    train_key = "train_nll" if same_measure else "train_loss"
    if same_measure:
        fig, ax = _axes()
        axes = [ax, ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, facecolor=SURFACE)
        for ax in axes:
            ax.set_facecolor(SURFACE)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
    labels = (("training (leave-one-out)" if same_measure else "training loss"), "holdout")
    for ax, key, label, color in zip(axes, (train_key, "holdout_nll"), labels, SERIES, strict=True):
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
        ax.margins(x=0.12)
    if selected is not None:
        axes[0].annotate(
            f"selected step {selected}",
            (selected, axes[0].get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            color=SECONDARY,
            fontsize=8,
        )
    if same_measure:
        axes[0].legend(frameon=False, labelcolor=SECONDARY, fontsize=8)
        _finish(fig, axes[0], path, "Count NLL per point during training", "step", "count NLL")
    else:
        axes[0].set_ylabel("training loss", color=SECONDARY)
        axes[1].set_ylabel("holdout count NLL", color=SECONDARY)
        _finish(fig, axes[1], path, "", "step", "holdout count NLL")


TARGET = {1: 0.6827, 2: 0.9545, 3: 0.9973}
# Same layout and colours as the RESUM_FLEX coverage and z-score plots (viz.dispatch,
# viz.zscore), reimplemented here so both models' plots compare one-to-one.
BAND_COLORS = {3: ("#d62728", 0.18), 2: ("#ffd700", 0.40), 1: ("#2ca02c", 0.50)}


def plot_coverage(
    arrays: dict, path: Path, label: str, ylim: tuple[float, float] | None = None
) -> dict[int, float]:
    """Voxels in input order with ±1/2/3σ count intervals and observed k/n; calibration bars."""
    observed = arrays["observed"]
    intervals = {k: (arrays[f"lower_{k}"], arrays[f"upper_{k}"]) for k in (1, 2, 3)}
    coverage = {
        k: float(((observed >= lo) & (observed <= hi)).mean()) for k, (lo, hi) in intervals.items()
    }
    x = np.arange(len(observed))
    fig, (ax_ts, ax_cal) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    for k in (3, 2, 1):
        color, alpha = BAND_COLORS[k]
        ax_ts.fill_between(
            x, *intervals[k], color=color, alpha=alpha, label=f"±{k}σ ({coverage[k] * 100:.0f}%)"
        )
    ax_ts.scatter(x, observed, c="black", s=14, alpha=0.85, label="Observed fraction", zorder=4)
    ax_ts.set_xlabel("Evaluation voxel index")
    ax_ts.set_ylabel("y")
    high = max(float(observed.max()), float(intervals[3][1].max()))
    ax_ts.set_ylim(*(ylim if ylim else (0, high * 1.05)))
    ax_ts.legend(loc="best", fontsize=9)
    ax_ts.grid(alpha=0.3)

    pos, w = np.arange(3), 0.38
    target = [TARGET[k] for k in (1, 2, 3)]
    actual = [coverage[k] for k in (1, 2, 3)]
    ax_cal.bar(
        pos - w / 2, target, w, color="lightgray", edgecolor="black", label="target (Gaussian)"
    )
    ax_cal.bar(pos + w / 2, actual, w, color="C0", edgecolor="black", label="actual")
    for p, t_, a_ in zip(pos, target, actual, strict=True):
        ax_cal.text(p - w / 2, t_ + 0.015, f"{t_ * 100:.1f}%", ha="center", fontsize=8)
        ax_cal.text(p + w / 2, a_ + 0.015, f"{a_ * 100:.1f}%", ha="center", fontsize=8)
    ax_cal.set_xticks(pos)
    ax_cal.set_xticklabels(["1σ", "2σ", "3σ"])
    ax_cal.set_ylim(0.0, 1.12)
    ax_cal.set_ylabel("coverage fraction")
    ax_cal.set_title("calibration")
    ax_cal.legend(loc="lower right", fontsize=8)
    ax_cal.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{label}: countfield observation-predictive coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return coverage


Z_LIMIT = 6.0  # fixed z axis range ±6 so plots of different runs compare directly


def plot_zscores(arrays: dict, path: Path, label: str, limit: float = Z_LIMIT) -> dict:
    """z histogram against N(0, 1), and z against the predicted rate with ±1/2/3σ bands.

    The z axis is fixed to ±limit; values beyond are drawn at the edge and counted in the
    legend. Writes PNG and PDF (``path`` without suffix)."""
    from scipy.stats import norm

    predicted = arrays["mean"]
    z = (arrays["observed"] - predicted) / arrays["sigma"]
    summary = dict(
        mean=float(z.mean()),
        std=float(z.std(ddof=1)) if len(z) > 1 else 0.0,
        outside_2=float(np.mean(np.abs(z) > 2)),
        voxels=len(z),
    )
    fig, (ax_hist, ax_rate) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    outside = int(np.sum(np.abs(z) > limit))
    shown = np.clip(z, -limit, limit)
    ax_hist.hist(
        shown,
        bins=np.linspace(-limit, limit, 41),
        density=True,
        color="C0",
        alpha=0.6,
        label=f"{len(z)} voxels" + (f" ({outside} beyond ±{limit:g}, at edge)" if outside else ""),
    )
    grid = np.linspace(-limit, limit, 400)
    ax_hist.plot(grid, norm.pdf(grid), "k--", label="N(0, 1)")
    ax_hist.set(
        xlabel="z = (observed − predicted) / σ",
        ylabel="Density",
        title=f"mean {summary['mean']:+.2f}, std {summary['std']:.2f}, "
        f"|z|>2: {summary['outside_2']:.0%} (4.6% expected)",
    )
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.2)
    for k, color, alpha in ((3, "#d62728", 0.15), (2, "#ffd700", 0.35), (1, "#2ca02c", 0.35)):
        inside = np.mean(np.abs(z) <= k)
        ax_rate.axhspan(
            -k,
            k,
            color=color,
            alpha=alpha,
            lw=0,
            label=f"±{k}σ: {inside:.0%} ({[68.3, 95.4, 99.7][k - 1]}%)",
        )
    ax_rate.scatter(predicted, shown, s=8, color="black", alpha=0.6, label="voxels", zorder=3)
    ax_rate.axhline(0, color="black", lw=1)
    ax_rate.set(
        xlabel="countfield predicted rate",
        ylabel="z [σ]",
        title="z vs predicted rate",
        ylim=(-limit, limit),
    )
    ax_rate.legend(fontsize=8)
    ax_rate.grid(alpha=0.2)
    fig.suptitle(f"{label}: z = (observed − predictive mean) / predictive sd")
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=150)
    plt.close(fig)
    return summary


def plot_run(directory: str | Path, coverage_ylim: tuple[float, float] | None = None) -> list[Path]:
    """All plots of a run; coverage_ylim None reads evaluation.coverage_ylim from config.json."""
    directory = Path(directory)
    written = []
    if coverage_ylim is None and (directory / "config.json").is_file():
        saved = json.loads((directory / "config.json").read_text())
        coverage_ylim = saved.get("evaluation", {}).get("coverage_ylim")
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
        written.append(directory / f"evaluation_{stem}_coverage.png")
        plot_coverage(arrays, written[-1], row["group"], coverage_ylim)
        plot_zscores(arrays, directory / f"evaluation_{stem}_zscore", row["group"])
        written += [directory / f"evaluation_{stem}_zscore.{s}" for s in ("png", "pdf")]
    return written
