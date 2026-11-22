"""Matplotlib figures. Imported lazily by the CLIs so training itself needs only NumPy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {"train": "#2a78d6", "validation": "#eb6834"}  # fixed colour per series, in every figure


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_learning_curves(history: list, path: Path, title: str) -> None:
    """Loss and accuracy per epoch, one panel each (they do not share a scale).

    Loss is on a log axis because it spans three orders of magnitude. Accuracy
    is drawn from epoch 1: the untrained epoch-0 value (chance level) would
    flatten everything else, so it is reported as text instead.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    panels = (
        (ax_loss, "loss", history, 1.0, "Cross-entropy loss", "loss (nats, log scale)", "{:.3f}"),
        (ax_acc, "acc", history[1:], 100.0, "Accuracy", "accuracy (%)", "{:.2f}%"),
    )
    for ax, key, records, scale, panel_title, ylabel, fmt in panels:
        epochs = [h["epoch"] for h in records]
        marker = {"marker": "o", "markersize": 3.5} if len(epochs) <= 30 else {}
        finals = {}
        for series, prefix in (("train", "train"), ("validation", "val")):
            values = [scale * h[f"{prefix}_{key}"] for h in records]
            ax.plot(epochs, values, color=SERIES[series], linewidth=2, label=series, **marker)
            finals[series] = values[-1]
        _style(ax, panel_title, ylabel)
        # Label the last point of each curve: the higher curve's label goes above, the lower one's below.
        upper = max(finals, key=finals.get)
        for series, value in finals.items():
            ax.annotate(
                fmt.format(value),
                xy=(epochs[-1], value),
                xytext=(6, 6 if series == upper else -6),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=TEXT,
            )
        ax.margins(x=0.1)
        ax.xaxis.get_major_locator().set_params(integer=True)
    # One legend serves both panels (same colours); the loss panel has free space top-right.
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    ax_loss.set_yscale("log")
    ax_loss.grid(axis="y", which="minor", visible=False)
    ax_loss.tick_params(which="minor", length=0)
    ax_acc.text(
        0.98, 0.05, f"epoch 0 (untrained): {100 * history[0]['val_acc']:.1f}% validation accuracy",
        transform=ax_acc.transAxes, ha="right", fontsize=8, color=TEXT_SECONDARY,
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


INK = "#0b0b0b"
NEUTRAL = "#8a8985"
LAYER_COLORS = ["#1baf7a", "#e87ba4", "#6250d6"]
# One colour per criterion, the same in every figure.
CRITERION_COLORS = {
    "magnitude": "#2a78d6",
    "fisher": "#eb6834",
    "taylor1": "#1baf7a",
    "movement": "#eda100",
    "taylor2": "#e87ba4",
    "random": NEUTRAL,
}


def plot_pruning_run(result: dict, path: Path, title: str) -> None:
    """Accuracy, sparsity and the loss around every mask update, on a shared epoch axis."""
    history, events = result["history"], result["pruning"]["events"]
    per_epoch = result["steps_per_epoch"]
    last_epoch = history[-1]["epoch"]
    start, end = events[0]["step"] / per_epoch, events[-1]["step"] / per_epoch

    fig, (ax_acc, ax_sparsity, ax_loss) = plt.subplots(3, 1, figsize=(9, 8.4), sharex=True, facecolor=SURFACE)

    epochs = [h["epoch"] for h in history[1:]]
    for series, prefix in (("train", "train"), ("validation", "val")):
        values = [100 * h[f"{prefix}_acc"] for h in history[1:]]
        ax_acc.plot(epochs, values, color=SERIES[series], linewidth=2, marker="o", markersize=3.5,
                    label=f"{series} · final {values[-1]:.2f}%")
    _style(ax_acc, "Accuracy", "accuracy (%)")
    ax_acc.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")

    steps = [e["step"] / per_epoch for e in events] + [last_epoch]
    overall = [100 * e["sparsity"] for e in events]
    ax_sparsity.plot(steps, overall + overall[-1:], color=INK, linewidth=2, drawstyle="steps-post",
                     label=f"all weights · final {overall[-1]:.2f}%")
    shapes = result["model"]["layer_shapes"]
    for index, (shape, color) in enumerate(zip(shapes, LAYER_COLORS)):
        values = [100 * e["layer_sparsity"][index] for e in events]
        ax_sparsity.plot(steps, values + values[-1:], color=color, linewidth=1.6, drawstyle="steps-post",
                         label=f"layer {index + 1} ({shape[0]}×{shape[1]}) · final {values[-1]:.1f}%")
    _style(ax_sparsity, "Sparsity (fraction of connections removed)", "sparsity (%)")
    ax_sparsity.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")

    xs, ys = [], []
    for e in events:  # (before, after) at the same x: the vertical jump is the update's immediate damage
        xs += [e["step"] / per_epoch] * 2
        ys += [e["probe_loss_before"], e["probe_loss_after"]]
    ax_loss.plot(xs, ys, color=TEXT_SECONDARY, linewidth=1.3)
    ax_loss.set_yscale("log")
    _style(ax_loss, "Probe loss just before and after each mask update (jump = immediate damage)",
           "loss (nats, log scale)")
    ax_loss.grid(axis="y", which="minor", visible=False)
    ax_loss.tick_params(which="minor", length=0)
    ax_loss.yaxis.set_minor_formatter(NullFormatter())

    for ax in (ax_acc, ax_sparsity, ax_loss):
        for boundary in (start, end):
            ax.axvline(boundary, color=GRID, linewidth=1.2, zorder=0)
        if ax is not ax_loss:
            ax.set_xlabel("")
    phases = (((1 + start) / 2, "dense"), ((start + end) / 2, "pruning"), ((end + last_epoch) / 2, "fine-tuning, mask frozen"))
    for x, text in phases:  # x in data units, y as a fraction of the panel height
        ax_loss.text(x, 0.9, text, transform=ax_loss.get_xaxis_transform(), ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax_loss.xaxis.get_major_locator().set_params(integer=True)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_criterion_check(result: dict, path: Path) -> None:
    """Predicted vs measured loss change for single connections, and the damage of removing many at once."""
    import numpy as np

    samples = result["samples"]
    measured = np.array(samples["true_loss_change"])
    taylor2, fisher, first_order = (np.array(samples[f"score_{k}"]) for k in ("taylor2", "fisher", "movement"))
    single = result["single_connection"]

    fig, (ax_full, ax_curv, ax_group) = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)

    ax_full.scatter(measured, taylor2, s=7, color=CRITERION_COLORS["taylor2"], alpha=0.6, linewidths=0)
    limit = float(np.abs(np.concatenate([measured, taylor2])).max()) * 1.5
    ax_full.plot([-limit, limit], [-limit, limit], color=NEUTRAL, linewidth=1, linestyle="--", zorder=0)
    ticks = [-1e-2, -1e-4, -1e-6, 0, 1e-6, 1e-4, 1e-2]
    for set_scale, set_ticks in ((ax_full.set_xscale, ax_full.set_xticks), (ax_full.set_yscale, ax_full.set_yticks)):
        set_scale("symlog", linthresh=1e-7)
        set_ticks(ticks)
    _style(ax_full, "One connection removed: full Taylor estimate", "predicted  −g·w + ½F·w²  (nats)")
    ax_full.set_xlabel("measured loss change (nats)", color=TEXT_SECONDARY, fontsize=9)
    ax_full.text(0.04, 0.93, f"rank correlation {single['spearman']['taylor2']['all_layers']:.3f}\n"
                 f"{100 * single['fraction_true_change_negative']:.0f}% of removals lower the loss",
                 transform=ax_full.transAxes, va="top", fontsize=8, color=TEXT)

    curvature = measured - first_order
    usable = (curvature > 0) & (fisher > 0)
    ax_curv.scatter(curvature[usable], fisher[usable], s=7, color=CRITERION_COLORS["fisher"], alpha=0.6, linewidths=0)
    low, high = curvature[usable].min() / 2, curvature[usable].max() * 2
    ax_curv.plot([low, high], [low, high], color=NEUTRAL, linewidth=1, linestyle="--", zorder=0)
    ax_curv.set_xscale("log")
    ax_curv.set_yscale("log")
    _style(ax_curv, "Curvature term alone: Fisher estimate", "predicted  ½F·w²  (nats)")
    ax_curv.set_xlabel("measured change minus first-order term (nats)", color=TEXT_SECONDARY, fontsize=9)
    c = single["curvature_term"]
    ax_curv.text(0.04, 0.93, f"rank correlation {c['spearman_fisher']:.2f}  (magnitude: {c['spearman_magnitude']:.2f})\n"
                 f"median predicted/measured {c['median_fisher_over_measured']:.2f}",
                 transform=ax_curv.transAxes, va="top", fontsize=8, color=TEXT)
    for ax in (ax_full, ax_curv):
        ax.grid(axis="both", which="minor", visible=False)
        ax.tick_params(which="minor", length=0)

    groups = result["one_shot_no_retraining"]
    positions = range(len(next(iter(groups.values()))))
    for name, rows in groups.items():
        ax_group.plot(positions, [100 * r["val_acc"] for r in rows], color=CRITERION_COLORS[name], linewidth=2,
                      marker="o", markersize=3.5, linestyle="--" if name == "random" else "-", label=name)
    ax_group.set_xticks(list(positions))
    ax_group.set_xticklabels([f"{100 * r['sparsity']:.0f}%" for r in next(iter(groups.values()))])
    _style(ax_group, "Many removed at once, no retraining", "validation accuracy (%)")
    ax_group.set_xlabel("fraction of all weights removed in one shot", color=TEXT_SECONDARY, fontsize=9)
    ax_group.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="center left", bbox_to_anchor=(1.01, 0.5))

    setup = result["setup"]
    fig.suptitle(
        f"Do the criteria predict the loss change they were derived to estimate?  MNIST MLP after {setup['dense_epochs']} dense "
        f"epochs, {single['n']} sampled connections",
        x=0.01, ha="left", fontsize=12, color=TEXT, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
