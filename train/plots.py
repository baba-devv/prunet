"""Matplotlib figures. Imported lazily by the CLIs so training itself needs only NumPy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
