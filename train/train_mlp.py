"""Train a dense MLP and report its learning curve.

    uv run python -m train.train_mlp                     # MNIST, 784-256-128-10
    uv run python -m train.train_mlp --dataset spirals   # 2-D synthetic, no download

Writes results/part2/<dataset>_history.json and <dataset>_learning_curve.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .loop import LR_SCHEDULES

DEFAULTS = {
    "mnist": {"hidden": [256, 128], "epochs": 12, "batch_size": 128, "lr": 1e-3},
    "spirals": {"hidden": [64, 64], "epochs": 150, "batch_size": 64, "lr": 3e-3},
}


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), default="mnist")
    parser.add_argument("--hidden", type=int, nargs="+", help="hidden layer widths")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-schedule", choices=sorted(LR_SCHEDULES), default="cosine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--normalize", choices=["standard", "unit"], default="standard")


def training_settings(args: argparse.Namespace, defaults: dict) -> dict:
    """Explicit command-line values override the per-dataset defaults."""
    settings = dict(defaults[args.dataset])
    for key in settings:
        if getattr(args, key, None) is not None:
            settings[key] = getattr(args, key)
    settings["hidden"] = tuple(settings["hidden"])
    settings.update(
        dataset=args.dataset, lr_schedule=args.lr_schedule, seed=args.seed, dtype=args.dtype, normalize=args.normalize
    )
    return settings


def describe(result: dict) -> str:
    c = result["config"]
    return (
        f"{c['dataset'].upper()} MLP {'-'.join(map(str, result['model']['sizes']))}, Adam lr={c['lr']:g} "
        f"({c['lr_schedule']}), batch {c['batch_size']}, seed {c['seed']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("results/part2"))
    args = parser.parse_args()

    result = run_experiment(ExperimentConfig(**training_settings(args, DEFAULTS)))
    print(f"test loss {result['test']['loss']:.4f}  test acc {result['test']['acc']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    history_path = args.out / f"{args.dataset}_history.json"
    history_path.write_text(json.dumps(result, indent=1) + "\n")

    from .plots import plot_learning_curves

    curve_path = args.out / f"{args.dataset}_learning_curve.png"
    title = f"{describe(result)}  ·  test accuracy {100 * result['test']['acc']:.2f}%"
    plot_learning_curves(result["history"], curve_path, title)
    print(f"wrote {history_path} and {curve_path}")


if __name__ == "__main__":
    main()
