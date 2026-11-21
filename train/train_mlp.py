"""Train a dense MLP and report its learning curve.

    uv run python -m train.train_mlp                     # MNIST, 784-256-128-10
    uv run python -m train.train_mlp --dataset spirals   # 2-D synthetic, no download

Writes results/part2/<dataset>_history.json and <dataset>_learning_curve.png.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from nn import MLP, Adam

from .data import load_dataset
from .loop import LR_SCHEDULES, evaluate, train

DEFAULTS = {
    "mnist": {"hidden": [256, 128], "epochs": 12, "batch_size": 128, "lr": 1e-3},
    "spirals": {"hidden": [64, 64], "epochs": 150, "batch_size": 64, "lr": 3e-3},
}


def run(
    dataset: str,
    hidden,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    lr_schedule: str = "cosine",
    dtype=np.float32,
    log=print,
) -> dict:
    """Train one model; return config, per-epoch history and final test metrics."""
    data = load_dataset(dataset, dtype=dtype)
    # Independent streams: changing the architecture does not change the batch order.
    init_seq, batch_seq = np.random.SeedSequence(seed).spawn(2)
    model = MLP([data.num_features, *hidden, data.num_classes], np.random.default_rng(init_seq), dtype=dtype)
    optimizer = Adam(model.parameters(), lr=lr)
    history = train(
        model, optimizer, data, epochs, batch_size, np.random.default_rng(batch_seq), lr_schedule=lr_schedule, log=log
    )
    test_loss, test_acc = evaluate(model, data.x_test, data.y_test)
    return {
        "config": {
            "dataset": dataset,
            "sizes": model.sizes,
            "activation": model.activation,
            "optimizer": "adam",
            "lr": lr,
            "lr_schedule": lr_schedule,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "dtype": np.dtype(dtype).name,
            "num_parameters": int(sum(p.size for p in model.parameters())),
            "train_examples": len(data.x_train),
            "val_examples": len(data.x_val),
            "test_examples": len(data.x_test),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
        "history": history,
        "test": {"loss": test_loss, "acc": test_acc},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), default="mnist")
    parser.add_argument("--hidden", type=int, nargs="+", help="hidden layer widths")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-schedule", choices=sorted(LR_SCHEDULES), default="cosine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--out", type=Path, default=Path("results/part2"))
    args = parser.parse_args()

    settings = dict(DEFAULTS[args.dataset])
    for key in settings:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)

    result = run(
        args.dataset, seed=args.seed, lr_schedule=args.lr_schedule, dtype=np.dtype(args.dtype).type, **settings
    )
    print(f"test loss {result['test']['loss']:.4f}  test acc {result['test']['acc']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    history_path = args.out / f"{args.dataset}_history.json"
    history_path.write_text(json.dumps(result, indent=2) + "\n")

    from .plots import plot_learning_curves

    config = result["config"]
    title = (
        f"{args.dataset.upper()} MLP {'-'.join(map(str, config['sizes']))}, Adam lr={config['lr']:g} "
        f"({config['lr_schedule']}), batch {config['batch_size']}, seed {config['seed']}  ·  test accuracy {100 * result['test']['acc']:.2f}%"
    )
    curve_path = args.out / f"{args.dataset}_learning_curve.png"
    plot_learning_curves(result["history"], curve_path, title)
    print(f"wrote {history_path} and {curve_path}")


if __name__ == "__main__":
    main()
