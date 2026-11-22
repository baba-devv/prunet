"""Does each importance criterion actually predict the loss change from removing a connection?

    uv run python -m train.criterion_check

Each criterion estimates score_i ~= L(w with w_i = 0) - L(w). This script
measures the right-hand side directly and compares.

1. Train the dense MNIST MLP for a few epochs (float64), then freeze it.
2. Estimate gradient statistics at those fixed weights over a fixed set of
   training examples (the "probe set"); L below is the loss on that same set.
3. Single connections: for a random sample of weights in each layer, set the
   weight to zero, re-evaluate L, restore it. Report the rank correlation
   between every criterion's score and the measured loss change, and for the
   criteria expressed in nats, how well the *values* agree.
4. Groups: with no retraining, remove the lowest-scoring fraction of all
   weights by each criterion and measure the damage.

Writes results/part3/criterion_check.json and criterion_check.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from engine import softmax_cross_entropy
from prune import CRITERIA, GradStats, select

from .experiment import ExperimentConfig, prunable_weights, setup_experiment
from .loop import evaluate, train

SCORED = ["magnitude", "fisher", "taylor1", "movement", "taylor2"]
IN_NATS = ["fisher", "taylor2"]  # criteria whose value, not just ranking, is a loss-change estimate


def rank(values: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 with ties given their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    sorted_values = values[order]
    boundaries = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(values)]):
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def gradient_statistics(model, weights, x, y, batch_size: int) -> list:
    """Plain means of g, g^2 and g*w over mini-batches of the probe set, at fixed weights."""
    stats = [GradStats(w.shape, batch_size, decay=None) for w in weights]
    for start in range(0, len(x), batch_size):
        model.zero_grad()
        softmax_cross_entropy(model(x[start : start + batch_size]), y[start : start + batch_size]).backward()
        for w, s in zip(weights, stats):
            s.update(w.grad, w.data)
    model.zero_grad()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3, help="dense training before the measurement")
    parser.add_argument("--probe", type=int, default=10000, help="training examples defining L")
    parser.add_argument("--batch-size", type=int, default=125, help="batch size for the gradient statistics")
    parser.add_argument("--per-layer", type=int, default=400, help="connections sampled per layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/part3"))
    args = parser.parse_args()

    config = ExperimentConfig(epochs=args.epochs, lr_schedule="constant", dtype="float64", seed=args.seed)
    if args.probe % args.batch_size:
        parser.error("--probe must be a multiple of --batch-size (the batches must partition the probe set)")
    data, model, optimizer, _, batch_rng, _ = setup_experiment(config)
    history = train(model, optimizer, data, config.epochs, config.batch_size, batch_rng, lr_schedule="constant")
    weights = prunable_weights(model)
    x, y = data.x_train[: args.probe], data.y_train[: args.probe]

    def loss() -> float:
        return evaluate(model, x, y, batch_size=args.probe)[0]

    base_loss = loss()
    stats = gradient_statistics(model, weights, x, y, args.batch_size)
    scores = {name: [CRITERIA[name].score(w.data, s) for w, s in zip(weights, stats)] for name in SCORED}

    # ---- single connections ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    samples = []  # (layer, flat index, true loss change)
    for layer, w in enumerate(weights):
        for flat in rng.choice(w.size, size=min(args.per_layer, w.size), replace=False):
            index = np.unravel_index(flat, w.shape)
            saved = w.data[index]
            w.data[index] = 0.0
            samples.append((layer, int(flat), loss() - base_loss))
            w.data[index] = saved
        print(f"layer {layer}: measured {min(args.per_layer, w.size)} single-connection removals")
    layers = np.array([s[0] for s in samples])
    true_change = np.array([s[2] for s in samples])
    sampled = {
        name: np.array([scores[name][layer].flat[flat] for layer, flat, _ in samples]) for name in SCORED
    }

    single = {"n": len(samples), "spearman": {}, "calibration": {}}
    for name in SCORED:
        single["spearman"][name] = {
            "all_layers": spearman(sampled[name], true_change),
            "per_layer": [spearman(sampled[name][layers == k], true_change[layers == k]) for k in range(len(weights))],
        }
    for name in IN_NATS:
        predicted = sampled[name]
        both_positive = (predicted > 0) & (true_change > 0)
        log_ratio = np.log10(predicted[both_positive] / true_change[both_positive])
        single["calibration"][name] = {
            "sum_predicted": float(predicted.sum()),
            "sum_true": float(true_change.sum()),
            "median_predicted_over_true": float(10 ** np.median(log_ratio)),
            "fraction_within_3x": float((np.abs(log_ratio) < np.log10(3)).mean()),
            "n_compared": int(both_positive.sum()),
        }
    single["fraction_true_change_negative"] = float((true_change < 0).mean())

    # The statistics were taken over batches that partition the probe set, so mean_g is the exact
    # gradient of L and `movement` (= -g_i w_i at fixed weights) is the exact first-order term.
    # What is left of the measured change is the curvature term, 1/2 H_ii w_i^2 + O(w^3): compare
    # it with the Fisher estimate 1/2 F_ii w_i^2 and with magnitude's 1/2 w_i^2.
    curvature_part = true_change - sampled["movement"]
    usable = (curvature_part > 0) & (sampled["fisher"] > 0)
    log_ratio = np.log10(sampled["fisher"][usable] / curvature_part[usable])
    single["curvature_term"] = {
        "fraction_positive": float((curvature_part > 0).mean()),
        "sum_measured": float(curvature_part.sum()),
        "sum_fisher": float(sampled["fisher"].sum()),
        "median_fisher_over_measured": float(10 ** np.median(log_ratio)),
        "fraction_within_3x": float((np.abs(log_ratio) < np.log10(3)).mean()),
        "spearman_fisher": spearman(sampled["fisher"], curvature_part),
        "spearman_magnitude": spearman(sampled["magnitude"], curvature_part),
        "spearman_fisher_per_layer": [
            spearman(sampled["fisher"][layers == k], curvature_part[layers == k]) for k in range(len(weights))
        ],
        "spearman_magnitude_per_layer": [
            spearman(sampled["magnitude"][layers == k], curvature_part[layers == k]) for k in range(len(weights))
        ],
    }

    # ---- groups, no retraining ----------------------------------------------
    sparsities = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    total = sum(w.size for w in weights)
    saved = [w.data.copy() for w in weights]
    everything = [np.ones(w.shape, dtype=bool) for w in weights]
    group_scores = dict(scores, random=[rng.random(w.shape) for w in weights])
    groups = {}
    for name, score in group_scores.items():
        rows = []
        for sparsity in sparsities:
            drop = select(score, everything, int(round(sparsity * total)), largest=False)
            for w, d in zip(weights, drop):
                w.data[d] = 0.0
            probe_loss = loss()
            val_acc = evaluate(model, data.x_val, data.y_val)[1]
            rows.append({"sparsity": sparsity, "probe_loss": probe_loss, "val_acc": val_acc})
            for w, s in zip(weights, saved):
                w.data[...] = s
        groups[name] = rows
        print(f"{name:10s} one-shot val acc: " + "  ".join(f"{r['sparsity']:.0%}:{100 * r['val_acc']:.1f}" for r in rows))

    result = {
        "setup": {
            "model": model.sizes,
            "dense_epochs": args.epochs,
            "val_acc_at_measurement": history[-1]["val_acc"],
            "probe_examples": args.probe,
            "stats_batch_size": args.batch_size,
            "base_probe_loss": base_loss,
            "seed": args.seed,
        },
        "single_connection": single,
        "samples": {
            "layer": layers.tolist(),
            "true_loss_change": true_change.tolist(),
            **{f"score_{name}": sampled[name].tolist() for name in SCORED},
        },
        "one_shot_no_retraining": groups,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "criterion_check.json"
    path.write_text(json.dumps(result, indent=1) + "\n")

    print("\nSpearman rank correlation, score vs measured loss change (all layers | per layer):")
    for name in SCORED:
        s = single["spearman"][name]
        print(f"  {name:10s} {s['all_layers']:+.3f} | " + " ".join(f"{v:+.3f}" for v in s["per_layer"]))
    for name in IN_NATS:
        print(f"  {name} calibration: {single['calibration'][name]}")
    print(f"  fraction of removals that lowered the loss: {single['fraction_true_change_negative']:.3f}")
    print(f"  curvature term (measured change + g*w) vs 1/2 F w^2: {single['curvature_term']}")

    from .plots import plot_criterion_check

    plot_criterion_check(result, args.out / "criterion_check.png")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
