"""Ablations: which pruning design choices matter, measured on validation accuracy.

    uv run python -m train.ablations              # 81 runs, ~15 minutes on 4 cores; resumable

Uses the validation split only; the test split is not evaluated here.

Writes results/part3/ablations.json (one record per run) and ablations.md
(the summary table). Finished runs are cached in the JSON, so an interrupted
sweep continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .experiment import ExperimentConfig
from .sweep import mean_std, run_all

BASE = ExperimentConfig(epochs=20)
SEEDS = (0, 1, 2)

# (group, label, sparsities, overrides)
VARIANTS = [
    ("dense reference", "no pruning", (0.0,), {}),
    ("criterion", "magnitude", (0.9, 0.95, 0.98), {"criterion": "magnitude"}),
    ("criterion", "fisher", (0.9, 0.95, 0.98), {"criterion": "fisher"}),
    ("criterion", "taylor1", (0.95,), {"criterion": "taylor1"}),
    ("criterion", "movement", (0.95,), {"criterion": "movement"}),
    ("criterion", "taylor2", (0.95,), {"criterion": "taylor2"}),
    ("criterion", "random", (0.95,), {"criterion": "random"}),
    ("schedule", "magnitude, one-shot", (0.95,), {"criterion": "magnitude", "schedule": "one_shot"}),
    ("schedule", "fisher, one-shot", (0.95,), {"criterion": "fisher", "schedule": "one_shot"}),
    ("scope", "magnitude, per-layer", (0.95,), {"criterion": "magnitude", "scope": "layer"}),
    ("scope", "fisher, per-layer", (0.95,), {"criterion": "fisher", "scope": "layer"}),
    ("regrowth", "fisher, regrow 0.2, no grace period", (0.95, 0.98), {"criterion": "fisher", "regrow": 0.2, "regrow_grace": 0}),
    ("regrowth", "fisher, regrow 0.2, grace 1 update", (0.95, 0.98), {"criterion": "fisher", "regrow": 0.2, "regrow_grace": 1}),
    ("regrowth", "fisher, regrow 0.2, grace 3 updates", (0.95, 0.98), {"criterion": "fisher", "regrow": 0.2, "regrow_grace": 3}),
    ("regrowth", "magnitude, regrow 0.2, grace 3 updates", (0.95, 0.98), {"criterion": "magnitude", "regrow": 0.2, "regrow_grace": 3}),
    ("preprocessing", "magnitude, inputs in [0,1]", (0.95, 0.98), {"criterion": "magnitude", "normalize": "unit"}),
    ("preprocessing", "fisher, inputs in [0,1]", (0.95, 0.98), {"criterion": "fisher", "normalize": "unit"}),
]


def summarise(result: dict) -> dict:
    """The few numbers kept per run (no test metrics: these runs inform design decisions)."""
    record = {
        "config": result["config"],
        "val_acc": result["val"]["acc"],
        "val_loss": result["val"]["loss"],
        "weight_sparsity": result["model"]["weight_sparsity"],
        "layer_sparsity": result["model"]["layer_sparsity"],
    }
    if "pruning" in result:
        events = result["pruning"]["events"]
        jumps = [e["probe_loss_after"] - e["probe_loss_before"] for e in events]
        record["mask_updates"] = len(events)
        record["largest_probe_loss_jump"] = max(jumps)
        record["sum_probe_loss_jumps"] = sum(jumps)
        record["regrowth"] = result["pruning"]["regrowth"]
    return record


def key(config: ExperimentConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/part3"))
    args = parser.parse_args()
    json_path = args.out / "ablations.json"

    plan = []  # (group, label, sparsity, config)
    for group, label, sparsities, overrides in VARIANTS:
        for sparsity in sparsities:
            for seed in SEEDS:
                plan.append((group, label, sparsity, replace(BASE, sparsity=sparsity, seed=seed, **overrides)))

    cache = json.loads(json_path.read_text())["runs"] if json_path.exists() else {}
    todo = [config for _, _, _, config in plan if key(config) not in cache]
    print(f"{len(plan)} runs planned, {len(plan) - len(todo)} cached, {len(todo)} to run")

    def save(result: dict) -> None:
        cache[key(ExperimentConfig(**{**result["config"], "hidden": tuple(result["config"]["hidden"])}))] = summarise(result)
        args.out.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"runs": cache}, indent=1) + "\n")

    if todo:
        run_all(todo, workers=args.workers, label="ablation", on_result=save)

    lines = [
        "| group | variant | target sparsity | validation accuracy, mean ± std (%) | per seed (%) | layer sparsity, seed 0 (%) |",
        "|---|---|---|---|---|---|",
    ]
    for group, label, sparsities, overrides in VARIANTS:
        for sparsity in sparsities:
            runs = [cache[key(c)] for g, lab, s, c in plan if (g, lab, s) == (group, label, sparsity)]
            mean, std = mean_std([100 * r["val_acc"] for r in runs])
            seeds = ", ".join(f"{100 * r['val_acc']:.2f}" for r in runs)
            layers = " / ".join(f"{100 * s:.1f}" for s in runs[0]["layer_sparsity"])
            lines.append(f"| {group} | {label} | {100 * sparsity:.0f}% | {mean:.2f} ± {std:.2f} | {seeds} | {layers} |")
    table = "\n".join(lines)
    header = (
        "# Ablations\n\n"
        f"MNIST 784-256-128-10, {BASE.epochs} epochs, seeds {list(SEEDS)}; validation split (5,000 images) only.\n"
        "Unless a variant says otherwise: global ranking, cubic schedule, no regrowth, standardised inputs.\n"
        "Generated by `uv run python -m train.ablations`; raw records in `ablations.json`.\n\n"
    )
    (args.out / "ablations.md").write_text(header + table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
