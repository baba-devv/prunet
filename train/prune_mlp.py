"""Train an MLP that prunes itself to a hard sparsity budget.

    uv run python -m train.prune_mlp                            # MNIST, 90% of connections removed
    uv run python -m train.prune_mlp --sparsity 0.95 --criterion magnitude
    uv run python -m train.prune_mlp --regrow 0.2               # allow pruned connections to return
    uv run python -m train.prune_mlp --dataset spirals          # no download

Training has three phases: dense warm-up, a pruning ramp along the schedule
(mask updated every --prune-interval steps), then fine-tuning with the mask
frozen. Writes results/part3/<dataset>_<criterion>_s<percent>[...].json and .png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prune import CRITERIA, SCHEDULES

from .experiment import ExperimentConfig, run_experiment
from .train_mlp import add_training_arguments, describe, training_settings

DEFAULTS = {
    "mnist": {"hidden": [256, 128], "epochs": 20, "batch_size": 128, "lr": 1e-3, "prune_interval": 100},
    "spirals": {"hidden": [64, 64], "epochs": 150, "batch_size": 64, "lr": 3e-3, "prune_interval": 20},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_arguments(parser)
    parser.add_argument("--sparsity", type=float, default=0.9, help="fraction of weights to remove (hard budget)")
    parser.add_argument("--criterion", choices=sorted(CRITERIA), default="fisher")
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), default="cubic")
    parser.add_argument("--scope", choices=["global", "layer"], default="global")
    parser.add_argument("--regrow", type=float, default=0.0, help="fraction of active connections swapped per update")
    parser.add_argument("--regrow-grace", type=int, default=1, help="updates a revived connection is protected for")
    parser.add_argument("--prune-start", type=float, default=0.1, help="fraction of training before pruning starts")
    parser.add_argument("--prune-end", type=float, default=0.7, help="fraction of training at which the mask freezes")
    parser.add_argument("--prune-interval", type=int, help="optimizer steps between mask updates")
    parser.add_argument("--out", type=Path, default=Path("results/part3"))
    args = parser.parse_args()

    config = ExperimentConfig(
        sparsity=args.sparsity,
        criterion=args.criterion,
        schedule=args.schedule,
        scope=args.scope,
        regrow=args.regrow,
        regrow_grace=args.regrow_grace,
        prune_start=args.prune_start,
        prune_end=args.prune_end,
        **training_settings(args, DEFAULTS),
    )
    result = run_experiment(config)

    model = result["model"]
    print(
        f"\nsparsity {model['weight_sparsity']:.4f}  ({model['nonzero_weights']:,} of {model['num_weights']:,} weights non-zero; "
        f"per layer {', '.join(f'{s:.3f}' for s in model['layer_sparsity'])})"
    )
    print(f"validation acc {result['val']['acc']:.4f}   test acc {result['test']['acc']:.4f}")
    if args.regrow > 0:
        print(f"regrowth: {result['pruning']['regrowth']}")

    name = f"{args.dataset}_{args.criterion}_s{round(100 * args.sparsity):02d}"
    name += f"_{args.schedule}" if args.schedule != "cubic" else ""
    name += f"_{args.scope}" if args.scope != "global" else ""
    name += f"_regrow{round(100 * args.regrow):02d}" if args.regrow > 0 else ""
    name += f"_{args.normalize}" if args.normalize != "standard" else ""
    name += f"_seed{args.seed}" if args.seed else ""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")

    from .plots import plot_pruning_run

    title = (
        f"{describe(result)}\n{args.criterion} pruning to {100 * model['weight_sparsity']:.1f}% sparsity"
        f"  ·  validation {100 * result['val']['acc']:.2f}%  ·  test {100 * result['test']['acc']:.2f}%"
    )
    plot_pruning_run(result, args.out / f"{name}.png", title)
    print(f"wrote {args.out / name}.json and .png")


if __name__ == "__main__":
    main()
