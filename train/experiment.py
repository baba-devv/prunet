"""One training run, dense or self-pruning, fully described by an `ExperimentConfig`.

Every entry point goes through `run_experiment`, so a result file's "config"
block is enough to reproduce it.
"""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

from nn import MLP, Adam
from prune import Pruner

from .data import load_dataset
from .loop import evaluate, train


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "mnist"
    hidden: tuple = (256, 128)
    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    seed: int = 0
    dtype: str = "float32"
    normalize: str = "standard"
    # --- pruning; sparsity == 0 means a dense run and the fields below are ignored ---
    sparsity: float = 0.0
    criterion: str = "fisher"
    schedule: str = "cubic"
    scope: str = "global"
    regrow: float = 0.0  # fraction of active connections swapped per mask update (cosine-decayed to 0)
    regrow_grace: int = 1  # mask updates during which a revived connection cannot be dropped again
    prune_start: float = 0.1  # pruning begins / ends at these fractions of training;
    prune_end: float = 0.7  # the remainder is fine-tuning with a frozen mask
    prune_interval: int = 100  # optimizer steps between mask updates
    stats_decay: float = 0.99  # EMA decay of the gradient statistics (~100-step window)


def setup_experiment(config: ExperimentConfig) -> tuple:
    """Build (data, model, optimizer, pruner-or-None, batch_rng, total_steps) for a config."""
    dtype = np.dtype(config.dtype).type
    data = load_dataset(config.dataset, dtype=dtype, normalize=config.normalize)
    # Independent streams, so e.g. changing the architecture does not change the batch order.
    init_seq, batch_seq, prune_seq = np.random.SeedSequence(config.seed).spawn(3)
    model = MLP([data.num_features, *config.hidden, data.num_classes], np.random.default_rng(init_seq), dtype=dtype)
    optimizer = Adam(model.parameters(), lr=config.lr)

    total_steps = config.epochs * math.ceil(len(data.x_train) / config.batch_size)
    pruner = None
    if config.sparsity > 0:
        pruner = Pruner(
            prunable_weights(model),
            criterion=config.criterion,
            target_sparsity=config.sparsity,
            schedule=config.schedule,
            start_step=int(config.prune_start * total_steps),
            end_step=int(config.prune_end * total_steps),
            interval=config.prune_interval,
            scope=config.scope,
            regrow_fraction=config.regrow,
            regrow_grace=config.regrow_grace,
            batch_size=config.batch_size,
            stats_decay=config.stats_decay,
            seed=int(prune_seq.generate_state(1)[0]),
        )
    return data, model, optimizer, pruner, np.random.default_rng(batch_seq), total_steps


def prunable_weights(model: MLP) -> list:
    """The weight matrices. Biases are never pruned and are not counted in sparsity."""
    return [layer.weight for layer in model.linears]


def run_experiment(config: ExperimentConfig, log: Optional[Callable[[str], None]] = print) -> dict:
    data, model, optimizer, pruner, batch_rng, total_steps = setup_experiment(config)
    weights = prunable_weights(model)
    step_losses: list = []
    history = train(
        model,
        optimizer,
        data,
        config.epochs,
        config.batch_size,
        batch_rng,
        lr_schedule=config.lr_schedule,
        pruner=pruner,
        step_losses=step_losses,
        log=log,
    )
    test_loss, test_acc = evaluate(model, data.x_test, data.y_test)

    # Measure sparsity on what the forward pass actually uses, not on the pruner's bookkeeping.
    effective = [w.effective().data for w in weights]
    nonzero = [int(np.count_nonzero(e)) for e in effective]
    total = sum(w.size for w in weights)
    result = {
        "config": asdict(config),
        "model": {
            "sizes": model.sizes,
            "num_weights": total,
            "num_biases": int(sum(layer.bias.size for layer in model.linears)),
            "nonzero_weights": sum(nonzero),
            "weight_sparsity": 1.0 - sum(nonzero) / total,
            "layer_shapes": [list(w.shape) for w in weights],
            "layer_nonzero": nonzero,
            "layer_sparsity": [1.0 - n / w.size for n, w in zip(nonzero, weights)],
            "stored_zeros_match_mask": all(
                bool(np.array_equal(w.data != 0, e != 0)) for w, e in zip(weights, effective)
            ),
        },
        "data": {"train": len(data.x_train), "val": len(data.x_val), "test": len(data.x_test)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
        "steps_per_epoch": total_steps // config.epochs,
        "history": history,
        "val": {"loss": history[-1]["val_loss"], "acc": history[-1]["val_acc"]},
        "test": {"loss": test_loss, "acc": test_acc},
        "step_losses": [round(x, 5) for x in step_losses],
    }
    if pruner is not None:
        result["pruning"] = {"events": pruner.events, "regrowth": pruner.regrowth_summary()}
    return result
