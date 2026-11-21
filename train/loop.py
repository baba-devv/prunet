"""Mini-batch training and evaluation."""

from __future__ import annotations

import math
import time
from typing import Callable, Iterator, Optional

import numpy as np

from engine import no_grad, softmax_cross_entropy
from nn import Module, Optimizer

from .data import Dataset


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator) -> Iterator[tuple]:
    """One pass over the data in a fresh random order; the last batch may be smaller."""
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        index = order[start : start + batch_size]
        yield x[index], y[index]


def evaluate(model: Module, x: np.ndarray, y: np.ndarray, batch_size: int = 2000) -> tuple:
    """(mean cross-entropy, accuracy) over a whole split, without building a graph."""
    total_loss, correct = 0.0, 0
    with no_grad():
        for start in range(0, len(x), batch_size):
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            logits = model(xb)
            total_loss += softmax_cross_entropy(logits, yb, reduction="sum").item()
            correct += int((logits.data.argmax(axis=1) == yb).sum())
    return total_loss / len(x), correct / len(x)


def global_grad_norm(params) -> float:
    return float(np.sqrt(sum(float((p.grad * p.grad).sum()) for p in params if p.grad is not None)))


# Learning-rate multiplier as a function of training progress in [0, 1).
LR_SCHEDULES = {
    "constant": lambda progress: 1.0,
    # Half a cosine from 1 to 0.
    "cosine": lambda progress: 0.5 * (1.0 + math.cos(math.pi * progress)),
}


def train(
    model: Module,
    optimizer: Optimizer,
    data: Dataset,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    lr_schedule: str = "constant",
    log: Optional[Callable[[str], None]] = print,
) -> list:
    """Train with mini-batches; return one record per epoch (epoch 0 = before training).

    `optimizer.lr` at call time is the base rate, scaled per step by `lr_schedule`.
    Train and validation metrics are measured with the same end-of-epoch weights.
    NumPy overflow / invalid / divide errors raise, and a non-finite loss aborts.
    """
    params = model.parameters()
    history = []
    schedule = LR_SCHEDULES[lr_schedule]
    base_lr = optimizer.lr
    total_steps = epochs * math.ceil(len(data.x_train) / batch_size)

    def record(epoch: int, grad_norms: list, seconds: float) -> None:
        train_loss, train_acc = evaluate(model, data.x_train, data.y_train)
        val_loss, val_acc = evaluate(model, data.x_val, data.y_val)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
            "lr": optimizer.lr,
            "seconds": seconds,
        }
        history.append(entry)
        if log is not None:
            log(
                f"epoch {epoch:3d}  train loss {train_loss:.4f} acc {train_acc:.4f}"
                f"  |  val loss {val_loss:.4f} acc {val_acc:.4f}  ({seconds:.1f}s)"
            )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        record(0, [], 0.0)
        step = 0
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            grad_norms = []
            for xb, yb in iterate_minibatches(data.x_train, data.y_train, batch_size, rng):
                optimizer.lr = base_lr * schedule(step / total_steps)
                optimizer.zero_grad()
                loss = softmax_cross_entropy(model(xb), yb)
                if not np.isfinite(loss.data):
                    raise FloatingPointError(f"non-finite loss {loss.item()} at step {step} (epoch {epoch})")
                loss.backward()
                grad_norms.append(global_grad_norm(params))
                optimizer.step()
                step += 1
            record(epoch, grad_norms, time.perf_counter() - started)
    return history
