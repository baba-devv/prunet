"""Importance criteria.

Every criterion estimates the change in loss if connection i is removed
(w_i -> 0), from a second-order Taylor expansion about the current weights:

    dL_i = L(w - w_i e_i) - L(w)  ~=  -g_i w_i  +  1/2 H_ii w_i^2

with g = dL/dw the expected gradient and H the Hessian. They differ in which
terms they keep and how they estimate g and H:

    magnitude   1/2 w^2          second-order term, assuming H_ii is the same for every connection
    fisher      1/2 F_ii w^2     second-order term, H_ii estimated by the Fisher information F_ii = E[g_sample^2]
    taylor1     |g_i w_i|        first-order term only, sign discarded
    movement    -g_i w_i         first-order term, sign kept (positive = the optimizer is growing |w_i|)
    taylor2     -g_i w_i + 1/2 F_ii w_i^2     both terms, sign kept
    random      -                control

A criterion is a function `score(weights, stats) -> array`; a higher score
means more important (pruned last).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from engine import Parameter


def dense_gradient(p: Parameter) -> np.ndarray:
    """dL/dw_eff for every coordinate of `p`, masked or not, from the last backward()."""
    return p.grad if p.mask is None else p.dense_grad


class GradStats:
    """Running averages of mini-batch gradient statistics for one weight tensor.

    Tracks E[g], E[g^2] and E[g * w], where g is the mini-batch *dense* gradient
    dL/dw_eff. With `decay` in (0, 1) these are bias-corrected exponential
    moving averages (recent steps matter most; the weights are moving); with
    `decay=None` they are plain means over all updates (for measurements at
    fixed weights).
    """

    def __init__(self, shape: tuple, batch_size: int, decay: Optional[float] = 0.99):
        self.batch_size = batch_size
        self.decay = decay
        self.steps = 0
        self._g = np.zeros(shape)
        self._g2 = np.zeros(shape)
        self._gw = np.zeros(shape)

    def update(self, grad: np.ndarray, weight: np.ndarray) -> None:
        self.steps += 1
        rate = 1.0 / self.steps if self.decay is None else 1.0 - self.decay
        self._g += rate * (grad - self._g)
        self._g2 += rate * (grad * grad - self._g2)
        self._gw += rate * (grad * weight - self._gw)

    @property
    def _correction(self) -> float:
        """An EMA started at 0 underestimates by (1 - decay^steps); undo that."""
        if self.decay is None or self.steps == 0:  # nothing observed yet: the statistics are all zero
            return 1.0
        return 1.0 - self.decay**self.steps

    @property
    def mean_g(self) -> np.ndarray:
        return self._g / self._correction

    @property
    def mean_g2(self) -> np.ndarray:
        return self._g2 / self._correction

    @property
    def mean_gw(self) -> np.ndarray:
        return self._gw / self._correction

    @property
    def fisher(self) -> np.ndarray:
        """Diagonal Fisher information E[g_sample^2], recovered from batch statistics.

        A batch gradient is the mean of B per-sample gradients, so
            E[g_batch^2] = mean^2 + Var(g_sample) / B
        and therefore
            E[g_sample^2] = mean^2 + Var(g_sample) = mean^2 + B * (E[g_batch^2] - mean^2).
        """
        mean_squared = self.mean_g**2
        return mean_squared + self.batch_size * np.maximum(self.mean_g2 - mean_squared, 0.0)


def magnitude(w: np.ndarray, stats: Optional[GradStats]) -> np.ndarray:
    return 0.5 * w * w


def fisher(w: np.ndarray, stats: GradStats) -> np.ndarray:
    return 0.5 * stats.fisher * w * w


def taylor1(w: np.ndarray, stats: GradStats) -> np.ndarray:
    return np.abs(stats.mean_g * w)


def movement(w: np.ndarray, stats: GradStats) -> np.ndarray:
    return -stats.mean_gw


def taylor2(w: np.ndarray, stats: GradStats) -> np.ndarray:
    return -stats.mean_g * w + 0.5 * stats.fisher * w * w


@dataclass(frozen=True)
class Criterion:
    score: Callable
    needs_gradients: bool


CRITERIA = {
    "magnitude": Criterion(magnitude, needs_gradients=False),
    "fisher": Criterion(fisher, needs_gradients=True),
    "taylor1": Criterion(taylor1, needs_gradients=True),
    "movement": Criterion(movement, needs_gradients=True),
    "taylor2": Criterion(taylor2, needs_gradients=True),
    # `random` is handled by the pruner (it needs a seeded generator), listed here so it is selectable.
    "random": Criterion(lambda w, stats: np.zeros_like(w), needs_gradients=False),
}


def regrowth_score(stats: GradStats) -> np.ndarray:
    """How much a currently-masked connection would help if switched back on.

    A revived weight starts at 0, where the loss changes at rate dL/dw_i per
    unit of weight. Adam moves every weight by about `lr` per step whatever
    the gradient's size, so the loss a new connection can recover in its first
    T steps is ~ lr * T * |E[g_i]|: rank candidates by the magnitude of the
    *mean* dense gradient. (A gradient that is large but flips sign from batch
    to batch averages out and correctly scores low.)
    """
    return np.abs(stats.mean_g)
