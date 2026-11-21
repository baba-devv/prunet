"""Optimizers whose per-coordinate state stays correct under pruning masks.

Optimizer state belongs to a connection and exists only while it does:

* A masked weight never moves: the final update is multiplied by the mask, so
  momentum, moments and weight decay can't leak into it.
* When a coordinate is pruned or revived, its state is reset to that of a
  fresh parameter (zero moments, age 0).
* Adam's bias correction uses each coordinate's own age, not a global step.
  With a global step, a coordinate revived late with m = v = 0 gets no bias
  correction and its first step is ~3.2x lr instead of lr.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from engine import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        if lr <= 0:
            raise ValueError(f"learning rate must be positive; got {lr}")
        self.lr = lr
        # State starts fresh, so earlier mask changes need no reset.
        for p in self.params:
            pop = getattr(p, "pop_mask_flips", None)
            if pop is not None:
                pop()

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def _sync_mask(self, index: int, p: Tensor) -> Optional[np.ndarray]:
        """Reset state where the mask flipped since the last step; return the current mask."""
        pop = getattr(p, "pop_mask_flips", None)
        if pop is None:
            return None  # a plain Tensor: never masked
        flips = pop()
        if flips is not None:
            self._reset_state(index, flips)
        return p.mask

    def _reset_state(self, index: int, flips: np.ndarray) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with (heavy-ball) momentum and L2 weight decay.

        v <- mu * v + (g + wd * w)
        w <- w - lr * v
    """

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def _reset_state(self, index, flips):
        self.velocity[index][flips] = 0

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            mask = self._sync_mask(i, p)
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v = self.velocity[i]
                v *= self.momentum
                v += g
                g = v
            update = self.lr * g
            if mask is not None:
                # Masked weights never move, whatever momentum or weight decay say.
                update *= mask
            p.data -= update


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with per-coordinate bias correction.

        m <- b1 * m + (1 - b1) * g
        v <- b2 * v + (1 - b2) * g^2
        w <- w - lr * [m / (1 - b1^age)] / (sqrt(v / (1 - b2^age)) + eps)  -  lr * wd * w

    `age` counts updates since a coordinate's state was last reset; without
    masks it equals the step count. Weight decay is decoupled (AdamW).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        # b1^age and b2^age per coordinate. 0-d while all coordinates share an
        # age; expanded to full shape the first time some are reset.
        self.b1_pow = [np.ones(()) for _ in self.params]
        self.b2_pow = [np.ones(()) for _ in self.params]

    def _reset_state(self, index, flips):
        self.m[index][flips] = 0
        self.v[index][flips] = 0
        for powers in (self.b1_pow, self.b2_pow):
            if powers[index].ndim == 0:
                powers[index] = np.full(flips.shape, float(powers[index]))
            powers[index][flips] = 1.0  # beta^0: age 0

    @staticmethod
    def _advance(power: np.ndarray, beta: float, dtype) -> object:
        """Age every coordinate by one step; return the correction 1 - beta^age."""
        power *= beta
        # Flush to 0 to avoid slow subnormals (1 - 1e-30 == 1.0 anyway).
        power[power < 1e-30] = 0.0
        if power.ndim == 0:
            return 1.0 - float(power)  # Python float: does not upcast float32 state
        return (1.0 - power).astype(dtype, copy=False)

    def step(self) -> None:
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            mask = self._sync_mask(i, p)
            g = p.grad

            m, v = self.m[i], self.v[i]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * (g * g)

            # Masked coordinates age too; harmless, since they're reset on revival.
            correction1 = self._advance(self.b1_pow[i], b1, p.data.dtype)
            correction2 = self._advance(self.b2_pow[i], b2, p.data.dtype)

            update = self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)
            if self.weight_decay:
                update += self.lr * self.weight_decay * p.data
            if mask is not None:
                # Masked weights never move, whatever momentum or weight decay say.
                update *= mask
            p.data -= update
