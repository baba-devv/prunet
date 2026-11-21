"""Trainable tensors with optional pruning masks.

The network uses w_eff = mask * theta, built as a graph node by effective():

* Forward: a masked connection contributes exactly 0, whatever is stored.
* Backward: dL/dtheta = mask * dL/dw_eff, exactly 0 at masked coordinates.

dL/dw_eff itself is kept as `dense_grad`. At a masked coordinate it is the
slope of the loss if that connection were switched back on, which is useful
for regrowth scores but must never drive an optimizer update.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .tensor import Tensor


class Parameter(Tensor):
    """A leaf tensor that requires grad and may carry a binary mask (True = active)."""

    __slots__ = ("_mask", "_flips", "dense_grad")

    def __init__(self, data, mask=None, name: Optional[str] = None):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(np.array(data), requires_grad=True, name=name)  # private copy
        self._mask: Optional[np.ndarray] = None
        self._flips: Optional[np.ndarray] = None
        self.dense_grad: Optional[np.ndarray] = None
        if mask is not None:
            self.set_mask(mask)

    # ------------------------------------------------------------------ mask

    @property
    def mask(self) -> Optional[np.ndarray]:
        """Read-only boolean array (True = active), or None if nothing is pruned."""
        return self._mask

    @mask.setter
    def mask(self, value) -> None:
        self.set_mask(value)

    def set_mask(self, mask) -> None:
        """Replace the mask (the only way a connection is pruned or revived).

        Newly masked entries are set to 0 in storage, so revived ones restart
        from 0. Flipped coordinates are recorded until an optimizer collects
        them with pop_mask_flips() and resets their state.
        """
        if mask is None:
            new = None
        else:
            new = np.array(mask)
            if new.shape != self.data.shape:
                raise ValueError(f"mask shape {new.shape} != parameter shape {self.data.shape}")
            if new.dtype != bool:
                if not np.all((new == 0) | (new == 1)):
                    raise ValueError("mask must be boolean (or contain only 0 and 1)")
                new = new.astype(bool)
            # In-place edits would bypass flip tracking and desynchronise the optimizer.
            new.flags.writeable = False

        was_active = True if self._mask is None else self._mask
        now_active = True if new is None else new
        flipped = np.broadcast_to(np.asarray(was_active != now_active), self.data.shape)
        if flipped.any():
            self._flips = flipped.copy() if self._flips is None else (self._flips | flipped)

        self._mask = new
        if new is not None:
            self.data[~new] = 0

    def pop_mask_flips(self) -> Optional[np.ndarray]:
        """Coordinates pruned or revived since the last call, or None.

        Flips are OR-ed, so a prune followed by a revive between two optimizer
        steps is still reported.
        """
        flips, self._flips = self._flips, None
        return flips

    @property
    def num_active(self) -> int:
        return self.data.size if self._mask is None else int(self._mask.sum())

    @property
    def sparsity(self) -> float:
        """Fraction of connections removed."""
        return 1.0 - self.num_active / self.data.size

    # ----------------------------------------------------------------- graph

    def effective(self) -> Tensor:
        """`mask * theta` as a graph node; layers should always use this."""
        mask = self._mask
        if mask is None:
            return self
        param = self

        # The mask is captured at forward time, so backward stays consistent
        # with this forward pass even if the mask changes in between.
        def backward(g):
            param.dense_grad = g.copy() if param.dense_grad is None else param.dense_grad + g
            param._accumulate(g * mask)  # dL/dtheta = m * dL/dw_eff

        # np.where rather than `data * mask`: stays 0 even if storage holds inf/nan.
        zero = self.data.dtype.type(0)
        return Tensor._from_op(np.where(mask, self.data, zero), (self,), backward, op="mask")

    def zero_grad(self) -> None:
        super().zero_grad()
        self.dense_grad = None
