"""Differentiable functions that are more than a single Tensor method."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def _log_softmax_array(z: np.ndarray, axis: int) -> np.ndarray:
    """log softmax via log-sum-exp: after subtracting the max, exp can't
    overflow and the sum is >= 1, so log never sees 0."""
    shifted = z - z.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    log_p = _log_softmax_array(x.data, axis)

    def backward(g):
        # y_i = x_i - logsumexp(x)  =>  dL/dx_j = g_j - softmax_j * sum_i g_i
        x._accumulate(g - np.exp(log_p) * g.sum(axis=axis, keepdims=True))

    return Tensor._from_op(log_p, (x,), backward, op="log_softmax")


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    s = np.exp(_log_softmax_array(x.data, axis))

    def backward(g):
        # ds_i/dx_j = s_i (delta_ij - s_j)  =>  dL/dx_j = s_j (g_j - sum_i g_i s_i)
        x._accumulate(s * (g - (g * s).sum(axis=axis, keepdims=True)))

    return Tensor._from_op(s, (x,), backward, op="softmax")


def softmax_cross_entropy(logits: Tensor, targets, reduction: str = "mean") -> Tensor:
    """Cross-entropy of (N, C) logits against (N,) integer labels.

    Fused so the loss comes straight from log-softmax (log(softmax) hits
    log(0) for confident predictions) and the gradient is softmax - onehot.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")
    z = logits.data
    if z.ndim != 2:
        raise ValueError(f"logits must be (N, C); got shape {z.shape}")
    y = np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(y == np.floor(y)):
            raise TypeError("targets must be integer class labels")
        y = y.astype(np.int64)
    n, c = z.shape
    if y.shape != (n,):
        raise ValueError(f"targets must have shape ({n},); got {y.shape}")
    if n == 0:
        raise ValueError("empty batch")
    if y.min() < 0 or y.max() >= c:
        raise ValueError(f"targets must lie in [0, {c})")

    log_p = _log_softmax_array(z, axis=1)
    rows = np.arange(n)
    nll = -log_p[rows, y]
    value = nll.mean() if reduction == "mean" else nll.sum() if reduction == "sum" else nll

    def backward(g):
        # d(nll_n)/d(z_n) = softmax(z_n) - onehot(y_n)
        d = np.exp(log_p)
        d[rows, y] -= 1
        if reduction == "mean":
            d *= g / n
        elif reduction == "sum":
            d *= g
        else:
            d *= g[:, None]
        logits._accumulate(d)

    return Tensor._from_op(value, (logits,), backward, op="softmax_cross_entropy")
