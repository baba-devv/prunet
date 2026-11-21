"""Finite-difference verification of analytic gradients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .tensor import Tensor, no_grad


class GradCheckError(AssertionError):
    pass


@dataclass
class GradCheckResult:
    ok: bool
    max_abs_err: float
    max_rel_err: float
    failures: list = field(default_factory=list)


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the scalar `f()` w.r.t. the array `x`.

    `x` is perturbed in place and restored, so `f` must read it on every call.
    """
    grad = np.zeros(x.shape, dtype=np.float64)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + eps
        f_plus = f()
        x[index] = original - eps
        f_minus = f()
        x[index] = original
        grad[index] = (f_plus - f_minus) / (2 * eps)
    return grad


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence,
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    seed: int = 0,
    raise_on_fail: bool = True,
) -> GradCheckResult:
    """Compare backward() against finite differences for `fn(*inputs)`.

    The output is contracted with a fixed random tensor R and the gradient of
    <fn(x), R> is checked. A plain sum would miss bugs (e.g. sum of a softmax
    is constant). Tensor inputs must be float64 leaves and are perturbed in place.
    """
    tensors = []
    for x in inputs:
        if isinstance(x, Tensor):
            if x.data.dtype != np.float64 or not x.requires_grad or not x.is_leaf:
                raise ValueError("Tensor inputs to gradcheck must be float64 leaves that require grad")
            x.zero_grad()
            tensors.append(x)
        else:
            tensors.append(Tensor(np.array(x, dtype=np.float64), requires_grad=True))

    out = fn(*tensors)
    projection = np.random.default_rng(seed).standard_normal(out.data.shape)
    out.backward(projection)
    analytic = [np.zeros(t.data.shape) if t.grad is None else t.grad.copy() for t in tensors]

    def scalar() -> float:
        with no_grad():
            return float((fn(*tensors).data * projection).sum())

    result = GradCheckResult(ok=True, max_abs_err=0.0, max_rel_err=0.0)
    for position, (tensor, a) in enumerate(zip(tensors, analytic)):
        n = numerical_gradient(scalar, tensor.data, eps)
        abs_err = np.abs(a - n)
        scale = np.maximum(np.abs(a), np.abs(n))
        result.max_abs_err = max(result.max_abs_err, float(abs_err.max(initial=0.0)))
        result.max_rel_err = max(result.max_rel_err, float((abs_err / np.maximum(scale, 1e-12)).max(initial=0.0)))
        bad = abs_err > atol + rtol * scale
        if bad.any():
            worst = np.unravel_index(np.argmax(np.where(bad, abs_err, -1.0)), abs_err.shape)
            result.ok = False
            result.failures.append(
                f"input {position} at index {tuple(int(i) for i in worst)}: "
                f"analytic={a[worst]:.10g} numerical={n[worst]:.10g} |diff|={abs_err[worst]:.3g}"
            )

    if not result.ok and raise_on_fail:
        raise GradCheckError("gradient check failed:\n  " + "\n  ".join(result.failures))
    return result
