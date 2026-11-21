"""Reverse-mode automatic differentiation on NumPy arrays.

Each op returns a Tensor that holds its parents and a closure mapping
dL/d(output) to dL/d(parent). backward() runs those closures once each, in
reverse topological order.
"""

from __future__ import annotations

import contextlib
import math
from typing import Callable, Optional

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Disable graph construction (inference, numerical differentiation)."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` (shaped like a broadcast result) back to an operand's `shape`.

    Broadcasting copies an operand along new leading axes and along its
    length-1 axes, so the gradient is summed over exactly those axes.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad


class Tensor:
    """An ndarray plus the bookkeeping needed to differentiate through it.

    Floating dtypes are kept as given; anything else becomes float64. On
    leaves, `grad` accumulates across backward() calls until zero_grad().
    """

    __slots__ = (
        "data",
        "grad",
        "requires_grad",
        "name",
        "_parents",
        "_backward",
        "_op",
        "_grad_owned",
        "_retain_grad",
    )

    # Make `ndarray <op> Tensor` dispatch to our reflected operators.
    __array_ufunc__ = None

    def __init__(self, data, requires_grad: bool = False, name: Optional[str] = None):
        if isinstance(data, Tensor):
            data = data.data
        array = np.asarray(data)
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float64)
        self.data = array
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = bool(requires_grad)
        self.name = name
        self._parents: tuple = ()
        self._backward: Optional[Callable[[np.ndarray], None]] = None
        self._op = ""
        self._grad_owned = False
        self._retain_grad = False

    # ------------------------------------------------------------------ graph

    @staticmethod
    def _from_op(data, parents: tuple, backward: Callable[[np.ndarray], None], op: str) -> "Tensor":
        """Wrap an op's result; record the graph edge only if a parent needs grad."""
        out = Tensor(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = tuple(p for p in parents if p.requires_grad)
            out._backward = backward
            out._op = op
        return out

    def _wrap(self, other) -> "Tensor":
        """Lift a constant operand to a Tensor of this tensor's dtype."""
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    def _accumulate(self, grad: np.ndarray) -> None:
        """Add `grad` into self.grad.

        Incoming arrays may be shared with other nodes (both operands of an add
        get the same array), so the first one is stored by reference and never
        written to; a private buffer is allocated on the second contribution.
        """
        if self.grad is None:
            self.grad = grad
            self._grad_owned = False
        elif self._grad_owned:
            self.grad += grad
        else:
            self.grad = self.grad + grad
            self._grad_owned = True

    def _topological_order(self) -> list:
        """Post-order DFS (iterative, so deep graphs don't hit the recursion limit)."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad=None) -> None:
        """Accumulate dL/d(leaf) into `.grad` of every leaf this tensor depends on.

        `grad` is dL/d(self) and defaults to 1 for a scalar.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on a non-scalar tensor needs an explicit seed gradient")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=self.data.dtype)
            if grad.shape != self.data.shape:
                raise ValueError(f"seed gradient shape {grad.shape} != tensor shape {self.data.shape}")

        order = self._topological_order()
        # Intermediate gradients are per-pass scratch space; only leaves accumulate across passes.
        for node in order:
            if node._backward is not None:
                node.grad = None
        self._accumulate(grad)

        for node in reversed(order):
            if node._backward is None or node.grad is None:
                continue
            node._backward(node.grad)
            if not node._retain_grad:
                node.grad = None  # free intermediates as soon as they are consumed

        # Give each leaf its own writable buffer so gradients never alias.
        for node in order:
            if node._backward is None and node.grad is not None and not node._grad_owned:
                node.grad = np.array(node.grad)
                node._grad_owned = True

    def zero_grad(self) -> None:
        self.grad = None
        self._grad_owned = False

    def retain_grad(self) -> "Tensor":
        """Keep this intermediate tensor's gradient after `backward()`."""
        self._retain_grad = True
        return self

    def detach(self) -> "Tensor":
        """A constant sharing this tensor's data, cut out of the graph."""
        return Tensor(self.data)

    # ------------------------------------------------------------- inspection

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        return self._backward is None

    def item(self) -> float:
        return self.data.item()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        extras = f", op={self._op!r}" if self._op else ""
        if self.requires_grad:
            extras += ", requires_grad=True"
        return f"{type(self).__name__}({self.data!r}{extras})"

    # ------------------------------------------------------------ arithmetic

    def __add__(self, other) -> "Tensor":
        a, b = self, self._wrap(other)

        def backward(g):
            if a.requires_grad:
                a._accumulate(unbroadcast(g, a.data.shape))
            if b.requires_grad:
                b._accumulate(unbroadcast(g, b.data.shape))

        return Tensor._from_op(a.data + b.data, (a, b), backward, op="add")

    def __sub__(self, other) -> "Tensor":
        a, b = self, self._wrap(other)

        def backward(g):
            if a.requires_grad:
                a._accumulate(unbroadcast(g, a.data.shape))
            if b.requires_grad:
                b._accumulate(unbroadcast(-g, b.data.shape))

        return Tensor._from_op(a.data - b.data, (a, b), backward, op="sub")

    def __mul__(self, other) -> "Tensor":
        a, b = self, self._wrap(other)

        def backward(g):
            if a.requires_grad:
                a._accumulate(unbroadcast(g * b.data, a.data.shape))
            if b.requires_grad:
                b._accumulate(unbroadcast(g * a.data, b.data.shape))

        return Tensor._from_op(a.data * b.data, (a, b), backward, op="mul")

    def __truediv__(self, other) -> "Tensor":
        a, b = self, self._wrap(other)
        out = a.data / b.data

        def backward(g):
            if a.requires_grad:
                a._accumulate(unbroadcast(g / b.data, a.data.shape))
            if b.requires_grad:
                # d(a/b)/db = -a / b^2 = -(a/b) / b
                b._accumulate(unbroadcast(-g * out / b.data, b.data.shape))

        return Tensor._from_op(out, (a, b), backward, op="div")

    def __neg__(self) -> "Tensor":
        a = self

        def backward(g):
            a._accumulate(-g)

        return Tensor._from_op(-a.data, (a,), backward, op="neg")

    def __pow__(self, exponent) -> "Tensor":
        if not isinstance(exponent, (int, float)):
            raise TypeError("only constant Python-number exponents are supported")
        a = self

        def backward(g):
            a._accumulate(g * exponent * a.data ** (exponent - 1))

        return Tensor._from_op(a.data**exponent, (a,), backward, op="pow")

    def __radd__(self, other) -> "Tensor":
        return self._wrap(other) + self

    def __rsub__(self, other) -> "Tensor":
        return self._wrap(other) - self

    def __rmul__(self, other) -> "Tensor":
        return self._wrap(other) * self

    def __rtruediv__(self, other) -> "Tensor":
        return self._wrap(other) / self

    def __matmul__(self, other) -> "Tensor":
        a, b = self, self._wrap(other)

        def backward(g):
            A, B = a.data, b.data
            # Promote 1-D operands (and g) the way np.matmul does, so
            # dA = g B^T and dB = A^T g cover every case.
            A2 = A[None, :] if A.ndim == 1 else A
            B2 = B[:, None] if B.ndim == 1 else B
            if A.ndim == 1 and B.ndim == 1:
                g2 = g.reshape(1, 1)
            elif A.ndim == 1:
                g2 = np.expand_dims(g, -2)
            elif B.ndim == 1:
                g2 = np.expand_dims(g, -1)
            else:
                g2 = g
            if a.requires_grad:
                grad_a = g2 @ np.swapaxes(B2, -1, -2)
                a._accumulate(unbroadcast(grad_a, A2.shape).reshape(A.shape))
            if b.requires_grad:
                grad_b = np.swapaxes(A2, -1, -2) @ g2
                b._accumulate(unbroadcast(grad_b, B2.shape).reshape(B.shape))

        return Tensor._from_op(a.data @ b.data, (a, b), backward, op="matmul")

    def __rmatmul__(self, other) -> "Tensor":
        return self._wrap(other) @ self

    # ------------------------------------------------------------ reductions

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)  # put the reduced axes back as length 1
            a._accumulate(np.broadcast_to(g, a.data.shape))

        return Tensor._from_op(a.data.sum(axis=axis, keepdims=keepdims), (a,), backward, op="sum")

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self
        out = a.data.mean(axis=axis, keepdims=keepdims)
        scale = out.size / a.data.size  # 1 / (number of elements averaged into each output)

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            a._accumulate(np.broadcast_to(g, a.data.shape) * scale)

        return Tensor._from_op(out, (a,), backward, op="mean")

    # ----------------------------------------------------------- elementwise

    def relu(self) -> "Tensor":
        a = self

        def backward(g):
            # Subgradient 0 at the kink x == 0.
            a._accumulate(g * (a.data > 0))

        return Tensor._from_op(np.maximum(a.data, 0), (a,), backward, op="relu")

    def tanh(self) -> "Tensor":
        a = self
        out = np.tanh(a.data)

        def backward(g):
            a._accumulate(g * (1 - out * out))

        return Tensor._from_op(out, (a,), backward, op="tanh")

    def sigmoid(self) -> "Tensor":
        a = self
        # exp(-|x|) never overflows; pick the algebraically equal branch per sign.
        e = np.exp(-np.abs(a.data))
        out = np.where(a.data >= 0, 1 / (1 + e), e / (1 + e))

        def backward(g):
            a._accumulate(g * out * (1 - out))

        return Tensor._from_op(out, (a,), backward, op="sigmoid")

    def gelu(self) -> "Tensor":
        """GELU, tanh approximation (NumPy has no erf)."""
        a = self
        x = a.data
        c, k = math.sqrt(2 / math.pi), 0.044715
        t = np.tanh(c * (x + k * x**3))

        def backward(g):
            dt = (1 - t * t) * c * (1 + 3 * k * x * x)
            a._accumulate(g * (0.5 * (1 + t) + 0.5 * x * dt))

        return Tensor._from_op(0.5 * x * (1 + t), (a,), backward, op="gelu")

    def exp(self) -> "Tensor":
        a = self
        out = np.exp(a.data)

        def backward(g):
            a._accumulate(g * out)

        return Tensor._from_op(out, (a,), backward, op="exp")

    def log(self) -> "Tensor":
        a = self

        def backward(g):
            a._accumulate(g / a.data)

        return Tensor._from_op(np.log(a.data), (a,), backward, op="log")

    # ----------------------------------------------------------------- shape

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self

        def backward(g):
            a._accumulate(g.reshape(a.data.shape))

        return Tensor._from_op(a.data.reshape(shape), (a,), backward, op="reshape")

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes or None
        a = self

        def backward(g):
            inverse = None if axes is None else np.argsort(axes)
            a._accumulate(g.transpose(inverse))

        return Tensor._from_op(a.data.transpose(axes), (a,), backward, op="transpose")

    @property
    def T(self) -> "Tensor":
        return self.transpose()
