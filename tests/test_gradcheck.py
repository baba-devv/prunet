"""Finite-difference gradient checks for every differentiable operation.

`test_every_op_is_gradchecked` scans the engine source for op names and fails
if any of them is not exercised by a case below, so an op cannot be added
without a gradient check.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import engine
from engine import Parameter, Tensor, gradcheck, log_softmax, softmax, softmax_cross_entropy

rng = np.random.default_rng(0)


def r(*shape):
    return rng.standard_normal(shape)


def away_from_zero(*shape):
    """Inputs for ops with a kink or pole at 0 (ReLU, division): |x| >= 0.1."""
    x = r(*shape)
    x[np.abs(x) < 0.1] += 0.5
    return x


def positive(*shape):
    return np.abs(r(*shape)) + 0.5


LABELS = np.array([0, 2, 1, 2, 0])
MASK = rng.random((4, 3)) < 0.5


def tiny_mlp(x, w1, b1, w2, b2):
    hidden = (x @ w1 + b1).relu()
    return softmax_cross_entropy(hidden @ w2 + b2, LABELS)


# (name, function of Tensors, inputs)
CASES = [
    # --- addition / subtraction, including broadcasting -------------------
    ("add", lambda a, b: a + b, [r(3, 4), r(3, 4)]),
    ("add_bias_broadcast_(N,D)+(D,)", lambda a, b: a + b, [r(5, 4), r(4)]),
    ("add_outer_broadcast_(N,1)+(1,D)", lambda a, b: a + b, [r(5, 1), r(1, 4)]),
    ("add_scalar_tensor_broadcast", lambda a, b: a + b, [r(3, 4), r()]),
    ("add_python_scalar", lambda a: a + 2.5, [r(3, 4)]),
    ("radd_python_scalar", lambda a: 2.5 + a, [r(3, 4)]),
    ("radd_ndarray", lambda a: np.ones((3, 4)) + a, [r(3, 4)]),
    ("sub", lambda a, b: a - b, [r(3, 4), r(3, 4)]),
    ("sub_broadcast_(D,)-(N,D)", lambda a, b: a - b, [r(4), r(5, 4)]),
    ("rsub_python_scalar", lambda a: 1.0 - a, [r(3, 4)]),
    ("neg", lambda a: -a, [r(3, 4)]),
    # --- multiplication / division / power --------------------------------
    ("mul", lambda a, b: a * b, [r(3, 4), r(3, 4)]),
    ("mul_broadcast_(N,1)*(D,)", lambda a, b: a * b, [r(5, 1), r(4)]),
    ("mul_same_tensor_twice", lambda a: a * a, [r(3, 4)]),
    ("rmul_python_scalar", lambda a: 3.0 * a, [r(3, 4)]),
    ("div", lambda a, b: a / b, [r(3, 4), away_from_zero(3, 4)]),
    ("div_broadcast_(N,D)/(D,)", lambda a, b: a / b, [r(5, 4), away_from_zero(4)]),
    ("rdiv_python_scalar", lambda a: 2.0 / a, [away_from_zero(3, 4)]),
    ("pow_integer", lambda a: a**3, [r(3, 4)]),
    ("pow_fractional", lambda a: a**0.5, [positive(3, 4)]),
    ("pow_negative", lambda a: a**-2, [positive(3, 4)]),
    # --- matmul ------------------------------------------------------------
    ("matmul_matrix_matrix", lambda a, b: a @ b, [r(5, 4), r(4, 3)]),
    ("matmul_matrix_vector", lambda a, b: a @ b, [r(5, 4), r(4)]),
    ("matmul_vector_matrix", lambda a, b: a @ b, [r(4), r(4, 3)]),
    ("matmul_vector_vector", lambda a, b: a @ b, [r(4), r(4)]),
    ("matmul_batched", lambda a, b: a @ b, [r(2, 5, 4), r(2, 4, 3)]),
    ("matmul_batch_broadcast", lambda a, b: a @ b, [r(2, 5, 4), r(4, 3)]),
    ("rmatmul_ndarray", lambda a: np.ones((2, 3)) @ a, [r(3, 4)]),
    # --- reductions --------------------------------------------------------
    ("sum_all", lambda a: a.sum(), [r(3, 4)]),
    ("sum_axis0", lambda a: a.sum(axis=0), [r(3, 4)]),
    ("sum_axis1_keepdims", lambda a: a.sum(axis=1, keepdims=True), [r(3, 4)]),
    ("sum_negative_axis", lambda a: a.sum(axis=-1), [r(2, 3, 4)]),
    ("sum_tuple_axes", lambda a: a.sum(axis=(0, 2)), [r(2, 3, 4)]),
    ("mean_all", lambda a: a.mean(), [r(3, 4)]),
    ("mean_axis0", lambda a: a.mean(axis=0), [r(3, 4)]),
    ("mean_axis1_keepdims", lambda a: a.mean(axis=1, keepdims=True), [r(3, 4)]),
    ("mean_tuple_axes", lambda a: a.mean(axis=(-1, -2)), [r(2, 3, 4)]),
    # --- non-linearities ---------------------------------------------------
    ("relu", lambda a: a.relu(), [away_from_zero(3, 4)]),
    ("tanh", lambda a: a.tanh(), [r(3, 4)]),
    ("sigmoid", lambda a: a.sigmoid(), [3 * r(3, 4)]),
    ("gelu", lambda a: a.gelu(), [2 * r(3, 4)]),
    ("exp", lambda a: a.exp(), [r(3, 4)]),
    ("log", lambda a: a.log(), [positive(3, 4)]),
    # --- shape -------------------------------------------------------------
    ("reshape", lambda a: a.reshape(4, 3) * np.arange(12.0).reshape(4, 3), [r(3, 4)]),
    ("transpose_default", lambda a: a.T @ np.ones((3, 2)), [r(3, 4)]),
    ("transpose_axes", lambda a: a.transpose(2, 0, 1), [r(2, 3, 4)]),
    # --- softmax family ----------------------------------------------------
    ("log_softmax", lambda a: log_softmax(a), [r(5, 3)]),
    ("log_softmax_axis0", lambda a: log_softmax(a, axis=0), [r(5, 3)]),
    ("softmax", lambda a: softmax(a), [r(5, 3)]),
    ("cross_entropy_mean", lambda z: softmax_cross_entropy(z, LABELS), [r(5, 3)]),
    ("cross_entropy_sum", lambda z: softmax_cross_entropy(z, LABELS, reduction="sum"), [r(5, 3)]),
    ("cross_entropy_none", lambda z: softmax_cross_entropy(z, LABELS, reduction="none"), [r(5, 3)]),
    ("cross_entropy_large_logits", lambda z: softmax_cross_entropy(z, LABELS), [30 * r(5, 3)]),
    # --- masked parameter --------------------------------------------------
    ("mask", lambda w, x: (x @ w.effective()).tanh(), [Parameter(r(4, 3), mask=MASK), r(5, 4)]),
    # --- everything together ----------------------------------------------
    ("mlp_end_to_end", tiny_mlp, [r(5, 6), r(6, 8), r(8), r(8, 3), r(3)]),
]


@pytest.mark.parametrize("name, fn, inputs", CASES, ids=[c[0] for c in CASES])
def test_gradcheck(name, fn, inputs):
    result = gradcheck(fn, inputs)
    assert result.ok


def _ops_in_graph(root: Tensor) -> set:
    return {node._op for node in root._topological_order() if node._op}


def test_every_op_is_gradchecked():
    source = "".join(p.read_text() for p in Path(engine.__file__).parent.glob("*.py"))
    declared = set(re.findall(r'op="(\w+)"', source))
    assert len(declared) >= 20, "op-name scan found suspiciously few operations"

    exercised = set()
    for _, fn, inputs in CASES:
        leaves = [x if isinstance(x, Tensor) else Tensor(x, requires_grad=True) for x in inputs]
        exercised |= _ops_in_graph(fn(*leaves))

    assert declared - exercised == set(), f"ops without a gradient check: {declared - exercised}"


# --- the checker itself must be able to fail ------------------------------


def _add_with_broken_broadcast(a: Tensor, b: Tensor) -> Tensor:
    """a + b whose backward forgets to scale the broadcast operand's gradient."""

    def backward(g):
        a._accumulate(g)
        b._accumulate(g.mean(axis=0))  # correct is g.sum(axis=0)

    return Tensor._from_op(a.data + b.data, (a, b), backward, op="broken_add")


def test_gradcheck_detects_a_wrong_gradient():
    with pytest.raises(engine.GradCheckError, match="input 1"):
        gradcheck(_add_with_broken_broadcast, [r(5, 4), r(4)])
    result = gradcheck(_add_with_broken_broadcast, [r(5, 4), r(4)], raise_on_fail=False)
    assert not result.ok and result.max_rel_err > 0.1


def test_gradcheck_needs_a_random_projection():
    """Summing a softmax gives a constant, so a sum-based check could not see its Jacobian."""
    x = Tensor(r(5, 3), requires_grad=True)
    softmax(x).sum().backward()
    np.testing.assert_allclose(x.grad, 0, atol=1e-15)
