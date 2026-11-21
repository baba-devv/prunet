"""backward(): topological order, accumulation, broadcasting, graph hygiene."""

import numpy as np
import pytest

from engine import Tensor, no_grad, softmax_cross_entropy, unbroadcast


def leaf(x):
    return Tensor(np.array(x, dtype=np.float64), requires_grad=True)


# --- broadcasting ----------------------------------------------------------


def test_bias_broadcast_sums_over_the_batch():
    """(N, D) + (D,): each bias element is used N times, so its gradient is a column sum."""
    n, d = 5, 4
    x, b = leaf(np.ones((n, d))), leaf(np.zeros(d))
    upstream = np.arange(n * d, dtype=np.float64).reshape(n, d)
    (x + b).backward(upstream)
    np.testing.assert_array_equal(x.grad, upstream)
    assert b.grad.shape == (d,)
    np.testing.assert_array_equal(b.grad, upstream.sum(axis=0))


def test_broadcast_over_size_one_axes_keeps_the_axis():
    a, b = leaf(np.full((3, 1), 2.0)), leaf(np.full((1, 4), 5.0))
    (a * b).sum().backward()
    assert a.grad.shape == (3, 1) and b.grad.shape == (1, 4)
    np.testing.assert_array_equal(a.grad, np.full((3, 1), 4 * 5.0))
    np.testing.assert_array_equal(b.grad, np.full((1, 4), 3 * 2.0))


def test_scalar_tensor_broadcast():
    x, s = leaf(np.ones((2, 3))), leaf(2.0)
    (x * s).sum().backward()
    assert s.grad.shape == ()
    assert s.grad == 6.0


@pytest.mark.parametrize(
    "grad_shape, target_shape",
    [((5, 4), (4,)), ((5, 4), (1, 4)), ((5, 4), (5, 1)), ((2, 5, 4), (5, 1)), ((5, 4), ()), ((5, 4), (5, 4))],
)
def test_unbroadcast_shapes_and_totals(grad_shape, target_shape):
    g = np.random.default_rng(0).standard_normal(grad_shape)
    reduced = unbroadcast(g, target_shape)
    assert reduced.shape == target_shape
    np.testing.assert_allclose(reduced.sum(), g.sum())


# --- accumulation and ordering ---------------------------------------------


def test_tensor_used_twice_accumulates():
    x = leaf([1.0, 2.0, 3.0])
    (x + x).sum().backward()
    np.testing.assert_array_equal(x.grad, [2.0, 2.0, 2.0])


def test_diamond_graph():
    """x feeds h, and h reaches y by two paths: dy/dx = (2h + 1) * 2 with h = 2x."""
    x = leaf([1.0, -2.0, 0.5])
    h = x * 2
    y = h * h + h
    y.sum().backward()
    np.testing.assert_allclose(x.grad, 8 * x.data + 2)


def test_shared_node_at_different_depths():
    """`a` is consumed directly and through a longer chain; its backward must
    run only after both consumers have contributed."""
    x = leaf(3.0)
    a = x * x  # a = x^2
    b = a * 2 + 1  # deeper consumer of a
    y = a * b  # y = a (2a + 1) = 2x^4 + x^2
    y.backward()
    np.testing.assert_allclose(x.grad, 8 * 3.0**3 + 2 * 3.0)


def test_exponentially_many_paths_are_handled_in_linear_time():
    """y = 2^50 x through 50 doublings: 2^50 paths, but each node's backward runs once."""
    x = leaf(1.0)
    y = x
    for _ in range(50):
        y = y + y
    y.backward()
    assert x.grad == 2.0**50


def test_each_backward_closure_runs_exactly_once():
    calls = []
    x = leaf([1.0, 2.0])
    h = x * 3
    original = h._backward

    def counting(g):
        calls.append(g.copy())
        original(g)

    h._backward = counting
    (h + h * h).sum().backward()
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], 1 + 2 * h.data)  # complete when it runs


def test_deep_graph_does_not_hit_the_recursion_limit():
    x = leaf(1.0)
    y = x
    for _ in range(20_000):
        y = y + 1.0
    y.backward()
    assert x.grad == 1.0


def test_leaf_gradients_accumulate_across_backward_calls_until_zeroed():
    x = leaf([1.0, 2.0])
    y = (x * x).sum()
    y.backward()
    np.testing.assert_array_equal(x.grad, [2.0, 4.0])
    y.backward()  # same graph again: intermediates must not double-count
    np.testing.assert_array_equal(x.grad, [4.0, 8.0])
    x.zero_grad()
    assert x.grad is None
    (x * x).sum().backward()
    np.testing.assert_array_equal(x.grad, [2.0, 4.0])


# --- graph hygiene -----------------------------------------------------------


def test_intermediate_gradients_are_freed_unless_retained():
    x = leaf([1.0, 2.0])
    h = x * 2
    kept = (h + 1).retain_grad()
    out = (kept * kept).sum()
    out.backward()
    assert h.grad is None and out.grad is None
    np.testing.assert_array_equal(kept.grad, 2 * kept.data)
    out.backward()  # a retained gradient reflects one pass; it is not doubled
    np.testing.assert_array_equal(kept.grad, 2 * kept.data)


def test_leaf_gradients_are_private_writable_arrays():
    x, y = leaf([1.0, 2.0]), leaf([3.0, 4.0])
    seed = np.array([1.0, 1.0])
    (x + y).backward(seed)
    assert x.grad is not y.grad and x.grad is not seed
    x.grad *= 10  # must not leak into y's gradient or the caller's seed
    np.testing.assert_array_equal(y.grad, [1.0, 1.0])
    np.testing.assert_array_equal(seed, [1.0, 1.0])

    z = leaf(np.ones((2, 3)))
    z.sum().backward()
    assert z.grad.flags.writeable and z.grad.shape == (2, 3)


def test_constants_get_no_gradient_and_no_graph():
    x, c = leaf([1.0, 2.0]), Tensor([5.0, 6.0])
    y = x * c
    assert y.requires_grad and y._parents == (x,)
    y.sum().backward()
    assert c.grad is None
    np.testing.assert_array_equal(x.grad, [5.0, 6.0])
    assert not (c * c).requires_grad and (c * c)._backward is None


def test_no_grad_builds_no_graph():
    x = leaf([1.0, 2.0])
    with no_grad():
        y = (x * x).sum()
    assert not y.requires_grad and y._parents == ()
    assert (x * x).requires_grad  # re-enabled afterwards


def test_backward_argument_validation():
    x = leaf([1.0, 2.0])
    with pytest.raises(RuntimeError, match="non-scalar"):
        (x * 2).backward()
    with pytest.raises(ValueError, match="shape"):
        (x * 2).backward(np.ones(3))
    with pytest.raises(RuntimeError, match="does not require grad"):
        Tensor([1.0]).backward()


def test_detach_cuts_the_graph():
    x = leaf([1.0, 2.0])
    y = (x * x.detach()).sum()  # treated as x * const
    y.backward()
    np.testing.assert_array_equal(x.grad, [1.0, 2.0])


# --- dtype -------------------------------------------------------------------


def test_float32_is_preserved_through_forward_and_backward():
    x = Tensor(np.ones((4, 3), dtype=np.float32), requires_grad=True)
    w = Tensor(np.full((3, 2), 0.5, dtype=np.float32), requires_grad=True)
    b = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
    h = ((x @ w + b) * 2.0 - 1.0) / 3.0
    h = h.relu() + h.tanh() + h.sigmoid() + h.gelu() + (h**2).mean(axis=0)
    loss = softmax_cross_entropy(h, np.array([0, 1, 0, 1]))
    assert h.dtype == np.float32 and loss.dtype == np.float32
    loss.backward()
    assert w.grad.dtype == np.float32 and b.grad.dtype == np.float32


def test_integer_input_becomes_float64():
    assert Tensor([1, 2, 3]).dtype == np.float64
    assert Tensor(np.arange(3)).dtype == np.float64


def test_ndarray_on_the_left_returns_a_tensor():
    x = leaf([1.0, 2.0])
    for result in (np.ones(2) + x, np.ones(2) - x, np.ones(2) * x, np.ones(2) / x, np.ones(2) @ x):
        assert isinstance(result, Tensor) and result.requires_grad
