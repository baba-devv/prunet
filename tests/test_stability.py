"""Numerical stability of the softmax / cross-entropy / sigmoid paths.

pytest is configured with `filterwarnings = error`, so any overflow or
invalid-value RuntimeWarning from NumPy fails these tests outright.
"""

import numpy as np
import pytest

from engine import Tensor, log_softmax, softmax, softmax_cross_entropy


def test_cross_entropy_survives_extreme_logits():
    z = Tensor(np.array([[1e4, -1e4, 0.0], [-1e4, 1e4, 0.0], [750.0, 750.0, 750.0]]), requires_grad=True)
    loss = softmax_cross_entropy(z, np.array([0, 0, 2]), reduction="none")
    # confident & right -> 0;  confident & wrong -> the logit gap;  uniform -> log C
    np.testing.assert_allclose(loss.data, [0.0, 2e4, np.log(3)], rtol=1e-12)
    loss.backward(np.ones(3))
    assert np.isfinite(z.grad).all()
    np.testing.assert_allclose(z.grad[0], [0.0, 0.0, 0.0], atol=1e-300)
    np.testing.assert_allclose(z.grad[1], [-1.0, 1.0, 0.0], atol=1e-300)


def test_naive_formula_fails_where_ours_does_not():
    z = np.array([[1000.0, 0.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    assert np.isnan(naive).any()
    assert softmax_cross_entropy(Tensor(z), np.array([1])).item() == 1000.0


def test_cross_entropy_matches_composed_ops_and_closed_form_gradient():
    rng = np.random.default_rng(0)
    z_value, y = rng.standard_normal((6, 4)), rng.integers(0, 4, size=6)

    z = Tensor(z_value, requires_grad=True)
    fused = softmax_cross_entropy(z, y)
    fused.backward()

    z2 = Tensor(z_value, requires_grad=True)
    onehot = np.eye(4)[y]
    composed = -(log_softmax(z2) * onehot).sum() / 6
    composed.backward()

    np.testing.assert_allclose(fused.data, composed.data, rtol=1e-13)
    np.testing.assert_allclose(z.grad, z2.grad, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(z.grad, (softmax(Tensor(z_value)).data - onehot) / 6, rtol=1e-12, atol=1e-15)
    # Shifting all logits of a row changes nothing, so each row of the gradient sums to 0.
    np.testing.assert_allclose(z.grad.sum(axis=1), 0, atol=1e-15)


def test_softmax_rows_sum_to_one_for_extreme_inputs():
    s = softmax(Tensor(np.array([[1e3, 1e3 + 1, -1e3], [-745.0, -746.0, -800.0]])))
    assert np.isfinite(s.data).all()
    np.testing.assert_allclose(s.data.sum(axis=1), 1.0, rtol=1e-12)


def test_sigmoid_saturates_without_overflow():
    x = Tensor(np.array([-1e4, -40.0, 0.0, 40.0, 1e4]), requires_grad=True)
    s = x.sigmoid()
    assert s.data[0] == 0.0 and s.data[2] == 0.5 and s.data[4] == 1.0
    assert 0 < s.data[1] < 1e-15 and 1 - 1e-15 < s.data[3] <= 1
    s.sum().backward()
    assert np.isfinite(x.grad).all() and x.grad[0] == 0.0 and x.grad[2] == 0.25


def test_cross_entropy_rejects_bad_targets():
    z = Tensor(np.zeros((3, 4)))
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        softmax_cross_entropy(z, np.array([0, 1, 4]))
    with pytest.raises(ValueError, match="shape"):
        softmax_cross_entropy(z, np.array([0, 1]))
    with pytest.raises(TypeError, match="integer"):
        softmax_cross_entropy(z, np.array([0.5, 1.0, 2.0]))
    with pytest.raises(ValueError, match="reduction"):
        softmax_cross_entropy(z, np.array([0, 1, 2]), reduction="avg")
