"""Forward value and gradients of a masked parameter."""

import numpy as np
import pytest

from engine import Parameter, Tensor, gradcheck, numerical_gradient, softmax_cross_entropy

rng = np.random.default_rng(0)
X = rng.standard_normal((6, 5))
Y = rng.integers(0, 3, size=6)
W0 = rng.standard_normal((5, 3))
MASK = rng.random((5, 3)) < 0.5


def loss_of(weight: Tensor) -> Tensor:
    return softmax_cross_entropy(Tensor(X) @ weight, Y)


# --- forward ---------------------------------------------------------------


def test_masked_weights_are_exact_zeros_in_storage_and_in_the_forward_pass():
    w = Parameter(W0, mask=MASK)
    assert (w.data[~MASK] == 0).all()
    np.testing.assert_array_equal(w.data[MASK], W0[MASK])
    effective = w.effective().data
    assert (effective[~MASK] == 0).all() and np.count_nonzero(effective) == MASK.sum()


def test_forward_ignores_whatever_is_stored_at_masked_coordinates():
    """Even if something corrupts the stored value, a pruned connection carries no signal."""
    w = Parameter(W0, mask=MASK)
    clean = loss_of(w.effective()).item()
    w.data[~MASK] = np.inf
    assert loss_of(w.effective()).item() == clean
    w.data[~MASK] = np.nan
    assert loss_of(w.effective()).item() == clean


def test_unmasked_parameter_is_used_directly():
    w = Parameter(W0)
    assert w.mask is None and w.effective() is w
    assert w.num_active == 15 and w.sparsity == 0.0


# --- backward --------------------------------------------------------------


def test_gradient_is_exactly_zero_at_masked_coordinates_and_dense_elsewhere():
    w = Parameter(W0, mask=MASK)
    loss_of(w.effective()).backward()

    # Reference: differentiate the same loss w.r.t. the effective weights as a plain tensor.
    reference = Tensor(np.where(MASK, W0, 0.0), requires_grad=True)
    loss_of(reference).backward()

    assert (w.grad[~MASK] == 0).all()
    np.testing.assert_array_equal(w.grad[MASK], reference.grad[MASK])
    # dense_grad is dL/dw_eff everywhere -- including where the connection is off.
    np.testing.assert_array_equal(w.dense_grad, reference.grad)
    assert np.abs(w.dense_grad[~MASK]).min() > 0


def test_both_gradients_agree_with_finite_differences():
    w = Parameter(W0, mask=MASK)

    # (a) dL/dtheta: perturbing a stored masked weight must not move the loss at all.
    gradcheck(lambda p: loss_of(p.effective()), [w])
    numeric_theta = numerical_gradient(lambda: loss_of(w.effective()).item(), w.data)
    assert (numeric_theta[~MASK] == 0).all()

    # (b) dense_grad: the slope of the loss if a masked connection were switched on.
    w.zero_grad()
    loss_of(w.effective()).backward()
    w_eff = np.where(MASK, W0, 0.0)
    numeric_dense = numerical_gradient(lambda: loss_of(Tensor(w_eff)).item(), w_eff)
    np.testing.assert_allclose(w.dense_grad, numeric_dense, rtol=1e-6, atol=1e-9)


def test_backward_uses_the_mask_from_the_forward_pass_that_built_the_graph():
    w = Parameter(W0, mask=MASK)
    loss = loss_of(w.effective())
    w.set_mask(np.ones_like(MASK))  # revive everything before backward
    loss.backward()
    assert (w.grad[~MASK] == 0).all()  # gradient of the loss that was actually computed


def test_zero_grad_clears_both_gradients_and_they_accumulate_together():
    w = Parameter(W0, mask=MASK)
    loss_of(w.effective()).backward()
    once_grad, once_dense = w.grad.copy(), w.dense_grad.copy()
    loss_of(w.effective()).backward()
    np.testing.assert_allclose(w.grad, 2 * once_grad)
    np.testing.assert_allclose(w.dense_grad, 2 * once_dense)
    w.zero_grad()
    assert w.grad is None and w.dense_grad is None


# --- mask bookkeeping ------------------------------------------------------


def test_set_mask_zeroes_newly_pruned_weights_and_revived_weights_restart_from_zero():
    w = Parameter(W0)
    w.set_mask(MASK)
    assert (w.data[~MASK] == 0).all()
    w.set_mask(None)
    assert w.mask is None
    assert (w.data[~MASK] == 0).all()  # the old value is gone for good
    np.testing.assert_array_equal(w.data[MASK], W0[MASK])


def test_mask_cannot_be_edited_behind_the_parameters_back():
    source = MASK.copy()
    w = Parameter(W0, mask=source)
    source[:] = True  # caller's array is not the parameter's mask
    np.testing.assert_array_equal(w.mask, MASK)
    with pytest.raises(ValueError, match="read-only"):
        w.mask[0, 0] = not w.mask[0, 0]
    w.mask = np.ones_like(MASK)  # assignment goes through set_mask, so the flip is tracked
    np.testing.assert_array_equal(w.pop_mask_flips(), ~MASK)


def test_mask_validation():
    w = Parameter(W0)
    with pytest.raises(ValueError, match="shape"):
        w.set_mask(np.ones((3, 5), dtype=bool))
    with pytest.raises(ValueError, match="boolean"):
        w.set_mask(np.full((5, 3), 0.5))
    w.set_mask(MASK.astype(np.float64))  # 0/1 floats are accepted
    assert w.mask.dtype == bool and w.num_active == MASK.sum()
    assert w.sparsity == pytest.approx(1 - MASK.mean())


def test_flips_accumulate_until_collected():
    w = Parameter(np.ones(4))
    assert w.pop_mask_flips() is None

    # A round trip leaves the mask unchanged, but the weight was zeroed in between:
    # comparing only the first and last mask would miss it.
    w.set_mask([True, False, True, True])
    w.set_mask(None)
    np.testing.assert_array_equal(w.pop_mask_flips(), [False, True, False, False])
    assert w.pop_mask_flips() is None

    # Flips from separate calls are OR-ed: the second call does not touch
    # coordinate 0, yet its revival must still be reported.
    w.set_mask([False, True, True, True])
    w.pop_mask_flips()
    w.set_mask([True, True, True, True])  # revive 0
    w.set_mask([True, True, False, True])  # prune 2
    np.testing.assert_array_equal(w.pop_mask_flips(), [True, False, True, False])

    w.set_mask([True, True, False, True])  # no change
    assert w.pop_mask_flips() is None


def test_parameter_owns_its_data():
    source = W0.copy()
    w = Parameter(source, mask=MASK)
    np.testing.assert_array_equal(source, W0)  # pruning did not write through to the caller's array
