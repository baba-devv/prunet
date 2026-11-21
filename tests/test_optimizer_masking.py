"""Optimizer state under pruning masks.

Each failure mode is reproduced with a plain, mask-unaware Adam on the same
engine, then shown not to occur with nn.optim.
"""

import math

import numpy as np
import pytest

from engine import Parameter, Tensor, softmax_cross_entropy
from nn import SGD, Adam

LR = 1e-2


class TextbookAdam:
    """Adam with one global step counter and no knowledge of masks.

    `reapply_mask=True` re-zeroes masked weights after every step, which hides
    ghost updates but not stale state on revival.
    """

    def __init__(self, params, lr=LR, betas=(0.9, 0.999), eps=1e-8, reapply_mask=False):
        self.params, self.lr, self.betas, self.eps = list(params), lr, betas, eps
        self.reapply_mask = reapply_mask
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        b1, b2 = self.betas
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.reapply_mask and getattr(p, "mask", None) is not None:
                p.data[~p.mask] = 0


def quadratic_step(p, target, opt):
    """One step on L = 0.5 * sum((w_eff - target)^2): separable, so dL/dw_i = w_i - target_i."""
    opt.zero_grad()
    diff = p.effective() - np.asarray(target, dtype=np.float64)
    (diff * diff * 0.5).sum().backward()
    opt.step()


OPTIMIZERS = {
    "adam": lambda params: Adam(params, lr=LR),
    "adam_weight_decay": lambda params: Adam(params, lr=LR, weight_decay=0.1),
    "sgd_momentum": lambda params: SGD(params, lr=LR, momentum=0.9),
    "sgd_momentum_weight_decay": lambda params: SGD(params, lr=LR, momentum=0.9, weight_decay=0.1),
}
all_optimizers = pytest.mark.parametrize("make_opt", OPTIMIZERS.values(), ids=OPTIMIZERS.keys())


# --- baseline: without masks, these are the standard algorithms ----------------


def test_adam_without_masks_is_textbook_adam():
    rng = np.random.default_rng(0)
    w0, target = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
    ours, reference = Parameter(w0), Parameter(w0)
    opt, ref_opt = Adam([ours], lr=LR), TextbookAdam([reference])
    for _ in range(200):
        quadratic_step(ours, target, opt)
        quadratic_step(reference, target, ref_opt)
        np.testing.assert_allclose(ours.data, reference.data, rtol=1e-11, atol=1e-13)


def test_adam_first_step_has_magnitude_lr():
    p = Parameter([0.0, 0.0])
    quadratic_step(p, [3.0, -700.0], Adam([p], lr=LR))  # bias correction makes step 1 = lr * sign(g)
    np.testing.assert_allclose(p.data, [LR, -LR], rtol=1e-6)


def test_sgd_momentum_matches_closed_form():
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1, momentum=0.9)
    w, v = 1.0, 0.0
    for _ in range(20):
        quadratic_step(p, [0.0], opt)
        v = 0.9 * v + w  # gradient of 0.5 w^2 is w
        w = w - 0.1 * v
        assert p.data[0] == pytest.approx(w, rel=1e-13)


# --- ghost updates to a pruned weight --------------------------------------------


def test_textbook_adam_moves_a_pruned_weight_despite_its_zero_gradient():
    p = Parameter([0.0, 0.0])
    opt = TextbookAdam([p])
    for _ in range(20):
        quadratic_step(p, [10.0, 10.0], opt)
    p.set_mask([True, False])
    assert p.data[1] == 0.0
    quadratic_step(p, [10.0, 10.0], opt)
    assert p.grad[1] == 0.0
    assert abs(p.data[1]) > 0.5 * LR  # first moment moved it anyway
    assert p.effective().data[1] == 0.0


@all_optimizers
def test_pruned_weight_and_its_optimizer_state_stay_exactly_zero(make_opt):
    p = Parameter([0.0, 0.0, 0.0])
    opt = make_opt([p])
    for _ in range(20):
        quadratic_step(p, [10.0, 10.0, 10.0], opt)
    p.set_mask([True, False, True])
    for _ in range(50):
        quadratic_step(p, [10.0, 10.0, 10.0], opt)
        assert p.data[1] == 0.0
        state = (opt.m[0], opt.v[0]) if isinstance(opt, Adam) else (opt.velocity[0],)
        assert all(s[1] == 0.0 for s in state)
    assert p.data[0] > 0 and p.data[2] > 0  # active neighbours keep training


def test_optimizer_ignores_a_nonzero_gradient_at_a_masked_coordinate():
    """Holds even if a layer bypasses effective() and the raw gradient is non-zero."""
    p = Parameter([1.0, 0.0], mask=[True, False])
    for opt in (Adam([p], lr=LR), SGD([p], lr=LR, momentum=0.9)):
        for _ in range(5):
            p.grad = np.array([1.0, 1.0])  # as if a layer had used the raw parameter
            opt.step()
            assert p.data[1] == 0.0


# --- static mask == the smaller dense problem ------------------------------------


@all_optimizers
def test_statically_masked_training_equals_dense_training_on_the_active_coordinates(make_opt):
    """The loss is separable, so each coordinate's trajectory is independent of the
    others; masking some coordinates must leave the remaining ones bit-for-bit alone."""
    rng = np.random.default_rng(1)
    w0, target = rng.standard_normal((6, 5)), rng.standard_normal((6, 5))
    mask = rng.random((6, 5)) < 0.5
    dense, masked = Parameter(w0), Parameter(w0, mask=mask)
    dense_opt, masked_opt = make_opt([dense]), make_opt([masked])
    for _ in range(100):
        quadratic_step(dense, target, dense_opt)
        quadratic_step(masked, target, masked_opt)
    np.testing.assert_allclose(masked.data[mask], dense.data[mask], rtol=0, atol=1e-14)
    assert (masked.data[~mask] == 0).all()


# --- stale state on revival ------------------------------------------------------


def _first_step_after_late_revival(make_opt):
    """Coordinate 1 is masked from the start, revived after 2000 steps; return its first step."""
    p = Parameter([0.0, 0.0], mask=[True, False])
    opt = make_opt([p])
    for _ in range(2000):
        quadratic_step(p, [5.0, 5.0], opt)
    p.set_mask(None)
    quadratic_step(p, [5.0, 5.0], opt)
    return p.data[1]


def test_global_bias_correction_gives_a_revived_weight_a_3x_first_step():
    """m = v = 0 after revival, but the global step is 2001, so there's no bias
    correction: step ~= lr * (1 - b1) / sqrt(1 - b2) ~= 3.16 lr."""
    step = _first_step_after_late_revival(lambda ps: TextbookAdam(ps, reapply_mask=True))
    assert step > 2.5 * LR
    limit = (1 - 0.9) / math.sqrt(1 - 0.999)
    expected = LR * limit * math.sqrt(1 - 0.999**2001) / (1 - 0.9**2001)
    assert step == pytest.approx(expected, rel=1e-6)


def test_revived_weight_takes_a_normal_first_adam_step():
    step = _first_step_after_late_revival(lambda ps: Adam(ps, lr=LR))
    assert step == pytest.approx(LR, rel=1e-6)


def _revive_after_the_loss_surface_moved(make_opt):
    p = Parameter([0.0])
    opt = make_opt([p])
    for _ in range(50):
        quadratic_step(p, [5.0], opt)  # history: gradient < 0, the weight climbs
    p.set_mask([False])
    for _ in range(3):
        quadratic_step(p, [-5.0], opt)  # pruned for a few steps
    p.set_mask(None)
    quadratic_step(p, [-5.0], opt)  # now dL/dw = 0 - (-5) = +5: the weight should go *down*
    return p.data[0]


def test_stale_first_moment_moves_a_revived_weight_against_its_gradient():
    assert _revive_after_the_loss_surface_moved(lambda ps: TextbookAdam(ps, reapply_mask=True)) > 0


@all_optimizers
def test_revived_weight_follows_its_current_gradient(make_opt):
    assert _revive_after_the_loss_surface_moved(make_opt) < 0


# --- revival == a fresh parameter with a fresh optimizer -------------------------


@all_optimizers
@pytest.mark.parametrize("masked_steps", [0, 1, 25])
def test_revived_connection_trains_exactly_like_a_new_parameter(make_opt, masked_steps):
    """masked_steps=0 prunes and revives between two optimizer steps -- the mask ends
    up unchanged, but the weight was zeroed, so its state must still be reset."""
    target = np.array([2.0, -3.0, 1.5])

    p = Parameter([1.0, -2.0, 0.5])
    opt = make_opt([p])
    for _ in range(40):
        quadratic_step(p, target, opt)
    p.set_mask([True, False, True])
    for _ in range(masked_steps):
        quadratic_step(p, target, opt)
    p.set_mask(None)
    assert p.data[1] == 0.0

    fresh = Parameter([0.0])
    fresh_opt = make_opt([fresh])
    for _ in range(60):
        quadratic_step(p, target, opt)
        quadratic_step(fresh, target[1:2], fresh_opt)
        np.testing.assert_allclose(p.data[1], fresh.data[0], rtol=0, atol=1e-14)


# --- everything at once ------------------------------------------------------------


def test_training_an_mlp_through_prune_and_regrow_events():
    rng = np.random.default_rng(0)
    centres = np.array([[2.0, 0.0], [-1.0, 2.0], [-1.0, -2.0]])
    y = rng.integers(0, 3, size=300)
    x = Tensor(centres[y] + 0.5 * rng.standard_normal((300, 2)))

    w1 = Parameter(rng.standard_normal((2, 16)) * np.sqrt(2 / 2))
    b1 = Parameter(np.zeros(16))
    w2 = Parameter(rng.standard_normal((16, 3)) * np.sqrt(1 / 16))
    b2 = Parameter(np.zeros(3))
    weights = [w1, w2]
    opt = Adam([w1, b1, w2, b2], lr=0.02, weight_decay=1e-3)

    def loss_fn():
        hidden = (x @ w1.effective() + b1).relu()
        return softmax_cross_entropy(hidden @ w2.effective() + b2, y)

    losses = []
    for step in range(300):
        if step == 100:  # prune the 60% smallest-magnitude weights of each layer
            for w in weights:
                w.set_mask(np.abs(w.data) > np.quantile(np.abs(w.data), 0.6))
        if step == 200:  # regrow the 5 masked connections with the largest dense gradient, drop 5 others
            for w in weights:
                mask = w.mask.copy()
                score = np.where(mask, -np.inf, np.abs(w.dense_grad))
                grow = np.unravel_index(np.argsort(score, axis=None)[-5:], mask.shape)
                keep = np.where(mask, np.abs(w.data), np.inf)
                drop = np.unravel_index(np.argsort(keep, axis=None)[:5], mask.shape)
                mask[grow], mask[drop] = True, False
                w.set_mask(mask)

        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        assert np.isfinite(loss.item())
        for w in weights:
            if w.mask is not None:
                off = ~w.mask
                assert (w.data[off] == 0).all() and (w.grad[off] == 0).all()
                index = opt.params.index(w)
                assert (opt.m[index][off] == 0).all() and (opt.v[index][off] == 0).all()

    assert all(w.sparsity == pytest.approx(0.6, abs=0.03) for w in weights)
    assert losses[-1] < 0.25 * losses[0]
    assert losses[-1] < 0.2  # still solves the task at 60% sparsity


def test_float32_parameters_stay_float32():
    for make_opt in OPTIMIZERS.values():
        p = Parameter(np.ones(4, dtype=np.float32), mask=[True, False, True, True])
        opt = make_opt([p])
        for _ in range(3):
            opt.zero_grad()
            (p.effective() * p.effective()).sum().backward()
            opt.step()
        p.set_mask(None)
        opt.zero_grad()
        (p.effective() * p.effective()).sum().backward()
        opt.step()
        assert p.data.dtype == np.float32 and p.grad.dtype == np.float32
        if isinstance(opt, Adam):
            assert opt.m[0].dtype == np.float32 and opt.v[0].dtype == np.float32
