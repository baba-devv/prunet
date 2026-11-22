"""Importance criteria, schedules and the pruner."""

import math

import numpy as np
import pytest

from engine import Parameter, Tensor, softmax_cross_entropy
from nn import Adam
from prune import CRITERIA, GradStats, Pruner, cubic, dense_gradient, linear, one_shot, regrowth_score, select


def rng(seed=0):
    return np.random.default_rng(seed)


# --- schedules ---------------------------------------------------------------


def test_cubic_schedule_endpoints_shape_and_monotonicity():
    start, end, final = 100, 1100, 0.9
    assert cubic(0, start, end, final) == 0.0 and cubic(start, start, end, final) == 0.0
    assert cubic(end, start, end, final) == final and cubic(end + 500, start, end, final) == final
    assert cubic(600, start, end, final) == pytest.approx(final * (1 - 0.5**3))  # 7/8 of the way at half time
    values = [cubic(t, start, end, final) for t in range(start, end + 1)]
    steps = np.diff(values)
    assert (steps >= 0).all()
    assert (np.diff(steps) <= 1e-12).all()  # the pruning rate only ever decreases ...
    assert steps[-1] < 1e-3 * steps[0]  # ... and is ~0 on arrival


def test_linear_and_one_shot_schedules():
    assert linear(600, 100, 1100, 0.8) == pytest.approx(0.4)
    assert linear(50, 100, 1100, 0.8) == 0.0 and linear(2000, 100, 1100, 0.8) == 0.8
    assert one_shot(1099, 100, 1100, 0.8) == 0.0 and one_shot(1100, 100, 1100, 0.8) == 0.8


# --- gradient statistics -------------------------------------------------------


def test_gradstats_running_mean_is_exact():
    grads = rng().standard_normal((50, 3, 2))
    weights = rng(1).standard_normal((50, 3, 2))
    stats = GradStats((3, 2), batch_size=4, decay=None)
    for g, w in zip(grads, weights):
        stats.update(g, w)
    np.testing.assert_allclose(stats.mean_g, grads.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats.mean_g2, (grads**2).mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats.mean_gw, (grads * weights).mean(axis=0), rtol=1e-12)


def test_gradstats_ema_is_bias_corrected_and_tracks_recent_values():
    stats = GradStats((2,), batch_size=1, decay=0.9)
    stats.update(np.array([3.0, -1.0]), np.array([2.0, 2.0]))
    np.testing.assert_allclose(stats.mean_g, [3.0, -1.0])  # one sample: the estimate is that sample, not 0.1x it
    np.testing.assert_allclose(stats.mean_g2, [9.0, 1.0])
    np.testing.assert_allclose(stats.mean_gw, [6.0, -2.0])
    for _ in range(200):
        stats.update(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(stats.mean_g, [1.0, 1.0], rtol=1e-6)  # old values are forgotten


def test_fisher_is_recovered_from_batch_gradients():
    """E[g_sample^2] from batch means: mean^2 + B * Var(batch mean). Checked against per-sample data."""
    r = rng()
    batch, n_batches = 16, 4000
    per_sample = 0.3 + 2.0 * r.standard_normal((n_batches, batch, 5)) * np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    stats = GradStats((5,), batch_size=batch, decay=None)
    for b in per_sample:
        stats.update(b.mean(axis=0), np.ones(5))
    truth = (per_sample**2).mean(axis=(0, 1))
    np.testing.assert_allclose(stats.fisher, truth, rtol=0.05)
    # Using E[g_batch^2] directly would be off by roughly the batch size where noise dominates.
    assert stats.mean_g2[-1] < truth[-1] / 10


# --- criteria -------------------------------------------------------------------


def make_stats(g, g2, gw, batch_size=1):
    stats = GradStats(np.shape(g), batch_size=batch_size, decay=None)
    stats.steps = 1
    stats._g, stats._g2, stats._gw = (np.asarray(a, dtype=np.float64) for a in (g, g2, gw))
    return stats


def test_criteria_formulas():
    w = np.array([2.0, -1.0, 0.5])
    g = np.array([0.1, 0.2, -0.4])
    stats = make_stats(g, g * g + 0.01, g * w)  # batch_size 1: fisher == E[g^2]
    np.testing.assert_allclose(stats.fisher, g * g + 0.01)
    np.testing.assert_allclose(CRITERIA["magnitude"].score(w, None), 0.5 * w**2)
    np.testing.assert_allclose(CRITERIA["fisher"].score(w, stats), 0.5 * (g * g + 0.01) * w**2)
    np.testing.assert_allclose(CRITERIA["taylor1"].score(w, stats), np.abs(g * w))
    np.testing.assert_allclose(CRITERIA["movement"].score(w, stats), -g * w)
    np.testing.assert_allclose(CRITERIA["taylor2"].score(w, stats), -g * w + 0.5 * (g * g + 0.01) * w**2)
    np.testing.assert_allclose(regrowth_score(stats), np.abs(g))
    assert not CRITERIA["magnitude"].needs_gradients and CRITERIA["fisher"].needs_gradients


def test_second_order_taylor_is_exact_on_a_quadratic():
    """For L = 1/2 sum h_i (w_i - a_i)^2 removing weight i changes the loss by exactly
    -g_i w_i + 1/2 h_i w_i^2: the expansion every criterion is derived from."""
    r = rng()
    h, a, w = r.uniform(0.1, 3.0, 6), r.standard_normal(6), r.standard_normal(6)
    loss = lambda v: 0.5 * np.sum(h * (v - a) ** 2)  # noqa: E731
    g = h * (w - a)
    for i in range(6):
        removed = w.copy()
        removed[i] = 0.0
        assert loss(removed) - loss(w) == pytest.approx(-g[i] * w[i] + 0.5 * h[i] * w[i] ** 2, rel=1e-12, abs=1e-12)


def _softmax_regression(scales):
    """A converged softmax-regression model whose input features have very different scales."""
    r = rng(3)
    n, d, c = 4000, len(scales), 3
    x = r.standard_normal((n, d)) * scales
    true_w = r.standard_normal((d, c)) / np.sqrt(scales)[:, None]  # large-scale features: small weights, big effect
    y = np.array([r.choice(c, p=p) for p in np.exp(x @ true_w) / np.exp(x @ true_w).sum(axis=1, keepdims=True)])
    w = Parameter(np.zeros((d, c)))
    opt = Adam([w], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        softmax_cross_entropy(Tensor(x) @ w, y).backward()
        opt.step()
    return x, y, w


def test_fisher_predicts_single_connection_loss_changes_where_magnitude_cannot():
    """With input scales spanning 100x, curvature differs by 10^4 between weights, so |w| is the
    wrong yardstick. The Fisher score should match the *measured* loss change, value for value."""
    scales = np.array([0.1, 0.1, 1.0, 1.0, 10.0, 10.0])
    x, y, w = _softmax_regression(scales)
    full_loss = lambda: softmax_cross_entropy(Tensor(x) @ w, y).item()  # noqa: E731
    base = full_loss()

    stats = GradStats(w.shape, batch_size=100, decay=None)
    for start in range(0, len(x), 100):
        w.zero_grad()
        softmax_cross_entropy(Tensor(x[start : start + 100]) @ w, y[start : start + 100]).backward()
        stats.update(dense_gradient(w), w.data)

    measured = np.zeros(w.shape)
    for index in np.ndindex(w.shape):
        saved = w.data[index]
        w.data[index] = 0.0
        measured[index] = full_loss() - base
        w.data[index] = saved

    predicted = CRITERIA["fisher"].score(w.data, stats)
    assert (measured > 0).all()  # converged: removing any weight hurts
    ratio = predicted / measured
    assert 0.5 < np.median(ratio) < 2.0 and ratio.min() > 0.2 and ratio.max() < 5.0

    order = lambda a: np.argsort(np.argsort(a.ravel()))  # noqa: E731
    rank_corr = lambda a, b: np.corrcoef(order(a), order(b))[0, 1]  # noqa: E731
    assert rank_corr(predicted, measured) > 0.9
    assert rank_corr(CRITERIA["magnitude"].score(w.data, None), measured) < 0.3


# --- select -------------------------------------------------------------------


def test_select_picks_exactly_k_across_tensors():
    scores = [np.array([[5.0, 1.0], [3.0, 9.0]]), np.array([2.0, 8.0, 0.5])]
    candidates = [np.array([[True, True], [False, True]]), np.array([True, True, True])]
    lowest = select(scores, candidates, 3, largest=False)
    np.testing.assert_array_equal(lowest[0], [[False, True], [False, False]])  # 1.0 (3.0 is not a candidate)
    np.testing.assert_array_equal(lowest[1], [True, False, True])  # 2.0 and 0.5
    largest = select(scores, candidates, 2, largest=True)
    np.testing.assert_array_equal(largest[0], [[False, False], [False, True]])
    np.testing.assert_array_equal(largest[1], [False, True, False])
    assert sum(int(c.sum()) for c in select(scores, candidates, 0, largest=False)) == 0
    with pytest.raises(ValueError):
        select(scores, candidates, 7, largest=False)


def test_select_returns_exactly_k_even_when_scores_tie():
    scores, candidates = [np.zeros(100)], [np.ones(100, dtype=bool)]
    assert int(select(scores, candidates, 37, largest=False)[0].sum()) == 37


# --- pruner -----------------------------------------------------------------------


def make_params(seed=0, shapes=((20, 10), (10, 5))):
    r = rng(seed)
    return [Parameter(r.standard_normal(s)) for s in shapes]


def mask_of(p):
    return np.ones(p.shape, dtype=bool) if p.mask is None else p.mask


def on_budget(event):
    """Achieved sparsity is at or above the schedule (up to float rounding), by less than one connection."""
    gap = event["sparsity"] - event["scheduled_sparsity"]
    return -1e-12 <= gap < 1 / 250 + 1e-12


def drive(pruner, steps, params=None, grad_fn=None):
    """Run the pruner's part of a training loop; optionally provide synthetic gradients."""
    for step in range(steps):
        if pruner.is_update_step(step):
            pruner.update(step)
        if grad_fn is not None:
            for p in params:
                p.zero_grad()
                grad_fn(p).backward()
            pruner.after_backward()


def test_budget_is_met_exactly_and_never_exceeded():
    assert Pruner.budget(1000, 0.9) == 100  # (1 - 0.9) * 1000 is 99.99999999999997 in floating point
    assert Pruner.budget(250, 0.9) == 25 and Pruner.budget(7, 0.5) == 3 and Pruner.budget(10, 0.0) == 10
    for target in (0.5, 0.9, 0.97):
        params = make_params()
        pruner = Pruner(params, "magnitude", target, start_step=10, end_step=110, interval=7)
        drive(pruner, 150)
        assert pruner.num_active == Pruner.budget(250, target)
        assert pruner.sparsity >= target
        assert pruner.sparsity - target < 1 / 250  # within one connection of the target


def test_sparsity_follows_the_schedule_and_the_mask_freezes_afterwards():
    params = make_params()
    pruner = Pruner(params, "magnitude", 0.8, start_step=10, end_step=110, interval=10)
    drive(pruner, 200)
    steps = [e["step"] for e in pruner.events]
    assert steps == list(range(10, 111, 10))  # nothing before start, nothing after end
    assert all(on_budget(event) for event in pruner.events)
    sparsities = [e["sparsity"] for e in pruner.events]
    assert sparsities == sorted(sparsities) and sparsities[-1] == pytest.approx(0.8)
    assert not pruner.is_update_step(111) and not pruner.is_update_step(9)


def test_final_update_happens_even_when_the_interval_does_not_divide_the_ramp():
    pruner = Pruner(make_params(), "magnitude", 0.9, start_step=0, end_step=95, interval=30)
    drive(pruner, 100)
    assert [e["step"] for e in pruner.events] == [0, 30, 60, 90, 95]
    assert pruner.num_active == 25


def test_without_regrowth_masks_are_nested_and_magnitude_keeps_the_largest_weights():
    params = make_params()
    original = [p.data.copy() for p in params]
    pruner = Pruner(params, "magnitude", 0.9, start_step=0, end_step=50, interval=10)
    previous = [np.ones(p.shape, dtype=bool) for p in params]
    for step in range(60):
        if pruner.is_update_step(step):
            pruner.update(step)
            for p, before in zip(params, previous):
                assert not (mask_of(p) & ~before).any()  # nothing ever comes back
            previous = [mask_of(p).copy() for p in params]
    kept = np.concatenate([np.abs(o[p.mask]) for o, p in zip(original, params)])
    dropped = np.concatenate([np.abs(o[~p.mask]) for o, p in zip(original, params)])
    assert kept.min() >= dropped.max()  # global magnitude ranking on untouched weights
    assert pruner.regrowth_summary()["regrow_events"] == 0
    for p in params:
        assert (p.data[~p.mask] == 0).all() and (p.effective().data[~p.mask] == 0).all()


def test_layer_scope_prunes_every_tensor_to_the_same_sparsity_and_global_does_not():
    r = rng()
    params = [Parameter(r.standard_normal((20, 10)) * 0.01), Parameter(r.standard_normal((10, 5)) * 10.0)]
    per_layer = Pruner(params, "magnitude", 0.8, start_step=0, end_step=10, interval=5, scope="layer")
    drive(per_layer, 11)
    assert per_layer.layer_sparsity() == pytest.approx([0.8, 0.8])

    params = [Parameter(r.standard_normal((20, 10)) * 0.01), Parameter(r.standard_normal((10, 5)) * 10.0)]
    overall = Pruner(params, "magnitude", 0.8, start_step=0, end_step=10, interval=5, scope="global")
    drive(overall, 11)
    first, second = overall.layer_sparsity()
    assert first > 0.98 and second < 0.1  # magnitude is not comparable across layers: one is wiped out
    assert overall.sparsity == pytest.approx(0.8)


def test_gradient_criteria_use_the_dense_gradient_statistics():
    """Two weights of equal magnitude; only one has gradient signal. Fisher must keep that one."""
    w = Parameter(np.array([1.0, 1.0, 1.0, 1.0]))
    sensitivity = np.array([10.0, 0.1, 5.0, 0.0])
    pruner = Pruner([w], "fisher", 0.5, start_step=20, end_step=20, interval=1, batch_size=1, stats_decay=None)
    signs = rng().choice([-1.0, 1.0], size=200)
    for step in range(21):
        if pruner.is_update_step(step):
            pruner.update(step)
        w.zero_grad()
        (w.effective() * Tensor(sensitivity * signs[step])).sum().backward()
        pruner.after_backward()
    np.testing.assert_array_equal(w.mask, [True, False, True, False])


def test_unknown_options_are_rejected():
    params = make_params()
    with pytest.raises(KeyError):
        Pruner(params, "no_such_criterion")
    with pytest.raises(KeyError):
        Pruner(params, schedule="no_such_schedule")
    with pytest.raises(ValueError):
        Pruner(params, scope="neuron")
    with pytest.raises(ValueError):
        Pruner(params, target_sparsity=1.0)


# --- regrowth --------------------------------------------------------------------


def test_regrowth_swaps_keep_the_budget_and_revive_the_highest_mean_gradient_at_zero():
    n = 40
    w = Parameter(np.linspace(4.0, 0.1, n))  # index 0 is the largest weight, index 39 the smallest
    pull = np.zeros(n)
    pull[28:] = [0.3, 2.0, 0.1, 5.0, 0.2, 0.05, 3.0, 0.4, 0.01, 4.0, 0.02, 0.03]  # mean dense gradient magnitudes
    pruner = Pruner(
        [w], "magnitude", 0.5, start_step=0, end_step=100, interval=25, regrow_fraction=0.4, batch_size=1, stats_decay=None
    )

    def step_gradients():
        w.zero_grad()
        (w.effective() * Tensor(pull)).sum().backward()  # dense gradient == pull, also at masked coordinates
        pruner.after_backward()

    pruner.update(0)  # scheduled sparsity 0: nothing to prune, nothing masked to regrow
    assert w.mask is None and pruner.events[-1]["regrown"] == 0
    step_gradients()

    pruner.update(25)  # budget floor((1 - 0.289) * 40) = 28: the 12 smallest go; nothing was masked before, so no regrowth
    assert pruner.events[-1]["pruned"] == 12 and pruner.events[-1]["regrown"] == 0
    assert w.mask[:28].all() and not w.mask[28:].any()
    step_gradients()

    # Budget floor((1 - 0.4375) * 40) = 22: prune 6 more (22..27). Regrow fraction 0.4 * 0.5 = 0.2 -> k = 4:
    # the 4 weakest survivors (18..21) are swapped for the 4 masked connections with the largest pull.
    pruner.update(50)
    event = pruner.events[-1]
    assert event["regrown"] == 4 and event["pruned"] == 10 and w.num_active == 22
    revived = {31, 37, 34, 29}  # pulls 5.0, 4.0, 3.0, 2.0
    assert set(np.flatnonzero(w.mask)) == set(range(18)) | revived
    assert (w.data[sorted(revived)] == 0.0).all()  # a revived connection restarts from exactly zero
    assert not w.mask[22:28].any()  # connections pruned in this very update are not candidates for regrowth
    summary = pruner.regrowth_summary()
    assert summary["regrow_events"] == 4 and summary["regrown_and_alive_at_end"] == 4


def test_regrowth_fraction_decays_to_zero_so_the_last_update_only_prunes():
    params = make_params()
    pruner = Pruner(params, "magnitude", 0.8, start_step=0, end_step=100, interval=10, regrow_fraction=0.3, stats_decay=0.9)
    drive(pruner, 101, params, grad_fn=lambda p: (p.effective() * p.effective()).sum())
    regrown = [e["regrown"] for e in pruner.events]
    assert regrown[0] == 0 and max(regrown) > 0 and regrown[-1] == 0
    assert pruner.num_active == Pruner.budget(250, 0.8)
    assert all(on_budget(event) for event in pruner.events)  # the budget holds at every update, swaps included
