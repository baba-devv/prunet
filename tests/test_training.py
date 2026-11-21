"""The training loop: it learns, it is reproducible, and it fails loudly on non-finite numbers."""

import math

import numpy as np
import pytest

from engine import softmax_cross_entropy
from nn import MLP, SGD, Adam
from train.data import load_dataset
from train.loop import LR_SCHEDULES, evaluate, iterate_minibatches, train
from train.train_mlp import run


def strip_timing(history):
    return [{k: v for k, v in h.items() if k != "seconds"} for h in history]


def test_minibatches_cover_every_example_exactly_once():
    x, y = np.arange(23).reshape(23, 1), np.arange(23)
    batches = list(iterate_minibatches(x, y, 5, np.random.default_rng(0)))
    assert [len(xb) for xb, _ in batches] == [5, 5, 5, 5, 3]
    seen = np.concatenate([yb for _, yb in batches])
    assert sorted(seen) == list(range(23)) and list(seen) != list(range(23))
    assert all((xb[:, 0] == yb).all() for xb, yb in batches)  # inputs and labels stay aligned


def test_evaluate_matches_a_direct_computation_and_builds_no_graph():
    rng = np.random.default_rng(0)
    model = MLP([4, 8, 3], rng)
    x, y = rng.standard_normal((50, 4)), rng.integers(0, 3, size=50)
    loss, acc = evaluate(model, x, y, batch_size=7)  # uneven batches must not change the mean
    logits = model(x)
    assert loss == pytest.approx(softmax_cross_entropy(logits, y).item(), rel=1e-12)
    assert acc == (logits.data.argmax(axis=1) == y).mean()
    assert all(p.grad is None for p in model.parameters())


def test_an_mlp_can_memorise_a_small_random_batch():
    """Capacity + optimizer sanity check: random labels can only be fit by actually learning them."""
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((32, 10)), rng.integers(0, 4, size=32)
    model = MLP([10, 64, 4], rng)
    opt = Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = softmax_cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2
    assert evaluate(model, x, y)[1] == 1.0


def test_training_on_spirals_learns_a_nonlinear_boundary_and_stays_finite():
    result = run("spirals", hidden=[64, 64], epochs=60, batch_size=64, lr=3e-3, seed=0, log=None)
    history = result["history"]
    assert len(history) == 61 and history[0]["epoch"] == 0
    # untrained: chance accuracy and a loss near ln(3)
    assert history[0]["val_acc"] < 0.5 and math.log(3) < history[0]["val_loss"] < math.log(3) + 1
    assert history[-1]["val_acc"] > 0.95 and result["test"]["acc"] > 0.95
    assert history[-1]["train_loss"] < 0.1 * history[0]["train_loss"]
    numbers = [h[k] for h in history[1:] for k in ("train_loss", "val_loss", "grad_norm_mean", "grad_norm_max")]
    assert np.isfinite(numbers).all()
    # cosine schedule: the learning rate only ever goes down, and ends near zero
    lrs = [h["lr"] for h in history[1:]]
    assert lrs == sorted(lrs, reverse=True) and lrs[-1] < 0.01 * 3e-3


def test_runs_are_reproducible_from_the_seed():
    settings = dict(hidden=[16], epochs=3, batch_size=64, lr=3e-3, log=None)
    first, second = run("spirals", seed=1, **settings), run("spirals", seed=1, **settings)
    other = run("spirals", seed=2, **settings)
    assert strip_timing(first["history"]) == strip_timing(second["history"])
    assert first["test"] == second["test"]
    assert strip_timing(first["history"]) != strip_timing(other["history"])
    # the data split is fixed; only initialisation and batch order depend on the run seed
    assert first["config"]["train_examples"] == other["config"]["train_examples"]


def test_a_diverging_run_raises_instead_of_training_on_nans():
    data = load_dataset("spirals")
    model = MLP([2, 32, 3], np.random.default_rng(0), dtype=np.float32)
    optimizer = SGD(model.parameters(), lr=1e12)
    with pytest.raises(FloatingPointError):
        train(model, optimizer, data, epochs=5, batch_size=64, rng=np.random.default_rng(0), log=None)


def test_a_nan_that_raises_no_floating_point_flag_is_still_caught():
    """A NaN already present in the data propagates quietly (no overflow/invalid flag), so the
    loop must also check the loss itself."""
    data = load_dataset("spirals")
    data.x_train[0, 0] = np.nan
    model = MLP([2, 32, 3], np.random.default_rng(0), dtype=np.float32)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(model, Adam(model.parameters()), data, epochs=1, batch_size=64, rng=np.random.default_rng(0), log=None)


def test_lr_schedules():
    assert LR_SCHEDULES["constant"](0.7) == 1.0
    cosine = LR_SCHEDULES["cosine"]
    assert cosine(0.0) == 1.0 and cosine(0.5) == pytest.approx(0.5) and cosine(1.0) == pytest.approx(0.0, abs=1e-12)
