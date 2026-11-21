"""Layers and initialisation."""

import math

import numpy as np
import pytest

from engine import Parameter, Tensor, gradcheck, softmax_cross_entropy
from nn import MLP, Linear, ReLU, Sequential, he_normal, lecun_normal, xavier_normal
from nn.init import INITIALIZERS


def rng(seed=0):
    return np.random.default_rng(seed)


# --- layers ------------------------------------------------------------------


def test_linear_computes_x_at_w_plus_b():
    layer = Linear(4, 3, rng())
    layer.bias.data[:] = [1.0, 2.0, 3.0]
    x = rng(1).standard_normal((5, 4))
    out = layer(x)  # ndarray input is wrapped
    assert isinstance(out, Tensor) and out.shape == (5, 3)
    np.testing.assert_allclose(out.data, x @ layer.weight.data + layer.bias.data)
    assert Linear(4, 3, rng(), bias=False).parameters()[0].shape == (4, 3)


def test_parameters_are_discovered_recursively_in_definition_order():
    model = MLP([4, 8, 6, 3], rng())
    names = [name for name, _ in model.named_parameters()]
    assert names == [
        "layers.0.weight", "layers.0.bias",
        "layers.2.weight", "layers.2.bias",
        "layers.4.weight", "layers.4.bias",
    ]  # fmt: skip
    assert all(isinstance(p, Parameter) for p in model.parameters())
    assert sum(p.size for p in model.parameters()) == (4 * 8 + 8) + (8 * 6 + 6) + (6 * 3 + 3)
    assert [layer.weight.shape for layer in model.linears] == [(4, 8), (8, 6), (6, 3)]


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu"])
def test_mlp_gradients_match_finite_differences(activation):
    model = MLP([5, 7, 6, 3], rng(), activation=activation)
    for p in model.parameters():  # non-zero biases, and keep ReLU inputs off the kink
        p.data += 0.1 * rng(2).standard_normal(p.shape)
    x, y = rng(3).standard_normal((6, 5)), rng(4).integers(0, 3, size=6)
    gradcheck(lambda *_: softmax_cross_entropy(model(x), y), model.parameters())


def test_linear_forward_and_gradient_respect_the_mask():
    layer = Linear(4, 3, rng())
    mask = rng(1).random((4, 3)) < 0.5
    layer.weight.set_mask(mask)
    x = rng(2).standard_normal((5, 4))
    out = layer(x)
    np.testing.assert_allclose(out.data, x @ np.where(mask, layer.weight.data, 0.0))
    out.sum().backward()
    assert (layer.weight.grad[~mask] == 0).all()
    assert layer.bias.mask is None  # biases are never pruned


def test_zero_grad_clears_every_parameter():
    model = MLP([4, 8, 3], rng())
    softmax_cross_entropy(model(rng(1).standard_normal((5, 4))), np.zeros(5, dtype=int)).backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_same_seed_gives_identical_models_and_dtype_is_respected():
    a, b, c = MLP([4, 8, 3], rng(7)), MLP([4, 8, 3], rng(7)), MLP([4, 8, 3], rng(8))
    for pa, pb in zip(a.parameters(), b.parameters()):
        np.testing.assert_array_equal(pa.data, pb.data)
    assert not np.array_equal(a.parameters()[0].data, c.parameters()[0].data)
    model32 = MLP([4, 8, 3], rng(), dtype=np.float32)
    assert all(p.dtype == np.float32 for p in model32.parameters())
    assert model32(np.ones((2, 4), dtype=np.float32)).dtype == np.float32


def test_mlp_needs_at_least_two_sizes():
    with pytest.raises(ValueError):
        MLP([4], rng())


# --- initialisation ------------------------------------------------------------


@pytest.mark.parametrize(
    "init, expected_std",
    [(he_normal, math.sqrt(2 / 400)), (lecun_normal, math.sqrt(1 / 400)), (xavier_normal, math.sqrt(2 / 700))],
)
def test_initialiser_scale(init, expected_std):
    w = init(rng(), 400, 300)
    assert w.shape == (400, 300)
    assert abs(w.mean()) < 5 * expected_std / math.sqrt(w.size)  # zero mean, within 5 standard errors
    assert w.std() == pytest.approx(expected_std, rel=0.02)


def _relu_stack(init: str, depth: int = 20, width: int = 256) -> Sequential:
    layers = []
    for _ in range(depth):
        layers += [Linear(width, width, rng(len(layers)), init=init), ReLU()]
    return Sequential(*layers)


def _second_moment_gain(init: str) -> float:
    """E[h_out^2] / E[x^2] through a 20-layer ReLU stack at initialisation."""
    x = rng(99).standard_normal((512, 256))
    out = _relu_stack(init)(x)
    return float((out.data**2).mean() / (x**2).mean())


def test_he_init_preserves_the_forward_signal_through_a_deep_relu_stack():
    assert 0.3 < _second_moment_gain("he") < 3.0


def test_lecun_init_halves_the_signal_at_every_relu_layer():
    """Var(w) = 1/fan_in loses the factor 2 that ReLU removes: gain ~ 2^-20 over 20 layers."""
    gain = _second_moment_gain("lecun")
    assert 0.3 * 2.0**-20 < gain < 3.0 * 2.0**-20


def test_he_init_keeps_gradients_the_same_scale_at_both_ends_of_the_stack():
    model = _relu_stack("he")
    x = rng(99).standard_normal((512, 256))
    (model(x) ** 2).mean().backward()
    first, last = model.layers[0].weight.grad, model.layers[-2].weight.grad
    ratio = np.linalg.norm(first) / np.linalg.norm(last)
    assert 0.2 < ratio < 5.0

    shrinking = _relu_stack("lecun")
    (shrinking(x) ** 2).mean().backward()
    assert np.linalg.norm(shrinking.layers[0].weight.grad) < 1e-4 * np.linalg.norm(first)


def test_mlp_uses_he_for_hidden_layers_and_lecun_for_the_logits():
    hidden, output = MLP([500, 400, 300], rng()).linears
    assert hidden.weight.data.std() == pytest.approx(math.sqrt(2 / 500), rel=0.02)
    assert output.weight.data.std() == pytest.approx(math.sqrt(1 / 400), rel=0.02)
    assert (hidden.bias.data == 0).all() and (output.bias.data == 0).all()
    tanh_hidden = MLP([500, 400, 300], rng(), activation="tanh").linears[0]
    assert tanh_hidden.weight.data.std() == pytest.approx(math.sqrt(2 / 900), rel=0.02)


def test_initial_loss_is_close_to_log_num_classes():
    """Unit-scale logits: the untrained loss is ln(C) plus a fraction of a nat, i.e. no saturated softmax."""
    x = rng(1).standard_normal((2000, 784))
    y = rng(2).integers(0, 10, size=2000)
    loss = softmax_cross_entropy(MLP([784, 256, 128, 10], rng(3))(x), y).item()
    assert math.log(10) < loss < math.log(10) + 0.6


def test_initialiser_registry_matches_the_functions():
    assert INITIALIZERS == {"he": he_normal, "lecun": lecun_normal, "xavier": xavier_normal}
