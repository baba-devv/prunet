"""Composable layers built only from engine operations."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from engine import Parameter, Tensor

from .init import INIT_FOR_ACTIVATION, INITIALIZERS


class Module:
    """A callable that owns Parameters, directly or through sub-modules.

    Parameters are discovered by walking instance attributes in definition
    order, so the order of `parameters()` is deterministic.
    """

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x) -> Tensor:
        return self.forward(x if isinstance(x, Tensor) else Tensor(x))

    def named_parameters(self, prefix: str = "") -> Iterator[tuple]:
        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                yield prefix + name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{name}.")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{index}.")

    def parameters(self) -> list:
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """y = x @ W + b, with W of shape (in_features, out_features).

    Uses `weight.effective()`, so pruned connections are absent from the
    forward pass and get zero gradient. Biases are never pruned.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        init: str = "he",
        bias: bool = True,
        dtype=np.float64,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(INITIALIZERS[init](rng, in_features, out_features, dtype), name="weight")
        self.bias = Parameter(np.zeros(out_features, dtype=dtype), name="bias") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.effective()
        return out if self.bias is None else out + self.bias


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid, "gelu": GELU}


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Sequential):
    """Linear -> activation -> ... -> Linear, returning logits.

    `sizes` = [inputs, hidden..., classes]. Hidden layers use the initialiser
    matched to the activation; the output layer uses LeCun.
    """

    def __init__(self, sizes: Sequence[int], rng: np.random.Generator, activation: str = "relu", dtype=np.float64):
        if len(sizes) < 2:
            raise ValueError("sizes must contain at least an input and an output size")
        layers = []
        last = len(sizes) - 2
        for index, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
            init = "lecun" if index == last else INIT_FOR_ACTIVATION[activation]
            layers.append(Linear(fan_in, fan_out, rng, init=init, dtype=dtype))
            if index != last:
                layers.append(ACTIVATIONS[activation]())
        super().__init__(*layers)
        self.sizes = list(sizes)
        self.activation = activation

    @property
    def linears(self) -> list:
        return [layer for layer in self.layers if isinstance(layer, Linear)]
