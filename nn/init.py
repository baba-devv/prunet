"""Weight initialisation for `Linear` layers (weights have shape (fan_in, fan_out)).

With zero-mean i.i.d. weights, Var(z) = fan_in * Var(w) * E[x^2].

* ReLU keeps half the second moment, so Var(w) = 2 / fan_in preserves the
  signal through depth (He). With 1 / fan_in it halves at every layer.
* tanh / sigmoid are ~linear near 0: Xavier, 2 / (fan_in + fan_out).
* The logit layer has no non-linearity after it: Var(w) = 1 / fan_in (LeCun),
  giving unit-scale logits and an initial loss near ln(C).

Biases start at 0.
"""

from __future__ import annotations

import math

import numpy as np


def he_normal(rng: np.random.Generator, fan_in: int, fan_out: int, dtype=np.float64) -> np.ndarray:
    return (rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)).astype(dtype)


def lecun_normal(rng: np.random.Generator, fan_in: int, fan_out: int, dtype=np.float64) -> np.ndarray:
    return (rng.standard_normal((fan_in, fan_out)) * math.sqrt(1.0 / fan_in)).astype(dtype)


def xavier_normal(rng: np.random.Generator, fan_in: int, fan_out: int, dtype=np.float64) -> np.ndarray:
    return (rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / (fan_in + fan_out))).astype(dtype)


INITIALIZERS = {"he": he_normal, "lecun": lecun_normal, "xavier": xavier_normal}

# Which initialiser preserves the signal through a layer followed by this activation.
INIT_FOR_ACTIVATION = {"relu": "he", "gelu": "he", "tanh": "xavier", "sigmoid": "xavier"}
