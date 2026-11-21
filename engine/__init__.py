"""A minimal reverse-mode autodiff engine on NumPy, with pruning-mask support."""

from .functional import log_softmax, softmax, softmax_cross_entropy
from .gradcheck import GradCheckError, GradCheckResult, gradcheck, numerical_gradient
from .parameter import Parameter
from .tensor import Tensor, no_grad, unbroadcast

__all__ = [
    "Tensor",
    "Parameter",
    "no_grad",
    "unbroadcast",
    "log_softmax",
    "softmax",
    "softmax_cross_entropy",
    "gradcheck",
    "numerical_gradient",
    "GradCheckError",
    "GradCheckResult",
]
