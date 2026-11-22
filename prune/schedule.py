"""Sparsity schedules: target fraction of connections removed as a function of the step."""

from __future__ import annotations


def cubic(step: int, start: int, end: int, final: float) -> float:
    """Zhu & Gupta (2017):  s(t) = s_f * (1 - (1 - (t - t0) / (t1 - t0))^3)  for t0 <= t <= t1.

    The pruning *rate* ds/dt is largest at t0 and falls to zero at t1: remove
    many connections early, while the network is highly redundant and has the
    most training left to recover, and ever fewer as each remaining connection
    carries more of the function.
    """
    if step < start:
        return 0.0
    if step >= end:
        return final
    remaining = 1.0 - (step - start) / (end - start)
    return final * (1.0 - remaining**3)


def linear(step: int, start: int, end: int, final: float) -> float:
    if step < start:
        return 0.0
    if step >= end:
        return final
    return final * (step - start) / (end - start)


def one_shot(step: int, start: int, end: int, final: float) -> float:
    """Train dense until `end`, then remove everything in a single mask update."""
    return final if step >= end else 0.0


SCHEDULES = {"cubic": cubic, "linear": linear, "one_shot": one_shot}
