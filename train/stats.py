"""Statistics for comparing two pruning criteria across seeds.

Runs with the same seed share initialisation and batch order, so the two
criteria are compared on paired differences. All tests here are exact or
resampling-based; none assumes normally distributed accuracies.
"""

from __future__ import annotations

import base64
import math

import numpy as np


def sign_flip_test(differences, exact_up_to: int = 20, n_random: int = 200_000, seed: int = 0) -> float:
    """Two-sided paired randomisation test of "mean difference = 0".

    Under the null hypothesis the two runs of a pair are exchangeable, so each
    paired difference is equally likely to have either sign. The p-value is the
    fraction of the 2^n sign assignments whose |mean| is at least the observed
    one (all of them enumerated for n <= `exact_up_to`, else sampled). With
    n = 10 the smallest possible p-value is 2/1024 ~= 0.002.
    """
    d = np.asarray(differences, dtype=np.float64)
    n = len(d)
    if n == 0:
        raise ValueError("no differences")
    observed = abs(d.mean())
    if n <= exact_up_to:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_random, n))
    means = np.abs(signs @ d) / n
    return float(np.mean(means >= observed * (1 - 1e-12) - 1e-15))


def binomial_sign_test(successes: int, n: int) -> float:
    """Exact two-sided sign test: P(a split at least this uneven) under Binomial(n, 1/2)."""
    k = max(successes, n - successes)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_mean_ci(values, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple:
    """Percentile interval for the mean, resampling the values (e.g. seeds) with replacement."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def two_level_bootstrap_ci(correct_a, correct_b, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple:
    """Interval for mean accuracy(a) - accuracy(b), resampling seeds AND test images.

    `correct_a`, `correct_b`: boolean arrays (n_seeds, n_images), paired by
    seed (row) and image (column). Resampling seeds captures training
    randomness; resampling images captures the fact that the test set is a
    finite sample of digits. Each bootstrap replicate draws both, and applies
    the same image draw to both criteria and all seeds, so the pairing is kept.
    """
    d = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    n_seeds, n_images = d.shape
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        seed_counts = np.bincount(rng.integers(0, n_seeds, n_seeds), minlength=n_seeds)
        image_counts = np.bincount(rng.integers(0, n_images, n_images), minlength=n_images)
        stats[b] = seed_counts @ d @ image_counts / (n_seeds * n_images)
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def holm(p_values) -> list:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def pack_bits(flags: np.ndarray) -> str:
    """A boolean vector as a compact base64 string (10,000 flags -> 1,668 characters)."""
    return base64.b64encode(np.packbits(np.asarray(flags, dtype=bool)).tobytes()).decode("ascii")


def unpack_bits(packed: str, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))[:n].astype(bool)
