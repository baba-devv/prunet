"""Statistics for the paired criterion comparison."""

import itertools

import numpy as np
import pytest

from train.stats import (
    binomial_sign_test,
    bootstrap_mean_ci,
    holm,
    pack_bits,
    sign_flip_test,
    two_level_bootstrap_ci,
    unpack_bits,
)


def brute_force_sign_flip(d):
    d = np.asarray(d, dtype=float)
    observed = abs(d.mean())
    means = [abs(np.mean(np.array(signs) * d)) for signs in itertools.product([-1, 1], repeat=len(d))]
    return np.mean([m >= observed - 1e-12 for m in means])


def test_sign_flip_known_values():
    assert sign_flip_test(np.ones(10)) == pytest.approx(2 / 1024)  # the smallest p-value 10 pairs can give
    assert sign_flip_test([1.0, 2.0, 3.0]) == pytest.approx(2 / 8)
    assert sign_flip_test([1.0, -1.0]) == 1.0
    assert sign_flip_test([0.3, -0.1, 0.2]) == sign_flip_test([-0.3, 0.1, -0.2])  # symmetric


@pytest.mark.parametrize("seed", range(5))
def test_sign_flip_matches_brute_force_enumeration(seed):
    d = np.random.default_rng(seed).normal(0.2, 1.0, size=8)
    assert sign_flip_test(d) == pytest.approx(brute_force_sign_flip(d))


def test_sign_flip_sampling_agrees_with_exact_enumeration():
    d = np.random.default_rng(0).normal(0.3, 1.0, size=12)
    exact = sign_flip_test(d)
    sampled = sign_flip_test(d, exact_up_to=5, n_random=200_000)
    assert sampled == pytest.approx(exact, abs=0.005)


def test_binomial_sign_test():
    assert binomial_sign_test(10, 10) == pytest.approx(2 / 1024)
    assert binomial_sign_test(0, 10) == pytest.approx(2 / 1024)
    assert binomial_sign_test(5, 10) == 1.0
    assert binomial_sign_test(8, 10) == pytest.approx(2 * (45 + 10 + 1) / 1024)


def test_bootstrap_mean_ci():
    values = np.random.default_rng(0).normal(5.0, 1.0, size=10)
    low, high = bootstrap_mean_ci(values)
    assert low < values.mean() < high
    assert bootstrap_mean_ci(np.full(10, 2.5)) == (2.5, 2.5)
    assert bootstrap_mean_ci(values) == bootstrap_mean_ci(values)  # seeded


def test_two_level_bootstrap_edge_cases_and_centre():
    r = np.random.default_rng(0)
    a = r.random((10, 2000)) < 0.9
    assert two_level_bootstrap_ci(a, a, n_boot=500) == (0.0, 0.0)
    assert two_level_bootstrap_ci(np.ones((4, 50), bool), np.zeros((4, 50), bool), n_boot=200) == (1.0, 1.0)

    b = r.random((10, 2000)) < 0.88
    observed = a.mean() - b.mean()
    low, high = two_level_bootstrap_ci(a, b, n_boot=2000)
    assert low < observed < high


def test_two_level_bootstrap_is_wider_than_resampling_seeds_alone():
    """Identical noise-free training runs (every seed the same) still leave test-set sampling noise."""
    r = np.random.default_rng(1)
    row_a, row_b = r.random(2000) < 0.9, r.random(2000) < 0.9
    a, b = np.tile(row_a, (10, 1)), np.tile(row_b, (10, 1))
    seed_only = bootstrap_mean_ci(a.mean(axis=1) - b.mean(axis=1))
    assert seed_only[0] == seed_only[1]  # zero width: the seeds agree perfectly
    low, high = two_level_bootstrap_ci(a, b, n_boot=2000)
    assert high - low > 0.01


def test_holm_adjustment():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert holm([0.5]) == [0.5]
    assert holm([0.6, 0.6]) == [1.0, 1.0]


def test_pack_bits_round_trip():
    flags = np.random.default_rng(0).random(10_000) < 0.97
    packed = pack_bits(flags)
    assert len(packed) == 1668
    np.testing.assert_array_equal(unpack_bits(packed, 10_000), flags)
