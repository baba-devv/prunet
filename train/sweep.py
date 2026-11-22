"""Run many experiment configs, in parallel processes, and summarise across seeds."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Iterable, Optional

import numpy as np

from .experiment import ExperimentConfig, run_experiment


def _run(config: ExperimentConfig) -> dict:
    result = run_experiment(config, log=None)
    result.pop("step_losses")  # large, and not needed for sweep summaries
    return result


def run_all(configs: Iterable[ExperimentConfig], workers: Optional[int] = None, label: str = "") -> list:
    """Run every config; results come back in input order whatever the completion order."""
    configs = list(configs)
    workers = workers or max(1, min(len(configs), (os.cpu_count() or 2) // 2))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_run, configs), start=1):
            c = result["config"]
            print(
                f"[{label} {index}/{len(configs)}] {c['criterion'] if c['sparsity'] else 'dense':9s} "
                f"sparsity {result['model']['weight_sparsity']:.4f} seed {c['seed']}  "
                f"val {100 * result['val']['acc']:.2f}%",
                flush=True,
            )
            results.append(result)
    return results


def over_seeds(base: ExperimentConfig, seeds: Iterable[int]) -> list:
    return [replace(base, seed=seed) for seed in seeds]


def mean_std(values) -> tuple:
    """Mean and sample standard deviation (ddof=1)."""
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
