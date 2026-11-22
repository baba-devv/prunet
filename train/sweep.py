"""Run many experiment configs, in parallel processes, and summarise across seeds."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Callable, Iterable, Optional

import numpy as np

from .experiment import ExperimentConfig, run_experiment


def _run(job: tuple) -> dict:
    config, model_path = job
    result = run_experiment(config, log=None, model_path=model_path)
    result.pop("step_losses")  # large, and not needed for sweep summaries
    return result


def run_all(
    configs: Iterable[ExperimentConfig],
    workers: Optional[int] = None,
    label: str = "",
    on_result: Optional[Callable[[dict], None]] = None,
    model_paths: Optional[list] = None,
) -> list:
    """Run every config; results come back in input order whatever the completion order.

    `on_result` is called with each result as soon as it is available (in input
    order), e.g. to checkpoint a long sweep. `model_paths`, parallel to
    `configs`, names where to save each trained network (None = do not save).
    """
    configs = list(configs)
    paths: list = list(model_paths) if model_paths is not None else [None] * len(configs)
    if len(paths) != len(configs):
        raise ValueError("model_paths must be parallel to configs")
    workers = workers or max(1, min(len(configs), (os.cpu_count() or 2) // 2))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_run, list(zip(configs, paths))), start=1):
            c = result["config"]
            print(
                f"[{label} {index}/{len(configs)}] {c['criterion'] if c['sparsity'] else 'dense':9s} "
                f"sparsity {result['model']['weight_sparsity']:.4f} seed {c['seed']}  "
                f"val {100 * result['val']['acc']:.2f}%",
                flush=True,
            )
            if on_result is not None:
                on_result(result)
            results.append(result)
    return results


def over_seeds(base: ExperimentConfig, seeds: Iterable[int]) -> list:
    return [replace(base, seed=seed) for seed in seeds]


def mean_std(values) -> tuple:
    """Mean and sample standard deviation (ddof=1)."""
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
