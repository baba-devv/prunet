"""Cost measurement: does a pruned network actually do less work?

    uv run python -m infer.benchmark     # uses the models saved by `uv run python -m train.pareto`

For the seed-100 network at every sparsity and criterion, runs the MNIST test
set through every execution path in infer/sparse.py and records

* multiply-accumulates per example and parameter bytes (exact, machine-independent);
* whether every path predicts the same class as the dense forward pass on every
  test image, and the largest logit difference;
* wall-clock time per batch at batch sizes 1, 32 and 256 (machine-specific).

BLAS is pinned to one thread, so times are per core: the unit a server would
schedule. (The script re-launches itself with the thread variables set, since
they only take effect before NumPy is first imported.) Each timing is the
median of 7 repeats after a warm-up, each repeat at least 50 ms long.

Writes results/part4/cost.json, cost.md and cost.png.
"""

from __future__ import annotations

import os
import sys

THREAD_VARIABLES = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

if __name__ == "__main__" and any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
    os.execve(
        sys.executable,
        [sys.executable, "-m", "infer.benchmark", *sys.argv[1:]],
        {**os.environ, **{name: "1" for name in THREAD_VARIABLES}},
    )

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .sparse import AlignedDensePath, CompactDensePath, CSRPath, DensePath, PrunedMLP, compact  # noqa: E402

BATCH_SIZES = (1, 32, 256)


def time_call(fn, x: np.ndarray, min_seconds: float = 0.05, repeats: int = 7) -> dict:
    """Seconds per call: median (and range) of `repeats` timings, each looping >= min_seconds."""
    fn(x)  # warm-up
    loops = 1
    while True:
        start = time.perf_counter()
        for _ in range(loops):
            fn(x)
        if time.perf_counter() - start >= min_seconds:
            break
        loops *= 2
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(loops):
            fn(x)
        samples.append((time.perf_counter() - start) / loops)
    return {"median": float(np.median(samples)), "min": float(min(samples)), "max": float(max(samples)), "loops": loops}


def paths_for(net: PrunedMLP) -> list:
    small = compact(net)
    if net.nnz == net.num_weights:  # unpruned: the other paths would do the same work
        return [DensePath(net)]
    return [DensePath(net), CompactDensePath(small), AlignedDensePath(small), CSRPath(small)]


def measure_model(net: PrunedMLP, x: np.ndarray, y: np.ndarray, batch_sizes=BATCH_SIZES, min_seconds: float = 0.05) -> dict:
    """Agreement and timing of every path for one network."""
    reference = net.forward(x)
    reference_pred = reference.argmax(axis=1)
    rows = []
    for path in paths_for(net):
        logits = path(x)
        pred = logits.argmax(axis=1)
        rows.append(
            {
                "path": path.name,
                "macs_per_example": path.macs,
                "parameter_bytes": path.nbytes,
                "test_acc": float((pred == y).mean()),
                "predictions_differing_from_dense": int((pred != reference_pred).sum()),
                "max_abs_logit_difference": float(np.abs(logits - reference).max()),
                "seconds_per_batch": {
                    str(b): time_call(path, np.ascontiguousarray(x[:b]), min_seconds=min_seconds) for b in batch_sizes
                },
            }
        )
    return {
        "criterion": "dense" if net.nnz == net.num_weights else net.meta["config"]["criterion"],
        "sparsity": 1 - net.nnz / net.num_weights,
        "nonzero_weights": net.nnz,
        "structure": compact(net).report,
        "saved_test_acc": net.meta.get("test_acc"),
        "paths": rows,
    }


def shape_diagnostic(net: PrunedMLP, x: np.ndarray, batch: int = 256, multiple: int = 32, min_seconds: float = 0.05) -> dict:
    """Why fewer multiply-accumulates need not mean less time, measured on one compacted network.

    Times (a) gathering the surviving input columns and (b) each compacted
    layer's matrix product, next to the same product with its dimensions
    zero-padded to a multiple of `multiple`.
    """
    small = compact(net)
    xb = np.ascontiguousarray(x[:batch])
    rng = np.random.default_rng(0)
    layers = []
    for w in small.weights:
        h = np.ascontiguousarray(rng.standard_normal((batch, w.shape[0])).astype(w.dtype))
        rows, cols = -(-w.shape[0] // multiple) * multiple, w.shape[1]
        if w is not small.weights[-1]:
            cols = -(-w.shape[1] // multiple) * multiple
        padded_w = np.zeros((rows, cols), dtype=w.dtype)
        padded_h = np.zeros((batch, rows), dtype=w.dtype)
        layers.append(
            {
                "shape": list(w.shape),
                "seconds": time_call(lambda a: a @ w, h, min_seconds=min_seconds)["median"],
                "padded_shape": [rows, cols],
                "padded_seconds": time_call(lambda a: a @ padded_w, padded_h, min_seconds=min_seconds)["median"],
            }
        )
    return {
        "batch": batch,
        "compact_sizes": small.report["compact_sizes"],
        "input_gather_seconds": time_call(lambda a: a[:, small.input_index], xb, min_seconds=min_seconds)["median"],
        "layers": layers,
    }


def environment() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }
    try:
        info["blas"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:  # older NumPy
        info["blas"] = "unknown"
    return info


def us(seconds: float) -> str:
    return f"{1e6 * seconds:,.1f}"


def markdown(report: dict) -> str:
    models = report["models"]
    reference = next(m for m in models if m["criterion"] == "dense")["paths"][0]
    ref_time = {b: reference["seconds_per_batch"][b]["median"] for b in reference["seconds_per_batch"]}
    env = report["environment"]
    lines = [
        "# What a pruned network costs to run",
        "",
        "Generated by `uv run python -m infer.benchmark` from the seed-100 networks saved by the Pareto sweep.",
        f"Machine: {env['platform']}, Python {env['python']}, NumPy {env['numpy']} ({env['blas']} BLAS), "
        "one BLAS thread. Times are medians of 7 repeats.",
        "",
        "*dense*: full-size matrices; on a pruned network this is the dense-times-zero shortcut. "
        "*compact_dense*: inputs and hidden units that cannot affect the output removed, then dense products. "
        "*compact_aligned*: the same hidden units removed, but all inputs kept and hidden widths zero-padded to a "
        "multiple of 32. "
        "*CSR*: only surviving weights multiplied. Speed-ups are relative to the **unpruned** network's dense path; "
        "below 1.0× means slower.",
        "",
        "| sparsity | criterion | path | MACs / example | parameter KB | batch 1 (µs) | speed-up | batch 256 (µs / example) | speed-up | predictions changed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        for row in m["paths"]:
            t1 = row["seconds_per_batch"]["1"]["median"]
            t256 = row["seconds_per_batch"]["256"]["median"]
            lines.append(
                f"| {100 * m['sparsity']:.0f}% | {m['criterion']} | {row['path']} | {row['macs_per_example']:,} | "
                f"{row['parameter_bytes'] / 1024:,.0f} | {us(t1)} | {ref_time['1'] / t1:.2f}× | {us(t256 / 256)} | "
                f"{ref_time['256'] / t256:.2f}× | {row['predictions_differing_from_dense']} |"
            )
    lines += [
        "",
        "## Why fewer MACs is not automatically less time (95% networks, batch 256)",
        "",
        "Each compacted layer's matrix product next to the same product with its dimensions zero-padded to a multiple "
        "of 32, and the cost of gathering the surviving input columns.",
        "",
        "| criterion | compacted layer | µs | padded to | µs |",
        "|---|---|---|---|---|",
    ]
    for criterion, d in report["shape_diagnostic_95"].items():
        lines.append(f"| {criterion} | input gather (784 → {d['compact_sizes'][0]} columns) | {us(d['input_gather_seconds'])} | – | – |")
        for layer in d["layers"]:
            lines.append(
                f"| {criterion} | {layer['shape'][0]}×{layer['shape'][1]} | {us(layer['seconds'])} | "
                f"{layer['padded_shape'][0]}×{layer['padded_shape'][1]} | {us(layer['padded_seconds'])} |"
            )
    h3 = report["h3"]
    lines += [
        "",
        f"### H3: **{h3['verdict'].upper()}**",
        "",
        f"The CSR path was slower than the unpruned dense path in {h3['csr_slower']} of {h3['comparisons']} "
        "comparisons (every pruned network at batch sizes 1 and 256). Prediction: slower in all of them.",
        "",
        "Every path predicted the same class as the dense forward pass on every one of the "
        f"{report['n_test']:,} test images: {report['all_predictions_identical']}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=Path("results/part4/models"))
    parser.add_argument("--out", type=Path, default=Path("results/part4"))
    parser.add_argument("--min-seconds", type=float, default=0.05)
    args = parser.parse_args()

    files = sorted(args.models.glob("*.npz"))
    if not files:
        sys.exit(f"no models in {args.models}: run `uv run python -m train.pareto` first")

    from train.data import load_dataset

    data = load_dataset("mnist", dtype=np.float32)
    x, y = np.ascontiguousarray(data.x_test), data.y_test

    models = []
    for f in files:
        net = PrunedMLP.load(f)
        result = measure_model(net, x, y, min_seconds=args.min_seconds)
        result["file"] = f.name
        models.append(result)
        dense = result["paths"][0]["seconds_per_batch"]
        print(f"{f.name:20s} nnz {net.nnz:7,}  dense b1 {us(dense['1']['median'])} µs", flush=True)
    models.sort(key=lambda m: (m["sparsity"], m["criterion"]))

    reference = next(m for m in models if m["criterion"] == "dense")["paths"][0]["seconds_per_batch"]
    comparisons = [
        row["seconds_per_batch"][b]["median"] > reference[b]["median"]
        for m in models
        if m["criterion"] != "dense"
        for row in m["paths"]
        if row["path"] == "csr"
        for b in ("1", "256")
    ]
    diagnostics = {}
    for f in files:
        net = PrunedMLP.load(f)
        if net.nnz < net.num_weights and abs(net.nnz / net.num_weights - 0.05) < 1e-3:  # the 95% networks
            diagnostics[net.meta["config"]["criterion"]] = shape_diagnostic(net, x, min_seconds=args.min_seconds)
    report = {
        "environment": environment(),
        "shape_diagnostic_95": diagnostics,
        "n_test": int(len(y)),
        "batch_sizes": list(BATCH_SIZES),
        "models": models,
        "all_predictions_identical": all(r["predictions_differing_from_dense"] == 0 for m in models for r in m["paths"]),
        "h3": {
            "csr_slower": int(sum(comparisons)),
            "comparisons": len(comparisons),
            "verdict": "holds" if all(comparisons) else "does not hold",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cost.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.out / "cost.md").write_text(markdown(report))

    from train.plots import plot_cost

    plot_cost(report, args.out / "cost.png")
    print((args.out / "cost.md").read_text())


if __name__ == "__main__":
    main()
