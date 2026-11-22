"""Where does a training step spend its time?

    uv run python -m train.profile_step                    # MNIST-sized MLP, float32
    uv run python -m train.profile_step --dtype float64 --sparsity 0.9

Uses random data of MNIST's shape, so it needs no download. Reports wall time
per phase, then a cProfile breakdown in which every backward closure is
attributed to the operation that created it.
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import pstats
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import engine
from engine import softmax_cross_entropy
from nn import MLP, Adam


def _closure_owners() -> dict:
    """Map (file, line) of each nested `backward` function to the name of the op defining it."""
    owners = {}
    for path in Path(engine.__file__).parent.glob("*.py"):
        for outer in ast.walk(ast.parse(path.read_text())):
            if isinstance(outer, ast.FunctionDef):
                for inner in outer.body:
                    if isinstance(inner, ast.FunctionDef) and inner.name == "backward":
                        owners[(str(path), inner.lineno)] = outer.name
    return owners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[784, 256, 128, 10])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--sparsity", type=float, default=0.0, help="fraction of each weight matrix to mask at random")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype).type
    rng = np.random.default_rng(0)
    model = MLP(args.sizes, rng, dtype=dtype)
    if args.sparsity > 0:
        for layer in model.linears:
            layer.weight.set_mask(rng.random(layer.weight.shape) >= args.sparsity)
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = rng.standard_normal((args.batch_size, args.sizes[0])).astype(dtype)
    y = rng.integers(0, args.sizes[-1], size=args.batch_size)

    def step(timer=None):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = softmax_cross_entropy(model(x), y)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        optimizer.step()
        t3 = time.perf_counter()
        if timer is not None:
            timer["forward"] += t1 - t0
            timer["backward"] += t2 - t1
            timer["optimizer"] += t3 - t2

    for _ in range(20):  # warm-up
        step()

    phases = defaultdict(float)
    for _ in range(args.steps):
        step(phases)
    total = sum(phases.values())
    print(f"sizes {args.sizes}, batch {args.batch_size}, {args.dtype}, sparsity {args.sparsity:.0%}")
    print(f"\n{1e3 * total / args.steps:.3f} ms per step")
    for name, seconds in phases.items():
        print(f"  {name:10s} {1e3 * seconds / args.steps:7.3f} ms  {100 * seconds / total:5.1f}%")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.steps):
        step()
    profiler.disable()

    owners = _closure_owners()
    rows = defaultdict(float)
    stats = pstats.Stats(profiler).stats  # {(file, line, name): (cc, nc, tottime, cumtime, callers)}
    profiled_total = sum(entry[2] for entry in stats.values())
    for (filename, line, name), entry in stats.items():
        if (filename, line) in owners:
            label = f"backward of {owners[(filename, line)]}"
        elif filename.startswith(str(Path(engine.__file__).parent.parent)):
            label = f"{Path(filename).parent.name}/{Path(filename).name}:{name}"
        else:
            label = f"numpy/builtin: {name}"
        rows[label] += entry[2]
    print("\nself time by function (under cProfile, which inflates Python-level overhead):")
    for label, seconds in sorted(rows.items(), key=lambda item: -item[1])[:14]:
        print(f"  {100 * seconds / profiled_total:5.1f}%  {label}")


if __name__ == "__main__":
    main()
