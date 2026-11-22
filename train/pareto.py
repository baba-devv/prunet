"""The sparsity-accuracy Pareto sweep: magnitude vs Fisher pruning, paired by seed.

    uv run python -m train.pareto            # 130 MNIST runs, ~35 minutes on 4 cores; resumable

The protocol, hypotheses and decision rules were fixed before the first run;
the constants below are that protocol.

Writes to results/part4/:
    runs.json    one record per run; also the cache, so an interrupted sweep resumes
    pareto.json  everything the tables and the figure are computed from
    pareto.md    the tables and the verdict on each hypothesis
    pareto.png   the figure
    models/      the seed-100 network at every sparsity, for `python -m infer.benchmark`
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .experiment import ExperimentConfig
from .stats import (
    binomial_sign_test,
    bootstrap_mean_ci,
    holm,
    sign_flip_test,
    two_level_bootstrap_ci,
    unpack_bits,
)
from .sweep import mean_std, run_all

# --- protocol -------------------------------------------------------------------------------------
BASE = ExperimentConfig(epochs=20)
SEEDS = tuple(range(100, 110))
SPARSITIES = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)
CRITERIA = ("magnitude", "fisher")
PRIMARY_SPARSITY = 0.95
ALPHA = 0.05
N_BOOT = 10_000


def label(config: dict) -> str:
    return "dense" if config["sparsity"] == 0 else config["criterion"]


def model_filename(name: str, sparsity: float) -> str:
    return "dense.npz" if name == "dense" else f"{name}_s{round(100 * sparsity):02d}.npz"


def plan(base: ExperimentConfig, seeds, sparsities) -> list:
    """(name, sparsity, seed, config) for every run: one dense run and one per criterion and sparsity."""
    runs = []
    for seed in seeds:
        runs.append(("dense", 0.0, seed, replace(base, seed=seed)))
        for criterion in CRITERIA:
            for sparsity in sparsities:
                runs.append((criterion, sparsity, seed, replace(base, sparsity=sparsity, criterion=criterion, seed=seed)))
    return runs


def key(config: ExperimentConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True)


def summarise(result: dict) -> dict:
    record = {
        "config": result["config"],
        "val_acc": result["val"]["acc"],
        "test_acc": result["test"]["acc"],
        "test_loss": result["test"]["loss"],
        "test_correct_packed": result["test"]["correct_packed"],
        "n_test": result["data"]["test"],
        "num_weights": result["model"]["num_weights"],
        "nonzero_weights": result["model"]["nonzero_weights"],
        "weight_sparsity": result["model"]["weight_sparsity"],
        "layer_sparsity": result["model"]["layer_sparsity"],
        "stored_zeros_match_mask": result["model"]["stored_zeros_match_mask"],
        "structure": result["structure"],
        "val_acc_by_epoch": [h["val_acc"] for h in result["history"]],
        "environment": result["environment"],
    }
    if "pruning" in result:
        events = result["pruning"]["events"]
        record["probe_loss_jump_total"] = sum(e["probe_loss_after"] - e["probe_loss_before"] for e in events)
    return record


# --- analysis ------------------------------------------------------------------------------------


def verdict(mean: float, p: float, low: float, high: float, alpha: float = ALPHA) -> str:
    """Decision rule for 'magnitude beats Fisher'."""
    if mean > 0 and p < alpha and low > 0:
        return "supported"
    if mean < 0 and p < alpha and high < 0:
        return "refuted"
    return "inconclusive"


def criterion_summary(records: list) -> dict:
    test = [100 * r["test_acc"] for r in records]
    val = [100 * r["val_acc"] for r in records]
    structure = [r["structure"] for r in records]
    return {
        "test_acc": test,
        "test_mean": mean_std(test)[0],
        "test_std": mean_std(test)[1],
        "val_mean": mean_std(val)[0],
        "val_std": mean_std(val)[1],
        "nonzero_weights": [r["nonzero_weights"] for r in records],
        "achieved_sparsity": [r["weight_sparsity"] for r in records],
        "layer_sparsity_mean": np.mean([r["layer_sparsity"] for r in records], axis=0).tolist(),
        "inputs_removed": [s["inputs_removed"] for s in structure],
        "hidden_units_removed": [s["hidden_units_removed"] for s in structure],
        "compact_dense_macs": [s["dense_macs_after"] for s in structure],
        "csr_macs": [s["nnz_after"] for s in structure],
        "stored_zeros_match_mask": all(r["stored_zeros_match_mask"] for r in records),
    }


def analyse(records: list, seeds, sparsities, primary: float = PRIMARY_SPARSITY, n_boot: int = N_BOOT) -> dict:
    by = {(label(r["config"]), r["config"]["sparsity"], r["config"]["seed"]): r for r in records}
    seeds, sparsities = list(seeds), list(sparsities)
    n_test = records[0]["n_test"]
    dense = criterion_summary([by[("dense", 0.0, s)] for s in seeds])

    levels = []
    for sparsity in sparsities:
        runs = {c: [by[(c, sparsity, s)] for s in seeds] for c in CRITERIA}
        diffs = [100 * (m["test_acc"] - f["test_acc"]) for m, f in zip(runs["magnitude"], runs["fisher"])]
        correct = {c: np.stack([unpack_bits(r["test_correct_packed"], n_test) for r in runs[c]]) for c in CRITERIA}
        # The run-level accuracies and the per-image flags must describe the same comparison.
        from_flags = 100 * (correct["magnitude"].mean() - correct["fisher"].mean())
        if abs(np.mean(diffs) - from_flags) > 1e-9:
            raise AssertionError(f"paired differences ({np.mean(diffs)}) disagree with per-image results ({from_flags})")
        low, high = two_level_bootstrap_ci(correct["magnitude"], correct["fisher"], n_boot=n_boot)
        seed_low, seed_high = bootstrap_mean_ci(diffs, n_boot=n_boot)
        levels.append(
            {
                "sparsity": sparsity,
                **{c: criterion_summary(runs[c]) for c in CRITERIA},
                "paired": {
                    "differences": diffs,
                    "mean": float(np.mean(diffs)),
                    "sign_flip_p": sign_flip_test(diffs),
                    "ci_two_level": [100 * low, 100 * high],
                    "ci_seeds_only": [seed_low, seed_high],
                    "magnitude_better_in": int(sum(d > 0 for d in diffs)),
                    "fisher_better_in": int(sum(d < 0 for d in diffs)),
                },
            }
        )
    for level, adjusted in zip(levels, holm([lv["paired"]["sign_flip_p"] for lv in levels])):
        level["paired"]["holm_p"] = adjusted

    h1, h2 = None, None
    if primary in sparsities:
        level = levels[sparsities.index(primary)]
        p = level["paired"]
        h1 = {"sparsity": primary, "verdict": verdict(p["mean"], p["sign_flip_p"], *p["ci_two_level"]), **p}
        macs = {c: level[c]["compact_dense_macs"] for c in CRITERIA}
        fisher_smaller = sum(f < m for f, m in zip(macs["fisher"], macs["magnitude"]))
        h2 = {
            "sparsity": primary,
            "compact_dense_macs": macs,
            "fisher_smaller_in": int(fisher_smaller),
            "n": len(seeds),
            "sign_test_p": binomial_sign_test(fisher_smaller, len(seeds)),
            "verdict": "supported" if fisher_smaller == len(seeds) else "not supported",
            "mean_ratio_fisher_over_magnitude": float(np.mean(macs["fisher"]) / np.mean(macs["magnitude"])),
        }
    return {
        "seeds": seeds,
        "sparsities": sparsities,
        "sizes": records[0]["structure"]["original_sizes"],
        "num_weights": records[0]["num_weights"],
        "n_test": n_test,
        "dense": dense,
        "levels": levels,
        "h1": h1,
        "h2": h2,
        "environment": records[0]["environment"],
    }


# --- report ----------------------------------------------------------------------------------------


def markdown(analysis: dict, config: dict) -> str:
    seeds, dense, levels, sizes = analysis["seeds"], analysis["dense"], analysis["levels"], analysis["sizes"]
    lines = [
        "# Sparsity-accuracy sweep: magnitude vs Fisher",
        "",
        "Protocol, hypotheses and decision rules were fixed before any of these runs.",
        f"Generated by `uv run python -m train.pareto` from [runs.json](runs.json) ({len(seeds) * (1 + 2 * len(levels))} runs).",
        "",
        f"{config['dataset'].upper()} {'-'.join(map(str, sizes))} MLP, {config['epochs']} epochs; seeds "
        f"{seeds[0]}–{seeds[-1]} ({len(seeds)}); test split, {analysis['n_test']:,} images, evaluated at the end of training.",
        f"Accuracies are mean ± standard deviation over seeds, in %. Dense reference: **{dense['test_mean']:.2f} ± {dense['test_std']:.2f}**.",
        "",
        "## Accuracy at each sparsity",
        "",
        "| sparsity | weights left | magnitude | Fisher saliency | magnitude − Fisher (points) |",
        "|---|---|---|---|---|",
    ]
    for lv in levels:
        m, f, p = lv["magnitude"], lv["fisher"], lv["paired"]
        lines.append(
            f"| {100 * lv['sparsity']:.0f}% | {m['nonzero_weights'][0]:,} | {m['test_mean']:.2f} ± {m['test_std']:.2f} | "
            f"{f['test_mean']:.2f} ± {f['test_std']:.2f} | {p['mean']:+.2f} |"
        )
    lines += [
        "",
        "## Is the difference real? (paired by seed)",
        "",
        "Two intervals: *seeds only* resamples training runs; *seeds + images* also resamples the test images,",
        "so it includes the uncertainty from the test set being a finite sample. The sign-flip test is exact.",
        "",
        f"| sparsity | mean difference | 95% CI, seeds + images | 95% CI, seeds only | sign-flip p | Holm p ({len(levels)} levels) "
        "| magnitude better in |",
        "|---|---|---|---|---|---|---|",
    ]
    for lv in levels:
        p = lv["paired"]
        lines.append(
            f"| {100 * lv['sparsity']:.0f}% | {p['mean']:+.3f} | [{p['ci_two_level'][0]:+.3f}, {p['ci_two_level'][1]:+.3f}] | "
            f"[{p['ci_seeds_only'][0]:+.3f}, {p['ci_seeds_only'][1]:+.3f}] | {p['sign_flip_p']:.4f} | {p['holm_p']:.4f} | "
            f"{p['magnitude_better_in']} of {len(seeds)} seeds |"
        )
    h1 = analysis["h1"]
    if h1:
        lines += [
            "",
            f"### H1 (primary): **{h1['verdict'].upper()}**",
            "",
            f"At {100 * h1['sparsity']:.0f}% sparsity the paired difference in test accuracy (magnitude − Fisher) is "
            f"{h1['mean']:+.3f} points, 95% CI [{h1['ci_two_level'][0]:+.3f}, {h1['ci_two_level'][1]:+.3f}] (seeds + images), "
            f"exact sign-flip p = {h1['sign_flip_p']:.4f}; magnitude was better in {h1['magnitude_better_in']} of {len(seeds)} seeds. "
            "Decision rule: supported if the mean is positive, p < 0.05 and the interval lies above 0.",
        ]
    lines += [
        "",
        "## What each network costs to run",
        "",
        "Multiply-accumulates per example. *Dense* runs the full matrices (the dense-times-zero shortcut costs the same as",
        f"the unpruned network, {analysis['num_weights']:,}). *Compact dense* first removes inputs and hidden units that cannot affect",
        "the output (exact, see `infer/sparse.py`). *CSR* multiplies only surviving weights. Means over seeds.",
        "",
        f"| sparsity | criterion | inputs removed (of {sizes[0]}) | hidden units removed (of {' / '.join(map(str, sizes[1:-1]))}) "
        "| compact dense MACs | CSR MACs |",
        "|---|---|---|---|---|---|",
    ]
    for lv in levels:
        for c in CRITERIA:
            s = lv[c]
            units = np.mean(s["hidden_units_removed"], axis=0)
            lines.append(
                f"| {100 * lv['sparsity']:.0f}% | {c} | {np.mean(s['inputs_removed']):.0f} | "
                f"{' / '.join(f'{u:.0f}' for u in units)} | {np.mean(s['compact_dense_macs']):,.0f} | {np.mean(s['csr_macs']):,.0f} |"
            )
    h2 = analysis["h2"]
    if h2:
        lines += [
            "",
            f"### H2: **{h2['verdict'].upper()}**",
            "",
            f"At {100 * h2['sparsity']:.0f}% sparsity the Fisher-pruned network compacts to fewer dense MACs than the "
            f"magnitude-pruned one in {h2['fisher_smaller_in']} of {h2['n']} seeds (exact sign test p = {h2['sign_test_p']:.4f}); "
            f"on average it needs {h2['mean_ratio_fisher_over_magnitude']:.2f}× the magnitude network's compact MACs.",
        ]
    env = analysis["environment"]
    lines += [
        "",
        f"Environment: Python {env['python']}, NumPy {env['numpy']}, {env['machine']}. "
        "Every pruned network was checked to hold exact zeros in storage wherever its mask is off.",
    ]
    return "\n".join(lines) + "\n"


# --- driver ---------------------------------------------------------------------------------------


def sweep(base: ExperimentConfig, seeds, sparsities, out: Path, workers: int, model_seed: int, n_boot: int = N_BOOT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.json"
    cache = json.loads(runs_path.read_text())["runs"] if runs_path.exists() else {}
    planned = plan(base, seeds, sparsities)
    models = out / "models"
    todo = [(c, s, seed, cfg) for c, s, seed, cfg in planned if key(cfg) not in cache]
    print(f"{len(planned)} runs planned, {len(planned) - len(todo)} cached, {len(todo)} to run", flush=True)

    def save(result: dict) -> None:
        config = ExperimentConfig(**{**result["config"], "hidden": tuple(result["config"]["hidden"])})
        cache[key(config)] = summarise(result)
        runs_path.write_text(json.dumps({"runs": cache}, indent=1) + "\n")

    if todo:
        paths = [models / model_filename(c, s) if seed == model_seed else None for c, s, seed, _ in todo]
        run_all([cfg for *_, cfg in todo], workers=workers, label="pareto", on_result=save, model_paths=paths)

    records = [cache[key(cfg)] for *_, cfg in planned]
    analysis = analyse(records, seeds, sparsities, n_boot=n_boot)
    analysis["config"] = asdict(base)
    (out / "pareto.json").write_text(json.dumps(analysis, indent=1) + "\n")
    (out / "pareto.md").write_text(markdown(analysis, asdict(base)))

    from .plots import plot_pareto

    plot_pareto(analysis, out / "pareto.png")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/part4"))
    args = parser.parse_args()
    analysis = sweep(BASE, SEEDS, SPARSITIES, args.out, args.workers, model_seed=SEEDS[0])
    print((args.out / "pareto.md").read_text())
    print(f"environment: {platform.platform()}")
    if analysis["h1"]:
        print(f"H1: {analysis['h1']['verdict']}   H2: {analysis['h2']['verdict']}")


if __name__ == "__main__":
    main()
