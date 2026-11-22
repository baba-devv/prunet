"""The Pareto sweep driver and its analysis, end to end on a small spirals version."""

import json
from dataclasses import replace

import numpy as np
import pytest

from infer.sparse import PrunedMLP
from train.experiment import ExperimentConfig
from train.pareto import BASE, PRIMARY_SPARSITY, SEEDS, SPARSITIES, plan, sweep, verdict

SMALL = ExperimentConfig(dataset="spirals", hidden=(32, 32), epochs=30, batch_size=64, lr=3e-3, prune_interval=10)


def test_protocol_has_not_drifted():
    """The defaults are the protocol fixed before the sweep was run."""
    assert SEEDS == tuple(range(100, 110))
    assert SPARSITIES == (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)
    assert PRIMARY_SPARSITY == 0.95
    assert BASE == replace(ExperimentConfig(), epochs=20)
    assert (BASE.dataset, BASE.hidden, BASE.schedule, BASE.scope, BASE.regrow) == ("mnist", (256, 128), "cubic", "global", 0.0)
    runs = plan(BASE, SEEDS, SPARSITIES)
    assert len(runs) == 130 and len({json.dumps(r[3].__dict__, sort_keys=True, default=str) for r in runs}) == 130


def test_decision_rule():
    assert verdict(0.3, 0.002, 0.1, 0.5) == "supported"
    assert verdict(-0.3, 0.002, -0.5, -0.1) == "refuted"
    assert verdict(0.3, 0.2, 0.1, 0.5) == "inconclusive"  # p too large
    assert verdict(0.3, 0.002, -0.05, 0.5) == "inconclusive"  # interval reaches 0
    assert verdict(0.0, 0.002, 0.1, 0.5) == "inconclusive"


@pytest.fixture(scope="module")
def small_sweep(tmp_path_factory):
    out = tmp_path_factory.mktemp("pareto")
    analysis = sweep(SMALL, seeds=(1, 2), sparsities=(0.5, 0.9), out=out, workers=1, model_seed=1, n_boot=200)
    return out, analysis


def test_sweep_writes_every_artifact(small_sweep):
    out, analysis = small_sweep
    for name in ("runs.json", "pareto.json", "pareto.md", "pareto.png"):
        assert (out / name).exists()
    models = sorted(p.name for p in (out / "models").iterdir())
    assert models == ["dense.npz", "fisher_s50.npz", "fisher_s90.npz", "magnitude_s50.npz", "magnitude_s90.npz"]
    for name in models:  # every saved network comes from the designated seed, not whichever run finished last
        assert PrunedMLP.load(out / "models" / name).meta["config"]["seed"] == 1
    assert len(json.loads((out / "runs.json").read_text())["runs"]) == 2 * (1 + 2 * 2)
    assert analysis["h1"] is None  # 95% was not in this small sweep
    assert "SPIRALS 2-32-32-3" in (out / "pareto.md").read_text()


def test_analysis_is_consistent_with_the_runs(small_sweep):
    out, analysis = small_sweep
    runs = list(json.loads((out / "runs.json").read_text())["runs"].values())
    for level in analysis["levels"]:
        s = level["sparsity"]
        for c in ("magnitude", "fisher"):
            accs = sorted(100 * r["test_acc"] for r in runs if r["config"]["sparsity"] == s and r["config"]["criterion"] == c)
            assert sorted(level[c]["test_acc"]) == accs
            assert all(abs(a - s) < 1e-3 for a in level[c]["achieved_sparsity"])
            assert level[c]["stored_zeros_match_mask"]
        p = level["paired"]
        np.testing.assert_allclose(p["mean"], np.mean(level["magnitude"]["test_acc"]) - np.mean(level["fisher"]["test_acc"]))
        assert 0 < p["sign_flip_p"] <= 1 and p["holm_p"] >= p["sign_flip_p"]
        assert p["ci_two_level"][0] <= p["ci_two_level"][1]


def test_sweep_resumes_from_its_cache(small_sweep, capsys):
    out, analysis = small_sweep
    again = sweep(SMALL, seeds=(1, 2), sparsities=(0.5, 0.9), out=out, workers=1, model_seed=1, n_boot=200)
    assert "0 to run" in capsys.readouterr().out
    assert again["levels"] == analysis["levels"]
