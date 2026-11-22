"""Inference paths for pruned networks: exactness, compaction, CSR, cost accounting."""

import numpy as np
import pytest

from engine import Tensor
from infer.sparse import (
    AlignedDensePath,
    CompactDensePath,
    CSRMatrix,
    CSRPath,
    DensePath,
    PrunedMLP,
    compact,
    relu_mlp_forward,
)
from nn import MLP


def rng(seed=0):
    return np.random.default_rng(seed)


def random_net(sizes, density, seed=0, dtype=np.float64):
    r = rng(seed)
    weights, biases, masks = [], [], []
    for n_in, n_out in zip(sizes, sizes[1:]):
        m = r.random((n_in, n_out)) < density
        weights.append((r.standard_normal((n_in, n_out)) * m).astype(dtype))
        biases.append(r.standard_normal(n_out).astype(dtype))
        masks.append(m)
    return PrunedMLP(weights, biases, masks)


# --- CSR ------------------------------------------------------------------------------


@pytest.mark.parametrize("density", [1.0, 0.3, 0.05, 0.0])
@pytest.mark.parametrize("batch", [1, 7, 300])
def test_csr_product_equals_dense_product(density, batch):
    r = rng()
    mask = r.random((40, 25)) < density
    mask[:, 3] = False  # an output with no inputs (an empty CSR row) ...
    mask[5, :] = False  # ... and an input that feeds nothing
    w = r.standard_normal((40, 25)) * mask
    x = r.standard_normal((batch, 40))
    csr = CSRMatrix(w, mask)
    assert csr.nnz == mask.sum()
    np.testing.assert_allclose(csr.matmul(x), x @ w, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(csr.to_dense(), w)


def test_csr_chunking_does_not_change_the_result():
    r = rng()
    mask = r.random((30, 20)) < 0.4
    w = r.standard_normal((30, 20)) * mask
    x = r.standard_normal((97, 30))
    csr = CSRMatrix(w, mask)
    np.testing.assert_array_equal(csr.matmul(x, chunk_elements=50), csr.matmul(x))  # chunks of 1-2 rows


def test_csr_stores_only_surviving_weights():
    mask = np.zeros((100, 50), dtype=bool)
    mask[np.arange(100), np.arange(100) % 50] = True  # 100 connections out of 5,000
    csr = CSRMatrix(np.where(mask, 1.5, 0.0).astype(np.float32), mask)
    assert csr.nnz == 100
    assert csr.nbytes == 100 * 4 + 100 * 4 + 51 * 4  # float32 values, int32 input indices, int32 row pointers


def test_csr_handles_a_matrix_with_no_weights():
    csr = CSRMatrix(np.zeros((4, 3)), np.zeros((4, 3), dtype=bool))
    np.testing.assert_array_equal(csr.matmul(np.ones((2, 4))), np.zeros((2, 3)))


# --- extraction from a trained model ------------------------------------------------------


def test_pruned_mlp_reproduces_the_autodiff_forward_pass():
    model = MLP([12, 9, 7, 4], rng(1))
    for layer in model.linears:
        layer.weight.set_mask(rng(2).random(layer.weight.shape) < 0.5)
        layer.bias.data += 0.1
    x = rng(3).standard_normal((20, 12))
    net = PrunedMLP.from_model(model)
    np.testing.assert_allclose(net.forward(x), model(Tensor(x)).data, rtol=1e-12, atol=1e-12)
    assert net.nnz == sum(int(layer.weight.mask.sum()) for layer in model.linears)
    for w, m in zip(net.weights, net.masks):
        assert (w[~m] == 0).all()


def test_save_and_load_round_trip(tmp_path):
    net = random_net([10, 8, 3], 0.3, dtype=np.float32)
    net.meta = {"criterion": "fisher", "sparsity": 0.7}
    net.save(tmp_path / "model.npz")
    back = PrunedMLP.load(tmp_path / "model.npz")
    assert back.meta == net.meta
    for a, b in zip(net.weights + net.biases + net.masks, back.weights + back.biases + back.masks):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype


def test_non_relu_models_are_rejected():
    with pytest.raises(ValueError, match="ReLU"):
        PrunedMLP.from_model(MLP([4, 3, 2], rng(), activation="tanh"))


# --- compaction ---------------------------------------------------------------------------


def handmade_net():
    """5 inputs -> 4 hidden -> 3 hidden -> 2 outputs, with every kind of dead unit:

    input 4:   feeds nothing                          -> removed
    input 0:   feeds only h1 unit 2                   -> removed on the second pass (cascade)
    h1 unit 0: no inputs, bias +2, feeds h2 unit 0    -> constant relu(2) = 2, folded into h2's bias
    h1 unit 1: no inputs, bias -1, feeds h2 unit 1    -> constant relu(-1) = 0, folded (adds nothing)
    h1 unit 2: has inputs, feeds nothing              -> removed
    h1 unit 3: inputs 1, 2, 3                         -> kept
    h2 unit 0: its only input was h1 unit 0           -> now a constant itself, folded into the output bias
    h2 unit 1: input from h1 unit 3, feeds the output -> kept
    h2 unit 2: input from h1 unit 3, feeds nothing    -> removed
    """
    m1 = np.zeros((5, 4), dtype=bool)
    m1[[0, 1], 2] = True
    m1[[1, 2, 3], 3] = True
    m2 = np.zeros((4, 3), dtype=bool)
    m2[0, 0] = True
    m2[1, 1] = True
    m2[3, 1] = m2[3, 2] = True
    m3 = np.zeros((3, 2), dtype=bool)
    m3[0, :] = True
    m3[1, :] = True
    r = rng(5)
    masks = [m1, m2, m3]
    weights = [r.standard_normal(m.shape) * m for m in masks]
    biases = [np.array([2.0, -1.0, 0.5, 0.3]), np.array([0.7, -0.2, 0.4]), np.array([0.1, -0.1])]
    return PrunedMLP(weights, biases, masks)


def test_compaction_removes_exactly_the_dead_units_and_keeps_the_function():
    net = handmade_net()
    small = compact(net)
    report = small.report
    assert report["compact_sizes"] == [3, 1, 1, 2]  # inputs 1,2,3 | h1 unit 3 | h2 unit 1 | both outputs
    np.testing.assert_array_equal(small.input_index, [1, 2, 3])
    assert report["inputs_removed"] == 2  # input 0 fed only h1 unit 2, which was removed: a cascade
    assert report["hidden_units_removed"] == [3, 2]
    assert report["hidden_units_folded_into_bias"] == [2, 1]
    x = rng(6).standard_normal((50, 5))
    np.testing.assert_allclose(CompactDensePath(small)(x), net.forward(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(CSRPath(small)(x), net.forward(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("density", [0.6, 0.15, 0.05, 0.01])
def test_compaction_is_exact_on_random_sparse_networks(density):
    net = random_net([60, 40, 30, 10], density, seed=7)
    small = compact(net)
    x = rng(8).standard_normal((64, 60))
    reference = net.forward(x)
    for path in (DensePath(net), CompactDensePath(small), AlignedDensePath(small), CSRPath(small)):
        np.testing.assert_allclose(path(x), reference, rtol=1e-10, atol=1e-10)
    assert small.report["compact_sizes"][-1] == 10  # outputs are never removed
    for w, m in zip(small.weights, small.masks):  # every surviving unit has an input and an output
        assert m.any(axis=0).all() or w.shape[1] == 10
    for m_in, m_out in zip(small.masks, small.masks[1:]):
        assert m_in.any(axis=0).all() and m_out.any(axis=1).all()


def test_a_network_with_no_path_to_the_output_becomes_a_constant():
    net = random_net([6, 5, 3], 0.5, seed=9)
    net.masks[1][:] = False
    net.weights[1][:] = 0.0
    small = compact(net)
    assert small.report["compact_sizes"] == [0, 0, 3]
    x = rng(10).standard_normal((4, 6))
    for path in (CompactDensePath(small), CSRPath(small)):
        np.testing.assert_allclose(path(x), net.forward(x), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(path(x), np.broadcast_to(net.biases[1], (4, 3)))


def test_an_unpruned_network_is_left_unchanged():
    net = random_net([8, 6, 3], 1.0, seed=11)
    small = compact(net)
    assert small.report["compact_sizes"] == [8, 6, 3]
    assert small.report["dense_macs_after"] == small.report["dense_macs_before"] == 8 * 6 + 6 * 3
    np.testing.assert_array_equal(small.input_index, np.arange(8))


# --- cost accounting ------------------------------------------------------------------------


def test_mac_and_byte_accounting():
    net = handmade_net()
    small = compact(net)
    dense, compact_dense, csr = DensePath(net), CompactDensePath(small), CSRPath(small)
    assert dense.macs == 5 * 4 + 4 * 3 + 3 * 2  # dense-times-zero pays for every pruned weight
    assert compact_dense.macs == 3 * 1 + 1 * 1 + 1 * 2
    assert csr.macs == small.report["nnz_after"] == 3 + 1 + 2  # h1u3 <- 3 inputs, h2u1 <- h1u3, outputs <- h2u1
    assert small.report["nnz_before"] == net.nnz == 13
    assert dense.nbytes == 8 * (5 * 4 + 4 + 4 * 3 + 3 + 3 * 2 + 2)  # float64 in this test
    assert csr.nbytes == sum(m.nbytes for m in csr.matrices) + 8 * (1 + 1 + 2) + 3 * 4


# --- end to end: a trained, saved, reloaded model ---------------------------------------------


def test_saved_model_reproduces_the_experiments_test_predictions_on_every_path(tmp_path):
    from train.data import load_dataset
    from train.experiment import ExperimentConfig, run_experiment
    from train.stats import unpack_bits

    config = ExperimentConfig(
        dataset="spirals", hidden=(32, 32), epochs=8, batch_size=64, lr=3e-3,
        sparsity=0.8, criterion="fisher", prune_interval=10,
    )  # fmt: skip
    result = run_experiment(config, log=None, model_path=tmp_path / "model.npz")
    net = PrunedMLP.load(tmp_path / "model.npz")
    assert net.meta["config"]["sparsity"] == 0.8 and net.nnz == result["model"]["nonzero_weights"]
    assert result["structure"] == compact(net).report

    data = load_dataset("spirals")
    expected = unpack_bits(result["test"]["correct_packed"], len(data.y_test))
    assert expected.mean() == result["test"]["acc"]
    small = compact(net)
    for path in (DensePath(net), CompactDensePath(small), CSRPath(small)):
        correct = path(data.x_test).argmax(axis=1) == data.y_test
        np.testing.assert_array_equal(correct, expected)


# --- the cost benchmark ---------------------------------------------------------------------


def test_benchmark_measures_every_path_and_checks_agreement():
    from infer.benchmark import measure_model, time_call

    timing = time_call(lambda x: x @ x.T, np.ones((8, 8)), min_seconds=0.001, repeats=3)
    assert 0 < timing["min"] <= timing["median"] <= timing["max"] and timing["loops"] >= 1

    net = random_net([30, 20, 10, 4], 0.1, seed=12, dtype=np.float32)
    net.meta = {"config": {"criterion": "fisher"}, "test_acc": None}
    x = rng(13).standard_normal((50, 30)).astype(np.float32)
    y = rng(14).integers(0, 4, size=50)
    result = measure_model(net, x, y, batch_sizes=(1, 4), min_seconds=0.001)
    assert [r["path"] for r in result["paths"]] == ["dense", "compact_dense", "compact_aligned", "csr"]
    assert result["criterion"] == "fisher" and result["nonzero_weights"] == net.nnz
    for row in result["paths"]:
        assert row["predictions_differing_from_dense"] == 0 and row["max_abs_logit_difference"] < 1e-4
        assert set(row["seconds_per_batch"]) == {"1", "4"}
    macs = {r["path"]: r["macs_per_example"] for r in result["paths"]}
    assert macs["dense"] == 30 * 20 + 20 * 10 + 10 * 4
    assert macs["dense"] >= macs["compact_dense"] >= macs["csr"]

    unpruned = random_net([6, 5, 3], 1.0, seed=15, dtype=np.float32)
    only = measure_model(unpruned, x[:, :6], y, batch_sizes=(1,), min_seconds=0.001)
    assert only["criterion"] == "dense" and [r["path"] for r in only["paths"]] == ["dense"]


def test_aligned_path_keeps_all_inputs_pads_hidden_widths_and_is_exact():
    net = handmade_net()
    small = compact(net)
    aligned = AlignedDensePath(small, multiple=4)
    assert aligned.widths == [5, 4, 4, 2]  # inputs kept at 5; hidden 1 -> 4 and 1 -> 4; outputs untouched
    x = rng(16).standard_normal((30, 5))
    np.testing.assert_allclose(aligned(x), net.forward(x), rtol=1e-12, atol=1e-12)
    assert aligned.macs == 5 * 4 + 4 * 4 + 4 * 2

    big = random_net([70, 50, 33, 10], 0.05, seed=17)
    path = AlignedDensePath(compact(big))
    assert path.widths[0] == 70 and all(w % 32 == 0 for w in path.widths[1:-1]) and path.widths[-1] == 10
    x = rng(18).standard_normal((20, 70))
    np.testing.assert_allclose(path(x), big.forward(x), rtol=1e-10, atol=1e-10)


def test_shape_diagnostic_reports_every_compacted_layer():
    from infer.benchmark import shape_diagnostic

    net = random_net([40, 30, 20, 4], 0.08, seed=19, dtype=np.float32)
    x = rng(20).standard_normal((16, 40)).astype(np.float32)
    d = shape_diagnostic(net, x, batch=8, multiple=16, min_seconds=0.001)
    small = compact(net)
    assert d["compact_sizes"] == small.report["compact_sizes"] and len(d["layers"]) == 3
    for layer, w in zip(d["layers"], small.weights):
        assert layer["shape"] == list(w.shape) and layer["padded_shape"][0] % 16 == 0
        assert layer["seconds"] > 0 and layer["padded_seconds"] > 0
    assert d["layers"][-1]["padded_shape"][1] == 4  # the output layer is never padded
