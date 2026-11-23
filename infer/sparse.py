"""Inference for pruned ReLU MLPs in plain NumPy, without the autodiff engine.

Four ways to run the same pruned network, giving the same outputs up to
floating-point rounding:

* `DensePath`: full-size matrices, i.e. the dense-times-zero shortcut.
* `CompactDensePath`: dense products after `compact` removes dead inputs and units.
* `AlignedDensePath`: compacted hidden units, full input width, hidden widths
  zero-padded to a multiple of 32 (shapes this BLAS runs fast).
* `CSRPath`: sparse products on the compacted network; only surviving weights
  are multiplied.

`macs` counts weight multiply-accumulates per example (biases and ReLUs not
counted); `nbytes` is the parameter memory a path holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Upper bound on the (batch x nnz) temporary the CSR product materialises at once.
CSR_CHUNK_ELEMENTS = 1 << 22


def relu_mlp_forward(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """Logits of Linear -> ReLU -> ... -> Linear with dense matrix products."""
    h = x
    last = len(weights) - 1
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w
        h += b
        if i != last:
            np.maximum(h, 0, out=h)
    return h


# --------------------------------------------------------------------------- the pruned network


@dataclass
class PrunedMLP:
    """A trained ReLU MLP as plain arrays: weights (n_in, n_out) with exact zeros where pruned."""

    weights: list
    biases: list
    masks: list  # bool, True = connection exists
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_model(cls, model, meta: Optional[dict] = None) -> "PrunedMLP":
        """Extract the arrays the forward pass of an `nn.MLP` actually uses."""
        if model.activation != "relu":
            raise ValueError("inference supports ReLU MLPs only")
        weights, biases, masks = [], [], []
        for layer in model.linears:
            w = layer.weight
            weights.append(np.array(w.effective().data))  # exact zeros where masked
            biases.append(np.array(layer.bias.data))
            masks.append(np.ones(w.shape, dtype=bool) if w.mask is None else np.array(w.mask))
        return cls(weights, biases, masks, dict(meta or {}))

    @property
    def sizes(self) -> list:
        return [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]

    @property
    def nnz(self) -> int:
        return int(sum(m.sum() for m in self.masks))

    @property
    def num_weights(self) -> int:
        return int(sum(m.size for m in self.masks))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return relu_mlp_forward(x, self.weights, self.biases)

    def save(self, path: Path) -> None:
        """Compressed .npz holding only surviving weights (flat int32 index + value per layer)."""
        arrays = {"meta": np.array(json.dumps(self.meta))}
        for i, (w, b, m) in enumerate(zip(self.weights, self.biases, self.masks)):
            index = np.flatnonzero(m).astype(np.int32)
            arrays[f"shape_{i}"] = np.array(w.shape, dtype=np.int64)
            arrays[f"index_{i}"] = index
            arrays[f"value_{i}"] = w.ravel()[index]
            arrays[f"bias_{i}"] = b
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "PrunedMLP":
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            weights, biases, masks = [], [], []
            i = 0
            while f"shape_{i}" in f:
                shape = tuple(int(n) for n in f[f"shape_{i}"])
                index, value = f[f"index_{i}"], f[f"value_{i}"]
                w = np.zeros(int(np.prod(shape)), dtype=value.dtype)
                m = np.zeros(w.size, dtype=bool)
                w[index], m[index] = value, True
                weights.append(w.reshape(shape))
                masks.append(m.reshape(shape))
                biases.append(np.array(f[f"bias_{i}"]))
                i += 1
        return cls(weights, biases, masks, meta)


# --------------------------------------------------------------------------- compaction


@dataclass
class CompactMLP:
    """An exactly equivalent, smaller network: inputs `input_index` of x feed `weights[0]`."""

    input_index: np.ndarray
    weights: list
    biases: list
    masks: list
    report: dict


def compact(net: PrunedMLP) -> CompactMLP:
    """Remove every input and hidden unit that cannot affect the output.

    Repeated until nothing changes (removing one unit can strand another):

    * An input with no surviving outgoing weight is never read: drop it.
    * A hidden unit with no surviving outgoing weight is never read: drop it.
    * A hidden unit with outgoing weights but no incoming ones outputs the
      constant relu(b_j). Fold it into the next layer's bias,
      b_next += relu(b_j) * W_next[j, :], then drop it. (Its outgoing weights
      no longer count as multiplications: they became one bias addition.)

    The output layer is never reduced. The result computes the same function
    as `net`; only the order of floating-point additions changes.
    """
    weights = [w.copy() for w in net.weights]
    biases = [b.copy() for b in net.biases]
    masks = [m.copy() for m in net.masks]
    input_index = np.arange(weights[0].shape[0])
    n_hidden = len(weights) - 1
    removed = [0] * n_hidden
    folded = [0] * n_hidden

    changed = True
    while changed:
        changed = False
        used = masks[0].any(axis=1)
        if not used.all():
            input_index = input_index[used]
            weights[0], masks[0] = weights[0][used], masks[0][used]
            changed = True
        for k in range(n_hidden):
            has_in = masks[k].any(axis=0)
            has_out = masks[k + 1].any(axis=1)
            constant = has_out & ~has_in
            if constant.any():
                value = np.maximum(biases[k][constant], 0)
                biases[k + 1] = biases[k + 1] + value @ weights[k + 1][constant]
                folded[k] += int(constant.sum())
            keep = has_in & has_out
            if not keep.all():
                removed[k] += int((~keep).sum())
                weights[k], masks[k] = weights[k][:, keep], masks[k][:, keep]
                biases[k] = biases[k][keep]
                weights[k + 1], masks[k + 1] = weights[k + 1][keep], masks[k + 1][keep]
                changed = True

    sizes = [len(input_index)] + [w.shape[1] for w in weights]
    report = {
        "original_sizes": net.sizes,
        "compact_sizes": sizes,
        "inputs_removed": net.sizes[0] - sizes[0],
        "hidden_units_removed": removed,
        "hidden_units_folded_into_bias": folded,
        "nnz_before": net.nnz,
        "nnz_after": int(sum(m.sum() for m in masks)),
        "dense_macs_before": int(sum(w.size for w in net.weights)),
        "dense_macs_after": int(sum(w.size for w in weights)),
    }
    return CompactMLP(input_index, weights, biases, masks, report)


# --------------------------------------------------------------------------- CSR


class CSRMatrix:
    """A sparse (n_in x n_out) weight matrix stored by output unit.

    For output j, `indices[indptr[j]:indptr[j+1]]` are the inputs feeding it
    and `data[...]` the weights. (This is CSR of W transposed.) The product
    x @ W gathers the needed input columns, multiplies by the weights and sums
    each output's segment: 2 * nnz flops per example, nothing for pruned
    weights.
    """

    def __init__(self, dense: np.ndarray, mask: np.ndarray):
        self.n_in, self.n_out = dense.shape
        out_index, in_index = np.nonzero(mask.T)  # row-major over outputs
        self.indices = in_index.astype(np.int32)
        self.data = np.ascontiguousarray(dense.T[out_index, in_index])
        counts = np.bincount(out_index, minlength=self.n_out)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        self.nonempty = counts > 0
        self.all_nonempty = bool(self.nonempty.all())
        self.starts = self.indptr[:-1][self.nonempty]

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes + self.indices.nbytes + self.indptr.nbytes)

    def to_dense(self) -> np.ndarray:
        w = np.zeros((self.n_out, self.n_in), dtype=self.data.dtype)
        rows = np.repeat(np.arange(self.n_out), np.diff(self.indptr))
        w[rows, self.indices] = self.data
        return w.T

    def matmul(self, x: np.ndarray, chunk_elements: int = CSR_CHUNK_ELEMENTS) -> np.ndarray:
        batch = x.shape[0]
        out = np.zeros((batch, self.n_out), dtype=np.result_type(x, self.data))
        if self.nnz == 0:
            return out
        rows = max(1, chunk_elements // self.nnz)
        for start in range(0, batch, rows):
            products = x[start : start + rows, self.indices]  # (rows, nnz) gather
            products *= self.data
            sums = np.add.reduceat(products, self.starts, axis=1)  # one segment per non-empty output
            if self.all_nonempty:
                out[start : start + rows] = sums
            else:
                out[start : start + rows, self.nonempty] = sums
        return out


# --------------------------------------------------------------------------- execution paths


class DensePath:
    """Dense matrix products on the full-size weights (zeros included)."""

    name = "dense"

    def __init__(self, net: PrunedMLP):
        self.weights, self.biases = net.weights, net.biases
        self.macs = int(sum(w.size for w in self.weights))
        self.nbytes = int(sum(w.nbytes + b.nbytes for w, b in zip(self.weights, self.biases)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return relu_mlp_forward(x, self.weights, self.biases)


class CompactDensePath:
    """Dense matrix products on the compacted network."""

    name = "compact_dense"

    def __init__(self, net: CompactMLP):
        self.net = net
        self.all_inputs = len(net.input_index) == net.report["original_sizes"][0]
        self.macs = int(sum(w.size for w in net.weights))
        self.nbytes = int(
            sum(w.nbytes + b.nbytes for w, b in zip(net.weights, net.biases))
            + (0 if self.all_inputs else len(net.input_index) * 4)
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.all_inputs:
            x = x[:, self.net.input_index]
        return relu_mlp_forward(x, self.net.weights, self.net.biases)


class AlignedDensePath:
    """Compacted hidden layers, full-width input, hidden widths zero-padded to `multiple`.

    Padding adds zero rows and columns only, so the function is unchanged:
    a padded unit has zero weights in and out and a zero bias, outputs relu(0) = 0,
    and is multiplied by zero downstream.
    """

    name = "compact_aligned"

    def __init__(self, net: CompactMLP, multiple: int = 32):
        n_inputs = net.report["original_sizes"][0]
        widths = [n_inputs] + [-(-w.shape[1] // multiple) * multiple for w in net.weights[:-1]] + [net.weights[-1].shape[1]]
        weights, biases = [], []
        for i, (w, b) in enumerate(zip(net.weights, net.biases)):
            padded = np.zeros((widths[i], widths[i + 1]), dtype=w.dtype)
            rows = net.input_index if i == 0 else np.arange(w.shape[0])
            padded[rows, : w.shape[1]] = w
            bias = np.zeros(widths[i + 1], dtype=b.dtype)
            bias[: b.shape[0]] = b
            weights.append(padded)
            biases.append(bias)
        self.weights, self.biases, self.widths = weights, biases, widths
        self.macs = int(sum(w.size for w in weights))
        self.nbytes = int(sum(w.nbytes + b.nbytes for w, b in zip(weights, biases)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return relu_mlp_forward(x, self.weights, self.biases)


class CSRPath:
    """Sparse matrix products on the compacted network: only surviving weights are multiplied."""

    name = "csr"

    def __init__(self, net: CompactMLP):
        self.net = net
        self.all_inputs = len(net.input_index) == net.report["original_sizes"][0]
        self.matrices = [CSRMatrix(w, m) for w, m in zip(net.weights, net.masks)]
        self.biases = net.biases
        self.macs = int(sum(m.nnz for m in self.matrices))
        self.nbytes = int(
            sum(m.nbytes + b.nbytes for m, b in zip(self.matrices, self.biases))
            + (0 if self.all_inputs else len(net.input_index) * 4)
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x if self.all_inputs else x[:, self.net.input_index]
        last = len(self.matrices) - 1
        for i, (m, b) in enumerate(zip(self.matrices, self.biases)):
            h = m.matmul(h)
            h += b
            if i != last:
                np.maximum(h, 0, out=h)
        return h
