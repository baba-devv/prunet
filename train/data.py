"""Datasets: MNIST (downloaded and parsed with the standard library) and synthetic spirals."""

from __future__ import annotations

import gzip
import hashlib
import shutil
import struct
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
# SHA-256 of the gzipped IDX files: a download that does not match is rejected,
# so every run trains on byte-identical data.
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
SPLIT_SEED = 0  # the train/validation split is fixed; run seeds vary only init and batch order


@dataclass
class Dataset:
    """Standardised inputs and integer labels. The test split is for final reporting only."""

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def num_features(self) -> int:
        return self.x_train.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y_train.max()) + 1


# ------------------------------------------------------------------- MNIST


def read_idx(path: Path) -> np.ndarray:
    """Parse a gzipped IDX file: big-endian magic (0, 0, dtype=0x08, ndim), dims, then uint8 data."""
    with gzip.open(path, "rb") as f:
        zero, dtype_code, ndim = struct.unpack(">HBB", f.read(4))
        if zero != 0 or dtype_code != 0x08:
            raise ValueError(f"{path} is not an unsigned-byte IDX file")
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: header promises {shape}, found {data.size} bytes")
    return data.reshape(shape)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str, target: Path) -> None:
    """Download with urllib, falling back to curl (python.org builds on macOS
    ship without CA certificates, so urllib can fail TLS there)."""
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as urllib_error:
        if shutil.which("curl") is None:
            raise
        try:
            subprocess.run(["curl", "-fsSL", "--max-time", "300", "-o", str(target), url], check=True)
        except subprocess.CalledProcessError as curl_error:
            raise RuntimeError(f"urllib: {urllib_error}; curl: exit {curl_error.returncode}") from None


def download_mnist(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected in MNIST_FILES.items():
        target = root / filename
        if target.exists() and _sha256(target) == expected:
            continue
        partial = root / (filename + ".part")
        errors = []
        for mirror in MNIST_MIRRORS:
            try:
                _fetch(mirror + filename, partial)
                if _sha256(partial) != expected:
                    raise ValueError("checksum mismatch")
                partial.replace(target)
                break
            except Exception as error:  # try the next mirror
                errors.append(f"{mirror}: {error}")
        else:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not download {filename} (you can also place the four MNIST .gz files in {root}/):\n  "
                + "\n  ".join(errors)
            )


def load_mnist_arrays(root: Path = Path("data/mnist")) -> tuple:
    """(train_images uint8 (60000, 784), train_labels, test_images (10000, 784), test_labels)."""
    download_mnist(root)
    train_x = read_idx(root / "train-images-idx3-ubyte.gz").reshape(60000, 784)
    train_y = read_idx(root / "train-labels-idx1-ubyte.gz").astype(np.int64)
    test_x = read_idx(root / "t10k-images-idx3-ubyte.gz").reshape(10000, 784)
    test_y = read_idx(root / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    return train_x, train_y, test_x, test_y


# ------------------------------------------------------------------ spirals


def make_spirals(
    rng: np.random.Generator, n_per_class: int = 500, n_classes: int = 3, noise: float = 0.15, turns: float = 1.25
) -> tuple:
    """Interleaved 2-D spiral arms: not linearly separable, so a hidden layer is required."""
    xs, ys = [], []
    for c in range(n_classes):
        radius = np.linspace(0.05, 1.0, n_per_class)
        angle = radius * turns * 2 * np.pi + c * 2 * np.pi / n_classes
        angle = angle + noise * rng.standard_normal(n_per_class)
        xs.append(np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1))
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


# ------------------------------------------------------------------ assembly


def _standardise(train: np.ndarray, *others: np.ndarray, dtype) -> list:
    """Shift and scale every split with the *training* split's global mean and std."""
    mean, std = train.mean(), train.std()
    return [((a - mean) / std).astype(dtype) for a in (train, *others)]


def _split(x: np.ndarray, y: np.ndarray, n_holdout: int, rng: np.random.Generator) -> tuple:
    order = rng.permutation(len(x))
    keep, held = order[n_holdout:], order[:n_holdout]
    return x[keep], y[keep], x[held], y[held]


def load_dataset(name: str, dtype=np.float32, root: Path = Path("data")) -> Dataset:
    split_rng = np.random.default_rng(SPLIT_SEED)
    if name == "mnist":
        train_x, train_y, test_x, test_y = load_mnist_arrays(root / "mnist")
        train_x, test_x = train_x / 255.0, test_x / 255.0
        train_x, train_y, val_x, val_y = _split(train_x, train_y, 5000, split_rng)
    elif name == "spirals":
        x, y = make_spirals(split_rng, n_per_class=600)
        x, y, test_x, test_y = _split(x, y, 300, split_rng)
        train_x, train_y, val_x, val_y = _split(x, y, 300, split_rng)
    else:
        raise ValueError(f"unknown dataset {name!r} (expected 'mnist' or 'spirals')")
    train_x, val_x, test_x = _standardise(train_x, val_x, test_x, dtype=dtype)
    return Dataset(name, train_x, train_y, val_x, val_y, test_x, test_y)
