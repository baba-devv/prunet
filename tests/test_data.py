"""Dataset loading: IDX parsing, download integrity, splits."""

import gzip
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

import train.data as data_module
from train.data import load_dataset, make_spirals, read_idx

MNIST_DIR = Path(__file__).resolve().parent.parent / "data" / "mnist"


def write_idx(path: Path, array: np.ndarray) -> None:
    header = struct.pack(">HBB", 0, 0x08, array.ndim) + struct.pack(f">{array.ndim}I", *array.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + array.astype(np.uint8).tobytes())


# --- IDX ---------------------------------------------------------------------


def test_read_idx_round_trip(tmp_path):
    images = np.random.default_rng(0).integers(0, 256, size=(7, 4, 3), dtype=np.uint8)
    labels = np.arange(7, dtype=np.uint8)
    write_idx(tmp_path / "images.gz", images)
    write_idx(tmp_path / "labels.gz", labels)
    np.testing.assert_array_equal(read_idx(tmp_path / "images.gz"), images)
    np.testing.assert_array_equal(read_idx(tmp_path / "labels.gz"), labels)


def test_read_idx_rejects_corrupt_files(tmp_path):
    with gzip.open(tmp_path / "bad_magic.gz", "wb") as f:
        f.write(struct.pack(">HBB", 0, 0x0D, 1) + struct.pack(">I", 2) + b"\x00\x00")
    with pytest.raises(ValueError, match="not an unsigned-byte IDX"):
        read_idx(tmp_path / "bad_magic.gz")

    with gzip.open(tmp_path / "truncated.gz", "wb") as f:
        f.write(struct.pack(">HBB", 0, 0x08, 1) + struct.pack(">I", 10) + b"\x00" * 4)
    with pytest.raises(ValueError, match="header promises"):
        read_idx(tmp_path / "truncated.gz")


# --- download integrity ------------------------------------------------------


def test_download_rejects_a_file_with_the_wrong_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "MNIST_FILES", {"file.gz": "0" * 64})
    monkeypatch.setattr(data_module, "_fetch", lambda url, target: target.write_bytes(b"tampered"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data_module.download_mnist(tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing unverified is left behind


def test_download_is_skipped_when_a_verified_file_is_present(tmp_path, monkeypatch):
    (tmp_path / "file.gz").write_bytes(b"payload")
    monkeypatch.setattr(data_module, "MNIST_FILES", {"file.gz": hashlib.sha256(b"payload").hexdigest()})

    def no_network(url, target):
        raise AssertionError("should not download")

    monkeypatch.setattr(data_module, "_fetch", no_network)
    data_module.download_mnist(tmp_path)


def test_download_falls_through_to_the_second_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "MNIST_FILES", {"file.gz": hashlib.sha256(b"payload").hexdigest()})
    calls = []

    def flaky(url, target):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("mirror down")
        target.write_bytes(b"payload")

    monkeypatch.setattr(data_module, "_fetch", flaky)
    data_module.download_mnist(tmp_path)
    assert len(calls) == 2 and (tmp_path / "file.gz").read_bytes() == b"payload"


# --- spirals -----------------------------------------------------------------


def test_spirals_are_balanced_bounded_and_seeded():
    x, y = make_spirals(np.random.default_rng(0), n_per_class=100, n_classes=3)
    assert x.shape == (300, 2) and y.dtype == np.int64
    assert np.bincount(y).tolist() == [100, 100, 100]
    assert np.linalg.norm(x, axis=1).max() <= 1.0 + 1e-12
    x2, _ = make_spirals(np.random.default_rng(0), n_per_class=100, n_classes=3)
    np.testing.assert_array_equal(x, x2)


def test_spirals_are_not_linearly_separable():
    """A least-squares linear classifier does poorly on them."""
    x, y = make_spirals(np.random.default_rng(0), n_per_class=300)
    design = np.hstack([x, np.ones((len(x), 1))])
    weights, *_ = np.linalg.lstsq(design, np.eye(3)[y], rcond=None)
    assert ((design @ weights).argmax(axis=1) == y).mean() < 0.6


def test_spirals_dataset_splits_are_disjoint_and_standardised():
    data = load_dataset("spirals", dtype=np.float64)
    assert (len(data.x_train), len(data.x_val), len(data.x_test)) == (1200, 300, 300)
    assert data.num_features == 2 and data.num_classes == 3
    rows = np.concatenate([data.x_train, data.x_val, data.x_test])
    assert len(np.unique(rows, axis=0)) == 1800  # no example appears in two splits
    assert data.x_train.mean() == pytest.approx(0.0, abs=1e-12)
    assert data.x_train.std() == pytest.approx(1.0, rel=1e-12)
    again = load_dataset("spirals", dtype=np.float64)
    np.testing.assert_array_equal(data.x_val, again.x_val)  # the split does not depend on the run seed
    assert load_dataset("spirals").x_train.dtype == np.float32


def test_other_splits_are_standardised_with_training_statistics():
    """Validation/test data must not influence (or be normalised by) their own statistics."""
    train, other = np.array([[0.0], [2.0]]), np.array([[10.0], [10.0]])
    train_std, other_std = data_module._standardise(train, other, dtype=np.float64)
    np.testing.assert_array_equal(train_std, [[-1.0], [1.0]])
    np.testing.assert_array_equal(other_std, [[9.0], [9.0]])  # (10 - train mean 1) / train std 1


def test_unknown_dataset_is_an_error():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("cifar")


# --- MNIST (only if already downloaded) ----------------------------------------


@pytest.mark.skipif(not (MNIST_DIR / "train-images-idx3-ubyte.gz").exists(), reason="MNIST not downloaded")
def test_mnist_splits():
    data = load_dataset("mnist", root=MNIST_DIR.parent)
    assert data.x_train.shape == (55000, 784) and data.x_val.shape == (5000, 784) and data.x_test.shape == (10000, 784)
    assert data.num_classes == 10 and data.x_train.dtype == np.float32
    assert np.bincount(data.y_test).tolist() == [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
    assert abs(float(data.x_train.mean())) < 1e-4 and float(data.x_train.std()) == pytest.approx(1.0, abs=1e-4)
