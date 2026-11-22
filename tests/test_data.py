"""Data preparation (test 13): the content hash of an npz, parquet to npz on a tiny pair, and (when MD_CIFAR_PARQUET
names a folder holding the two CIFAR-10 parquet files, and MD_CIFAR_NPZ the npz) the real file's content."""
import io
import os

import numpy as np
import pytest

from geolip.modular_diffusion.data import cifar_npz, npz_content_hash


def _arrays(seed=0, n=5, m=3):
    rng = np.random.default_rng(seed)
    return dict(x_train=rng.integers(0, 256, (n, 32, 32, 3)).astype(np.uint8), y_train=rng.integers(0, 10, n).astype(np.int64),
                x_test=rng.integers(0, 256, (m, 32, 32, 3)).astype(np.uint8), y_test=rng.integers(0, 10, m).astype(np.int64))


def test_13_content_hash_ignores_the_container(tmp_path):
    arr = _arrays()
    np.savez(tmp_path / "a.npz", **arr)
    np.savez_compressed(tmp_path / "b.npz", **arr)
    assert open(tmp_path / "a.npz", "rb").read() != open(tmp_path / "b.npz", "rb").read()
    assert npz_content_hash(str(tmp_path / "a.npz")) == npz_content_hash(str(tmp_path / "b.npz"))
    other = dict(arr, y_test=arr["y_test"] + 1)
    np.savez(tmp_path / "c.npz", **other)
    assert npz_content_hash(str(tmp_path / "c.npz")) != npz_content_hash(str(tmp_path / "a.npz"))


def test_13_cifar_npz_on_a_tiny_parquet_pair(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image
    arr = _arrays(1)
    src = tmp_path / "parquet"
    src.mkdir()
    for split in ("train", "test"):
        imgs = []
        for x in arr[f"x_{split}"]:
            buf = io.BytesIO()
            Image.fromarray(x).save(buf, format="PNG")
            imgs.append({"bytes": buf.getvalue(), "path": None})
        table = pa.table({"img": imgs, "label": arr[f"y_{split}"].tolist()})
        pq.write_table(table, str(src / f"{split}-00000-of-00001.parquet"))
    out = str(tmp_path / "out" / "tiny.npz")
    with pytest.raises(ValueError):
        cifar_npz(str(src), out)
    assert cifar_npz(str(src), out, expect=None) == out
    with np.load(out) as d:
        for k, v in arr.items():
            assert np.array_equal(d[k], v), k
    np.savez(tmp_path / "direct.npz", **arr)
    assert npz_content_hash(out) == npz_content_hash(str(tmp_path / "direct.npz"))


def test_13_the_real_parquet_files_give_the_stored_npz(tmp_path):
    folder = os.environ.get("MD_CIFAR_PARQUET", "")
    npz = os.environ.get("MD_CIFAR_NPZ", "")
    if not folder or not npz:
        pytest.skip("MD_CIFAR_PARQUET and MD_CIFAR_NPZ are not both set")
    pytest.importorskip("pyarrow")
    out = cifar_npz(folder, str(tmp_path / "cifar10_32.npz"))
    assert npz_content_hash(out) == npz_content_hash(npz)
