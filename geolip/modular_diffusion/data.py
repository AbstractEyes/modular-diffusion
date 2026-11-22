"""Data preparation: CIFAR-10 from parquet files (the Hugging Face Hub layout) to one npz of uint8 arrays.

Needs pyarrow (imported when called).
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
from typing import Optional, Tuple

import numpy as np


def _load(path: str):
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path).to_pydict()
    key = "img" if "img" in t else "image"
    imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t[key]])
    return imgs.astype(np.uint8), np.array(t["label"], dtype=np.int64)


NPZ_ARRAYS = ("x_train", "y_train", "x_test", "y_test")


def npz_content_hash(path: str) -> str:
    """SHA-256 over the arrays x_train, y_train, x_test, y_test (each: its name, dtype, shape and raw little-endian
    bytes). It does not depend on the zip container, so two saves of the same arrays give the same hash."""
    h = hashlib.sha256()
    with np.load(path) as d:
        for name in NPZ_ARRAYS:
            a = np.ascontiguousarray(d[name])
            if a.dtype.byteorder == ">":
                a = a.astype(a.dtype.newbyteorder("<"))
            h.update(b"\0" + name.encode("ascii") + b"\0" + str(a.dtype).encode("ascii") + b"\0"
                     + json.dumps(list(a.shape)).encode("ascii") + b"\0")
            h.update(a.tobytes())
    return h.hexdigest()


def cifar_npz(parquet_dir: str, out: str, expect: Optional[Tuple[int, int]] = (50000, 10000)) -> str:
    """parquet_dir holds exactly one train-*.parquet and one test-*.parquet. Writes x_train [N, 32, 32, 3], y_train,
    x_test [M, 32, 32, 3], y_test to ``out`` and returns its path. ``expect`` (N, M) is checked unless None."""
    tr = glob.glob(os.path.join(parquet_dir, "train-*.parquet"))
    te = glob.glob(os.path.join(parquet_dir, "test-*.parquet"))
    if len(tr) != 1 or len(te) != 1:
        raise FileNotFoundError(f"expected one train and one test parquet file in {parquet_dir}: {tr}, {te}")
    xtr, ytr = _load(tr[0])
    xte, yte = _load(te[0])
    if expect is not None and (xtr.shape != (expect[0], 32, 32, 3) or xte.shape != (expect[1], 32, 32, 3)):
        raise ValueError(f"unexpected shapes {xtr.shape}, {xte.shape}")
    folder = os.path.dirname(out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez(out, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    return out
