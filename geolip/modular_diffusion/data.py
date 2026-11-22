"""Data preparation: CIFAR-10 from parquet files (the Hugging Face Hub layout) to one npz of uint8 arrays.

Needs pyarrow (imported when called).
"""
from __future__ import annotations

import glob
import io
import os

import numpy as np


def _load(path: str):
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path).to_pydict()
    key = "img" if "img" in t else "image"
    imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t[key]])
    return imgs.astype(np.uint8), np.array(t["label"], dtype=np.int64)


def cifar_npz(parquet_dir: str, out: str) -> str:
    """parquet_dir holds exactly one train-*.parquet and one test-*.parquet. Writes x_train [50000, 32, 32, 3],
    y_train, x_test [10000, 32, 32, 3], y_test to ``out`` and returns its path."""
    tr = glob.glob(os.path.join(parquet_dir, "train-*.parquet"))
    te = glob.glob(os.path.join(parquet_dir, "test-*.parquet"))
    if len(tr) != 1 or len(te) != 1:
        raise FileNotFoundError(f"expected one train and one test parquet file in {parquet_dir}: {tr}, {te}")
    xtr, ytr = _load(tr[0])
    xte, yte = _load(te[0])
    if xtr.shape != (50000, 32, 32, 3) or xte.shape != (10000, 32, 32, 3):
        raise ValueError(f"unexpected shapes {xtr.shape}, {xte.shape}")
    folder = os.path.dirname(out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez(out, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    return out
