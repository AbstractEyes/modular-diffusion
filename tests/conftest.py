"""Shared fixtures.

The parity tests compare the package with the original research script and the block library it was written against.
Neither is part of this repository; they are found through environment variables, and the tests that need them skip
when a variable is unset:

    MD_LEGACY_TOOLS     the folder holding the original script (found there by its SHA-256)
    MD_GEOFRACTAL_SRC   source folder(s) of the block library, joined by the OS path separator
    MD_CIFAR_NPZ        the CIFAR-10 npz (uint8 x_train, y_train, x_test, y_test)
    MD_STORED_RUNS      the folder of stored results
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

SCRIPT_SHA256 = "361319f196c520a9b15b8395dd7050759ae5999f2f2c24500fc2c2cb07c66460"
SHA256 = {
    "blocks.py": "978c1521127b9a66028d967091978c55472c3aebd3f6a92d5ab4c14b0b93c86c",
    "towers.py": "f89e3d1bf67f14b993f62a494762a718f2365fc0042be5ed5396fe830c139da5",
    "model.py": "8c0061cc17021c25cc791dec63b8c2d5f7ce5e4a992bfbc696abab52f03dfb54",
}
NPZ_SHA256 = "27d3cc7c2c65fe18d7d11df0f96e5914faa5294ad5cda9c3fd882d6c03e4f3da"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        pytest.skip(f"{name} is not set")
    return v


@pytest.fixture(scope="session")
def geofractal_src():
    paths = [p for p in _env("MD_GEOFRACTAL_SRC").split(os.pathsep) if p]
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    import geofractal.router.prefab.agatha_v2 as pkg
    folder = os.path.dirname(pkg.__file__)
    for name in ("blocks.py", "towers.py", "model.py"):
        assert sha256(os.path.join(folder, name)) == SHA256[name], f"the library's {name} is not the expected version"
    return paths


@pytest.fixture(scope="session")
def gf(geofractal_src):
    """The library's blocks, towers and model modules."""
    from geofractal.router.prefab.agatha_v2 import blocks, model, towers
    return {"blocks": blocks, "towers": towers, "model": model}


@pytest.fixture(scope="session")
def legacy(geofractal_src):
    """The original script, loaded as a module (its top level imports only the standard library, numpy and torch)."""
    folder = _env("MD_LEGACY_TOOLS")
    found = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
             if n.endswith(".py") and sha256(os.path.join(folder, n)) == SCRIPT_SHA256]
    assert found, "the original script (by its SHA-256) is not in MD_LEGACY_TOOLS"
    path = found[0]
    spec = importlib.util.spec_from_file_location("legacy_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def cifar_npz():
    path = _env("MD_CIFAR_NPZ")
    assert sha256(path) == NPZ_SHA256, "the CIFAR-10 npz is not the expected file"
    return path


STORED_BASE_PT = "155f9e590a2f6fb8ecf15027095be27a92b6f200751993f691a0d0fcf8d3311e"   # the 6,000-step base weights
STORED_SMOKE_PT = "9abde5083c1d98a643e146d5fd35754b6a5cd24a47a8a536b508b15b0e06a309"  # the recorded 4-step CPU run


def find_stored(root: str, suffix: str, digest: str) -> str:
    """A stored file in a direct subfolder of ``root`` whose name ends with ``suffix`` and whose SHA-256 is ``digest``
    (a run repeated to the bit may be stored twice; any copy will do)."""
    for sub in sorted(os.listdir(root)):
        folder = os.path.join(root, sub)
        if os.path.isdir(folder):
            for n in sorted(os.listdir(folder)):
                if n.endswith(suffix) and sha256(os.path.join(folder, n)) == digest:
                    return os.path.join(folder, n)
    raise AssertionError(f"no stored file ending {suffix!r} with SHA-256 {digest}")


@pytest.fixture(scope="session")
def stored_runs():
    return _env("MD_STORED_RUNS")


@pytest.fixture(scope="session")
def synthetic_npz(tmp_path_factory):
    """A small npz in the CIFAR layout: 600 training and 300 test pictures of 32 x 32 x 3, ten classes."""
    rng = np.random.default_rng(12345)
    path = tmp_path_factory.mktemp("npz") / "synthetic32.npz"
    yy, xx = np.mgrid[0:32, 0:32]
    def pictures(n):
        base = rng.integers(0, 256, size=(n, 32, 32, 3)).astype(np.float64)
        smooth = 127 + 100 * np.sin(xx / 5.0)[None, :, :, None] * np.cos(yy / 7.0)[None, :, :, None]
        return np.clip(0.5 * base + 0.5 * smooth + rng.normal(0, 10, size=(n, 1, 1, 3)), 0, 255).astype(np.uint8)
    np.savez(path, x_train=pictures(600), y_train=rng.integers(0, 10, 600).astype(np.int64),
             x_test=pictures(300), y_test=rng.integers(0, 10, 300).astype(np.int64))
    return str(path)


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4") or 4))
    yield
