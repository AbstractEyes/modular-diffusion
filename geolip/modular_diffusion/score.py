"""The picture score: the Frechet distance between the features of generated pictures and of the training pictures.

    d(mu1, C1, mu2, C2) = |mu1 - mu2|^2 + tr(C1) + tr(C2) - 2 sum_i sqrt(lambda_i(C1 C2))

(the eigenvalues of the product, real parts, negatives clipped to 0), in float64. A judge maps uint8 pictures
[N, 3, H, W] to features: ``InceptionJudge`` is the usual one (torch-fidelity's Inception-v3 at its 2048-feature output;
``pip install modular-diffusion[judge]``; its weights are fetched on first use); ``FakeJudge`` is a fixed random
projection for tests.

Pictures are sampled with Euler steps (``sample.euler``), classes i mod 10 (balanced), every noise batch drawn in order
from one generator seeded ``seed``, no caption drop and no guidance.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .sample import euler
from .spectral import token_filter

SCORE_SEED = 20260101
INCEPTION_BATCH = 500


def frechet_distance(mu1, cov1, mu2, cov2) -> float:
    mu1, mu2 = np.asarray(mu1, dtype=np.float64), np.asarray(mu2, dtype=np.float64)
    cov1, cov2 = np.asarray(cov1, dtype=np.float64), np.asarray(cov2, dtype=np.float64)
    eig = np.linalg.eigvals(cov1 @ cov2).real
    root = np.sqrt(np.clip(eig, 0.0, None)).sum()
    d = mu1 - mu2
    return float(d @ d + np.trace(cov1) + np.trace(cov2) - 2.0 * root)


def feature_stats(feats) -> Tuple[np.ndarray, np.ndarray, int]:
    """(mean, covariance with n - 1, n) of features [N, D], in float64."""
    f = np.asarray(feats, dtype=np.float64)
    return f.mean(axis=0), np.cov(f, rowvar=False, ddof=1), int(f.shape[0])


def to_pictures(z: Tensor, bed) -> Tensor:
    """Model-space fields [N, C, H, W] -> uint8 pictures [N, C, H, W] on the CPU: the tilt undone when the bed has one,
    times the std plus the mean, clamped to [0, 1], times 255, rounded."""
    if bed.tilt is not None:
        z = token_filter(z, 1.0 / bed.tilt.to(z.device), bed.grid)
    x = (z * bed.std.to(z.device) + bed.mean.to(z.device)).clamp(0, 1)
    return (x * 255).round().to(torch.uint8).cpu()


def score_classes(n: int, device="cpu") -> Tensor:
    """The classes of the scored pictures: i mod 10."""
    return torch.arange(n, device=device) % 10


@torch.no_grad()
def sample_pictures(model, bed, n: int, steps: int = 50, batch: int = 1000, seed: int = SCORE_SEED, device="cpu") -> Tensor:
    """n generated pictures, uint8 [n, C, H, W] on the CPU (classes i mod 10)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    cls_all = score_classes(n, device)
    out = []
    for i in range(0, n, batch):
        cls = cls_all[i:i + batch]
        z = torch.randn((cls.shape[0],) + tuple(bed.field_shape), generator=gen, device=device)
        out.append(to_pictures(euler(model, z, cls, steps), bed))
    return torch.cat(out) if out else torch.zeros((0,) + tuple(bed.field_shape), dtype=torch.uint8)


class FakeJudge:
    """A fixed random projection of the pictures (scaled to [0, 1]) to ``dim`` features, in float64. For tests."""

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim, self.seed = int(dim), int(seed)
        self.name = f"fake-projection-{self.dim}-{self.seed}"
        self._w = {}

    def features(self, pictures: Tensor) -> np.ndarray:
        x = pictures.reshape(pictures.shape[0], -1).cpu().numpy().astype(np.float64) / 255.0
        d = x.shape[1]
        if d not in self._w:
            self._w[d] = np.random.default_rng(self.seed).standard_normal((d, self.dim)) / np.sqrt(d)
        return x @ self._w[d]


class InceptionJudge:
    """torch-fidelity's Inception-v3 feature extractor at its 2048-feature output (uint8 input, eval, no gradient,
    batches of 500). Imported and built when constructed; its weights are fetched on first use."""

    def __init__(self, device="cpu"):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.device = device
        self.name = "inception-v3-compat-2048"
        self.net = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()

    @torch.no_grad()
    def features(self, pictures: Tensor) -> np.ndarray:
        out = []
        for i in range(0, pictures.shape[0], INCEPTION_BATCH):
            f = self.net(pictures[i:i + INCEPTION_BATCH].to(self.device))[0]
            out.append(f.double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048))


def make_judge(kind: str, device="cpu"):
    """"inception" or "fake"."""
    if kind == "inception":
        return InceptionJudge(device)
    if kind == "fake":
        return FakeJudge()
    raise ValueError(f"unknown judge {kind!r}")


def reference_stats(bed, judge, path: Optional[str] = None, batch: int = 1000) -> Tuple[np.ndarray, np.ndarray, int]:
    """The judge's statistics of the training pictures (unflipped). Cached in an npz at ``path`` with the judge's name
    and the picture count; a cache made with another judge or count raises ValueError."""
    x = bed.train_pictures()
    n = int(x.shape[0])
    if path and os.path.exists(path):
        with np.load(path) as d:
            judge_name, count = str(d["judge"]), int(d["n"])
            if judge_name != judge.name or count != n:
                raise ValueError(f"the reference cache {path} was made with judge {judge_name!r} on {count} pictures, "
                                 f"not {judge.name!r} on {n}")
            return d["mu"].astype(np.float64), d["cov"].astype(np.float64), count
    feats = np.concatenate([judge.features(to_pictures(x[i:i + batch], bed)) for i in range(0, n, batch)])
    mu, cov, n = feature_stats(feats)
    if path:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        # written under a process-own name and moved into place, so jobs sharing the cache never read a partial file
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, mu=mu, cov=cov, n=np.int64(n), judge=np.str_(judge.name))
        os.replace(tmp, path)
    return mu, cov, n


def picture_score(model, bed, judge, ref, n: int = 10000, steps: int = 50, batch: int = 1000, seed: int = SCORE_SEED,
                  device="cpu") -> dict:
    """The Frechet distance of n generated pictures against ``ref`` = (mu, cov, n) of the training pictures."""
    t0 = time.time()
    pics = sample_pictures(model, bed, n, steps, batch, seed, device)
    feats = np.concatenate([judge.features(pics[i:i + batch]) for i in range(0, pics.shape[0], batch)])
    mu, cov, _ = feature_stats(feats)
    return {"score": frechet_distance(mu, cov, ref[0], ref[1]), "n": int(n), "steps": int(steps), "seed": int(seed),
            "judge": judge.name, "seconds": time.time() - t0}
