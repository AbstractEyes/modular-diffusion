"""The picture score: the Frechet distance between the features of generated pictures and of the training pictures.

    d(mu1, C1, mu2, C2) = |mu1 - mu2|^2 + tr(C1) + tr(C2) - 2 sum_i sqrt(lambda_i(C1 C2))

(the eigenvalues of the product, real parts, negatives clipped to 0), in float64. A judge maps uint8 pictures
[N, 3, H, W] to features: ``InceptionJudge`` is the usual one (torch-fidelity's Inception-v3 at its 2048-feature output;
``pip install modular-diffusion[judge]``; its weights are fetched on first use); ``FakeJudge`` is a fixed random
projection for tests.

Pictures are sampled with Euler steps (``sample.euler``, or ``sample.euler_grid`` on a given time grid), classes i mod 10
(balanced), every noise batch drawn in order from one generator seeded ``seed``, no caption drop and no guidance.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .sample import check_grid, euler, euler_grid, picture_symbols
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
def sample_pictures(model, bed, n: int, steps: int = 50, batch: int = 1000, seed: int = SCORE_SEED, device="cpu",
                    grid: Optional[Sequence[float]] = None) -> Tensor:
    """n generated pictures, uint8 [n, C, H, W] on the CPU (classes i mod 10). With ``grid`` the pictures take the steps
    of that time grid (``sample.euler_grid``) and ``steps`` is ignored."""
    pts = None if grid is None else check_grid(grid)
    model.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    cls_all = score_classes(n, device)
    sym_all = picture_symbols(model, n, seed, device)          # None for a model without symbol-gated units
    out = []
    for i in range(0, n, batch):
        cls = cls_all[i:i + batch]
        sym = None if sym_all is None else sym_all[i:i + batch]
        z = torch.randn((cls.shape[0],) + tuple(bed.field_shape), generator=gen, device=device)
        z = euler(model, z, cls, steps, symbols=sym) if pts is None else euler_grid(model, z, cls, pts, symbols=sym)
        out.append(to_pictures(z, bed))
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


CLASS_KEYS = ("class_mu", "class_n")


def _labels_of(bed):
    get = getattr(bed, "train_labels", None)
    if get is None or not getattr(bed, "n_classes", 0):
        return None
    return get().cpu().numpy().astype(np.int64)


def _compute_reference(bed, judge, batch: int) -> dict:
    """The judge's features of every training picture -> {mu, cov, n} and, on a bed with classes, {class_mu [K, D],
    class_n [K]}, in float64."""
    x = bed.train_pictures()
    n = int(x.shape[0])
    feats = np.concatenate([judge.features(to_pictures(x[i:i + batch], bed)) for i in range(0, n, batch)])
    mu, cov, n = feature_stats(feats)
    out = {"mu": mu, "cov": cov, "n": n}
    labels = _labels_of(bed)
    if labels is not None:
        k = int(bed.n_classes)
        class_n = np.bincount(labels, minlength=k)[:k].astype(np.int64)
        class_mu = np.zeros((k, feats.shape[1]), dtype=np.float64)
        for c in range(k):
            if class_n[c]:
                class_mu[c] = feats[labels == c].mean(axis=0)
        out.update(class_mu=class_mu, class_n=class_n)
    return out


def _write_reference(path: str, ref: dict, judge) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    # written under a process-own name and moved into place, so jobs sharing the cache never read a partial file
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    extra = {k: ref[k] for k in CLASS_KEYS if k in ref}
    np.savez(tmp, mu=ref["mu"], cov=ref["cov"], n=np.int64(ref["n"]), judge=np.str_(judge.name), **extra)
    os.replace(tmp, path)


def _read_reference(path: str, judge, n: int) -> dict:
    with np.load(path) as d:
        judge_name, count = str(d["judge"]), int(d["n"])
        if judge_name != judge.name or count != n:
            raise ValueError(f"the reference cache {path} was made with judge {judge_name!r} on {count} pictures, "
                             f"not {judge.name!r} on {n}")
        out = {"mu": d["mu"].astype(np.float64), "cov": d["cov"].astype(np.float64), "n": count}
        if all(k in d.files for k in CLASS_KEYS):
            out["class_mu"] = d["class_mu"].astype(np.float64)
            out["class_n"] = d["class_n"].astype(np.int64)
    return out


def reference_stats(bed, judge, path: Optional[str] = None, batch: int = 1000) -> Tuple[np.ndarray, np.ndarray, int]:
    """The judge's statistics of the training pictures (unflipped). Cached in an npz at ``path`` with the judge's name
    and the picture count; a cache made with another judge or count raises ValueError. A cache written here also holds
    the class means of the features (see ``reference_classes``)."""
    n = int(bed.train_pictures().shape[0])
    if path and os.path.exists(path):
        r = _read_reference(path, judge, n)
        return r["mu"], r["cov"], r["n"]
    r = _compute_reference(bed, judge, batch)
    if path:
        _write_reference(path, r, judge)
    return r["mu"], r["cov"], r["n"]


def reference_classes(bed, judge, path: Optional[str] = None, batch: int = 1000) -> dict:
    """The reference statistics with the class means of the judge's features: {mu, cov, n, class_mu [K, D], class_n
    [K]}, float64. Cached in the same npz as ``reference_stats`` under the keys class_mu and class_n; a cache without
    them is recomputed and rewritten whole (under a process-own name, then moved into place). ``fresh`` in the result
    is True when this call computed the statistics (no usable cache)."""
    n = int(bed.train_pictures().shape[0])
    if _labels_of(bed) is None:
        raise ValueError("the class reference needs a bed with classes")
    if path and os.path.exists(path):
        r = _read_reference(path, judge, n)
        if all(k in r for k in CLASS_KEYS):
            r["fresh"] = False
            return r
    r = _compute_reference(bed, judge, batch)
    if path:
        _write_reference(path, r, judge)
    r["fresh"] = True
    return r


def class_reader(ref: dict, ridge: float = 1e-3) -> dict:
    """A ridge map from features to classes, in closed form from the reference statistics -> {w [D, K], mu [D],
    prior [K]} (float64).

    With p = class_n / n and M the [D, K] matrix whose column c is p_c (class_mu_c - mu):

        w = (n / (n - 1)) (cov + ridge mean(diag(cov)) I)^-1 M

    Derivation. Let Fc be the [n, D] centered features (rows f_i - mu) and Yc the centered one-hot classes (rows
    y_i - p). Then Fc^T Fc = (n - 1) cov, and since the rows of Fc sum to zero, Fc^T Yc = Fc^T Y, whose column c is the
    sum over class c's rows of (f_i - mu) = n_c (class_mu_c - mu) = n p_c (class_mu_c - mu): Fc^T Yc = n M (centering
    the classes changes nothing, because M's columns sum to the zero vector). The ridge regression of Yc on Fc with the
    penalty (n - 1) ridge mean(diag(cov)) is (Fc^T Fc + that I)^-1 Fc^T Yc = (n / (n - 1)) (cov + ridge
    mean(diag(cov)) I)^-1 M. The read class of a feature f is argmax_c [(f - mu) w + p]_c."""
    mu = np.asarray(ref["mu"], dtype=np.float64)
    cov = np.asarray(ref["cov"], dtype=np.float64)
    n = float(ref["n"])
    class_mu = np.asarray(ref["class_mu"], dtype=np.float64)
    class_n = np.asarray(ref["class_n"], dtype=np.float64)
    p = class_n / n
    m = (class_mu - mu[None, :]).T * p[None, :]                            # [D, K]
    a = cov + float(ridge) * float(np.mean(np.diag(cov))) * np.eye(cov.shape[0])
    w = (n / (n - 1.0)) * np.linalg.solve(a, m)
    return {"w": w, "mu": mu, "prior": p}


def read_classes(features, reader: dict) -> np.ndarray:
    """The read class of every feature row: argmax_c [(f - mu) w + p]_c."""
    f = np.asarray(features, dtype=np.float64)
    return np.argmax((f - reader["mu"][None, :]) @ reader["w"] + reader["prior"][None, :], axis=1)


def class_read(features, labels, reader: dict) -> dict:
    """{"accuracy": the share of rows whose read class is the label, "by_class": that share per class (None for a class
    without rows), "n"}."""
    labels = np.asarray(labels, dtype=np.int64)
    pred = read_classes(features, reader)
    k = reader["w"].shape[1]
    hit = pred == labels
    by_class = [float(hit[labels == c].mean()) if bool((labels == c).any()) else None for c in range(k)]
    return {"accuracy": float(hit.mean()) if hit.size else None, "by_class": by_class, "n": int(labels.shape[0])}


def reader_check(bed, judge, reader: dict, batch: int = 1000) -> dict:
    """The class reader on the real test pictures: {"test_accuracy", "n"}."""
    x, y = bed.test_pictures()
    feats = np.concatenate([judge.features(to_pictures(x[i:i + batch], bed)) for i in range(0, x.shape[0], batch)])
    r = class_read(feats, y.cpu().numpy(), reader)
    return {"test_accuracy": r["accuracy"], "n": r["n"]}


def picture_score(model, bed, judge, ref, n: int = 10000, steps: int = 50, batch: int = 1000, seed: int = SCORE_SEED,
                  device="cpu", grid: Optional[Sequence[float]] = None, reader: Optional[dict] = None) -> dict:
    """The Frechet distance of n generated pictures against ``ref`` = (mu, cov, n) of the training pictures. With
    ``grid`` the pictures are sampled on that time grid; the result then records "steps": K (the grid's step count) and
    "grid": its points, and the ``steps`` argument is ignored. With a class ``reader`` the result also holds
    "class_read": ``class_read`` of the same features against the classes asked for (``score_classes``)."""
    t0 = time.time()
    pts = None if grid is None else check_grid(grid)
    pics = sample_pictures(model, bed, n, steps, batch, seed, device, grid=pts)
    feats = np.concatenate([judge.features(pics[i:i + batch]) for i in range(0, pics.shape[0], batch)])
    mu, cov, _ = feature_stats(feats)
    out = {"score": frechet_distance(mu, cov, ref[0], ref[1]), "n": int(n),
           "steps": int(steps) if pts is None else len(pts) - 1, "seed": int(seed), "judge": judge.name,
           "seconds": time.time() - t0}
    # the distance's two parts: the squared distance of the feature means, and the rest (the covariances' part)
    gap = np.asarray(mu, dtype=np.float64) - np.asarray(ref[0], dtype=np.float64)
    out["mean_term"] = float(gap @ gap)
    out["spread_term"] = out["score"] - out["mean_term"]
    if pts is not None:
        out["grid"] = pts
    if reader is not None:
        out["class_read"] = class_read(feats, score_classes(pics.shape[0]).numpy(), reader)
    return out
