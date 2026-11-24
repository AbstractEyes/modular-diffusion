"""Pictures of an assembly and their scores. Nothing here trains anything.

    decode(vae, z, batch=8)          uint8 pictures [N, 3, H, W] on the CPU from latents [N, 4, h, w]:
                                     x = vae.decode(z / LATENT_SCALE).sample, clamped to [-1, 1], then
                                     round((x + 1) * 127.5) as uint8, ``batch`` rows a decode, no gradient
    sample_latents(model, states, empty, noise_seed, group=...)
                                     the final latents of ``assemble.sample`` for every row of ``states``, ``group``
                                     rows at a time; group j starts from initial_noise(group_seed(noise_seed, j), ...),
                                     the rule of ``reads.sample_probe`` (the same rows and seed give every model the same
                                     start noise, and the result equals ``sample_probe`` on a probe of those rows)
    kid(a, b, blocks=10)             the kernel distance of two feature sets (below)
    ClipJudge(path)                  a text-to-picture score from a CLIP model kept in a local folder (below)
    FakeClipJudge(seed)              the same interface on fixed random projections, for tests
    make_clip_judge(kind)            "" -> None, "fake" -> FakeClipJudge(), anything else -> ClipJudge(kind)
    contact_sheet(pictures)          one PIL picture of a grid of pictures

THE KERNEL DISTANCE. With features a [Na, D], b [Nb, D] and the kernel k(x, y) = (x . y / D + 1) ** 3,

    full = sum_{i != j} k(a_i, a_j) / (Na (Na - 1)) + sum_{i != j} k(b_i, b_j) / (Nb (Nb - 1))
           - 2 mean_{i, j} k(a_i, b_j)

the unbiased estimate of the squared maximum mean discrepancy (it can be slightly below 0). The same estimate is also
taken on ``blocks`` consecutive equal blocks of the rows in order: r = min(Na, Nb) // blocks rows a block, block k of
``a`` (rows k r .. (k + 1) r - 1) against block k of ``b``; rows past blocks x r are dropped. The result holds the
blocks' values in block order, their mean and their standard deviation (ddof 1). All in float64.

THE TEXT-TO-PICTURE SCORE. score = 100 cos(image embedding, text embedding) of the CLIP model's projected features
(``get_image_features``, ``get_text_features``), NOT clipped at 0 (some published scores clip it; this one does not, so
a mean over rows is not lifted by the clip). A picture is prepared in torch: float in [0, 1], its shorter side resized
to the model's input size (bicubic, antialiased), cropped at the center, normalized with the mean and standard
deviation of the folder's ``preprocessor_config.json`` (CLIP's published numbers when the file is not there). A prompt
is the token ids [N, 77] the teacher's tokenizer made (padded with the end token), passed with no attention mask.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .assemble import _default_shape, initial_noise, sample
from .data import LATENT_SCALE
from .reads import group_seed, sample_groups

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ------------------------------------------------------------------ pictures
@torch.no_grad()
def decode(vae, z: Tensor, batch: int = 8) -> Tensor:
    """uint8 pictures [N, 3, H, W] on the CPU of the latents ``z`` (already multiplied by the latent scale)."""
    batch = int(batch)
    if batch < 1:
        raise ValueError("batch is at least 1")
    p = next(vae.parameters())
    out = []
    for i in range(0, z.shape[0], batch):
        x = vae.decode(z[i:i + batch].to(device=p.device, dtype=p.dtype) / LATENT_SCALE).sample
        x = ((x.float().clamp(-1, 1) + 1) * 127.5).round()
        out.append(x.to(torch.uint8).cpu())
    if not out:
        raise ValueError("no latents to decode")
    return torch.cat(out)


@torch.no_grad()
def sample_latents(model, states: Tensor, empty: Optional[Tensor], noise_seed: int, *, group: int, steps: int = 50,
                   guidance: float = 7.5, latent_shape: Optional[Sequence[int]] = None,
                   alphas_cumprod: Optional[Tensor] = None, log: Optional[Callable[[str], None]] = None) -> Tensor:
    """float32 [N, C, h, w] on the CPU: the sampler's final latents of every row of ``states``, ``group`` rows at a
    time (0: one group), group j from the noise of ``group_seed(noise_seed, j)`` (see the module's text). The seed of
    group j depends only on j, so row i starts from the same noise whatever the number of rows, as long as the group
    size is the same: a job of fewer rows is a prefix of a job of more (exactly so when every group of the shorter job
    is whole; a short last group puts fewer rows through a forward, which may change the last digits)."""
    shape = tuple(int(x) for x in (latent_shape or _default_shape(model)))
    n_rows = int(states.shape[0])
    out = []
    done = 0
    for j, (first, n) in enumerate(sample_groups(n_rows, group)):
        t0 = time.time()
        z = initial_noise(group_seed(noise_seed, j), n, shape)
        out.append(sample(model, states[first:first + n], empty, int(noise_seed), steps, guidance, latent_shape=shape,
                          alphas_cumprod=alphas_cumprod, z=z).float().cpu())
        done += n
        if log:
            log(f"sampled rows {done:,} of {n_rows:,} ({time.time() - t0:.1f} s for this group)")
    return torch.cat(out)


def contact_sheet(pictures: Tensor, columns: int = 4, size: int = 256):
    """A PIL RGB picture of the uint8 pictures [N, 3, H, W] in a grid, row by row, each resized to ``size`` x
    ``size`` (LANCZOS), no gaps. The grid has min(columns, N) columns."""
    from PIL import Image
    n = int(pictures.shape[0])
    if n < 1:
        raise ValueError("a contact sheet needs at least one picture")
    cols = max(1, min(int(columns), n))
    rows = math.ceil(n / cols)
    size = int(size)
    sheet = Image.new("RGB", (cols * size, rows * size))
    arr = pictures.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
    for k in range(n):
        img = Image.fromarray(arr[k]).resize((size, size), Image.LANCZOS)
        sheet.paste(img, ((k % cols) * size, (k // cols) * size))
    return sheet


# ------------------------------------------------------------------ the kernel distance
def _kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x @ y.T / x.shape[1] + 1.0) ** 3


def _mmd2(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.shape[0], b.shape[0]
    kaa, kbb, kab = _kernel(a, a), _kernel(b, b), _kernel(a, b)
    within_a = (kaa.sum() - np.trace(kaa)) / (na * (na - 1))
    within_b = (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1))
    return float(within_a + within_b - 2.0 * kab.mean())


def kid(a, b, blocks: int = 10) -> dict:
    """The kernel distance of features ``a`` [Na, D] and ``b`` [Nb, D] (see the module's text): {"full",
    "block_mean", "block_std", "block_values" (the per-block estimates in block order), "blocks", "block_rows", "na",
    "nb"}. ValueError when ``blocks`` < 2 or a block would
    hold fewer than 2 rows."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"two feature sets [N, D] of the same D: {a.shape}, {b.shape}")
    blocks = int(blocks)
    if blocks < 2:
        raise ValueError(f"blocks is at least 2, not {blocks}")
    rows = min(a.shape[0], b.shape[0]) // blocks
    if rows < 2:
        raise ValueError(f"{blocks} blocks of {min(a.shape[0], b.shape[0])} rows hold {rows} rows each; at least 2 "
                         f"are needed")
    per = np.array([_mmd2(a[k * rows:(k + 1) * rows], b[k * rows:(k + 1) * rows]) for k in range(blocks)])
    return {"full": _mmd2(a, b), "block_mean": float(per.mean()), "block_std": float(per.std(ddof=1)),
            "block_values": [float(v) for v in per], "blocks": blocks, "block_rows": int(rows),
            "na": int(a.shape[0]), "nb": int(b.shape[0])}


# ------------------------------------------------------------------ the text-to-picture judges
def _cosine_score(x: Tensor, y: Tensor) -> np.ndarray:
    x = x.double()
    y = y.double()
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    return (100.0 * (x * y).sum(-1)).cpu().numpy()


def _features(out) -> Tensor:
    """The projected features of a CLIP feature call (a tensor, or an output object holding them as its pooled
    output, depending on the library's version)."""
    if isinstance(out, Tensor):
        return out
    return out.pooler_output


def _preprocessor_config(path: str) -> Optional[dict]:
    """The folder's preprocessor_config.json, or the cached copy of a hub id's (never downloaded), or None."""
    if os.path.isdir(path):
        cfg = os.path.join(path, "preprocessor_config.json")
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
            cfg = try_to_load_from_cache(path, "preprocessor_config.json")
        except ImportError:
            cfg = None
        if not isinstance(cfg, str):
            return None
    if not os.path.isfile(cfg):
        return None
    with open(cfg, encoding="utf-8") as fh:
        return json.load(fh)


def prepare_pictures(pictures: Tensor, size: int, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    """uint8 [N, 3, H, W] -> float32 [N, 3, size, size]: in [0, 1], the shorter side resized to ``size`` (bicubic,
    antialiased), cropped at the center, normalized by ``mean`` and ``std``."""
    x = pictures.float() / 255.0
    h, w = int(x.shape[-2]), int(x.shape[-1])
    scale = size / min(h, w)
    nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
    if (nh, nw) != (h, w):
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
    top, left = (nh - size) // 2, (nw - size) // 2
    x = x[..., top:top + size, left:left + size]
    m = torch.tensor(list(mean), dtype=x.dtype).reshape(1, 3, 1, 1)
    s = torch.tensor(list(std), dtype=x.dtype).reshape(1, 3, 1, 1)
    return (x - m) / s


class ClipJudge:
    """A text-to-picture judge on ``transformers.CLIPModel`` from a local folder or a cached hub id
    (``local_files_only=local_only``; eval, frozen). ``scores(pictures, ids)`` -> float64 [N]: 100 cos(image
    embedding, text embedding), NOT clipped at 0 (see the module's text)."""

    def __init__(self, path: str, device="cpu", dtype: torch.dtype = torch.float32, local_only: bool = True):
        from transformers import CLIPModel
        self.path = str(path)
        self.name = os.path.basename(os.path.normpath(self.path)) if os.path.isdir(self.path) else self.path
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = CLIPModel.from_pretrained(self.path, local_files_only=bool(local_only))
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.size = int(self.model.config.vision_config.image_size)
        cfg = _preprocessor_config(self.path) or {}
        self.mean = tuple(cfg.get("image_mean", CLIP_MEAN))
        self.std = tuple(cfg.get("image_std", CLIP_STD))

    @torch.no_grad()
    def scores(self, pictures: Tensor, ids: Tensor, batch: int = 64) -> np.ndarray:
        if pictures.shape[0] != ids.shape[0]:
            raise ValueError(f"{pictures.shape[0]} pictures and {ids.shape[0]} prompts")
        out = []
        for i in range(0, pictures.shape[0], int(batch)):
            x = prepare_pictures(pictures[i:i + batch], self.size, self.mean, self.std)
            img = _features(self.model.get_image_features(pixel_values=x.to(self.device, self.dtype)))
            txt = _features(self.model.get_text_features(input_ids=ids[i:i + batch].to(self.device).long()))
            out.append(_cosine_score(img, txt))
        return np.concatenate(out) if out else np.zeros((0,), dtype=np.float64)


class FakeClipJudge:
    """A judge for tests with the interface of ``ClipJudge``: a picture's feature is a fixed random projection of its
    8 x 8 average-pooled pixels; a prompt's is the sum over positions of a fixed random embedding of each id weighted
    by a fixed random number of its position; the score is 100 cos of the two. Deterministic."""

    VOCAB = 49408
    DIM = 64

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.name = "fake-clip"
        g = torch.Generator().manual_seed(self.seed + 4_242)
        self._image = torch.randn(3 * 8 * 8, self.DIM, generator=g, dtype=torch.float64)
        self._embed = torch.randn(self.VOCAB, self.DIM, generator=g, dtype=torch.float64)
        self._position = torch.randn(77, generator=g, dtype=torch.float64)

    @torch.no_grad()
    def scores(self, pictures: Tensor, ids: Tensor, batch: int = 64) -> np.ndarray:
        if pictures.shape[0] != ids.shape[0]:
            raise ValueError(f"{pictures.shape[0]} pictures and {ids.shape[0]} prompts")
        x = torch.nn.functional.adaptive_avg_pool2d(pictures.double().cpu() / 255.0, (8, 8))
        img = x.reshape(x.shape[0], -1) @ self._image
        ids = ids.cpu().long() % self.VOCAB
        w = self._position[: ids.shape[1]].reshape(1, -1, 1)
        txt = (self._embed[ids] * w).sum(1)
        return _cosine_score(img, txt)


def make_clip_judge(kind: str, device="cpu"):
    """"" -> None; "fake" -> ``FakeClipJudge()``; anything else -> ``ClipJudge(kind, device)`` (a local folder or a
    cached hub id)."""
    if not kind:
        return None
    if kind == "fake":
        return FakeClipJudge()
    return ClipJudge(kind, device)
