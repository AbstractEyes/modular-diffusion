"""The judge of a scene's cells: for a cell of 16 x 16 pixels, which of the nine things or NOTHING it holds, and its
color (``scenes``, ``scene_reads``).

THE NETWORK: three 3 x 3 convolutions (32, 64, 128 channels; strides 1, 2, 2; padding 1; GELU), a flatten (no
pooling: 128 x 4 x 4 numbers), a linear head of ten (nine things and NOTHING = 9) and one of nine colors. Input: a crop
in model space [N, 3, 16, 16].

TRAINING (``train_judge``): cross-entropy on the thing head, plus cross-entropy on the color head over the crops that
hold a thing (their mean; zero when a batch holds none); ``steps`` steps (2,000) of pure Adam 1e-3 (betas 0.9 / 0.999,
eps 1e-8, no weight decay, no warm-up); a batch of 256 crops: step s draws, from ``train.step_generator(seed, s,
"cpu")``, 256 fresh real ``mix`` scenes of the bed (``ScenesBed.train_batch``), then one cell of each (randint 0..8 [256]).
THE CEILING (``ceiling``): 4,500 held-out crops, every cell of 500 ``mix`` scenes drawn from a generator seeded
``HELDOUT_SEED``: the thing rate (ten classes) and the color rate (over the crops that hold a thing).
Save and load with safetensors and a json card.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .scene_reads import NOTHING, rate
from .scenes import N_CELLS, ScenesBed, cell_crops
from .train import step_generator
from .units import name_seed

HELDOUT_SEED = 31337
HELDOUT_SCENES = 500
BETAS = (0.9, 0.999)
EPS = 1e-8


class SceneJudge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.things = nn.Linear(128 * 4 * 4, NOTHING + 1)
        self.colors = nn.Linear(128 * 4 * 4, 9)

    def forward(self, crops: Tensor) -> Tuple[Tensor, Tensor]:
        h = F.gelu(self.conv1(crops))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        h = h.flatten(1)
        return self.things(h), self.colors(h)


def born_judge(seed: int) -> SceneJudge:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(name_seed(int(seed), "scene_judge"))
        return SceneJudge()


def labels(cells: Tensor, colors: Tensor) -> Tuple[Tensor, Tensor]:
    """Thing labels (NOTHING where a cell is empty) and color labels (-1 where a cell is empty)."""
    return torch.where(cells >= 0, cells, torch.full_like(cells, NOTHING)), colors


def judge_loss(judge: SceneJudge, crops: Tensor, things: Tensor, colors: Tensor) -> Tensor:
    lt, lc = judge(crops)
    loss = F.cross_entropy(lt, things)
    held = things != NOTHING
    if bool(held.any()):
        loss = loss + F.cross_entropy(lc[held], colors[held])
    return loss


@torch.no_grad()
def predict(judge: SceneJudge, x: Tensor, chunk: int = 4096) -> Tuple[Tensor, Tensor]:
    """Pictures in model space [B, 3, 48, 48] -> (things [B, 9], colors [B, 9]) long, the judge's reads of every cell."""
    was = judge.training
    judge.eval()
    dev = next(judge.parameters()).device
    crops = cell_crops(x).reshape(-1, 3, 16, 16)
    ts, cs = [], []
    for a in range(0, crops.shape[0], chunk):
        lt, lc = judge(crops[a:a + chunk].to(dev))
        ts.append(lt.argmax(dim=1).cpu())
        cs.append(lc.argmax(dim=1).cpu())
    judge.train(was)
    b = x.shape[0]
    return torch.cat(ts).reshape(b, N_CELLS), torch.cat(cs).reshape(b, N_CELLS)


def ceiling(judge: SceneJudge, bed: ScenesBed, scenes: int = HELDOUT_SCENES, seed: int = HELDOUT_SEED) -> dict:
    """The thing rate (ten classes) and the color rate on every cell of ``scenes`` held-out mix scenes."""
    sb = bed.heldout("mix", int(scenes), int(seed))
    pt, pc = predict(judge, sb.x)
    things, colors = labels(sb.cells.cpu(), sb.colors.cpu())
    held = things != NOTHING
    return {"crops": int(things.numel()), "thing_rate": rate(pt == things), "color_rate": rate((pc == colors)[held]),
            "thing_rate_on_things": rate((pt == things)[held]), "nothing_rate": rate((pt == things)[~held])}


def train_judge(bed: ScenesBed, seed: int, out: str, steps: int = 2000, batch: int = 256, lr: float = 1e-3,
                log: Callable[[str], None] = print, log_every: int = 250) -> dict:
    """Train the judge of a bed into ``out`` (judge.safetensors, judge.json) -> its result with the ceiling."""
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    dev = bed.device
    judge = born_judge(seed).to(dev)
    judge.train()
    opt = torch.optim.Adam(judge.parameters(), lr=float(lr), betas=BETAS, eps=EPS, weight_decay=0.0)
    curve, secs = [], []
    t0 = time.time()
    for s in range(int(steps)):
        s0 = time.perf_counter()
        gen = step_generator(int(seed), s, "cpu")
        sb = bed.train_batch("mix", int(batch), gen)
        pick = torch.randint(0, N_CELLS, (int(batch),), generator=gen).to(dev)
        rows = torch.arange(int(batch), device=dev)
        crops = cell_crops(sb.x)[rows, pick]
        things, colors = labels(sb.cells[rows, pick], sb.colors[rows, pick])
        loss = judge_loss(judge, crops, things, colors)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        secs.append(time.perf_counter() - s0)
        if (s + 1) % int(log_every) == 0 or s + 1 == int(steps):
            curve.append({"step": s + 1, "loss": float(loss.detach())})
            log(f"judge step {s + 1:,}/{int(steps):,} loss {float(loss.detach()):.4f}; spent {time.time() - t0:.0f} s")
    ceil = ceiling(judge, bed)
    save_file({k: v.detach().cpu().contiguous() for k, v in judge.state_dict().items()},
              os.path.join(out, "judge.safetensors"))
    res = {"bed": bed.bed, "seed": int(seed), "steps": int(steps), "batch": int(batch), "lr": float(lr),
           "curve": curve, "ceiling": ceil, "train_seconds": float(sum(secs)),
           "seconds_a_step": sorted(secs)[len(secs) // 2] if secs else None,
           "trained": int(sum(p.numel() for p in judge.parameters()))}
    with open(os.path.join(out, "judge.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    return res


def load_judge(folder: str, device="cpu") -> SceneJudge:
    from safetensors.torch import load_file
    j = SceneJudge()
    j.load_state_dict(load_file(os.path.join(folder, "judge.safetensors")), strict=True)
    return j.to(device).eval()
