"""The feature cache: a frozen model's states over the steps of a training stream, so that output units (a head, band
members, content experts) can be trained without running the stem.

For every step k the cache replays the training stream exactly as the trainer draws it (one generator per step, seeded
1,000,003 (seed + 1) + k; the batch, the noise, t, the class drop) and stores, in float32:

    t [B], cls [B] (after the class drop), target [B, C, H, W]     the batch
    h [B, N, d], ctx [B, 1, d], e [B, d]                             the trunk's state after the stem and the stage units
    part.<label> [B, C, H, W]                                        the output of every frozen output unit

A step trained from the cache recomputes only the units being trained, from the stored h, and sums the parts in the
model's own order, so K steps from the cache equal K steps of the trainer bit for bit. The cache covers steps 0..K-1 of
one stream; training longer than the cache, or with member dropout, is not supported.

    cache = FeatureCache.write(model, bed, seed=0, steps=200, batch=128, folder="cache", train=["x_b0_0-1"])
    cache = FeatureCache.open("cache", model)
    cache.fit(model, lr=1e-3, warmup=100)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch import Tensor

from .build import Build, content_hash, read_weights
from .linker import StaleUnit

CACHE_FORMAT = "modular-diffusion/cache/1"
META_FILE = "cache.json"
TRAINABLE = ("Head", "BandMember", "ContentExpert")


def _step_file(folder: str, step: int) -> str:
    return os.path.join(folder, f"step_{step:07d}.safetensors")


def _step_generator(seed: int, step: int, device) -> torch.Generator:
    from .train import step_generator
    return step_generator(seed, step, device)


class FeatureCache:
    def __init__(self, folder: str, meta: Dict[str, Any]):
        self.folder = folder
        self.meta = meta

    @property
    def steps(self) -> int:
        return int(self.meta["steps"])

    @property
    def trained(self) -> List[str]:
        return list(self.meta["train"])

    # ------------------------------------------------------------------ writing
    @staticmethod
    def write(model, bed, seed: int, steps: int, batch: int, folder: str, train: Iterable[str] = (),
              cond_drop: Optional[float] = None, t_sampling: str = "uniform", log: Optional[Callable[[str], None]] = None
              ) -> "FeatureCache":
        """Store the model's states for steps 0..steps-1 of the stream (seed, batch). ``train`` names the output units
        that will be trained from the cache (their outputs are not stored); every other unit must stay frozen.
        cond_drop defaults to the trainer's 0.1 on a bed with classes and 0 otherwise."""
        from safetensors.torch import save_file
        train = list(train)
        for n in train:
            if model.unit(n).unit_type not in TRAINABLE:
                raise ValueError(f"{n!r} is a {model.unit(n).unit_type}; the cache trains only {TRAINABLE}")
        if cond_drop is None:
            cond_drop = 0.1 if bed.n_classes else 0.0
        device = next(model.parameters()).device
        os.makedirs(folder, exist_ok=True)
        head_label = model.head.part if model.head is not None else None
        skip_labels = {model.unit(n).part for n in train}
        t0 = time.time()
        with torch.no_grad():
            for step in range(steps):
                gen = _step_generator(seed, step, device)
                x, cls = bed.train_batch(batch, gen)
                zt, t, cls_d, target = model.draw(x, cls, gen, cond_drop, t_sampling)
                parts, h, ctx, e = model.components(zt, t, cls_d)
                rec = {"t": t, "cls": cls_d, "target": target, "h": h, "ctx": ctx, "e": e}
                for label, v in parts.items():
                    if label not in skip_labels:
                        rec["part." + label] = v
                save_file({k: v.detach().to("cpu").contiguous() for k, v in rec.items()}, _step_file(folder, step))
                if log is not None and (step % 100 == 0 or step == steps - 1):
                    spent = time.time() - t0
                    log(f"cache step {step + 1}/{steps}; spent {spent:.0f} s; left about "
                        f"{spent / (step + 1) * (steps - step - 1):.0f} s")
        meta = {"format": CACHE_FORMAT, "seed": int(seed), "steps": int(steps), "batch": int(batch),
                "cond_drop": float(cond_drop), "t_sampling": t_sampling, "train": train,
                "hashes": {u.name: content_hash(u) for u in model.units if u.name not in train},
                "build": model.build.names(), "head": head_label, "torch": torch.__version__, "device": str(device)}
        with open(os.path.join(folder, META_FILE), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, indent=1)
            fh.write("\n")
        return FeatureCache(folder, meta)

    # ------------------------------------------------------------------ reading
    @classmethod
    def open(cls, folder: str, build: Union[Build, Any]) -> "FeatureCache":
        """Open a cache for a build (or a model). Every unit the cache holds frozen must have the hash it had when the
        cache was written; StaleUnit otherwise."""
        with open(os.path.join(folder, META_FILE), encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("format") != CACHE_FORMAT:
            raise ValueError(f"not a feature cache: {meta.get('format')!r}")
        if isinstance(build, Build):
            names, current = build.names(), {e.name: e.hash for e in build.units}
        else:
            names, current = build.build.names(), {u.name: content_hash(u) for u in build.units}
        if names != meta["build"]:
            raise StaleUnit(folder, f"the cache was written for the units {meta['build']}, not {names}")
        for n, h in meta["hashes"].items():
            if current.get(n) is None:
                raise StaleUnit(n, "the build entry has no hash to check against the cache")
            if current[n] != h:
                raise StaleUnit(n, "it changed since the cache was written")
        return cls(folder, meta)

    def batch(self, step: int, device="cpu") -> Dict[str, Tensor]:
        """The stored tensors of one step."""
        if not 0 <= step < self.steps:
            raise IndexError(f"the cache holds steps 0..{self.steps - 1}, not {step}")
        rec = read_weights(_step_file(self.folder, step))
        return {k: v.to(device) for k, v in rec.items()}

    # ------------------------------------------------------------------ training from the cache
    def parts(self, model, rec: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """The parts of one step in the model's order: stored for the frozen units, computed for the trained ones."""
        train = set(self.trained)
        h, ctx, e, t, cls = rec["h"], rec["ctx"], rec["e"], rec["t"], rec["cls"]
        out = {}
        anc = model.anchor
        if anc is not None and "part." + anc.part in rec:
            out[anc.part] = rec["part." + anc.part]
        if model.head is not None:
            if model.head.name in train:
                out[model.head.part] = model.head_part(h)
            elif "part." + model.head.part in rec:
                out[model.head.part] = rec["part." + model.head.part]
        for m in model.members:
            if m.name in train:
                y = model.member_part(m, h, ctx, e, t, cls)
                if y is not None:
                    out[m.part] = y
            elif "part." + m.part in rec:
                out[m.part] = rec["part." + m.part]
        return out

    def loss(self, model, step: int) -> Tensor:
        rec = self.batch(step, next(model.parameters()).device)
        parts = self.parts(model, rec)
        v = torch.stack(list(parts.values())).sum(dim=0) if parts else torch.zeros_like(rec["target"])
        return ((v - rec["target"]) ** 2).mean()

    def fit(self, model, lr: float = 1e-3, warmup: int = 100, steps: Optional[int] = None, equalize: bool = False,
            log_every: int = 500, log: Optional[Callable[[str], None]] = None) -> List[list]:
        """Train the cache's units with the trainer's recipe (Adam, betas 0.9 / 0.999, eps 1e-8, no weight decay, the
        learning rate warmed up linearly and set before every step); every other parameter is frozen. Returns the
        curve [[step, loss], ...]."""
        from .train import equalize as equalize_grads
        steps = self.steps if steps is None else int(steps)
        if steps > self.steps:
            raise ValueError(f"the cache holds {self.steps} steps, not {steps}")
        train = set(self.trained)
        for u in model.units:
            for p in u.parameters():
                p.requires_grad_(u.name in train)
        params = [p for p in model.parameters() if p.requires_grad]
        curve = []
        if not params or steps <= 0:
            return curve
        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        model.train()
        t0 = time.time()
        for step in range(steps):
            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
            loss = self.loss(model, step)
            if not loss.requires_grad:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if equalize:
                equalize_grads(model)
            opt.step()
            if step % log_every == 0 or step == steps - 1:
                curve.append([step, float(loss.detach())])
                if log is not None:
                    spent = time.time() - t0
                    log(f"  cached step {step + 1}/{steps}  loss {float(loss.detach()):.4f}  spent {spent:.0f} s  "
                        f"left about {spent / (step + 1) * (steps - step - 1):.0f} s")
        return curve
