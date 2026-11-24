"""Small engines that read the address frame on image tokens, their training batches and trainer, and the collective
that serves a picture's requests by place and thing addresses (``scenes``, ``frame``).

AN ENGINE (``AddressedEngine``; one class for every kind of part). z [B, 3, 48, 48] in model space is cut into 144
tokens of 48 numbers (``spectral.patchify``, patch 4). Every token's embedding is

    h = embed(token) + code_in(code)        code [B, T, 28]: the 16 place numbers of ``frame.TokenFrame.place_code``
                                            and, on the tokens of a request the engine is told about, the 12 numbers
                                            of ``frame.content_code`` (zeros elsewhere)

The cross-attention context is ``ctx_in`` of [NULL, the request codes]: by rule the first context token of every
forward is a NULL token (a zero code, always valid), so a forward without a request still has one valid context
token; the request codes (``frame.request_code``) are those of the cells the engine is TOLD about (``told`` [B, 9]),
compacted in cell order with a mask (``told_slots``). This is the EMBEDDING read; the ROTARY read (``read="rotary"``)
puts the told requests into the sequence as tokens instead (see ``AddressedEngine``). Then ``depth`` of
the library's blocks (``blocks.Block``) with the library's rotary positions (``rotary.AxialBandCode`` over the whole
band 2 to 64 tokens), the time embedding (``blocks.TimeEmbedding``, 64 frequencies) and the key mask (``key_mask``
[B, T] bool: True = this token may be read; a token always reads itself); the library's zero-born head
(``blocks.ZeroHead``), so a newborn engine answers exactly 0; ``spectral.unpatchify``. An engine's starting numbers
come from (seed, its name) through ``units.name_seed``: the global generator is seeded from them inside a forked
random state, the layers are made in the order embed, code_in, ctx_in, time, blocks, head, and the caller's state is
put back. Weights are saved with safetensors (``save_engine``), with a json card beside them.

THE KINDS AND THEIR TRAINING BATCHES (``kind_batch``)

| kind                  | scenes                | code                               | context              | key mask                                  | loss mask        |
|-----------------------|-----------------------|------------------------------------|----------------------|-------------------------------------------|------------------|
| addressed_thing (k)   | one of class k        | place; content on the own cell     | null + the request   | own cell + request-free cells, h hidden   | own cell         |
| addressed_base        | empty                 | place                              | null                 | the cells not hidden (h of the nine)      | the same cells   |
| whole_thing (k)       | one of class k        | place; content on the own cell     | null + the request   | none                                      | every token      |
| whole_empty           | empty                 | place                              | null                 | none                                      | every token      |
| whole_unconditional   | one, a uniform class  | place only (told nothing)          | null                 | none                                      | every token      |
| model_one             | one, a uniform class  | place; content on the thing's cell | null + 9 slots       | none                                      | every token      |
| model_mix             | mix                   | place; content on every request    | null + 9 slots       | none                                      | every token      |

h is uniform on 0..8: for a thing part h of the other eight cells are hidden, for the base h of the nine. The nine
slots of the whole models are the cells in cell order, masked where a cell has no request (``frame.layout_codes``).

THE DRAWS of a training step s, all from ``train.step_generator(seed, s, "cpu")``: the scenes (``ScenesBed.
train_batch``: the layouts, then the render), then for the addressed kinds the hidden cells (h [B] randint 0..8, then
keys [B, 9] rand: the hidden cells are the h smallest keys, the own cell's key set above every other), then the noise
[B, 3, 48, 48] (randn) and t [B] (rand), the order of ``Model.draw``; everything is moved to the device after it is
drawn. z_t = (1 - t) noise + t picture, target = picture - noise (t = 0 is pure noise).

THE LOSS (``masked_loss``): per row, the sum over the masked tokens of the squared velocity error divided by 48 x the
mask's token count (asserted above zero for every row), then the mean over the rows; with a full mask it is the plain
mean squared error.

THE TRAINER (``train_engine``): pure Adam (betas 0.9 / 0.999, eps 1e-8, no weight decay, no clipping: the library's
recipe clips nothing) at lr min(1, (s + 1) / warmup) x 1e-3; the averaged weights of record "g8" (after optimizer step
k = 0, 1, ...: avg <- d_k avg + (1 - d_k) w with d_k = (1 + k) / (k + 10), so the newest weights' share is 9 / (10 +
k)), the arithmetic of ``train.fit_units``; a curve every ``every`` steps of the training loss and of THE OWN READ (the
masked loss on ``heldout`` scenes of the kind's own batch law, drawn once from a generator seeded ``OWN_READ_SEED``
with their noise and t, on the averaged weights), the newborn engine's read first; the seconds of each step from
per-step stamps (the card synchronized at the stamp); a checkpoint (safetensors and json, no pickle) every
``save_every`` steps; a job started on a folder holding a checkpoint continues from it, and on the CPU reaches the same
numbers bit for bit; a folder holding train.json and the engine files is finished and is read back.

THE COLLECTIVE (``Collective.answer``): joins of the parts' answers at one Euler step.

    gate               the address gate (``frame.token_owner``): for each class present, ONE batched forward over the
                       (scene, request) pairs of that class in (scene, cell) order, each pair with its own code, context
                       and (addressed parts) key mask = its own cell and the request-free cells; the pair's answer is
                       copied to the output on its own cell's tokens only (selection, never a sum); the base (or the
                       empty-scene engine, for whole-picture parts) answers the request-free tokens in one forward over
                       the scenes that have any (addressed base: key mask = the request-free cells). An engine with no
                       request in the batch is not run, and no engine's batch depends on which other engines exist
    sum_empty          v = v_E + sum over requests r of (v_r - v_E), the requests taken in cell order, from the same
                       pair forwards; v_E is the background engine's forward over every scene
    sum_unconditional  the same with the unconditional engine as the background
``WholeModel.answer``: one forward with every request's code and context.

``sample(answer, cells, colors, seed, steps=50)``: the start noise drawn on the CPU from a generator seeded ``seed``
(so it does not depend on the device), Euler steps z <- z + v(z, i / steps) / steps, no guidance; the lists are
answered in chunks of ``chunk`` (the chunks are fixed by the list order alone).
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from . import frame
from .blocks import Block, TimeEmbedding, ZeroHead
from .compat import TIME_FREQ
from .rotary import ANCHORED_HEAD_DIM, WHOLE, AxialBandCode, anchored_rotary
from .scenes import CANVAS, CHANNELS, N_CELLS, PATCH, TOKEN_GRID, ScenesBed
from .spectral import patchify, unpatchify
from .train import step_generator, use_average
from .units import name_seed

KINDS = ("addressed_thing", "addressed_base", "whole_thing", "whole_empty", "whole_unconditional", "model_one",
         "model_mix")
THING_KINDS = ("addressed_thing", "whole_thing")
ADDRESSED_KINDS = ("addressed_thing", "addressed_base")
MODEL_KINDS = ("model_one", "model_mix")
JOINS = ("gate", "sum_empty", "sum_unconditional")
READS = ("embedding", "rotary")
TOKEN_DIM = CHANNELS * PATCH * PATCH          # 48
PLACE_DIM = 16
CONTENT_DIM = 12
CODE_DIM = PLACE_DIM + CONTENT_DIM            # 28
CTX_DIM = 18
SHAPE = {"dim": 128, "heads": 2, "depth": 3}
BETAS = (0.9, 0.999)
EPS = 1e-8
OWN_READ_SEED = 424242
ENGINE_FILES = ("engine.safetensors", "engine_avg.safetensors", "engine.json")
CHECKPOINT_FILES = ("checkpoint.safetensors", "checkpoint.json")
TRAIN_JSON = "train.json"


# ------------------------------------------------------------------ the engine
def engine_name(kind: str, cls: Optional[int] = None) -> str:
    """The name of an engine's random stream: the kind, with ``_c<k>`` for a thing part."""
    if kind not in KINDS:
        raise ValueError(f"the kinds are {KINDS}, not {kind!r}")
    if kind in THING_KINDS:
        if cls is None or not 0 <= int(cls) < 9:
            raise ValueError(f"a {kind} engine needs a class 0..8")
        return f"{kind}_c{int(cls)}"
    if cls is not None:
        raise ValueError(f"a {kind} engine has no class")
    return kind


def told_slots(told: Tensor) -> Tuple[Tensor, Tensor]:
    """The cells an engine is told about, compacted: (slots [B, R] long, the told cells in cell order, padded with 0;
    mask [B, R] bool), R = the largest count of the batch (0 when nothing is told)."""
    count = told.sum(dim=1)
    r = int(count.max()) if told.numel() else 0
    order = torch.argsort((~told).to(torch.int8), dim=1, stable=True)[:, :r]
    mask = torch.arange(r, device=told.device)[None] < count[:, None]
    return torch.where(mask, order, torch.zeros_like(order)), mask


class AddressedEngine(nn.Module):
    """See the module's text. ``settings``: dim, heads, depth, read ("embedding" or "rotary") and the fixed token, grid
    and code sizes.

    forward(z, t, cells, colors, told, key_mask=None): ``told`` [B, 9] bool marks the cells whose requests the engine
    is told about (their codes; for a thing part its own cell, none for a base, every request for a whole model);
    ``key_mask`` [B, 144] bool over the image tokens (None: every token visible).

    THE EMBEDDING READ: h = embed(token) + code_in(place code and the content code on the told cells' tokens); the
    context [NULL, the told requests' codes]; the library's axial rotary over the whole band.
    THE ROTARY READ: one REQUEST TOKEN a told request, req_in(its 12 content numbers), placed before the 144 image
    tokens; no code is added to an image token; the context holds the NULL token only; the rotary is
    ``rotary.anchored_rotary``: an image token at (row iy, column ix) with its cell's digits, a request token at its
    cell's center (row 4k + 1.5, column 4j + 1.5) with its cell's digits. A request token is always a visible key
    (padding slots never are). The head reads the image tokens only."""

    def __init__(self, dim: int = 128, heads: int = 2, depth: int = 3, read: str = "embedding"):
        super().__init__()
        if read not in READS:
            raise ValueError(f"the read forms are {READS}, not {read!r}")
        if read == "rotary" and int(dim) // int(heads) != ANCHORED_HEAD_DIM:
            raise ValueError(f"the rotary read needs a head of {ANCHORED_HEAD_DIM} numbers")
        self.read = read
        self.settings = {"dim": int(dim), "heads": int(heads), "depth": int(depth), "read": read,
                         "token_dim": TOKEN_DIM, "grid": TOKEN_GRID, "patch": PATCH, "code_dim": CODE_DIM,
                         "ctx_dim": CTX_DIM, "time_freq": TIME_FREQ, "band": list(WHOLE)}
        d = int(dim)
        self.embed = nn.Linear(TOKEN_DIM, d)
        if read == "embedding":
            self.code_in = nn.Linear(CODE_DIM, d)
        else:
            self.req_in = nn.Linear(CONTENT_DIM, d)
        self.ctx_in = nn.Linear(CTX_DIM, d)
        self.time = TimeEmbedding(TIME_FREQ, d, 1000.0)
        self.blocks = nn.ModuleList([Block(d, int(heads)) for _ in range(int(depth))])
        self.head = ZeroHead(d, TOKEN_DIM)
        self.code = AxialBandCode(d // int(heads), TOKEN_GRID, float(WHOLE[0]), float(WHOLE[1]))

    def forward(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor, told: Tensor,
                key_mask: Optional[Tensor] = None) -> Tensor:
        b, dev = z.shape[0], z.device
        cells, colors = cells.to(dev), colors.to(dev)
        told = told.to(dev).bool() & (cells >= 0)
        slots, smask = told_slots(told)
        codes, _ = frame.layout_codes(cells, colors)                             # [B, 9, 18]
        req = torch.gather(codes, 1, slots[..., None].expand(-1, -1, CTX_DIM)) if slots.shape[1] else \
            codes[:, :0]
        null = torch.zeros(b, 1, CTX_DIM, device=dev, dtype=z.dtype)
        valid = torch.ones(b, 1, dtype=torch.bool, device=dev)
        tokens = patchify(z, PATCH)
        e = self.time(t)
        if self.read == "embedding":
            h = self.embed(tokens) + self.code_in(token_codes(cells, colors, told))
            c = self.ctx_in(torch.cat([null, req.to(z.dtype)], dim=1))
            m = torch.cat([valid, smask], dim=1)
            rot = self.code.rotary()
            for blk in self.blocks:
                h = blk(h, c, m, e, rot, key_mask)
            return unpatchify(self.head(h), (CHANNELS, CANVAS, CANVAS), PATCH, TOKEN_GRID)
        r = slots.shape[1]
        h = torch.cat([self.req_in(req[..., 6:].to(z.dtype)), self.embed(tokens)], dim=1)
        fr = _FRAME
        img_rows = (torch.arange(fr.n_tokens, device=dev) // fr.grid).double()
        img_cols = (torch.arange(fr.n_tokens, device=dev) % fr.grid).double()
        sj, sk = (slots % fr.cells), (slots // fr.cells)
        rows = torch.cat([(sk * fr.side).double() + (fr.side - 1) / 2, img_rows.expand(b, -1)], dim=1)
        cols = torch.cat([(sj * fr.side).double() + (fr.side - 1) / 2, img_cols.expand(b, -1)], dim=1)
        dcol = torch.cat([sj, fr.column.to(dev).expand(b, -1)], dim=1)
        drow = torch.cat([sk, fr.row.to(dev).expand(b, -1)], dim=1)
        rot = anchored_rotary(rows, cols, dcol, drow)
        if key_mask is None and (r == 0 or bool(smask.all())):
            km = None
        else:
            img = torch.ones(b, fr.n_tokens, dtype=torch.bool, device=dev) if key_mask is None else key_mask.to(dev)
            km = torch.cat([smask, img], dim=1)
        c = self.ctx_in(null)
        for blk in self.blocks:
            h = blk(h, c, valid, e, rot, km)
        return unpatchify(self.head(h[:, r:]), (CHANNELS, CANVAS, CANVAS), PATCH, TOKEN_GRID)


def param_count(engine: nn.Module) -> int:
    return int(sum(p.numel() for p in engine.parameters()))


def born(kind: str, cls: Optional[int], seed: int, dim: int = 128, heads: int = 2, depth: int = 3,
         read: str = "embedding") -> AddressedEngine:
    """A newborn engine whose numbers depend only on (seed, its name) (and its shape and read form); the caller's
    random state is put back."""
    name = engine_name(kind, cls)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(name_seed(int(seed), name))
        eng = AddressedEngine(dim, heads, depth, read)
    eng.kind, eng.cls, eng.seed = kind, cls, int(seed)
    return eng


def shape_numbers(dim: int, heads: int, depth: int, read: str = "embedding") -> int:
    """The trained numbers of an engine of this shape."""
    with torch.random.fork_rng(devices=[]):
        return param_count(AddressedEngine(dim, heads, depth, read))


def depth_to_reach(total: int, dim: int = 256, heads: int = 4, max_depth: int = 64, read: str = "embedding") -> int:
    """The smallest depth whose engine of width ``dim`` holds at least ``total`` trained numbers."""
    for d in range(1, int(max_depth) + 1):
        if shape_numbers(dim, heads, d, read) >= int(total):
            return d
    raise ValueError(f"no depth up to {max_depth} reaches {total} numbers")


def save_engine(engine: AddressedEngine, folder: str, average: Optional[Dict[str, Tensor]] = None,
                card: Optional[dict] = None) -> None:
    """engine.safetensors (the last numbers), engine_avg.safetensors (the averaged parameters, when given) and
    engine.json (kind, class, settings, trained numbers and the given card)."""
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in engine.state_dict().items()},
              os.path.join(folder, "engine.safetensors"))
    if average is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in average.items()},
                  os.path.join(folder, "engine_avg.safetensors"))
    info = {"kind": getattr(engine, "kind", None), "cls": getattr(engine, "cls", None),
            "seed": getattr(engine, "seed", None), "settings": engine.settings, "trained": param_count(engine)}
    info.update(card or {})
    with open(os.path.join(folder, "engine.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(info, fh, indent=1)
        fh.write("\n")


def load_engine(folder: str, average: bool = True, device="cpu") -> AddressedEngine:
    """An engine written by ``save_engine``: the averaged parameters when ``average`` and the file is there."""
    from safetensors.torch import load_file
    with open(os.path.join(folder, "engine.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    s = info["settings"]
    with torch.random.fork_rng(devices=[]):
        eng = AddressedEngine(s["dim"], s["heads"], s["depth"], s.get("read", "embedding"))
    state = load_file(os.path.join(folder, "engine.safetensors"))
    avg = os.path.join(folder, "engine_avg.safetensors")
    if average and os.path.isfile(avg):
        state.update(load_file(avg))
    eng.load_state_dict(state, strict=True)
    eng.kind, eng.cls, eng.seed = info.get("kind"), info.get("cls"), info.get("seed")
    eng.card = info
    return eng.to(device).eval()


# ------------------------------------------------------------------ codes
_FRAME = frame.TokenFrame()
_PLACE = _FRAME.place_code()                                     # [T, 16]


def token_codes(cells: Tensor, colors: Tensor, content: Tensor) -> Tensor:
    """[B, T, 28]: the place code on every token and the content code of a cell's request on the tokens of the cells
    marked in ``content`` [B, 9] bool (zeros elsewhere)."""
    dev = cells.device
    codes, mask = frame.layout_codes(cells, colors)
    content = content.to(dev) & mask
    c12 = (codes[..., 6:] * content[..., None].float())[:, _FRAME.cell.to(dev)]       # [B, T, 12]
    place = _PLACE.to(dev).expand(cells.shape[0], -1, -1)
    return torch.cat([place, c12], dim=-1)


def _own_cell(cells: Tensor) -> Tensor:
    if bool(((cells >= 0).sum(dim=1) != 1).any()):
        raise ValueError("a one-thing batch holds exactly one request a scene")
    return (cells >= 0).long().argmax(dim=1)


def _hidden(own: Optional[Tensor], b: int, gen: torch.Generator) -> Tensor:
    """[B, 9] bool: h uniform on 0..8 cells hidden, the h smallest of keys drawn uniform (the own cell's key above all)."""
    h = torch.randint(0, N_CELLS, (b,), generator=gen)
    keys = torch.rand(b, N_CELLS, generator=gen)
    if own is not None:
        keys[torch.arange(b), own.cpu()] = 2.0
    order = torch.argsort(keys, dim=1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(N_CELLS).expand(b, N_CELLS).contiguous())
    return rank < h[:, None]


def kind_batch(kind: str, cls: Optional[int], bed: ScenesBed, n: int, gen: torch.Generator,
               allowed_cells: Optional[Sequence[int]] = None) -> dict:
    """The scenes of a training batch of a kind with what the engine is told (``told`` [B, 9]: the cells whose
    requests it reads; the codes and context follow from it in either read form), its key mask and loss mask (see the
    module's table); the scenes' and the hidden cells' draws come from ``gen`` (a CPU generator). ``allowed_cells``:
    the thing kinds' one-thing scenes use only these cells (None: every cell, the default path)."""
    engine_name(kind, cls)
    dev = bed.device
    if allowed_cells is not None and kind not in THING_KINDS:
        raise ValueError("allowed_cells applies to the thing kinds")
    if kind in THING_KINDS:
        sb = bed.train_batch("one", n, gen, only_class=int(cls), allowed_cells=allowed_cells)
    elif kind in ("addressed_base", "whole_empty"):
        sb = bed.train_batch("empty", n, gen)
    elif kind in ("whole_unconditional", "model_one"):
        sb = bed.train_batch("one", n, gen)
    else:
        sb = bed.train_batch("mix", n, gen)
    cells, colors = sb.cells, sb.colors
    T = _FRAME.n_tokens
    none = torch.zeros(n, N_CELLS, dtype=torch.bool, device=dev)
    out = {"x": sb.x, "cells": cells, "colors": colors, "params": sb.params, "key_mask": None, "told": none,
           "loss_mask": torch.ones(n, T, dtype=torch.bool, device=dev), "hidden": None, "serving": None}
    if kind in THING_KINDS:
        own = _own_cell(cells)
        out["told"] = torch.nn.functional.one_hot(own, N_CELLS).bool()
        out["serving"] = own
        if kind == "addressed_thing":
            hidden = _hidden(own, n, gen).to(dev)
            out["hidden"] = hidden
            out["key_mask"] = frame.visible_keys(cells, own, hidden)
            out["loss_mask"] = (_FRAME.cell.to(dev)[None] == own[:, None])
    elif kind == "addressed_base":
        hidden = _hidden(None, n, gen).to(dev)
        out["hidden"] = hidden
        out["key_mask"] = frame.visible_keys(cells, torch.full((n,), -1, dtype=torch.long, device=dev), hidden)
        out["loss_mask"] = out["key_mask"].clone()
    elif kind in MODEL_KINDS:
        out["told"] = cells >= 0
    return out


def noisy(x: Tensor, gen: torch.Generator) -> Tuple[Tensor, Tensor, Tensor]:
    """(z_t, t, target): noise [B, 3, 48, 48] then t [B] drawn from ``gen`` on the CPU, moved to x's device."""
    b = x.shape[0]
    z0 = torch.randn(tuple(x.shape), generator=gen).to(x.device)
    t = torch.rand(b, generator=gen).to(x.device)
    tt = t.view(b, 1, 1, 1)
    return (1 - tt) * z0 + tt * x, t, x - z0


def masked_loss(v: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
    """The mean over rows of (sum over masked tokens of the squared error) / (48 x the row's masked token count)."""
    count = loss_mask.sum(dim=1)
    if bool((count == 0).any()):
        raise ValueError("a row of the batch has no token in its loss mask")
    d = patchify(v - target, PATCH)
    sq = (d * d).sum(dim=-1)
    per = (sq * loss_mask.to(sq.dtype)).sum(dim=1) / (TOKEN_DIM * count.to(sq.dtype))
    return per.mean()


def engine_forward(engine: AddressedEngine, batch: dict, z: Tensor, t: Tensor) -> Tensor:
    return engine(z, t, batch["cells"], batch["colors"], batch["told"], batch["key_mask"])


# ------------------------------------------------------------------ the trainer
def _median(values) -> Optional[float]:
    v = sorted(float(x) for x in values)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def own_read_set(kind: str, cls: Optional[int], bed: ScenesBed, n: int, seed: int = OWN_READ_SEED,
                 allowed_cells: Optional[Sequence[int]] = None) -> dict:
    """The held-out set of the own read: n scenes of the kind's batch law with their noise and t, from one generator
    seeded ``seed`` (drawn in chunks of 256 scenes, each chunk's scenes, hidden cells, noise and t in that order)."""
    gen = torch.Generator().manual_seed(int(seed))
    parts = []
    for k in range(0, int(n), 256):
        m = min(256, int(n) - k)
        b = kind_batch(kind, cls, bed, m, gen, allowed_cells)
        z, t, target = noisy(b["x"], gen)
        parts.append({"batch": b, "z": z, "t": t, "target": target})
    return {"parts": parts, "n": int(n)}


@torch.no_grad()
def own_read(engine: AddressedEngine, rs: dict) -> float:
    """The masked loss on the held-out set (the mean over its rows)."""
    total, rows = 0.0, 0
    was = engine.training
    engine.eval()
    for p in rs["parts"]:
        v = engine_forward(engine, p["batch"], p["z"], p["t"])
        m = p["z"].shape[0]
        total += float(masked_loss(v, p["target"], p["batch"]["loss_mask"])) * m
        rows += m
    engine.train(was)
    return total / max(1, rows)


def _adam_tensors(opt, params) -> Tuple[Dict[str, Tensor], List[dict]]:
    sd = opt.state_dict()
    tensors = {}
    for idx, st in sd["state"].items():
        for key, val in st.items():
            tensors[f"adam.{idx}.{key}"] = val.detach().cpu().reshape(-1).clone() if key == "step" else val.detach().cpu().contiguous()
    return tensors, sd["param_groups"]


def _adam_load(opt, tensors: Dict[str, Tensor], groups: List[dict], device) -> None:
    state: Dict[int, dict] = {}
    for name, val in tensors.items():
        if not name.startswith("adam."):
            continue
        _, idx, key = name.split(".", 2)
        st = state.setdefault(int(idx), {})
        st[key] = val.reshape(()) if key == "step" else val.to(device)
    opt.load_state_dict({"state": state, "param_groups": groups})


def train_engine(kind: str, cls: Optional[int], bed: ScenesBed, seed: int, out: str, *, steps: int = 3000,
                 batch: int = 128, lr: float = 1e-3, warmup: int = 100, every: int = 250, heldout: int = 2048,
                 dim: int = 128, heads: int = 2, depth: int = 3, save_every: int = 500, until: Optional[int] = None,
                 read: str = "embedding", allowed_cells: Optional[Sequence[int]] = None,
                 log: Callable[[str], None] = print) -> dict:
    """Train one engine alone (see the module's text) into ``out`` -> its result (also written as train.json): the
    settings, the curve [{"step", "loss", "own_read"}], the newborn and the last own read, "descended" (the last own
    read below the newborn's), trained numbers, train seconds (the sum of the step stamps), seconds a step (median).
    ``allowed_cells``: a thing part's one-thing scenes (and its own read's) use only these cells."""
    engine_name(kind, cls)
    device = bed.device
    steps, batch = int(steps), int(batch)
    allowed = None if allowed_cells is None else sorted({int(c) for c in allowed_cells})
    settings = {"kind": kind, "cls": None if cls is None else int(cls), "bed": bed.bed, "seed": int(seed),
                "steps": steps, "batch": batch, "lr": float(lr), "warmup": int(warmup), "every": int(every),
                "heldout": int(heldout), "dim": int(dim), "heads": int(heads), "depth": int(depth), "read": read,
                "allowed_cells": allowed, "average": "g8", "optimizer": "Adam", "betas": list(BETAS), "eps": EPS,
                "weight_decay": 0.0, "clip": None}
    os.makedirs(out, exist_ok=True)
    tj = os.path.join(out, TRAIN_JSON)
    ckj = os.path.join(out, "checkpoint.json")
    if os.path.isfile(tj) and all(os.path.isfile(os.path.join(out, f)) for f in ENGINE_FILES):
        with open(tj, encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored["settings"] != settings:
            raise ValueError(f"{out} holds a finished engine made with other settings")
        log(f"FINISHED already: {out}")
        stored["restored"] = True
        return stored
    from safetensors.torch import load_file, save_file
    engine = born(kind, cls, seed, dim, heads, depth, read).to(device)
    engine.train()
    params = list(engine.parameters())
    names = [n for n, _ in engine.named_parameters()]
    opt = torch.optim.Adam(params, lr=float(lr), betas=BETAS, eps=EPS, weight_decay=0.0)
    average = {n: p.detach().clone().float() for n, p in engine.named_parameters()}
    rs = own_read_set(kind, cls, bed, heldout, allowed_cells=allowed)
    curve: List[dict] = []
    step_seconds: List[float] = []
    done = 0
    newborn = None
    if os.path.isfile(ckj):
        with open(ckj, encoding="utf-8") as fh:
            ck = json.load(fh)
        if ck["settings"] != settings:
            raise ValueError(f"the checkpoint in {out} was made with other settings")
        tensors = load_file(os.path.join(out, "checkpoint.safetensors"))
        engine.load_state_dict({k[len("engine."):]: v.to(device) for k, v in tensors.items() if k.startswith("engine.")})
        average = {k[len("avg."):]: v.to(device) for k, v in tensors.items() if k.startswith("avg.")}
        _adam_load(opt, tensors, ck["param_groups"], device)
        curve, step_seconds, done, newborn = ck["curve"], ck["step_seconds"], int(ck["step"]), ck["newborn_read"]
        log(f"RESUME from step {done:,} of {steps:,}")
    trained = param_count(engine)
    log(f"JOB {engine_name(kind, cls)} bed {bed.bed} seed {seed}: {steps:,} steps x batch {batch} = {steps * batch:,} "
        f"scenes; {trained:,} trained numbers; pure Adam lr {lr:g}; device {device}")
    if newborn is None:
        newborn = own_read(engine, rs)
        curve.append({"step": 0, "loss": None, "own_read": newborn})
        log(f"newborn own read {newborn:.6f}")

    def checkpoint(step: int) -> None:
        tensors, groups = _adam_tensors(opt, params)
        tensors.update({"engine." + k: v.detach().cpu().contiguous() for k, v in engine.state_dict().items()})
        tensors.update({"avg." + k: v.detach().cpu().contiguous() for k, v in average.items()})
        save_file(tensors, os.path.join(out, "checkpoint.safetensors.part"))
        os.replace(os.path.join(out, "checkpoint.safetensors.part"), os.path.join(out, "checkpoint.safetensors"))
        with open(ckj + ".part", "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"settings": settings, "step": step, "param_groups": groups, "curve": curve,
                       "step_seconds": step_seconds, "newborn_read": newborn, "names": names}, fh)
        os.replace(ckj + ".part", ckj)

    stop = steps if until is None else min(int(until), steps)
    is_cuda = str(device).startswith("cuda")
    t0 = time.time()
    loss_v = None
    for s in range(done, stop):
        s0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = float(lr) * min(1.0, (s + 1) / max(1, int(warmup)))
        gen = step_generator(int(seed), s, "cpu")
        b = kind_batch(kind, cls, bed, batch, gen, allowed)
        z, t, target = noisy(b["x"], gen)
        loss = masked_loss(engine_forward(engine, b, z, t), target, b["loss_mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        k = s
        dk = (1 + k) / (k + 8 + 2)
        with torch.no_grad():
            for n_, p in engine.named_parameters():
                average[n_].copy_(dk * average[n_] + (1.0 - dk) * p.detach().float())
        n = s + 1
        if every and (n % int(every) == 0 or n == steps):
            loss_v = float(loss.detach())
        elif is_cuda:
            torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - s0)
        if every and (n % int(every) == 0 or n == steps):
            with use_average(engine, average):
                r = own_read(engine, rs)
            curve.append({"step": n, "loss": loss_v, "own_read": r})
            spent = time.time() - t0
            log(f"step {n:,}/{steps:,} loss {loss_v:.5f} own read {r:.5f}; spent {spent:.0f} s, left about "
                f"{spent / max(1, n - done) * (steps - n):.0f} s")
        if save_every and n % int(save_every) == 0 and n != stop:
            checkpoint(n)
    checkpoint(stop)
    result = {"settings": settings, "step": stop, "finished": stop >= steps}
    if stop >= steps:
        last = curve[-1]["own_read"] if curve and curve[-1]["step"] == steps else None
        if last is None:
            with use_average(engine, average):
                last = own_read(engine, rs)
        card = {"bed": bed.bed, "steps": steps, "seed": int(seed)}
        save_engine(engine, out, average=average, card=card)
        result.update({"curve": curve, "newborn_read": newborn, "last_read": last, "descended": bool(last < newborn),
                       "trained": trained, "train_seconds": float(sum(step_seconds)),
                       "seconds_a_step": _median(step_seconds), "step_seconds_count": len(step_seconds),
                       "device": str(device), "torch": torch.__version__})
        with open(tj + ".part", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=1)
            fh.write("\n")
        os.replace(tj + ".part", tj)
        log(f"DONE {engine_name(kind, cls)}: own read {newborn:.5f} -> {last:.5f} (descended {last < newborn}); "
            f"{sum(step_seconds):.1f} s training, {_median(step_seconds):.4f} s a step")
    result["engine"] = engine
    result["average"] = average
    return result


def averaged_copy(engine: AddressedEngine, average: Dict[str, Tensor]) -> AddressedEngine:
    e = copy.deepcopy(engine)
    with torch.no_grad():
        for n, p in e.named_parameters():
            p.copy_(average[n])
    for a in ("kind", "cls", "seed"):
        setattr(e, a, getattr(engine, a, None))
    return e.eval()


# ------------------------------------------------------------------ the collective
class Collective:
    """Thing parts {class: engine}, a base (the addressed base, or the empty-scene engine for whole-picture parts),
    ``addressed`` (whether the thing parts and the base read through their key masks) and optionally the unconditional
    engine (the background of ``sum_unconditional``). Every engine is of one read form (mixed forms raise)."""

    def __init__(self, things: Dict[int, AddressedEngine], base: Optional[AddressedEngine], addressed: bool,
                 unconditional: Optional[AddressedEngine] = None):
        self.things = {int(k): v for k, v in things.items()}
        self.base = base
        self.addressed = bool(addressed)
        self.unconditional = unconditional
        reads = {getattr(e, "read", "embedding") for e in list(self.things.values()) + [base, unconditional]
                 if e is not None}
        if len(reads) > 1:
            raise ValueError(f"a collective is built from engines of one read form, not {sorted(reads)}")
        self.read = reads.pop() if reads else "embedding"

    def owners(self) -> List[List[int]]:
        return [[k] for k in sorted(self.things)]

    def pair_answers(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor) -> Dict[int, Tuple[Tensor, Tensor, Tensor]]:
        """{class: (rows, cols, answers [P, 3, 48, 48])}: one batched forward a class present, its pairs in (scene,
        cell) order. A requested class without a part raises (``frame.token_owner``)."""
        frame.token_owner(cells, self.owners())
        out = {}
        for k in sorted(self.things):
            rows, cols = torch.nonzero(cells == k, as_tuple=True)
            if rows.numel() == 0:
                continue
            c, kk = cells[rows], colors[rows]
            told = torch.nn.functional.one_hot(cols, N_CELLS).bool()
            km = frame.visible_keys(c, cols) if self.addressed else None
            out[k] = (rows, cols, self.things[k](z[rows], t[rows], c, kk, told, km))
        return out

    def background(self, eng: AddressedEngine, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor) -> Tensor:
        """A background engine's forward over every scene: told nothing, no key mask."""
        return eng(z, t, cells, colors, torch.zeros_like(cells, dtype=torch.bool), None)

    def answer(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor, join: str = "gate") -> Tensor:
        if join not in JOINS:
            raise ValueError(f"the joins are {JOINS}, not {join!r}")
        cells = cells.to(z.device)
        colors = colors.to(z.device)
        pairs = self.pair_answers(z, t, cells, colors)
        if join == "gate":
            owner = frame.token_owner(cells, self.owners())                  # [B, T]
            base_idx = len(self.things)
            out = torch.zeros(z.shape[0], _FRAME.n_tokens, TOKEN_DIM, device=z.device, dtype=z.dtype)
            free_rows = torch.nonzero((cells < 0).any(dim=1), as_tuple=True)[0]
            if free_rows.numel():
                if self.base is None:
                    raise ValueError("a scene has cells without a request and the collective has no base")
                c, k = cells[free_rows], colors[free_rows]
                none = torch.zeros_like(c, dtype=torch.bool)
                km = frame.visible_keys(c, torch.full((c.shape[0],), -1, dtype=torch.long, device=z.device)) \
                    if self.addressed else None
                vb = patchify(self.base(z[free_rows], t[free_rows], c, k, none, km), PATCH)
                sel = (owner[free_rows] == base_idx)[..., None]
                out[free_rows] = torch.where(sel, vb, out[free_rows])
            cell_of = _FRAME.cell.to(z.device)
            for i, k in enumerate(sorted(self.things)):
                if k not in pairs:
                    continue
                rows, cols, v = pairs[k]
                vt = patchify(v, PATCH)
                sel = (cell_of[None] == cols[:, None]) & (owner[rows] == i)      # [P, T]: the pair's own tokens
                p_idx, tok = torch.nonzero(sel, as_tuple=True)
                out[rows[p_idx], tok] = vt[p_idx, tok]                          # each (scene, token) written once
            return unpatchify(out, (CHANNELS, CANVAS, CANVAS), PATCH, TOKEN_GRID)
        bg = self.base if join == "sum_empty" else self.unconditional
        if bg is None:
            raise ValueError(f"the join {join} needs its background engine")
        ve = self.background(bg, z, t, cells, colors)
        v = ve.clone()
        slot = torch.zeros((z.shape[0], N_CELLS) + tuple(z.shape[1:]), device=z.device, dtype=z.dtype)
        for k, (rows, cols, va) in pairs.items():
            slot[rows, cols] = va
        for c in range(N_CELLS):                                            # the requests in cell order
            rows = torch.nonzero(cells[:, c] >= 0, as_tuple=True)[0]
            if rows.numel():
                v[rows] = v[rows] + (slot[rows, c] - ve[rows])
        return v


class WholeModel:
    """One engine told every request: the content code on every requested cell, one context slot a cell."""

    def __init__(self, engine: AddressedEngine):
        self.engine = engine

    def answer(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor, join: str = "") -> Tensor:
        cells, colors = cells.to(z.device), colors.to(z.device)
        return self.engine(z, t, cells, colors, cells >= 0, None)


def start_noise(n: int, seed: int) -> Tensor:
    """[n, 3, 48, 48] drawn on the CPU from a generator seeded ``seed``."""
    return torch.randn(int(n), CHANNELS, CANVAS, CANVAS, generator=torch.Generator().manual_seed(int(seed)))


@torch.no_grad()
def sample(answer: Callable, cells: Tensor, colors: Tensor, seed: int, steps: int = 50, device="cpu",
           chunk: int = 0, noise: Optional[Tensor] = None) -> Tensor:
    """Euler sampling of every list (rows of cells, colors) -> z [N, 3, 48, 48] in model space on the CPU.
    ``answer(z, t, cells, colors)``; the start noise from ``start_noise(N, seed)`` unless given; lists in chunks of
    ``chunk`` (0: all at once)."""
    n = cells.shape[0]
    z_all = start_noise(n, seed) if noise is None else noise.clone()
    size = n if not chunk else int(chunk)
    outs = []
    for a in range(0, n, max(1, size)):
        c, k = cells[a:a + size].to(device), colors[a:a + size].to(device)
        z = z_all[a:a + size].to(device)
        for i in range(int(steps)):
            t = torch.full((z.shape[0],), i / int(steps), device=device)
            z = z + answer(z, t, c, k) / int(steps)
        outs.append(z.cpu())
    return torch.cat(outs) if outs else z_all


# ------------------------------------------------------------------ reads that need engines
def outside_inside(a: Tensor, b: Tensor, cells: Tensor, cls: int) -> Tuple[float, float]:
    """(the largest |a - b| over pixels outside the cells holding ``cls``, the mean |a - b| inside them; None when no
    cell holds it)."""
    from .scenes import cell_crops
    d = cell_crops((a - b).abs())                                    # [B, 9, C, 16, 16]
    inside = (cells == int(cls)).to(d.device)
    out_v = d[~inside]
    in_v = d[inside]
    return (float(out_v.max()) if out_v.numel() else 0.0, float(in_v.mean()) if in_v.numel() else None)


@torch.no_grad()
def leak_read(things: Dict[int, AddressedEngine], empty: AddressedEngine, bed: ScenesBed, n_things: int,
              t_value: float, scenes: int, seed: int) -> dict:
    """THE LEAK of whole-picture parts on real composed scenes of ``n_things`` things at noise level ``t_value``: for
    every request r, the mean square of v_r - v_E on the tokens of the other requested cells and on the request-free
    cells, each over its mean square on r's own cell; pooled as (the mean over requests of the numerator) / (the mean
    over requests of the own-cell value). The scenes and their noise come from a CPU generator seeded ``seed``."""
    gen = torch.Generator().manual_seed(int(seed))
    sb = bed.train_batch("composed", int(scenes), gen, count=int(n_things))
    z0 = torch.randn(tuple(sb.x.shape), generator=gen).to(sb.x.device)
    t = torch.full((sb.x.shape[0],), float(t_value), device=sb.x.device)
    z = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * sb.x
    col = Collective(things, empty, addressed=False)
    ve = col.background(empty, z, t, sb.cells, sb.colors)
    own_s, other_s, free_s, n_req, n_free = 0.0, 0.0, 0.0, 0, 0
    cell_of = _FRAME.cell.to(z.device)
    for k, (rows, cols, va) in col.pair_answers(z, t, sb.cells, sb.colors).items():
        d = patchify(va - ve[rows], PATCH).pow(2).mean(dim=-1)        # [P, T]
        c = sb.cells[rows]
        own = cell_of[None] == cols[:, None]
        req_tok = (c >= 0)[:, cell_of]
        other = req_tok & ~own
        free = ~req_tok
        own_s += float((d * own).sum(1).div(own.sum(1)).sum())
        if bool(other.any()):
            other_s += float((d * other).sum(1).div(other.sum(1).clamp_min(1)).sum())
        fr = free.sum(1) > 0
        if bool(fr.any()):
            free_s += float(((d * free).sum(1)[fr] / free.sum(1)[fr]).sum())
            n_free += int(fr.sum())
        n_req += rows.shape[0]
    own_m = own_s / max(1, n_req)
    return {"n": int(n_things), "t": float(t_value), "requests": n_req,
            "own": own_m, "other_cells": other_s / max(1, n_req) / own_m if n_things > 1 and own_m > 0 else None,
            "free_cells": (free_s / n_free) / own_m if n_free and own_m > 0 else None, "free_requests": n_free}
