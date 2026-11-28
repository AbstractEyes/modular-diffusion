"""Several member engines of one variant trained as ONE batch: the parameters of G engines stacked on a leading axis,
one forward written out over that axis, one optimizer, and each engine's rows, loss, averages and reads its own.

WHY THE FORWARD IS WRITTEN OUT. ``torch.func.vmap`` has no batching rule for scaled-dot-product attention and runs it
one example at a time; here every parameter carries the leading axis G, every linear layer is one einsum over it, and
attention is called once a block on all G x B rows.

THE STACK (``stack_state``, ``unstack_state``): parameter p of engine g is state[p][g] (float32, detached copies), the
names of ``members.MemberEngine``. The depth is read from the names (blocks.<i>.*).

THE STACKED FORWARD (``stacked_forward``), the arithmetic of ``members.MemberEngine.forward_tokens`` layer for layer.
Shapes: z [G, B, 3, 48, 48], t [G, B], content [G, B, 9, W], angles [G, B, 153, 32], key_mask [G, B, 153] bool.

    lin(x; L)       = x W_L^T + b_L, as einsum("g...i,goi->g...o") over the leading G, then + b_L
    LN(x)           = layer_norm over the last axis, no affine, eps 1e-6
    tokens          = patchify(z) (patch 4)                                        [G, B, 144, 48]
    h               = [lin(content; req_in), lin(tokens; embed)]                     [G, B, 153, d]
    time            f = exp(-log(10000) k / 32), k = 0 .. 31 (float32); a = (1000 t) f;
                    e = lin(SiLU(lin([sin a, cos a]; time.mlp.0)); time.mlp.2)       [G, B, d]
    context         c = lin(0 (one null token of 18 numbers); ctx_in), always valid
    rotary          R = multirotary.angle_rotary(query angles, key angles) on q, k [G, B, heads, N, 64]; block 1 with
                    head 0 turned by the told angles when given
    each block      (ds1, dh1, ds2, dh2, ds3, dh3) = chunks of lin(SiLU(e); modulation.1)
                    y = LN(h)(1 + ds1) + dh1;  q, k, v = lin(y; attn.qkv) split into heads;  q, k = R(q, k)
                    h = h + lin(SDPA(q, k, v, mask: key j read by query i when key_mask[j] or j = i); attn.out)
                    y = LN(h)(1 + ds2) + dh2;  h = h + lin(SDPA(lin(y; cross.q), the context's k, v); cross.out)
                    h = h + lin(GELU_tanh(lin(LN(h)(1 + ds3) + dh3; mlp.0)); mlp.2)
    head            out = unpatchify(lin(LN(h[:, :, 9:]); head.proj))              [G, B, 3, 48, 48]
SDPA runs once a block on the G x B rows together (the leading two axes merged into one).

THE TRAINER (``train_members``). Engine g is ``members.born(variant, classes[g], seed, dim, heads, depth)``. A step s:
for EACH engine in turn, gen = ``train.step_generator(seed, s, "cpu")`` (a fresh generator an engine, the same (seed, s)
for all, the draws of ``collective.train_engine`` for one engine); the layout ``members.member_layout(variant, cls, the
training bed, batch, gen, p_point, exclude, held_pairs)`` (every draw on the CPU generator, the render's numbers
included); then the noise, then t: noise "stream" (the default) draws them from gen on the CPU, as
``collective.noisy(x, gen)`` does (the numbers of ``member_batch`` then ``noisy``, bit for bit); noise "device" draws
them on the training device from a fresh ``noise_generator(seed, s, device)`` an engine (so a stacked engine still
draws what it would draw alone). Then for the whole stack at once on the training device: ONE render of every engine's
rows (``scenes.render_drawn``), z and the target, and ONE ``members.prepare`` of every engine's layout (checked once on
the CPU, so that nothing is read back), reshaped to the leading axis G (``stack_step_rows``). One stacked forward; the loss of
engine g is the masked loss of ``collective.masked_loss`` (per row the sum over its masked tokens of the squared error
over 48 x the mask's token count, then the mean over rows; the mask's counts checked above zero from the CPU layout: a
thing's class stands in the row, the base has a free cell); the step's loss is the SUM over g, so each engine's gradient is its own. ONE ``torch.optim.Adam`` over the
stacked leaves (betas 0.9 / 0.999, eps 1e-8, no weight decay, no clipping) at lr min(1, (s + 1) / warmup) x lr; Adam acts
element by element, so a stacked engine's update is the one it would get alone. THE GROUPS (``group``, default 1):
the rows are built once for the stack; after ``zero_grad`` the forward and backward run on each run of ``group``
consecutive engines (a slice of the state and of the rows, its summed loss backward into the stacked leaves), then
ONE Adam step; ``group`` = the number of engines is the single stacked pass, bit for bit. With group 1 an engine's
arithmetic does not depend on the other engines: a stacked engine's training is EXACTLY its solo run (torch.equal
on the CPU for kinds T1, T2 and T2K, both beds, both noises; ``tests/test_member_train.py``). The averaged weights "g8", element by
element: after step k, avg <- d_k avg + (1 - d_k) w, d_k = (1 + k) / (k + 10). Nothing in a step reads a number back
from the card, and ``train_members`` waits for the card only at its read points: the curve's steps (the losses and the
own reads), the snapshots, the saves and the last step run. THE SECONDS A STEP it records are the wall time between
two read points (the reads and saves themselves left out) divided by the number of steps between them, one such value
for each of those steps; ``time_stack`` waits for the card after every step (and after every phase when asked).

THE OWN READ of engine g: ``heldout`` rows of its own batch law with their noise and t, drawn once from a CPU generator
seeded ``collective.OWN_READ_SEED`` in chunks of 256 (each chunk's rows, then noise, then t, as
``collective.own_read_set``; rendered on the training bed's device, where the pictures, noise and targets stay), read
with the averaged weights through the stack (one stacked forward a chunk; the inputs prepared from the CPU layouts
once, when the sets are built, and every chunk stacked and kept on the device, ``own_read_chunks``): the newborn read first, then every ``every`` steps and at the end. The curve of
engine g: [{"step", "loss" (its own loss at that step), "own_read"}].

SNAPSHOTS: at each step k of ``snapshots`` (increasing, below ``steps``; (1500, 3000) by default when steps is 6000),
every engine's folder is written with k as its steps (in its folder name and card), its averaged weights and the own
read of step k. The learning rate after warm-up does not depend on the run's length and a step's rows depend only on
(seed, step), so a snapshot at step k is the result of a run of k steps.

FILES: engine g in ``<out>/<member_tag(...)>/``, in this order: engine.safetensors, engine_avg.safetensors, engine.json
(through ``members.save_member``; the card: bed, steps, seed, p_point, exclude, held_pairs, noise), then train.json
LAST (written to train.json.part and renamed; the settings, the curve, newborn_read, last_read, descended, trained
numbers, seconds a step of the stack and of an engine, device, torch version): a folder counts as a finished engine
only when its train.json is there. Snapshots are written the same way. THE STACK'S CHECKPOINT, every ``save_every``
steps and at the last step run, one GENERATION a save: the tensors (parameters, averages, the optimizer's state) to
``stack_checkpoint_<step>.safetensors`` (written to a ``.part`` name, then renamed), THEN the record
``stack_checkpoint.json`` (the settings, the step, the name of that tensor file, the curves; to ``.part``, renamed
LAST), then the older generations' tensor files are removed. A job killed anywhere in a save leaves the previous record
naming the previous, whole file. A job started again on the same folder loads the file its record names (a record
naming a missing file is an error) and continues, bit for bit on the CPU. The settings are compared as JSON reads them
back. ``<out>/stack_train.json`` (written after every engine's folder, renamed into place) marks a finished stack,
which is read back.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from . import members, scenes
from .collective import OWN_READ_SEED, TOKEN_DIM, _adam_load, _adam_tensors, noisy
from .compat import TIME_FREQ
from .members import SLOTS, TOLD_BLOCK, TOLD_HEAD, Variant
from .multirotary import angle_rotary
from .scenes import CANVAS, CHANNELS, PATCH, TOKEN_GRID, ScenesBed
from .spectral import patchify, unpatchify
from .train import step_generator

BETAS = (0.9, 0.999)
EPS = 1e-8
LN_EPS = 1e-6
READ_CHUNK = 256
STACK_FILES = ("stack_checkpoint_<step>.safetensors", "stack_checkpoint.json")
_GENERATION = re.compile(r"stack_checkpoint(_\d+)?\.safetensors(\.part)?")


def checkpoint_name(step: int) -> str:
    """The tensor file of the stack's checkpoint generation of ``step``."""
    return f"stack_checkpoint_{int(step)}.safetensors"
STACK_DONE = "stack_train.json"
NOISES = ("stream", "device")
NOISE_SEED_OFFSET = 1 << 40           # the step offset of the noise generator of noise "device"
PHASES = ("layouts", "noise", "render", "prepare", "forward", "backward", "adam_and_averages")


# ------------------------------------------------------------------ the stack
def stack_state(engines: Sequence[torch.nn.Module]) -> Dict[str, Tensor]:
    """{parameter name: [G, ...] float32}: each engine's parameters stacked on a new first axis (detached copies)."""
    named = [dict(e.named_parameters()) for e in engines]
    names = list(named[0])
    for n in named[1:]:
        if list(n) != names:
            raise ValueError("the engines of a stack have the same parameters")
    return {k: torch.stack([n[k].detach().float() for n in named]).clone() for k in names}


def unstack_state(state: Dict[str, Tensor], g: int) -> Dict[str, Tensor]:
    """The state dict of engine g (detached copies)."""
    return {k: v[int(g)].detach().clone() for k, v in state.items()}


def stack_depth(state: Dict[str, Tensor]) -> int:
    idx = {int(m.group(1)) for m in (re.match(r"blocks\.(\d+)\.", k) for k in state) if m}
    return max(idx) + 1 if idx else 0


def _lin(x: Tensor, state: Dict[str, Tensor], name: str) -> Tensor:
    w, b = state[name + ".weight"], state[name + ".bias"]
    y = torch.einsum("g...i,goi->g...o", x, w)
    return y + b.view(b.shape[0], *([1] * (y.dim() - 2)), b.shape[-1])


def _ln(x: Tensor) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=LN_EPS)


def _sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Attention on [G, B, heads, N, hd] with a bool mask [G, B, 1, N, L] (None: every key): one call on the G x B
    rows."""
    g, b = q.shape[:2]
    y = F.scaled_dot_product_attention(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
                                       attn_mask=None if mask is None else mask.flatten(0, 1))
    return y.view(g, b, *y.shape[1:])


def stacked_forward(state: Dict[str, Tensor], heads: int, z: Tensor, t: Tensor, content: Tensor,
                    query_angles: Tensor, key_angles: Tensor, key_mask: Tensor,
                    told_angles: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
    """See the module's text -> [G, B, 3, 48, 48]."""
    g, b = z.shape[:2]
    d = state["embed.weight"].shape[1]
    hd = d // int(heads)
    tokens = patchify(z.reshape(g * b, CHANNELS, CANVAS, CANVAS), PATCH).reshape(g, b, -1, TOKEN_DIM)
    h = torch.cat([_lin(content.to(z.dtype), state, "req_in"), _lin(tokens, state, "embed")], dim=2)
    n = h.shape[2]
    half = TIME_FREQ // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=z.device) / half)
    arg = (t.float() * 1000.0)[..., None] * freqs
    e = _lin(F.silu(_lin(torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1), state, "time.mlp.0")), state,
             "time.mlp.2")                                                             # [G, B, d]
    ctx = _lin(torch.zeros(g, b, 1, state["ctx_in.weight"].shape[-1], device=z.device, dtype=z.dtype), state, "ctx_in")
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    allowed = key_mask.to(device=z.device, dtype=torch.bool)[:, :, None, None, :] | eye
    rot = angle_rotary(query_angles, key_angles)
    rot_told = rot if told_angles is None else angle_rotary(query_angles, key_angles, {TOLD_HEAD: told_angles})
    for i in range(stack_depth(state)):
        p = f"blocks.{i}."
        ds1, dh1, ds2, dh2, ds3, dh3 = _lin(F.silu(e), state, p + "modulation.1").unsqueeze(2).chunk(6, dim=-1)
        y = _ln(h) * (1 + ds1) + dh1
        qkv = _lin(y, state, p + "attn.qkv").view(g, b, n, 3, heads, hd).permute(3, 0, 1, 4, 2, 5)
        q, k = (rot_told if i == TOLD_BLOCK else rot)(qkv[0], qkv[1])
        att = _sdpa(q, k, qkv[2], allowed).transpose(2, 3).reshape(g, b, n, d)
        h = h + _lin(att, state, p + "attn.out")
        y = _ln(h) * (1 + ds2) + dh2
        cq = _lin(y, state, p + "cross.q").view(g, b, n, heads, hd).transpose(2, 3)
        kv = _lin(ctx, state, p + "cross.kv").view(g, b, 1, 2, heads, hd).permute(3, 0, 1, 4, 2, 5)
        catt = _sdpa(cq, kv[0], kv[1], None).transpose(2, 3).reshape(g, b, n, d)     # one valid context token
        h = h + _lin(catt, state, p + "cross.out")
        y = _ln(h) * (1 + ds3) + dh3
        h = h + _lin(F.gelu(_lin(y, state, p + "mlp.0"), approximate="tanh"), state, p + "mlp.2")
    out = _lin(_ln(h[:, :, SLOTS:]), state, "head.proj")
    return unpatchify(out.reshape(g * b, -1, TOKEN_DIM), (CHANNELS, CANVAS, CANVAS), PATCH,
                      TOKEN_GRID).view(g, b, CHANNELS, CANVAS, CANVAS)


# ------------------------------------------------------------------ rows
def _held_text(held_pairs) -> str:
    pairs = sorted((int(a), int(q)) for a, q in held_pairs)
    if not pairs:
        return "none"
    text = ".".join(f"{a}{q}" for a, q in pairs)
    return text if len(pairs) <= 4 else f"{len(pairs)}x{zlib.crc32(text.encode()):08x}"


def member_tag(variant: Variant, cls: Optional[int], bed, seed: int, steps: int, dim: int, heads: int, depth: int,
               exclude: Optional[int] = None, held_pairs=(), noise: str = "stream") -> str:
    """A folder name that changes with every argument: variant, class (or base), bed, seed, steps, shape, the excluded
    class, the held pairs (written out up to four pairs, else their count and checksum) and the noise when it is not
    "stream" (``_noise<noise>`` at the end)."""
    if noise not in NOISES:
        raise ValueError(f"noise is one of {NOISES}, not {noise!r}")
    who = "base" if cls is None else f"c{int(cls)}"
    bed_name = bed.bed if isinstance(bed, ScenesBed) else str(bed)
    ex = "none" if exclude is None else str(int(exclude))
    tail = "" if noise == "stream" else f"_noise{noise}"
    return (f"{variant.name}_{who}_bed{bed_name}_seed{int(seed)}_steps{int(steps)}_d{int(dim)}h{int(heads)}"
            f"x{int(depth)}_ex{ex}_held{_held_text(held_pairs)}{tail}")


def _batch_args(variant: Variant, exclude, held_pairs) -> Tuple[Optional[int], tuple]:
    if variant.told == "own":
        return None, ()                                          # the stored rows: no clutter, no pointers
    return exclude, tuple(held_pairs)


def _loss_ok(variant: Variant, cls: Optional[int], b: dict) -> bool:
    """Every row's loss mask holds a token, read from the CPU layout: a thing's class stands in the row; the base has a
    request-free cell (told "all") or a visible cell (told "own", the stored base's loss = its visible cells)."""
    if cls is not None:
        return bool((b["cells_cpu"] == int(cls)).any(dim=1).all())
    free = b["visible_cpu"] if variant.told == "own" else (b["cells_cpu"] < 0)
    return bool(free.any(dim=1).all())


def _prepare_cpu(variant: Variant, b: dict) -> dict:
    return members.prepare(variant, b["cells_cpu"], b["colors_cpu"], b["points_cpu"], b["told_cpu"], b["visible_cpu"])


def engine_rows(variant: Variant, cls: Optional[int], bed: ScenesBed, n: int, gen: torch.Generator, p_point: float,
                exclude, held_pairs) -> dict:
    """One engine's rows from the CPU generator ``gen``: member_batch (its draws on the CPU, its render on ``bed``'s
    device), then noise and t (drawn on the CPU, moved to the picture's device), then the inputs prepared from the
    layout on the CPU (moved by ``stack_rows``)."""
    ex, held = _batch_args(variant, exclude, held_pairs)
    b = members.member_batch(variant, cls, bed, n, gen, p_point, ex, held)
    z, t, target = noisy(b["x"], gen)
    return {"batch": b, "z": z, "t": t, "target": target, "prep": _prepare_cpu(variant, b),
            "loss_mask": b["loss_mask"], "loss_ok": _loss_ok(variant, cls, b)}


def step_rows(variant: Variant, classes: Sequence[Optional[int]], bed: ScenesBed, seed: int, step: int, n: int,
              p_point: float = 0.5, exclude=None, held_pairs=()) -> List[dict]:
    """The rows of every engine at a step: a fresh ``step_generator(seed, step, "cpu")`` an engine."""
    return [engine_rows(variant, cls, bed, n, step_generator(int(seed), int(step), "cpu"), p_point, exclude, held_pairs)
            for cls in classes]


def stack_rows(rows: Sequence[dict], device) -> dict:
    """The rows of G engines stacked on G and moved to ``device`` (the loss masks' counts checked on the CPU layout
    first: ``loss_ok``)."""
    for r in rows:
        if not r["loss_ok"]:
            raise ValueError("a row of the batch has no token in its loss mask")

    def st(key):
        return torch.stack([r["prep"][key] for r in rows]).to(device)

    told = None
    if rows[0]["prep"]["told_angles"] is not None:
        told = tuple(torch.stack([r["prep"]["told_angles"][j] for r in rows]).to(device) for j in range(2))
    return {"z": torch.stack([r["z"] for r in rows]).to(device), "t": torch.stack([r["t"] for r in rows]).to(device),
            "target": torch.stack([r["target"] for r in rows]).to(device),
            "loss_mask": torch.stack([r["loss_mask"] for r in rows]).to(device),
            "inputs": {"content": st("content"), "query_angles": st("query_angles"), "key_angles": st("key_angles"),
                       "key_mask": st("key_mask"), "told_angles": told}}


def noise_generator(seed: int, step: int, device) -> torch.Generator:
    """The generator of a step's noise and t for noise "device": on ``device``, seeded by the derivation of
    ``train.step_generator`` with the step moved by ``NOISE_SEED_OFFSET``."""
    return step_generator(int(seed), int(step) + NOISE_SEED_OFFSET, device)


def _tick(clock: Optional[dict], key: str, since: float, sync: bool) -> float:
    """Add the seconds since ``since`` to clock[key] (the card waited for first when ``sync``) -> now."""
    if clock is None:
        return since
    if sync:
        torch.cuda.synchronize()
    now = time.perf_counter()
    clock[key] = clock.get(key, 0.0) + (now - since)
    return now


def stack_step_rows(variant: Variant, classes: Sequence[Optional[int]], bed: ScenesBed, seed: int, step: int, n: int,
                    p_point: float = 0.5, exclude=None, held_pairs=(), device=None, noise: str = "stream",
                    clock: Optional[dict] = None) -> dict:
    """The rows of every engine at a step, stacked on G, on ``device`` (default the bed's): for each engine on the CPU,
    gen = ``step_generator(seed, step, "cpu")``, ``members.member_layout`` (its loss mask checked on the CPU layout),
    then the noise [n, 3, 48, 48] and t [n]: noise "stream" draws them from gen on the CPU (the numbers of
    ``step_rows``), noise "device" from a fresh ``noise_generator(seed, step, device)`` an engine, on the device. Then
    ONE ``scenes.render_drawn`` of all engines' rows on the device, z = (1 - t) noise + t x and target = x - noise (the
    expressions of ``collective.noisy``; bed C's texture engine by engine, the one piece of the render whose rounding
    may depend on its batch), and ONE ``members.prepare`` of all engines' layouts on the device (the
    layouts checked once on the CPU), reshaped to a leading axis of engines; when its query and key angles are one
    tensor the stacked forward receives that one tensor for both. Nothing is read back from the device. -> the dict of
    ``stack_rows``. ``clock``: a dict that takes the seconds of the phases "layouts", "noise", "render" and "prepare"
    (the card waited for at the end of each)."""
    if noise not in NOISES:
        raise ValueError(f"noise is one of {NOISES}, not {noise!r}")
    device = bed.device if device is None else device
    sync = clock is not None and str(device).startswith("cuda")
    ex, held = _batch_args(variant, exclude, held_pairs)
    n, g_count = int(n), len(classes)
    lays, z0s, ts = [], [], []
    for cls in classes:
        a = time.perf_counter()
        gen = step_generator(int(seed), int(step), "cpu")
        lay = members.member_layout(variant, cls, bed, n, gen, p_point, ex, held)
        if not _loss_ok(variant, cls, {"cells_cpu": lay["cells"], "visible_cpu": lay["visible"]}):
            raise ValueError("a row of the batch has no token in its loss mask")
        lays.append(lay)
        a = _tick(clock, "layouts", a, False)
        if noise == "stream":
            z0s.append(torch.randn((n, CHANNELS, CANVAS, CANVAS), generator=gen))
            ts.append(torch.rand(n, generator=gen))
        else:
            ng = noise_generator(seed, step, device)
            z0s.append(torch.randn((n, CHANNELS, CANVAS, CANVAS), generator=ng, device=device))
            ts.append(torch.rand(n, generator=ng, device=device))
        _tick(clock, "noise", a, False)
    a = time.perf_counter()

    def cat(key):
        return torch.cat([lay[key] for lay in lays])

    cells, colors, points, told = cat("cells"), cat("colors"), cat("points"), cat("told")
    visible = None if lays[0]["visible"] is None else cat("visible")
    params = {k: torch.cat([lay["params"][k] for lay in lays]) for k in lays[0]["params"]}
    loss_mask = torch.stack([lay["loss_mask"] for lay in lays])
    members.check_layout(cells, colors, points)
    a = _tick(clock, "layouts", a, False)
    z0 = torch.stack(z0s).to(device)
    t = torch.stack(ts).to(device)
    a = _tick(clock, "noise", a, sync)
    pixels, _ = scenes.render_drawn(cells, colors, params, bed.bed, device, parts=[n] * g_count)
    x = scenes.to_model(pixels).view(g_count, n, CHANNELS, CANVAS, CANVAS)
    tt = t.view(g_count, n, 1, 1, 1)
    z, target = (1 - tt) * z0 + tt * x, x - z0
    a = _tick(clock, "render", a, sync)
    prep = members.prepare(variant, cells.to(device), colors.to(device), points.to(device), told.to(device),
                           None if visible is None else visible.to(device), check=False)

    def per_engine(v: Tensor) -> Tensor:
        return v.reshape(g_count, n, *v.shape[1:]).contiguous()

    ka = per_engine(prep["key_angles"])
    qa = ka if prep["query_angles"] is prep["key_angles"] else per_engine(prep["query_angles"])
    told_angles = None if prep["told_angles"] is None else tuple(per_engine(v) for v in prep["told_angles"])
    inputs = {"content": per_engine(prep["content"]), "query_angles": qa, "key_angles": ka,
              "key_mask": per_engine(prep["key_mask"]), "told_angles": told_angles}
    out = {"z": z, "t": t, "target": target, "loss_mask": loss_mask.to(device), "inputs": inputs}
    _tick(clock, "prepare", a, sync)
    return out


def masked_loss_unchecked(v: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
    """``collective.masked_loss`` without its check (the counts were checked on the CPU), the same expressions."""
    count = loss_mask.sum(dim=1)
    d = patchify(v - target, PATCH)
    sq = (d * d).sum(dim=-1)
    per = (sq * loss_mask.to(sq.dtype)).sum(dim=1) / (TOKEN_DIM * count.to(sq.dtype))
    return per.mean()


def check_group(group, engines: int) -> int:
    """``group`` as an int, or a ValueError unless 1 <= group <= the number of engines."""
    g = int(group)
    if not 1 <= g <= int(engines):
        raise ValueError(f"group is 1 .. {int(engines)} (the engines of the stack), not {group}")
    return g


def slice_rows(sb: dict, a: int, b: int) -> dict:
    """The stacked rows of engines a .. b - 1 (views); when the query and key angles are one tensor, one view for
    both."""
    inp = sb["inputs"]
    ka = inp["key_angles"][a:b]
    qa = ka if inp["query_angles"] is inp["key_angles"] else inp["query_angles"][a:b]
    told = None if inp["told_angles"] is None else tuple(x[a:b] for x in inp["told_angles"])
    return {"z": sb["z"][a:b], "t": sb["t"][a:b], "target": sb["target"][a:b], "loss_mask": sb["loss_mask"][a:b],
            "inputs": {"content": inp["content"][a:b], "query_angles": qa, "key_angles": ka,
                       "key_mask": inp["key_mask"][a:b], "told_angles": told}}


def stack_losses(state: Dict[str, Tensor], heads: int, sb: dict) -> List[Tensor]:
    v = stacked_forward(state, heads, sb["z"], sb["t"], **sb["inputs"])
    return [masked_loss_unchecked(v[g], sb["target"][g], sb["loss_mask"][g]) for g in range(v.shape[0])]


# ------------------------------------------------------------------ the own read
def own_read_sets(variant: Variant, classes, bed: ScenesBed, n: int, p_point: float, exclude, held_pairs) -> list:
    """For each engine, its held-out chunks (rows, then noise, then t, a chunk of 256 at a time) from one CPU generator
    seeded ``collective.OWN_READ_SEED``, rendered on ``bed``'s device (the pictures, noise and targets stay there);
    the layouts kept on the CPU, the inputs prepared from them at the read."""
    out = []
    ex, held = _batch_args(variant, exclude, held_pairs)
    for cls in classes:
        gen = torch.Generator().manual_seed(OWN_READ_SEED)
        chunks = []
        for k in range(0, int(n), READ_CHUNK):
            m = min(READ_CHUNK, int(n) - k)
            b = members.member_batch(variant, cls, bed, m, gen, p_point, ex, held)
            z, t, target = noisy(b["x"], gen)
            chunks.append({"batch": {k2: b[k2] for k2 in ("cells_cpu", "colors_cpu", "points_cpu", "told_cpu",
                                                           "visible_cpu")},
                           "z": z, "t": t, "target": target, "loss_mask": b["loss_mask"],
                           "loss_ok": _loss_ok(variant, cls, b), "prep": _prepare_cpu(variant, b)})
        out.append(chunks)
    return out


def own_read_chunks(sets: list, device) -> List[dict]:
    """The held-out chunks of every engine stacked on G and moved to ``device`` once (``stack_rows`` of each chunk's
    rows and the inputs prepared when the sets were built); when every engine's query and key angles are one tensor,
    the stack holds one tensor for both (the same numbers, half the memory)."""
    out = []
    for c in range(len(sets[0])):
        parts = [s[c] for s in sets]
        sb = stack_rows(parts, device)
        if all(p["prep"]["query_angles"] is p["prep"]["key_angles"] for p in parts):
            sb["inputs"]["query_angles"] = sb["inputs"]["key_angles"]
        sb["rows"] = int(parts[0]["z"].shape[0])
        out.append(sb)
    return out


@torch.no_grad()
def stack_own_read(state: Dict[str, Tensor], heads: int, variant: Variant, sets: list, device,
                   chunks: Optional[List[dict]] = None) -> List[float]:
    """The masked loss of each engine on its held-out set (the mean over its rows), one stacked forward a chunk;
    ``chunks``: ``own_read_chunks(sets, device)`` built once (built here when not given)."""
    totals = [0.0] * len(sets)
    rows = 0
    for sb in (own_read_chunks(sets, device) if chunks is None else chunks):
        m = sb["rows"]
        for g, loss in enumerate(stack_losses(state, heads, sb)):
            totals[g] += float(loss) * m
        rows += m
    return [x / max(1, rows) for x in totals]


# ------------------------------------------------------------------ the trainer
def _median(values) -> Optional[float]:
    v = sorted(float(x) for x in values)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def _engine_settings(variant, cls, bed, seed, steps, settings) -> dict:
    s = dict(settings)
    s.update({"variant": variant.settings(), "cls": None if cls is None else int(cls), "steps": int(steps)})
    return s


def _write_engine(folder: str, variant: Variant, cls, seed: int, state: Dict[str, Tensor], avg: Dict[str, Tensor],
                  g: int, dim: int, heads: int, depth: int, card: dict, train_info: dict) -> None:
    eng = members.born(variant, cls, seed, dim, heads, depth)
    eng.load_state_dict(unstack_state(state, g), strict=True)
    members.save_member(eng, folder, average=unstack_state(avg, g), card=card)
    with open(os.path.join(folder, "train.json") + ".part", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(train_info, fh, indent=1)
        fh.write("\n")
    os.replace(os.path.join(folder, "train.json") + ".part", os.path.join(folder, "train.json"))


def _step(variant: Variant, classes, bed: ScenesBed, seed: int, s: int, batch: int, p_point: float, exclude,
          held, state: Dict[str, Tensor], heads: int, opt, average: Dict[str, Tensor], device, lr: float,
          warmup: int, rows_clock: Optional[List[float]] = None, noise: str = "stream",
          clock: Optional[dict] = None, group: int = 1) -> List[Tensor]:
    """One training step s of the stack (the learning rate, the rows of ``stack_step_rows``, the summed loss, Adam,
    the averages) -> the engines' losses (on the device, not read). ``rows_clock``: a list that takes the seconds spent
    building the step's rows (without ``clock``, on a card the render is queued, so this reads the time to queue it).
    ``clock``: a dict that takes the seconds of every phase of ``PHASES``, the card waited for at the end of each
    (the groups' forwards and backwards added up). ``group``: the forward and backward run on runs of ``group``
    consecutive engines (a slice of the state and of the rows), each run's summed loss backward into the stacked
    leaves, then ONE Adam step; group = the number of engines is the single stacked pass."""
    sync = clock is not None and str(device).startswith("cuda")
    for grp in opt.param_groups:
        grp["lr"] = float(lr) * min(1.0, (s + 1) / max(1, int(warmup)))
    r0 = time.perf_counter()
    sb = stack_step_rows(variant, classes, bed, seed, s, batch, p_point, exclude, held, device, noise, clock)
    if rows_clock is not None:
        rows_clock.append(time.perf_counter() - r0)
    g_count = len(classes)
    group = check_group(group, g_count)
    opt.zero_grad(set_to_none=True)
    losses: List[Tensor] = []
    for a0 in range(0, g_count, group):
        b0 = min(g_count, a0 + group)
        a = time.perf_counter()
        if a0 == 0 and b0 == g_count:                        # the whole stack: one pass, nothing sliced
            part = stack_losses(state, heads, sb)
        else:
            part = stack_losses({k: v[a0:b0] for k, v in state.items()}, heads, slice_rows(sb, a0, b0))
        loss = part[0]
        for x in part[1:]:
            loss = loss + x
        a = _tick(clock, "forward", a, sync)
        loss.backward()                                      # the gradients accumulate in the stacked leaves
        _tick(clock, "backward", a, sync)
        losses += part
    a = time.perf_counter()
    opt.step()
    dk = (1 + s) / (s + 8 + 2)
    with torch.no_grad():
        for k in state:
            average[k].copy_(dk * average[k] + (1.0 - dk) * state[k].detach())
    _tick(clock, "adam_and_averages", a, sync)
    return losses


def time_stack(variant: Variant, classes: Sequence[Optional[int]], bed: ScenesBed, seed: int = 0, *,
               batch: int = 128, steps: int = 6, from_step: int = 2, p_point: float = 0.5,
               exclude: Optional[int] = None, held_pairs=(), dim: int = 128, heads: int = 2, depth: int = 3,
               lr: float = 1e-3, warmup: int = 100, noise: str = "stream", phases: bool = False,
               group: int = 1) -> dict:
    """The seconds a training step of a stack of these engines on ``bed``'s device (the steps of ``train_members``,
    nothing written, no reads): the stamps of every step (the card synchronized at each), the median over the steps
    from ``from_step`` on, and that median divided by the number of engines. ``phases``: also "phases", the median
    over the same steps of the seconds of each phase of ``PHASES`` (the card waited for at the end of each phase, so
    those steps are a little slower than unclocked ones)."""
    if noise not in NOISES:
        raise ValueError(f"noise is one of {NOISES}, not {noise!r}")
    classes = [None if c is None else int(c) for c in classes]
    group = check_group(group, len(classes))
    device = bed.device
    held = sorted((int(a), int(q)) for a, q in held_pairs)
    engines = [members.born(variant, c, seed, dim, heads, depth) for c in classes]
    state = {k: v.to(device).requires_grad_(True) for k, v in stack_state(engines).items()}
    opt = torch.optim.Adam(list(state.values()), lr=float(lr), betas=BETAS, eps=EPS, weight_decay=0.0)
    average = {k: v.detach().clone() for k, v in state.items()}
    is_cuda = str(device).startswith("cuda")
    stamps = []
    rows_clock: List[float] = []
    clocks: List[dict] = []
    for s in range(int(steps)):
        clock = {} if phases else None
        s0 = time.perf_counter()
        _step(variant, classes, bed, seed, s, int(batch), p_point, exclude, held, state, heads, opt, average,
              device, lr, warmup, rows_clock, noise, clock, group)
        if is_cuda:
            torch.cuda.synchronize()
        stamps.append(time.perf_counter() - s0)
        if phases:
            clocks.append(clock)
    med = _median(stamps[int(from_step):])
    out = {"engines": len(classes), "batch": int(batch), "steps": int(steps), "from_step": int(from_step),
           "stamps": stamps, "seconds_a_step_stack": med, "seconds_rows": _median(rows_clock[int(from_step):]),
           "seconds_a_step_engine": None if med is None else med / len(classes), "device": str(device),
           "noise": noise, "group": group}
    if phases:
        out["phases"] = {k: _median(c.get(k, 0.0) for c in clocks[int(from_step):]) for k in PHASES}
    return out


def train_members(variant: Variant, classes: Sequence[Optional[int]], bed: ScenesBed, seed: int, out: str, *,
                  steps: int = 3000, batch: int = 128, lr: float = 1e-3, warmup: int = 100, every: int = 250,
                  heldout: int = 2048, save_every: int = 500, p_point: float = 0.5, exclude: Optional[int] = None,
                  held_pairs=(), dim: int = 128, heads: int = 2, depth: int = 3, until: Optional[int] = None,
                  snapshots: Optional[Sequence[int]] = None, noise: str = "stream", group: int = 1,
                  log: Callable[[str], None] = print) -> dict:
    """Train the engines of ``classes`` (0..8 and/or None for the base) as one stack (see the module's text) into
    ``out`` -> {"settings", "step", "finished", "engines": [{"cls", "tag", "curve", "newborn_read", "last_read",
    "descended"}], "seconds_a_step_stack", "seconds_a_step_engine", "state", "average"}. ``noise``: where a step's
    noise and t are drawn (``stack_step_rows``); it is part of the settings and of the engines' names. ``group``: the
    engines run forward and backward ``group`` at a time (``_step``); part of the settings, not of the names."""
    if noise not in NOISES:
        raise ValueError(f"noise is one of {NOISES}, not {noise!r}")
    classes = [None if c is None else int(c) for c in classes]
    group = check_group(group, len(classes))
    if len(set(classes)) != len(classes) or not classes:
        raise ValueError("a stack holds one engine a class (and at most one base)")
    steps, batch = int(steps), int(batch)
    if snapshots is None:
        snapshots = (1500, 3000) if steps == 6000 else ()
    snapshots = tuple(int(k) for k in snapshots)
    if list(snapshots) != sorted(set(snapshots)) or any(not 0 < k < steps for k in snapshots):
        raise ValueError("snapshots are increasing steps between 0 and steps")
    exclude = None if exclude is None else int(exclude)
    held = sorted((int(a), int(q)) for a, q in held_pairs)
    device = bed.device
    G = len(classes)
    held_lists = [[a, q] for a, q in held]
    # the settings as JSON reads them back (lists, never tuples), so that a restart compares like with like
    settings = json.loads(json.dumps({
        "variant": variant.settings(), "classes": classes, "bed": bed.bed, "seed": int(seed), "steps": steps,
        "batch": batch, "lr": float(lr), "warmup": int(warmup), "every": int(every), "heldout": int(heldout),
        "save_every": int(save_every), "p_point": float(p_point), "exclude": exclude, "held_pairs": held_lists,
        "dim": int(dim), "heads": int(heads), "depth": int(depth), "snapshots": list(snapshots), "average": "g8",
        "optimizer": "Adam", "betas": list(BETAS), "eps": EPS, "weight_decay": 0.0, "clip": None,
        "stack": "written-out stacked forward", "noise": noise, "group": group}))
    tags = [member_tag(variant, c, bed, seed, steps, dim, heads, depth, exclude, held, noise) for c in classes]
    os.makedirs(out, exist_ok=True)
    done_path = os.path.join(out, STACK_DONE)
    if os.path.isfile(done_path) and all(os.path.isfile(os.path.join(out, tg, f)) for tg in tags
                                         for f in members.MEMBER_FILES + ("train.json",)):
        with open(done_path, encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored["settings"] != settings:
            raise ValueError(f"{out} holds a finished stack made with other settings")
        log(f"FINISHED already: {out}")
        stored["restored"] = True
        return stored
    from safetensors.torch import load as load_bytes, save_file
    engines = [members.born(variant, c, seed, dim, heads, depth) for c in classes]
    trained = int(sum(p.numel() for p in engines[0].parameters()))
    state = {k: v.to(device).requires_grad_(True) for k, v in stack_state(engines).items()}
    names = list(state)
    params = [state[k] for k in names]
    opt = torch.optim.Adam(params, lr=float(lr), betas=BETAS, eps=EPS, weight_decay=0.0)
    average = {k: v.detach().clone() for k, v in state.items()}
    sets = own_read_sets(variant, classes, bed, heldout, p_point, exclude, held)
    read_chunks = own_read_chunks(sets, device)                    # prepared and moved once, kept on the device
    sets = [None] * len(sets)                                      # the chunks hold all a read needs: the sets' tensors are let go
    curves: List[List[dict]] = [[] for _ in classes]
    step_seconds: List[float] = []
    done = 0
    newborn: Optional[List[float]] = None
    ck_json = os.path.join(out, STACK_FILES[1])
    if os.path.isfile(ck_json):
        with open(ck_json, encoding="utf-8") as fh:
            ck = json.load(fh)
        if ck["settings"] != settings:
            raise ValueError(f"the stack checkpoint in {out} was made with other settings")
        named = ck.get("tensors")
        if not named or not os.path.isfile(os.path.join(out, named)):
            raise FileNotFoundError(f"the stack checkpoint record in {out} names the tensor file {named!r}, which is "
                                    "not there")
        with open(os.path.join(out, named), "rb") as fh:         # read whole, not mapped: the file can be removed
            tensors = load_bytes(fh.read())
        with torch.no_grad():
            for k in names:
                state[k].copy_(tensors["param." + k].to(device))
                average[k].copy_(tensors["avg." + k].to(device))
        _adam_load(opt, tensors, ck["param_groups"], device)
        curves, step_seconds, done, newborn = ck["curves"], ck["step_seconds"], int(ck["step"]), ck["newborn_reads"]
        log(f"RESUME from step {done:,} of {steps:,}")
    log(f"JOB stack {variant.name} classes {classes} bed {bed.bed} seed {seed}: {steps:,} steps x batch {batch} x "
        f"{G} engines; {trained:,} trained numbers an engine; pure Adam lr {lr:g}; noise {noise}; device {device}")
    if newborn is None:
        newborn = stack_own_read(average, heads, variant, sets, device, read_chunks)
        for g in range(G):
            curves[g].append({"step": 0, "loss": None, "own_read": newborn[g]})
        log("newborn own reads " + " ".join(f"{x:.6f}" for x in newborn))

    def checkpoint(step: int) -> None:
        tensors, groups = _adam_tensors(opt, params)
        tensors.update({"param." + k: v.detach().cpu().contiguous() for k, v in state.items()})
        tensors.update({"avg." + k: v.detach().cpu().contiguous() for k, v in average.items()})
        name = checkpoint_name(step)
        save_file(tensors, os.path.join(out, name) + ".part")
        os.replace(os.path.join(out, name) + ".part", os.path.join(out, name))
        with open(ck_json + ".part", "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"settings": settings, "step": step, "tensors": name, "param_groups": groups, "curves": curves,
                       "step_seconds": step_seconds, "newborn_reads": newborn, "names": names}, fh)
        os.replace(ck_json + ".part", ck_json)                         # the record last: it names a whole file
        for f in os.listdir(out):
            if f != name and _GENERATION.fullmatch(f):
                os.remove(os.path.join(out, f))

    def write_engines(at: int, reads: List[float], curve_upto: List[List[dict]]) -> List[dict]:
        rows = []
        per_stack = _median(step_seconds)
        for g, cls in enumerate(classes):
            tag = member_tag(variant, cls, bed, seed, at, dim, heads, depth, exclude, held, noise)
            info = {"settings": _engine_settings(variant, cls, bed, seed, at, settings), "curve": curve_upto[g],
                    "newborn_read": newborn[g], "last_read": reads[g], "descended": bool(reads[g] < newborn[g]),
                    "trained": trained, "seconds_a_step_stack": per_stack,
                    "seconds_a_step_engine": None if per_stack is None else per_stack / G,
                    "step_seconds_count": len(step_seconds), "device": str(device), "torch": torch.__version__}
            card = {"bed": bed.bed, "steps": int(at), "seed": int(seed), "p_point": float(p_point), "exclude": exclude,
                    "held_pairs": held_lists, "noise": noise}
            _write_engine(os.path.join(out, tag), variant, cls, seed, state, average, g, dim, heads, depth, card, info)
            rows.append({"cls": cls, "tag": tag, "curve": curve_upto[g], "newborn_read": newborn[g],
                         "last_read": reads[g], "descended": bool(reads[g] < newborn[g])})
        return rows

    stop = steps if until is None else min(int(until), steps)
    is_cuda = str(device).startswith("cuda")
    t0 = time.time()
    mark, since = time.perf_counter(), done
    for s in range(done, stop):
        losses = _step(variant, classes, bed, seed, s, batch, p_point, exclude, held, state, heads, opt, average,
                       device, lr, warmup, noise=noise, group=group)
        n = s + 1
        read_now = (every and (n % int(every) == 0 or n == steps)) or n in snapshots
        save_now = bool(save_every) and n % int(save_every) == 0 and n != stop
        if not (read_now or save_now or n == stop):
            continue                                                 # no wait for the card between the read points
        if is_cuda:
            torch.cuda.synchronize()
        per_step = (time.perf_counter() - mark) / (n - since)
        step_seconds.extend([per_step] * (n - since))
        since = n
        loss_v = [float(x.detach()) for x in losses] if read_now else None
        if read_now:
            reads = stack_own_read(average, heads, variant, sets, device, read_chunks)
            for g in range(G):
                curves[g].append({"step": n, "loss": loss_v[g], "own_read": reads[g]})
            spent = time.time() - t0
            log(f"step {n:,}/{steps:,} losses " + " ".join(f"{x:.5f}" for x in loss_v) + " own reads " +
                " ".join(f"{x:.5f}" for x in reads) + f"; spent {spent:.0f} s, left about "
                f"{spent / max(1, n - done) * (steps - n):.0f} s")
            if n in snapshots:
                write_engines(n, reads, [[c for c in cv if c["step"] <= n] for cv in curves])
                log(f"SNAPSHOT at step {n:,}")
        if save_now:
            checkpoint(n)
        mark = time.perf_counter()                                   # the reads and saves are not step time
    checkpoint(stop)
    result = {"settings": settings, "step": stop, "finished": stop >= steps}
    if stop >= steps:
        last = curves[0][-1]["step"] == steps
        reads = [cv[-1]["own_read"] for cv in curves] if last else stack_own_read(average, heads, variant, sets, device, read_chunks)
        rows = write_engines(steps, reads, curves)
        per_stack = _median(step_seconds)
        result.update({"engines": rows, "trained": trained, "seconds_a_step_stack": per_stack,
                       "seconds_a_step_engine": None if per_stack is None else per_stack / G,
                       "train_seconds": float(sum(step_seconds)), "device": str(device), "torch": torch.__version__})
        with open(done_path + ".part", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=1)
            fh.write("\n")
        os.replace(done_path + ".part", done_path)
        log(f"DONE stack {variant.name}: " + " ".join(f"{r['cls']}: {r['newborn_read']:.5f} -> {r['last_read']:.5f}"
                                                      for r in rows) +
            f"; {per_stack:.4f} s a step of the stack")
    result["state"] = {k: v.detach() for k, v in state.items()}
    result["average"] = {k: v.detach() for k, v in average.items()}
    return result
