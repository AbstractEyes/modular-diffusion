"""Band engines: smaller U-Nets made from the teacher by paired block removal, every number inherited exactly.

An engine is a diffusers ``UNet2DConditionModel`` built from the teacher's own settings with ``layers_per_block`` set
per stage (``shapes.Shape``), no middle block when the shape drops it, and the block lists cut to three stages when
asked. Its numbers are then copied from the teacher, tensor by tensor, by this map:

    a down stage keeping one layer       the teacher's FIRST residual block of that stage (and its attention block)
    an up stage keeping two layers       the teacher's FIRST and THIRD residual blocks (and their attention blocks)
    a stage keeping all its layers       every block, in order
    everything else                      as it is: the input convolution, the time embedding, the down and up samplers,
                                         the middle block when kept, the output norm and convolution

A stage that becomes the last down stage (with three stages) has no down sampler in the engine, so the teacher's is
not used. Every engine tensor is filled by exactly one teacher tensor of exactly the same shape; nothing is resized or
initialized afresh (``build_engine`` checks both and raises otherwise). The engine of the shape ``full`` is the
teacher, bit for bit.

These copies keep their channels: the first residual block of a down stage reads the stage's input channels, as the
teacher's does, and the third residual block of an up stage reads the skip of the stage below, as the teacher's does,
so each copy fits its place with no change.

``text=False`` makes an engine that ignores the prompt: its cross-attention reads one constant context stored in the
engine (a trainable [1, 77, D] tensor, born as the teacher's empty-prompt states).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn

from .shapes import Shape, shape_from

# settings of the U-Net that hold one entry per down stage, and one entry per up stage
PER_DOWN = ("down_block_types", "block_out_channels", "layers_per_block", "attention_head_dim", "num_attention_heads",
            "only_cross_attention", "transformer_layers_per_block", "cross_attention_dim")
PER_UP = ("up_block_types", "reverse_transformer_layers_per_block")

_DOWN = re.compile(r"^down_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.*)$")
_UP = re.compile(r"^up_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.*)$")
_UP_OTHER = re.compile(r"^up_blocks\.(\d+)\.(.*)$")


def _unet_of(x) -> nn.Module:
    """The diffusers U-Net of a teacher, an engine or a U-Net."""
    if isinstance(x, Engine):
        return x.unet
    return getattr(x, "unet", x)


def engine_config(teacher_unet, shape: Shape) -> dict:
    """The teacher's settings changed to the shape's: layers per stage, the middle block, the number of stages."""
    unet = _unet_of(teacher_unet)
    cfg = {k: v for k, v in dict(unet.config).items() if not k.startswith("_")}
    n = len(cfg["down_block_types"])
    if n != 4:
        raise ValueError(f"the teacher must have four stages, not {n}")
    lpb = cfg["layers_per_block"]
    lpb = [lpb] * n if isinstance(lpb, int) else list(lpb)
    if any(x != 2 for x in lpb):
        raise ValueError(f"the teacher must keep two layers a down stage, not {lpb}")
    s = shape.stages
    for k in PER_DOWN:
        v = cfg.get(k)
        if isinstance(v, (list, tuple)) and len(v) == n:
            cfg[k] = list(v)[:s]
    for k in PER_UP:
        v = cfg.get(k)
        if isinstance(v, (list, tuple)) and len(v) == n:
            cfg[k] = list(v)[n - s:]
    cfg["layers_per_block"] = list(shape.kept)
    if not shape.mid:
        cfg["mid_block_type"] = None
    return cfg


def down_kept(n: int):
    """The teacher's layer indices a down stage keeps when it keeps n of its two."""
    return [0] if n == 1 else [0, 1]


def up_kept(n: int):
    """The teacher's layer indices an up stage keeps when its down stage keeps n (it keeps n + 1 of its three)."""
    return [0, 2] if n == 1 else [0, 1, 2]


def teacher_key(key: str, shape: Shape) -> str:
    """The teacher tensor an engine tensor is copied from."""
    s = shape.stages
    off = 4 - s
    m = _DOWN.match(key)
    if m:
        i, kind, j, rest = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        return f"down_blocks.{i}.{kind}.{down_kept(shape.kept[i])[j]}.{rest}"
    m = _UP.match(key)
    if m:
        k, kind, j, rest = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        stage = s - 1 - k
        return f"up_blocks.{k + off}.{kind}.{up_kept(shape.kept[stage])[j]}.{rest}"
    m = _UP_OTHER.match(key)
    if m:
        return f"up_blocks.{int(m.group(1)) + off}.{m.group(2)}"
    return key


def inherit_map(teacher_unet, shape: Union[str, Shape]) -> Dict[str, str]:
    """{engine tensor name: teacher tensor name} for every tensor of the shape's engine U-Net."""
    from diffusers import UNet2DConditionModel
    shape = shape_from(shape)
    with torch.device("meta"):
        skeleton = UNet2DConditionModel.from_config(engine_config(teacher_unet, shape))
    return {k: teacher_key(k, shape) for k in skeleton.state_dict()}


class Engine(nn.Module):
    """A band engine: ``unet`` (a diffusers U-Net) and, for an engine that ignores the prompt, ``context``.

    forward(z, t, states) -> the predicted noise. With ``text`` False the prompt's states are not read (only their
    batch size is): every row reads the stored context."""

    def __init__(self, unet: nn.Module, shape: Shape, text: bool = True, context: Optional[Tensor] = None):
        super().__init__()
        self.unet = unet
        self.shape = shape
        self.text = bool(text)
        if self.text:
            self.context = None
        else:
            if context is None:
                raise ValueError("an engine that ignores the prompt needs its context (the empty prompt's states)")
            c = context.detach().float().clone()
            if c.dim() == 2:
                c = c.unsqueeze(0)
            self.context = nn.Parameter(c[:1].clone())

    def forward(self, z: Tensor, t: Tensor, states: Optional[Tensor] = None) -> Tensor:
        # The inputs are cast to the engine's own precision and the answer is returned in the latents' precision,
        # so a full-precision engine and a half-precision teacher can be read on the same probe.
        own = next(self.unet.parameters()).dtype
        if not self.text:
            states = self.context.expand(z.shape[0], -1, -1)
        return self.unet(z.to(own), t, encoder_hidden_states=states.to(own)).sample.to(z.dtype)


def _context_of(teacher_unet, context: Optional[Tensor]) -> Optional[Tensor]:
    if context is not None:
        return context
    if hasattr(teacher_unet, "empty_states"):
        return teacher_unet.empty_states(1)
    return None


def _meta_unet(teacher_unet, shape: Shape):
    from diffusers import UNet2DConditionModel
    with torch.device("meta"):
        return UNet2DConditionModel.from_config(engine_config(teacher_unet, shape))


def _blank(teacher_unet, shape: Shape):
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel.from_config(engine_config(teacher_unet, shape)).float()


def build_engine(teacher_unet, shape: Union[str, Shape], text: bool = True, context: Optional[Tensor] = None) -> Engine:
    """The engine of a shape, every tensor copied from the teacher (``teacher_unet``: the teacher's U-Net or the
    teacher itself). Its numbers are kept in float32. ``text=False``: the context is ``context`` ([1, 77, D] or
    [77, D]) or, when the teacher itself is given, its empty prompt's states."""
    shape = shape_from(shape)
    t_unet = _unet_of(teacher_unet)
    device = next(t_unet.parameters()).device
    unet = _meta_unet(teacher_unet, shape)
    src = t_unet.state_dict()
    dst = unet.state_dict()
    filled = {}
    for k, v in dst.items():
        tk = teacher_key(k, shape)
        if tk not in src:
            raise KeyError(f"engine tensor {k} maps to {tk}, which the teacher does not have")
        if tuple(src[tk].shape) != tuple(v.shape):
            raise ValueError(f"engine tensor {k} {tuple(v.shape)} and teacher tensor {tk} {tuple(src[tk].shape)} differ")
        dtype = torch.float32 if v.is_floating_point() else v.dtype
        filled[k] = src[tk].detach().to(device=device, dtype=dtype, copy=True)
    missing = sorted(set(dst) - set(filled))
    if missing:
        raise ValueError(f"engine tensors not inherited: {missing[:5]}")
    unet.load_state_dict(filled, strict=True, assign=True)
    for p in unet.parameters():
        p.requires_grad_(True)
    ctx = None if text else _context_of(teacher_unet, context)
    return Engine(unet, shape, text, ctx).to(device)


def fresh_engine(teacher_unet, shape: Union[str, Shape], seed: int = 0, text: bool = True,
                 context: Optional[Tensor] = None) -> Engine:
    """An engine of the same shape with random numbers (the library's own initialization from the seed), nothing
    inherited: a control for the born read only."""
    shape = shape_from(shape)
    torch.manual_seed(int(seed))
    unet = _blank(teacher_unet, shape)
    device = next(_unet_of(teacher_unet).parameters()).device
    ctx = None if text else _context_of(teacher_unet, context)
    return Engine(unet, shape, text, ctx).to(device)


class UNetPart(nn.Module):
    """A diffusers U-Net as a part: forward(z, t, states) -> its predicted noise."""

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, z: Tensor, t: Tensor, states: Tensor) -> Tensor:
        # Inputs in the U-Net's own precision; the answer back in the latents' precision (see Engine.forward).
        own = next(self.unet.parameters()).dtype
        return self.unet(z.to(own), t, encoder_hidden_states=states.to(own)).sample.to(z.dtype)


def as_part(x) -> nn.Module:
    """Anything that answers forward(z, t, states) -> noise: an engine or an assembly as it is, a teacher or a diffusers
    U-Net wrapped."""
    if isinstance(x, Engine) or getattr(x, "is_part", False):
        return x
    if hasattr(x, "unet") and not isinstance(x, nn.Module):
        return UNetPart(x.unet)
    if hasattr(x, "config") and hasattr(x, "down_blocks"):
        return UNetPart(x)
    if isinstance(x, nn.Module):
        return x
    raise TypeError(f"not a part: {type(x).__name__}")


def param_count(module) -> int:
    """The number of numbers in a module's parameters (a teacher counts its U-Net)."""
    m = module if isinstance(module, nn.Module) else _unet_of(module)
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def forward_seconds(part, z: Tensor, t: Tensor, states: Tensor, repeats: int = 3, warmup: int = 1) -> float:
    """The measured seconds of one forward (the median of ``repeats`` after ``warmup`` untimed ones)."""
    p = as_part(part)
    sync = torch.cuda.synchronize if z.is_cuda else (lambda: None)
    for _ in range(int(warmup)):
        p(z, t, states)
    sync()
    times = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        p(z, t, states)
        sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def shape_report(teacher_unet, shape: Union[str, Shape], batch: int = 1, size: Optional[int] = None,
                 repeats: int = 3, step: int = 500, teacher_seconds: Optional[float] = None) -> dict:
    """Parameters of the shape's engine, their share of the teacher's, and its forward cost against the teacher's
    (measured here, one forward at ``batch`` rows of ``size`` x ``size`` latents, at one step)."""
    shape = shape_from(shape)
    unet = _unet_of(teacher_unet)
    eng = build_engine(unet, shape)
    dev = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    size = int(size or unet.config.sample_size)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(batch, unet.config.in_channels, size, size, generator=g).to(dev, dtype)
    ctx = torch.randn(batch, 77, int(unet.config.cross_attention_dim), generator=g).to(dev, dtype)
    t = torch.full((batch,), int(step), device=dev, dtype=torch.long)
    eng = eng.to(dtype)
    ts = forward_seconds(unet, z, t, ctx, repeats) if teacher_seconds is None else float(teacher_seconds)
    es = forward_seconds(eng, z, t, ctx, repeats)
    pt, pe = param_count(unet), param_count(eng)
    return {"shape": shape.name, "params": pe, "teacher_params": pt, "share": pe / pt, "seconds": es,
            "teacher_seconds": ts, "cost": es / ts, "batch": int(batch), "size": size}


# ------------------------------------------------------------------ files
def save_engine(engine: Engine, folder: str, name: str = "engine", average: Optional[Dict[str, Tensor]] = None,
                extra: Optional[dict] = None) -> None:
    """``<name>.safetensors`` (the engine's numbers), ``<name>_avg.safetensors`` (an averaged copy, when given) and
    ``<name>.json`` (its shape and whether it reads the prompt)."""
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in engine.state_dict().items()},
              os.path.join(folder, name + ".safetensors"))
    if average is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in average.items()},
                  os.path.join(folder, name + "_avg.safetensors"))
    card = {"shape": engine.shape.text, "shape_name": engine.shape.name, "text": engine.text}
    card.update(extra or {})
    with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(card, fh, indent=1)
        fh.write("\n")


def load_engine(teacher_unet, folder: str, name: str = "engine", average: bool = True) -> Engine:
    """An engine written by ``save_engine``: its averaged numbers when ``average`` and the file is there, else its last
    numbers (the averaged file holds the parameters; anything else comes from the last numbers)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with open(os.path.join(folder, name + ".json"), encoding="utf-8") as fh:
        card = json.load(fh)
    eng_path = os.path.join(folder, name + ".safetensors")
    avg_path = os.path.join(folder, name + "_avg.safetensors")
    if average and os.path.isfile(avg_path):
        state = load_file(avg_path)
        with safe_open(eng_path, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                if k not in state:
                    state[k] = fh.get_tensor(k)
    else:
        state = load_file(eng_path)
    shape = Shape.parse(card["shape"])
    text = bool(card["text"])
    ctx = state.get("context")
    device = next(_unet_of(teacher_unet).parameters()).device
    unet = _meta_unet(teacher_unet, shape)
    eng = Engine(unet, shape, text=text, context=ctx)
    expected = eng.state_dict()
    loaded = {}
    for k, v in state.items():
        if k not in expected:
            loaded[k] = v
            continue
        ev = expected[k]
        if tuple(v.shape) != tuple(ev.shape):
            raise RuntimeError(f"size mismatch for {k}: copying a param with shape {tuple(v.shape)} from checkpoint, "
                               f"the shape in current model is {tuple(ev.shape)}.")
        dtype = torch.float32 if ev.is_floating_point() else ev.dtype
        loaded[k] = v.detach().to(device=device, dtype=dtype, copy=True)
    eng.load_state_dict(loaded, strict=True, assign=True)
    for p in eng.parameters():
        p.requires_grad_(True)
    return eng
