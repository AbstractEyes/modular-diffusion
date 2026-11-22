"""Units: the independently trainable parts a model is assembled from.

Every unit has a ``name`` (unique in a build), a ``unit_type``, an ``address`` (where it acts: its site, band, window of
the noise level, soft half-width, class set) and a ``spec`` (its structure). Its owned state is its saved state dict,
with keys relative to the unit.

| type          | owns                                        | acts                                                   |
|---------------|---------------------------------------------|--------------------------------------------------------|
| Anchor        | spectrum [C, H, W]                          | (z_t, t) -> the closed-form linear velocity            |
| Stem          | embed, time, cls, blocks                    | tokens, t, class -> state h, context, time embedding    |
| StageBody     | block                                       | h -> h inside its window of t                           |
| GrowthBlock   | block (output layers zero at birth)         | as StageBody; the identity at birth                     |
| Body          | blocks                                      | h -> h on every row (whole-range) or in its window      |
| Head          | proj (zero at birth)                        | h -> a field over every band                            |
| BandMember    | blocks, head.proj (zero at birth)           | (h, context, e) -> a field kept to one band and window  |
| ContentExpert | as BandMember                               | as BandMember, for its class set only                   |

Birth. A unit's starting numbers depend only on the run seed and the name of its random stream (by default the unit's
name), never on which other units exist: ``born`` seeds the global generator from (seed, stream), constructs the unit on
the CPU in float32 and puts the caller's generator state back.
"""
from __future__ import annotations

import zlib
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .blocks import Block, TimeEmbedding, ZeroHead, zero_linear
from .rotary import AxialBandCode
from .spectral import anchor_velocity
from .windows import window_weight

NULL_CLASS = 10                      # the class index that means "no caption"


def name_seed(seed: int, stream: str) -> int:
    """The generator seed of a named random stream: (1,000,003 (seed + 1) + crc32(stream)) mod 2^31."""
    return (1_000_003 * (seed + 1) + zlib.crc32(stream.encode())) % (2 ** 31)


def _reseeder(seed: int) -> Callable[[str], None]:
    def reseed(stream: str) -> None:
        torch.manual_seed(name_seed(seed, stream))
    return reseed


class Unit(nn.Module):
    unit_type = "Unit"
    part = None                      # the label of the unit's output in a model's parts (None: it writes no field)

    def __init__(self, name: str, address: Dict[str, Any], spec: Dict[str, Any], geometry: Dict[str, Any]):
        super().__init__()
        self.name = str(name)
        self.address = dict(address)
        self.spec = dict(spec)
        self.geometry = dict(geometry)

    def owned_state(self) -> Dict[str, Tensor]:
        return dict(self.state_dict())

    def load_owned(self, sd: Dict[str, Tensor]) -> None:
        self.load_state_dict(sd, strict=True)

    @classmethod
    def _birth(cls, seed: int, name, address, spec, geometry, stream: str, index: int, **kw) -> "Unit":
        _reseeder(seed)(stream)
        return cls(name, address, spec, geometry, **kw)

    # window helpers shared by windowed units
    @property
    def lo(self) -> float:
        return float(self.address.get("window", ["0", "1"])[0])

    @property
    def hi(self) -> float:
        return float(self.address.get("window", ["0", "1"])[1])

    @property
    def soft(self) -> float:
        return float(self.address.get("soft", "0"))


class Anchor(Unit):
    """The closed-form linear predictor from a measured spectrum. spec: {"output": bool} (whether its answer is a part
    of the prediction; a model may also use it only as an input to the stem, or only for reads)."""
    unit_type = "Anchor"
    part = "anchor"

    def __init__(self, name, address, spec, geometry, spectrum: Optional[Tensor] = None):
        super().__init__(name, address, spec, geometry)
        if spectrum is None:
            spectrum = torch.ones(tuple(geometry["field"]))
        self.register_buffer("spectrum", spectrum.float(), persistent=True)

    @property
    def output(self) -> bool:
        return bool(self.spec.get("output", True))

    @classmethod
    def _birth(cls, seed, name, address, spec, geometry, stream, index, **kw):
        return cls(name, address, spec, geometry, **kw)          # closed form: no random draws

    def forward(self, zt: Tensor, t: Tensor) -> Tensor:
        return anchor_velocity(zt, t, self.spectrum)


class Stem(Unit):
    """Token embedding (optionally widened by the anchor's tokens), time and class embeddings, and the shared blocks.
    spec: {"depth", "anchor_in" (0, 1 or 2), "code": [lo, hi]}. Random streams: "embed" for the embedding, then "body"
    for the time embedding, the class table and the blocks, in that order."""
    unit_type = "Stem"

    def __init__(self, name, address, spec, geometry, reseed: Optional[Callable[[str], None]] = None):
        super().__init__(name, address, spec, geometry)
        d, heads = geometry["dim"], geometry["heads"]
        self.anchor_in = int(spec.get("anchor_in", 0))
        if reseed is not None:
            reseed("embed")
        self.embed = nn.Linear(geometry["token_dim"] * (1 + self.anchor_in), d)
        if reseed is not None:
            reseed("body")
        self.time = TimeEmbedding(geometry["time_freq"], d, 1000.0)
        self.cls = nn.Embedding(geometry["classes"], d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(int(spec["depth"]))])
        lo, hi = spec["code"]
        self.code = AxialBandCode(d // heads, geometry["grid"], float(lo), float(hi))

    @classmethod
    def _birth(cls, seed, name, address, spec, geometry, stream, index, **kw):
        return cls(name, address, spec, geometry, reseed=_reseeder(seed), **kw)

    def rotary(self):
        return self.code.rotary()

    def forward(self, x: Tensor, t: Tensor, cls_idx: Tensor):
        """tokens [B, N, P (1 + anchor_in)] -> (h [B, N, d], ctx [B, 1, d], e [B, d], the rotary)."""
        e = self.time(t)
        ctx = self.cls(cls_idx)[:, None, :]
        h = self.embed(x)
        rot = self.code.rotary()
        for blk in self.blocks:
            h = blk(h, ctx, None, e, rot)
        return h, ctx, e, rot


class StageBody(Unit):
    """One block after the stem, on inside its window of t. spec: {"index": s}. Random stream "stage_blocks" by default,
    shared by a partition's blocks: the block of index s is the (s + 1)-th block drawn from the stream."""
    unit_type = "StageBody"
    zero_birth = False

    def __init__(self, name, address, spec, geometry):
        super().__init__(name, address, spec, geometry)
        self.block = Block(geometry["dim"], geometry["heads"])
        if self.zero_birth:
            self.block.zero_outputs()

    @classmethod
    def _birth(cls, seed, name, address, spec, geometry, stream, index, **kw):
        _reseeder(seed)(stream)
        for _ in range(index):                 # the earlier blocks of the shared stream, drawn and discarded
            Block(geometry["dim"], geometry["heads"])
        return cls(name, address, spec, geometry, **kw)

    def forward(self, h, ctx, e, rot):
        return self.block(h, ctx, None, e, rot)


class GrowthBlock(StageBody):
    """A StageBody whose three residual output layers start at zero: the identity at birth, removable exactly."""
    unit_type = "GrowthBlock"
    zero_birth = True


class Body(StageBody):
    """``depth`` blocks after the stem, run with the stem's rotary. spec: {"depth": k}.

    Whole-range (no window, or the window ["0", "1"], and soft 0): runs on every row, t = 1 included. Windowed: acts as a
    stage unit (hard: the rows with lo <= t < hi; with a half-width: a weighted mix of states). Random stream: its name
    unless the build entry gives one; the blocks are drawn in order from the stream (a position ``index`` in a shared
    stream first draws and discards ``index * depth`` blocks)."""
    unit_type = "Body"

    def __init__(self, name, address, spec, geometry):
        Unit.__init__(self, name, address, spec, geometry)
        self.blocks = nn.ModuleList([Block(geometry["dim"], geometry["heads"]) for _ in range(int(spec["depth"]))])

    @property
    def whole_range(self) -> bool:
        return self.lo <= 0.0 and self.hi >= 1.0 and self.soft == 0.0

    @classmethod
    def _birth(cls, seed, name, address, spec, geometry, stream, index, **kw):
        _reseeder(seed)(stream)
        for _ in range(index * int(spec["depth"])):
            Block(geometry["dim"], geometry["heads"])
        return cls(name, address, spec, geometry, **kw)

    def forward(self, h, ctx, e, rot):
        for blk in self.blocks:
            h = blk(h, ctx, None, e, rot)
        return h


class Head(Unit):
    """The trunk's own output: LayerNorm then a zero-born projection to token values, over every band."""
    unit_type = "Head"
    part = "body"                            # the label of its output in parts and result files

    def __init__(self, name, address, spec, geometry):
        super().__init__(name, address, spec, geometry)
        self.norm = nn.LayerNorm(geometry["dim"], elementwise_affine=False, eps=1e-6)
        self.proj = zero_linear(geometry["dim"], geometry["token_dim"])

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(self.norm(h))


class BandMember(Unit):
    """Blocks with their own position code and a zero-born head; the model keeps its output to one band of the token
    grid's frequencies (address "band": 0, 1, 2 or "all") and gates it by its window of t.
    spec: {"depth", "code": [lo, hi], "keep": bool}."""
    unit_type = "BandMember"

    def __init__(self, name, address, spec, geometry):
        super().__init__(name, address, spec, geometry)
        d, heads = geometry["dim"], geometry["heads"]
        classes = address.get("classes")
        self.classes = None if classes is None else tuple(int(c) for c in classes)
        mask = torch.ones(geometry["classes"]) if self.classes is None else torch.zeros(geometry["classes"])
        if self.classes is not None:
            mask[list(self.classes)] = 1.0
        self.register_buffer("cls_mask", mask, persistent=False)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(int(spec["depth"]))])
        self.head = ZeroHead(d, geometry["token_dim"])
        lo, hi = spec["code"]
        self.code = AxialBandCode(d // heads, geometry["grid"], float(lo), float(hi))

    @property
    def part(self) -> str:
        return self.name

    @property
    def band(self):
        return self.address["band"]

    @property
    def keep(self) -> bool:
        return bool(self.spec.get("keep", False)) and self.band != "all"

    def gate(self, t: Tensor, cls_idx: Tensor) -> Tensor:
        """[B]: the window weight times the class mask."""
        w = window_weight(t, self.lo, self.hi, self.soft)
        if self.classes is not None:
            w = w * self.cls_mask[cls_idx]
        return w

    def forward(self, h: Tensor, ctx: Tensor, e: Tensor):
        """-> (token output [B, N, P], the state after the blocks)."""
        rot = self.code.rotary()
        for blk in self.blocks:
            h = blk(h, ctx, None, e, rot)
        return self.head(h), h


class ContentExpert(BandMember):
    """A BandMember for a set of classes (address "classes"); silent for every other class and for the null class."""
    unit_type = "ContentExpert"


UNIT_TYPES = {c.unit_type: c for c in (Anchor, Stem, StageBody, GrowthBlock, Body, Head, BandMember, ContentExpert)}


def born(cls, seed: int, name: str, address: Dict[str, Any], spec: Dict[str, Any], geometry: Dict[str, Any],
         stream: Optional[str] = None, index: int = 0, **kw) -> Unit:
    """A newborn unit: seeded from (seed, stream or name), built on the CPU in float32. The caller's CPU generator state
    is restored afterwards."""
    if isinstance(cls, str):
        cls = UNIT_TYPES[cls]
    state = torch.get_rng_state()
    try:
        with torch.device("cpu"):
            unit = cls._birth(int(seed), name, address, spec, geometry, stream or name, int(index), **kw)
    finally:
        torch.set_rng_state(state)
    return unit


def make(cls, name: str, address, spec, geometry, **kw) -> Unit:
    """A unit constructed without a defined birth (for loading saved weights into)."""
    if isinstance(cls, str):
        cls = UNIT_TYPES[cls]
    state = torch.get_rng_state()
    try:
        with torch.device("cpu"):
            return cls(name, address, spec, geometry, **kw)
    finally:
        torch.set_rng_state(state)
