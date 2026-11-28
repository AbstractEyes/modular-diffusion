"""The model assembled from a build.

    h      = stem(tokens of z_t [+ the anchor's tokens], t, class), then the whole-range body (if any), then the stage
             units of its window of t, then the depth blocks (if any) one after another
    v_hat  = anchor(z_t, t [, class]) + head(h) + sum_m gate_m(t, class) x keep_band(m)[ member_m(h, t, class) ]

A per-class anchor (spec "symbols") reads the row's class; the stem's anchor input is that answer. A stem with a class
window reads "no class" outside its window; the anchor and the members' class gates keep the class they are given.

Parts. ``components`` returns each unit's own contribution as a field [B, C, H, W], in the order anchor, head, members
(build order); v_hat is ``torch.stack(parts).sum(0)``. A gated unit runs only on the samples whose gate is above zero,
so it is exactly silent elsewhere, and a unit that is absent from the build is simply not run (nothing is multiplied by
zero to switch a unit off).

Stage units. With hard windows a stage block replaces the rows of h whose t lies in [lo, hi) (t = 1 is in no stage
window). With a crossfade half-width w > 0 every stage block whose weight is above zero runs, and h becomes
sum_s w_s(t) block_s(h); for growth blocks (identity at birth) it becomes h + sum_s w_s(t) (block_s(h) - h), which is
exactly h at birth even inside an overlap. Windowed Body units are stage units of the first kind (several blocks each).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from .build import Build
from .spectral import band_masks, keep_band, patchify, unpatchify
from .units import (Anchor, BandMember, Body, DepthBlock, GrowthBlock, Head, StageBody, Stem, Unit, UNIT_TYPES, born,
                    make)
from . import frame
from .windows import window_weight  # noqa: F401  (kept importable from here, as in earlier versions)

NULL_CLASS = 10
SYMBOLS = 10                         # symbols are drawn uniformly from 0 .. SYMBOLS - 1; NULL_CLASS is "no symbol"


def symbol_gated(model) -> bool:
    """Whether a model (or a staged model) holds a symbol-gated unit (see ``units.BandMember``)."""
    return any(getattr(m, "symbol_gate", False) for m in getattr(model, "members", []))


def draw_symbols(cls_idx: Tensor, gen: torch.Generator) -> Tensor:
    """One symbol per row, uniform over 0 .. SYMBOLS - 1, from ``gen``; a row whose class is the null class (its class
    was dropped) gets the null index."""
    s = torch.randint(0, SYMBOLS, (cls_idx.shape[0],), generator=gen, device=cls_idx.device)
    return torch.where(cls_idx == NULL_CLASS, torch.full_like(s, NULL_CLASS), s)


def fixed_symbols(n: int, seed: int, device="cpu") -> Tensor:
    """n symbols drawn on the CPU from a generator seeded ``seed`` (so they do not depend on the device), moved to the
    device: the fixed symbols of the read and of the sampled pictures."""
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, SYMBOLS, (int(n),), generator=gen).to(device)


class Model(nn.Module):
    def __init__(self, build: Build, units: Union[Sequence[Unit], Dict[str, Unit]]):
        super().__init__()
        build.validate()
        if isinstance(units, dict):
            units = [units[n] for n in build.names()]
        units = list(units)
        if [u.name for u in units] != build.names():
            raise ValueError(f"units {[u.name for u in units]} do not match the build {build.names()}")
        for u, entry in zip(units, build.units):
            if u.unit_type != entry.type:
                raise ValueError(f"unit {u.name!r} is a {u.unit_type}, the build says {entry.type}")
        self.build = build
        g = build.geometry
        self.field_shape = tuple(int(x) for x in g["field"])
        self.grid = int(g["grid"])
        self.patch = int(g["patch"])
        self.token_dim = int(g["token_dim"])
        self.units = nn.ModuleList(units)
        by_type = lambda *ts: [u for u in units if isinstance(u, ts)]
        anchors, stems, heads = by_type(Anchor), by_type(Stem), by_type(Head)
        if len(stems) != 1:
            raise ValueError("a build needs exactly one Stem")
        if len(anchors) > 1 or len(heads) > 1:
            raise ValueError("a build holds at most one Anchor and one Head")
        bodies = by_type(Body)
        whole = [u for u in bodies if u.whole_range]
        windowed = [u for u in bodies if not u.whole_range]
        others = [u for u in by_type(StageBody) if not isinstance(u, Body)]
        self.depth_blocks: List[DepthBlock] = by_type(DepthBlock)
        if len(whole) > 1:
            raise ValueError("a build holds at most one whole-range Body")
        if whole and (windowed or others):
            raise ValueError("a whole-range Body cannot be combined with windowed stage units")
        if windowed and others:
            raise ValueError("windowed Body units cannot be mixed with StageBody or GrowthBlock units in one build")
        # plain lists and tuples, so the units stay registered once (under self.units) and the state dict has no aliases
        self._roles = (anchors[0] if anchors else None, stems[0], heads[0] if heads else None, whole[0] if whole else None)
        self.stage_units: List[StageBody] = [u for u in by_type(StageBody) if not (isinstance(u, Body) and u.whole_range)]
        self.members: List[BandMember] = by_type(BandMember)
        if self.stem.anchor_in and self.anchor is None:
            raise ValueError("the stem reads the anchor, and the build has none")
        softs = {u.soft for u in self.stage_units}
        if len(softs) > 1:
            raise ValueError(f"stage units disagree on their crossfade half-width: {sorted(softs)}")
        self.stage_soft = softs.pop() if softs else 0.0
        if self.stage_soft > 0 and len({type(u) for u in self.stage_units}) > 1:
            raise ValueError("a crossfaded stage holds either StageBody or GrowthBlock units, not both")
        nb = int(g.get("bands", 1))
        self.all_band = nb
        masks, _ = band_masks(self.grid, nb)
        masks = list(masks) + [torch.ones_like(masks[0])]                         # index nb = every band
        self.register_buffer("masks", torch.stack(masks).float(), persistent=False)   # [nb + 1, G, G]

    # ------------------------------------------------------------------ construction
    @classmethod
    def born(cls, build: Build, seed: int, spectrum: Optional[Tensor] = None,
             class_stats: Optional[Sequence[Tensor]] = None) -> "Model":
        """Every unit newborn from (seed, its stream), on the CPU. An anchor with symbols takes ``class_stats`` =
        (means [K, C, H, W], spectra [K, C, H, W]) (``PicturesBed.class_stats()``); it is required then."""
        units = []
        for e in build.units:
            kw = {"spectrum": spectrum} if e.type == "Anchor" else {}
            if e.type == "Anchor" and int(e.spec.get("symbols", 0) or 0) > 0:
                if class_stats is None:
                    raise ValueError(f"the anchor {e.name!r} has symbols and no class statistics were given")
                kw["class_stats"] = class_stats
            units.append(born(e.type, seed, e.name, e.address, e.spec, build.geometry, stream=e.stream, index=e.index, **kw))
        return cls(build, units)

    @classmethod
    def empty(cls, build: Build) -> "Model":
        """Every unit constructed without a defined birth, for loading saved states into."""
        return cls(build, [make(e.type, e.name, e.address, e.spec, build.geometry) for e in build.units])

    @property
    def anchor(self) -> Optional[Anchor]:
        return self._roles[0]

    @property
    def stem(self) -> Stem:
        return self._roles[1]

    @property
    def head(self) -> Optional[Head]:
        return self._roles[2]

    @property
    def body(self) -> Optional[Body]:
        """The whole-range Body (None when the build has none)."""
        return self._roles[3]

    def unit(self, name: str) -> Unit:
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(name)

    @property
    def edges(self) -> list:
        return list(self.build.info.get("stage_edges", []))

    # ------------------------------------------------------------------ shapes
    def patchify(self, z: Tensor) -> Tensor:
        return patchify(z, self.patch)

    def unpatchify(self, x: Tensor) -> Tensor:
        return unpatchify(x, self.field_shape, self.patch, self.grid)

    def keep_band(self, x: Tensor, band: int) -> Tensor:
        return keep_band(x, self.masks[band], self.grid)

    def band_index(self, m: BandMember) -> int:
        return self.all_band if m.band == "all" else int(m.band)

    # ------------------------------------------------------------------ the parts
    def anchor_velocity(self, zt: Tensor, t: Tensor, cls_idx: Optional[Tensor] = None) -> Tensor:
        """The anchor's answer; a per-class anchor reads the row's class (None: "no class" for every row), the pooled
        anchor ignores it."""
        if self.anchor is None:
            raise ValueError("the build has no Anchor")
        return self.anchor(zt, t, cls_idx)

    def gate(self, m: BandMember, t: Tensor, cls_idx: Tensor) -> Tensor:
        return m.gate(t, cls_idx)

    def _stages(self, h, t, ctx, e, rot, skip):
        live = [u for u in self.stage_units if u.name not in skip]
        if not live:
            return h
        if self.stage_soft > 0:
            growth = isinstance(live[0], GrowthBlock)
            mixed = h if growth else torch.zeros_like(h)
            for u in live:
                w = frame.stage_factor(frame.owner_of(u), t)          # the window weight of the unit's address
                idx = torch.nonzero(w > 0, as_tuple=False).squeeze(1)
                if idx.numel():
                    hi_ = h[idx]
                    y = u(hi_, ctx[idx], e[idx], rot)
                    if growth:
                        y = y - hi_
                    mixed = mixed.index_add(0, idx, w[idx].view(-1, 1, 1).to(h.dtype) * y)
            return mixed
        h_new = h.clone()
        for u in live:
            idx = torch.nonzero(frame.stage_rows(frame.owner_of(u), t), as_tuple=False).squeeze(1)
            if idx.numel():
                h_new[idx] = u(h[idx], ctx[idx], e[idx], rot)
        return h_new

    def components(self, zt: Tensor, t: Tensor, cls_idx: Tensor, only: Optional[Iterable[str]] = None,
                   scale: Optional[Dict[str, Tensor]] = None, skip: Iterable[str] = (),
                   symbols: Optional[Tensor] = None, _va: Optional[Tensor] = None):
        """-> (parts {label: field [B, C, H, W]}, h, ctx, e).

        only: member names to run (the others are left out); scale: {member name: [B] weights} (member dropout);
        skip: unit names to leave out (stage units, depth blocks or members); symbols: [B] the rows' symbols for the
        symbol-gated members (None: every symbol-gated member is off)."""
        skip = set(skip)
        out = {}
        anc = self.anchor
        use_out = anc is not None and anc.output and anc.name not in skip
        if _va is not None:
            va = _va
        elif use_out or self.stem.anchor_in:
            va = self.anchor_velocity(zt, t, cls_idx)
        else:
            va = None
        if use_out:
            out[anc.part] = va
        x = self.patchify(zt)
        if self.stem.anchor_in:                                  # the stem reads what the anchor answered
            x = torch.cat([x, self.patchify(va)], dim=-1)
            if self.stem.anchor_in > 1:
                x = torch.cat([x, self.patchify(zt + (1 - t.view(-1, 1, 1, 1)) * va)], dim=-1)
        h, ctx, e, rot = self.stem(x, t, cls_idx)
        if self.body is not None and self.body.name not in skip:
            h = self.body(h, ctx, e, rot)
        h = self._stages(h, t, ctx, e, rot, skip)
        for u in self.depth_blocks:
            if u.name not in skip:
                h = u(h, ctx, e, rot)
        if self.head is not None and self.head.name not in skip:
            out[self.head.part] = self.head_part(h)
        only = None if only is None else set(only)
        for m in self.members:
            if (only is not None and m.name not in only) or m.name in skip:
                continue
            y = self.member_part(m, h, ctx, e, t, cls_idx, None if scale is None else scale.get(m.name),
                                 symbols=symbols)
            if y is not None:
                out[m.part] = y
        return out, h, ctx, e

    def head_part(self, h: Tensor) -> Tensor:
        """The head's field from the trunk state h."""
        return self.unpatchify(self.head(h).float())

    def member_part(self, m: BandMember, h: Tensor, ctx: Tensor, e: Tensor, t: Tensor, cls_idx: Tensor,
                    scale: Optional[Tensor] = None, symbols: Optional[Tensor] = None) -> Optional[Tensor]:
        """One member's field [B, C, H, W] from the trunk state (None when its gate is zero for every sample): the
        member runs on the gated rows only, its output is weighted by the gate, kept to its band, and zero elsewhere."""
        if (scale is None and m.classes is None and not m.symbol_gate
                and m.lo <= 0.0 and m.hi >= 1.0 and m.soft == 0.0):
            y, _ = m(h, ctx, e)
            y = y.float()
            if m.keep:
                y = self.keep_band(y, self.band_index(m))
            return self.unpatchify(y)
        g = m.gate(t, cls_idx, symbols) if m.symbol_gate else m.gate(t, cls_idx)
        if scale is not None:
            g = g * scale
        idx = torch.nonzero(g > 0, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return None
        y, _ = m(h[idx], ctx[idx], e[idx])
        y = y.float() * g[idx].view(-1, 1, 1)
        if m.keep:
            y = self.keep_band(y, self.band_index(m))
        full = y.new_zeros(h.shape[0], y.shape[1], y.shape[2])
        full[idx] = y
        return self.unpatchify(full)

    def velocity(self, zt: Tensor, t: Tensor, cls_idx: Tensor, symbols: Optional[Tensor] = None) -> Tensor:
        parts, _, _, _ = self.components(zt, t, cls_idx, symbols=symbols)
        if not parts:
            return torch.zeros_like(zt)
        return torch.stack(list(parts.values())).sum(dim=0)

    def loss(self, z1: Tensor, cls_idx: Tensor, gen: torch.Generator, cond_drop: float, mdrop: float = 0.0,
             t_sampling: str = "uniform", t_range: Sequence[float] = (0.0, 1.0), t_mix: float = 0.0) -> Tensor:
        """The flow-matching loss of one batch. Random draws from ``gen`` in this order: noise, t, the class drop (only
        when cond_drop > 0), then one draw per member (only for member dropout in training), then, only when the model
        holds a symbol-gated member, one symbol per row (``draw_symbols``). t_range, t_mix: see ``draw``."""
        b = z1.shape[0]
        zt, t, cls_idx, target = self.draw(z1, cls_idx, gen, cond_drop, t_sampling, t_range, t_mix)
        scale = None
        if self.training and mdrop > 0 and len(self.members):
            scale = {m.name: (torch.rand(b, generator=gen, device=z1.device) >= mdrop).to(z1.dtype) for m in self.members}
        symbols = draw_symbols(cls_idx, gen) if symbol_gated(self) else None
        parts, _, _, _ = self.components(zt, t, cls_idx, scale=scale, symbols=symbols)
        v = torch.stack(list(parts.values())).sum(dim=0) if parts else torch.zeros_like(z1)
        return ((v - target) ** 2).mean()

    @staticmethod
    def draw(z1: Tensor, cls_idx: Tensor, gen: torch.Generator, cond_drop: float, t_sampling: str = "uniform",
             t_range: Sequence[float] = (0.0, 1.0), t_mix: float = 0.0):
        """The noisy input of one batch -> (z_t, t, class after the drop, target). Draws from ``gen``: noise, t, then the
        class drop (only when cond_drop > 0).

        t_range (lo, hi): uniform t is drawn as before and mapped to lo + (hi - lo) t when the range is not exactly
        (0, 1), so the draws and their order do not depend on the range. logit_normal takes the whole range only.
        t_mix p in [0, 1): with p > 0 and a range other than (0, 1), two more draws right after t: m = rand < p, then a
        whole-range t' = rand, and t = t' where m; otherwise nothing more is drawn."""
        if t_sampling not in ("uniform", "logit_normal"):
            raise ValueError("t_sampling must be uniform or logit_normal")
        lo, hi = (float(x) for x in t_range)
        whole = (lo, hi) == (0.0, 1.0)
        p = float(t_mix)
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"t_range must satisfy 0 <= lo < hi <= 1: {(lo, hi)}")
        if not 0.0 <= p < 1.0:
            raise ValueError(f"t_mix must lie in [0, 1): {p}")
        if t_sampling == "logit_normal" and not whole:
            raise ValueError("t_sampling logit_normal takes the whole range (0, 1) only")
        if t_sampling == "logit_normal" and p > 0:
            raise ValueError("t_mix applies to uniform t_sampling only")
        b = z1.shape[0]
        z0 = torch.randn(z1.shape, generator=gen, device=z1.device)
        if t_sampling == "logit_normal":
            t = torch.sigmoid(torch.randn(b, generator=gen, device=z1.device))
        else:
            t = torch.rand(b, generator=gen, device=z1.device)
            if not whole:
                t = lo + (hi - lo) * t
                if p > 0:
                    m = torch.rand(b, generator=gen, device=z1.device) < p
                    t_whole = torch.rand(b, generator=gen, device=z1.device)
                    t = torch.where(m, t_whole, t)
        if cond_drop > 0:
            drop = torch.rand(b, generator=gen, device=z1.device) < cond_drop
            cls_idx = torch.where(drop, torch.full_like(cls_idx, NULL_CLASS), cls_idx)
        tt = t.view(b, 1, 1, 1)
        return (1 - tt) * z0 + tt * z1, t, cls_idx, z1 - z0

    def trained_numbers(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
