"""The address frame: the gate every part already uses, written once as one projective gate over declared addresses.

An address has digits. The STAGE digit of a row at noise level t is a unit vector on the stage anchors e_0 .. e_{K-1}:
on e_k inside stage k; across the crossfade around the cut c between stage k and stage k + 1 (half-width w) it turns
from e_k to e_{k+1} at constant angular speed,

    a(t) = cos(phi) e_k + sin(phi) e_{k+1},      phi = (pi / 2) (t - (c - w)) / (2 w)

A DISCRETE digit (the class, a random label drawn per row, a band) sits on one anchor, or on none (the null class, no
label).

A part OWNS a set of anchors per digit it is addressed by. Its factor for a digit is the squared length of the row's
digit projected on its anchors,

    g = sum over the owned anchors e of <digit, e>^2

and its gate is the product of its digits' factors (``projective``). For a discrete digit g is set membership: 1.0 or
0.0, exactly. For the stage digit of a part owning the window [lo, hi) g is the window weight
``windows.window_weight(t, lo, hi, soft)``: across the crossfade at an edge of the owned set g = sin^2(phi) (rising) or
cos^2(phi) (falling), and ``windows.ramp`` computes sin^2(phi) in its double-angle form 0.5 - 0.5 cos(2 phi) (its u is
2 phi / pi); between two owned stages cos^2 + sin^2 = 1. So the gates the library has always used ARE this gate; this
module computes the stage factor by calling ``window_weight`` and the class factor by the part's class mask, with the
expressions and their order of the units themselves, so no stored number moves. ``stage_digit`` and ``digit_factor``
compute the frame's own form, for the tests that prove the identity.

    owners = [owner_of(u) for u in model.members]      the address every unit already carries (nothing is changed)
    projective(owners, t, cls_idx, symbols)            [B, n]: every part's gate for a batch of rows
    aleph(owners, t, cls_idx, symbols, U=5.0, form="plain" | "discounted")
                                                       [B, n]: the amplitude forms over the same addresses
    own_weight(U, n_parts, form)                       a part's weight on its address, the others silent
    cover([(lo, hi), ...]) -> (holes, overlaps)        how a set of stage windows covers [0, 1)
    check_cover(model_or_staged)                       the same for a model's stage owners

The amplitude forms (tested and tabulated only; no part of a model uses them): with u_i = U x prod over digits of
sqrt(g) (here U sqrt(gate), the gate being the product of the digits' g),

    plain:       w_i = sinh(u_i) / sum_j cosh(u_j)
    discounted:  w_i = sinh(u_i) / (1 + sum_j (cosh(u_j) - 1))

Both are exactly 0.0 off a part's address (sinh(0) = 0). A silent part adds cosh(0) - 1 = 0 to the discounted
denominator and 1 to the plain one, so removing silent parts leaves the discounted weights of the others bit-identical
(the sums run one part at a time, in order) and changes the plain ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .windows import window_weight

CLASSES = 11                          # the class digit's anchors: the ten classes and the null class (index 10)


@dataclass
class Owner:
    """The anchors a part owns: the stage window [lo, hi) with its crossfade half-width, and optionally a class list
    (or, with ``symbol``, a list of random labels). ``mask`` is the class digit's membership vector (1.0 on the owned
    classes, 0.0 elsewhere); a unit's own buffer is used when the owner is read from a unit."""
    lo: float = 0.0
    hi: float = 1.0
    soft: float = 0.0
    classes: Optional[Tuple[int, ...]] = None
    symbol: bool = False
    mask: Optional[Tensor] = None
    name: str = ""

    def class_mask(self, device=None) -> Tensor:
        if self.mask is not None:
            return self.mask if device is None else self.mask.to(device)
        m = torch.zeros(CLASSES, device=device)
        m[list(self.classes)] = 1.0
        return m


def owner(lo: float = 0.0, hi: float = 1.0, soft: float = 0.0, classes: Optional[Sequence[int]] = None,
          symbol: bool = False, name: str = "") -> Owner:
    """An owner given by its window, half-width and class (or label) list."""
    cls = None if classes is None else tuple(int(c) for c in classes)
    if symbol and cls is None:
        raise ValueError("a label-addressed owner needs its list of labels")
    return Owner(float(lo), float(hi), float(soft), cls, bool(symbol), None, str(name))


def owner_of(unit) -> Owner:
    """The address a unit already carries, as an owner: its window and half-width (``Unit.lo``, ``hi``, ``soft``) and,
    for a band member or content expert, its class list (its symbol list for a symbol-gated unit) with its class mask.
    The unit is not changed."""
    classes = getattr(unit, "classes", None)
    return Owner(unit.lo, unit.hi, unit.soft, classes, bool(getattr(unit, "symbol_gate", False)),
                 getattr(unit, "cls_mask", None) if classes is not None else None, unit.name)


def stage_factor(o: Owner, t: Tensor) -> Tensor:
    """[B]: the stage digit's factor of an owner, the window weight of [lo, hi) (computed by ``window_weight``)."""
    return window_weight(t, o.lo, o.hi, o.soft)


def stage_rows(o: Owner, t: Tensor) -> Tensor:
    """[B] bool: the rows a HARD stage unit replaces, lo <= t < hi. A hard stage of a model is not its window weight at
    t = 1 (the weight counts t = 1 into a window ending at 1; a stage unit never runs there)."""
    return (t >= o.lo) & (t < o.hi)


def factor(o: Owner, t: Tensor, cls_idx: Optional[Tensor] = None, symbols: Optional[Tensor] = None) -> Tensor:
    """[B]: one part's gate, the product of its digits' factors: the stage factor, times the class factor (the class
    mask at the row's class; at the row's symbol for a label-addressed owner, which is off for every row when no symbols
    are given). The expressions and their order are those of ``units.BandMember.gate``."""
    w = stage_factor(o, t)
    if o.classes is not None:
        if o.symbol:
            if symbols is None:
                return torch.zeros_like(w)
            w = w * o.class_mask(w.device)[symbols]
        else:
            w = w * o.class_mask(w.device)[cls_idx]
    return w


def projective(owners: Sequence[Owner], t: Tensor, cls_idx: Optional[Tensor] = None,
               symbols: Optional[Tensor] = None) -> Tensor:
    """[B, n]: the gate of every owner for a batch of rows (column i is ``factor(owners[i], ...)``)."""
    return torch.stack([factor(o, t, cls_idx, symbols) for o in owners], dim=1)


def stage_weights(owners: Sequence[Owner], t: Tensor) -> Tensor:
    """[S, B]: the stage factors of stage owners (the layout of ``Staged.weights``)."""
    return torch.stack([stage_factor(o, t) for o in owners])


# ------------------------------------------------------------------ the frame's own form (for the identity)
def stage_digit(t: Tensor, cuts: Sequence[float], soft: float) -> Tensor:
    """[B, K]: the stage digit a(t) of every row on the K = len(cuts) - 1 stage anchors (one-hot inside a stage,
    cos(phi) / sin(phi) on the two neighbors across a crossfade). The crossfades must not overlap (soft at most half
    the narrowest stage)."""
    cuts = [float(c) for c in cuts]
    k = len(cuts) - 1
    w = float(soft)
    if w > 0 and any(b - a < 2 * w - 1e-12 for a, b in zip(cuts, cuts[1:])):
        raise ValueError("the crossfades overlap: soft exceeds half the narrowest stage")
    td = t.double()
    out = torch.zeros(t.shape[0], k, dtype=torch.float64, device=t.device)
    idx = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
    for s, c in enumerate(cuts[1:-1]):
        idx = torch.where(td >= c, torch.full_like(idx, s + 1), idx)
    out[torch.arange(t.shape[0], device=t.device), idx] = 1.0
    if w > 0:
        for s, c in enumerate(cuts[1:-1]):
            inside = (td > c - w) & (td < c + w)
            if bool(inside.any()):
                phi = (math.pi / 2) * (td[inside] - (c - w)) / (2 * w)
                rows = torch.nonzero(inside, as_tuple=False).squeeze(1)
                out[rows] = 0.0
                out[rows, s] = torch.cos(phi)
                out[rows, s + 1] = torch.sin(phi)
    return out


def digit_factor(digit: Tensor, owned: Sequence[int]) -> Tensor:
    """[B]: the squared length of each row's digit projected on the owned anchors (one-hot anchors)."""
    return (digit[:, list(owned)] ** 2).sum(dim=1)


# ------------------------------------------------------------------ the amplitude forms
FORMS = ("plain", "discounted")


def aleph(owners: Sequence[Owner], t: Tensor, cls_idx: Optional[Tensor] = None, symbols: Optional[Tensor] = None,
          U: float = 5.0, form: str = "plain") -> Tensor:
    """[B, n]: the amplitude form over the owners' addresses, u_i = U sqrt(gate_i) (the product of the digits' sqrt(g)),
    plain sinh(u_i) / sum_j cosh(u_j) or discounted sinh(u_i) / (1 + sum_j (cosh(u_j) - 1)); float64. Each part's
    column is computed alone and the sums run one part at a time in the owners' order, so a silent part changes no
    digit of the discounted weights of the others."""
    if form not in FORMS:
        raise ValueError(f"form is plain or discounted, not {form!r}")
    # every operation runs on one part's column at a time, so a part's numbers never depend on the roster's width
    us = [float(U) * torch.sqrt(factor(o, t, cls_idx, symbols).double()) for o in owners]
    b = t.shape[0]
    if form == "plain":
        den = torch.zeros(b, dtype=torch.float64, device=t.device)
        for u in us:
            den = den + torch.cosh(u)
    else:
        den = torch.ones(b, dtype=torch.float64, device=t.device)
        for u in us:
            den = den + (torch.cosh(u) - 1.0)
    return torch.stack([torch.sinh(u) / den for u in us], dim=1)


def own_weight(U: float, n_parts: int, form: str = "plain") -> float:
    """A part's weight on its own address (u = U) when the other n_parts - 1 parts are silent (u = 0): plain
    sinh(U) / (cosh(U) + n_parts - 1), discounted tanh(U) at any count."""
    if form not in FORMS:
        raise ValueError(f"form is plain or discounted, not {form!r}")
    if int(n_parts) < 1:
        raise ValueError("a roster holds at least one part")
    U = float(U)
    if form == "plain":
        return math.sinh(U) / (math.cosh(U) + (int(n_parts) - 1))
    return math.sinh(U) / (1.0 + (math.cosh(U) - 1.0))


# ------------------------------------------------------------------ coverage of the stage digit
def _merge(spans: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in spans:
        if out and abs(out[-1][1] - a) < 1e-12:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def cover(windows: Sequence[Sequence[float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """How stage windows [lo, hi) cover [0, 1) -> (holes, overlaps): the stretches covered by no window and those
    covered by more than one (merged, in order). Two empty lists when the windows partition [0, 1)."""
    wins = [(float(lo), float(hi)) for lo, hi in windows]
    edges = sorted({0.0, 1.0} | {min(max(x, 0.0), 1.0) for w in wins for x in w})
    holes, overlaps = [], []
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        n = sum(1 for lo, hi in wins if lo <= mid < hi)
        if n == 0:
            holes.append((a, b))
        elif n > 1:
            overlaps.append((a, b))
    return _merge(holes), _merge(overlaps)


def stage_owners(model) -> List[Owner]:
    """The stage owners of a model: a staged model's stages (its cuts and half-width), or a model's stage units (none
    for a model whose trunk runs on every row)."""
    if hasattr(model, "clusters"):
        return [owner(lo, hi, model.soft, name=f"stage_{s}") for s, (lo, hi) in enumerate(model.windows())]
    return [owner_of(u) for u in model.stage_units]


def check_cover(model) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """``cover`` of a model's (or a staged model's) stage owners. A model without stage units covers the range with
    its trunk: two empty lists. Nothing calls this by default."""
    owners = stage_owners(model)
    if not owners:
        return [], []
    return cover([(o.lo, o.hi) for o in owners])
