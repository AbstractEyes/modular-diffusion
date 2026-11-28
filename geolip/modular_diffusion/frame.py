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


# ------------------------------------------------------------------ proposed forms (tested; used by no model yet)
def overlap_weights(u: Tensor) -> Tuple[Tensor, Tensor]:
    """The overlap form over owners' arguments u >= 0 (the last axis is the owners) -> (weights, base), float64:

        w_i  = sinh(u_i) / (sum_j sinh(u_j) + exp(-sum_j u_j))
        base = exp(-sum_j u_j) / (the same denominator)

    The weights and the base weight sum to one. An owner at rest (u_i = 0) weighs exactly 0.0. With at most one owner
    above zero the denominator is cosh(u) and the form equals the discounted form sinh(u_i) / (1 + sum_j (cosh(u_j) - 1));
    with several owners above zero they share the mass instead of each taking up to one."""
    u = u.double()
    if bool((u < 0).any()):
        raise ValueError("the overlap form takes non-negative arguments")
    s = torch.sinh(u)
    rest = torch.exp(-u.sum(dim=-1, keepdim=True))
    den = s.sum(dim=-1, keepdim=True) + rest
    return s / den, (rest / den).squeeze(-1)


def request_arguments(requests: Sequence[Sequence[int]], owners: Sequence[Sequence[int]], U: float = 5.0) -> Tensor:
    """[n]: the argument of each owner's gate for a set of requests, float64. An owner's address and every request are
    tuples of digits of one length L. The argument is U times the SUM over requests of the PRODUCT over digits of the
    exact digit match, so each request fires on the one address that matches it on every digit: k requests fire on at
    most k addresses (never on the k^L addresses that a product over digits of per-digit sums would light)."""
    req = torch.as_tensor([tuple(int(d) for d in r) for r in requests], dtype=torch.long)
    own = torch.as_tensor([tuple(int(d) for d in o) for o in owners], dtype=torch.long)
    if req.dim() != 2 or own.dim() != 2 or req.shape[1] != own.shape[1]:
        raise ValueError("requests and owners are lists of digit tuples of one length")
    match = (req[:, None, :] == own[None, :, :]).double()      # [k, n, L]: the exact match of each digit
    return float(U) * match.prod(dim=2).sum(dim=0)


def request_weights(requests: Sequence[Sequence[int]], owners: Sequence[Sequence[int]],
                    U: float = 5.0) -> Tuple[Tensor, Tensor]:
    """The overlap form (``overlap_weights``) over the owners' arguments for a set of requests
    (``request_arguments``) -> (weights [n], base)."""
    return overlap_weights(request_arguments(requests, owners, U))


# ------------------------------------------------------------------ the address frame on image tokens
# A picture of a 3 x 3 grid of cells is cut into a 12 x 12 grid of tokens (patch 4; a cell is 4 x 4 tokens), row-major
# like ``spectral.patchify``. Every token carries four PLACE digits: its column j = ix div 4 and row k = iy div 4 (base
# 3) and its place inside the cell, ix mod 4 and iy mod 4 (base 5, four values used). A REQUEST asks for one thing in
# one cell: six digits of base 3, (column, row, family, member, color high, color low), the thing being 3 family +
# member and the color 3 high + low; at most one request a cell.
#
# A digit of base b = 2P + 1 sits on one of b orthonormal anchors (``digit_code``):
#
#     u_b(a) = [1/sqrt(b), sqrt(2/b) cos(2 pi p a / b), sqrt(2/b) sin(2 pi p a / b) for p = 1..P],   a = 0..b-1
#
# THE GATE (declared; nothing is compared, nothing learned). Part i owns a set T_i of things (and every band); the base
# (index n, the last) owns "no thing":
#
#     w_i(token)    = sum over requests r of <u_3(j), u_3(j_r)> <u_3(k), u_3(k_r)> [thing_r in T_i]
#     w_base(token) = 1 - sum over requests r of <u_3(j), u_3(j_r)> <u_3(k), u_3(k_r)>
#
# one product a request, summed over requests (never a product of per-digit sums, which would light every cell whose
# column is some request's column and whose row is some request's row). Each inner product is 1 or 0, so every token has
# exactly one owner at weight 1: the part owning the thing of its cell's request, or the base when its cell has none.
# ``token_weights`` computes the gate by integers or by the code kernels in float64 (the thing bracket then also by its
# kernel, sum over the owned things tau of <u_3(f_r), u_3(f_tau)> <u_3(m_r), u_3(m_tau)>); the two agree to rounding.
TOKEN_GRID = 12
GRID_CELLS = 3
REQUEST_DIGITS = 6


_DIGIT_CODE_CACHE: dict = {}
_DIGIT_CODE_DEV_CACHE: dict = {}
_ARANGE9_CACHE: dict = {}


def _digit_code_cached(b: int, device=None, dtype: torch.dtype = torch.float64) -> Tensor:
    base = _DIGIT_CODE_CACHE.get(b)
    if base is None:
        if b < 1 or b % 2 == 0:
            raise ValueError(f"a digit's base is odd (2P + 1), not {b}")
        p_count = (b - 1) // 2
        a = torch.arange(b, dtype=torch.float64)
        cols = [torch.full((b,), 1.0 / math.sqrt(b), dtype=torch.float64)]
        for p in range(1, p_count + 1):
            ang = 2.0 * math.pi * p * a / b
            cols.append(math.sqrt(2.0 / b) * torch.cos(ang))
            cols.append(math.sqrt(2.0 / b) * torch.sin(ang))
        base = torch.stack(cols, dim=1)
        _DIGIT_CODE_CACHE[b] = base
    dev = torch.device("cpu" if device is None else device)
    if dev.type == "cpu" and dtype == torch.float64:
        return base
    key = (b, dev, dtype)
    cached = _DIGIT_CODE_DEV_CACHE.get(key)
    if cached is None:
        cached = base.to(device=dev, dtype=dtype)
        _DIGIT_CODE_DEV_CACHE[key] = cached
    return cached


def _arange_n_on(n: int, device: torch.device) -> Tensor:
    if n == 9:
        cached = _ARANGE9_CACHE.get(device)
        if cached is None:
            cached = torch.arange(9, device=device)
            _ARANGE9_CACHE[device] = cached
        return cached
    return torch.arange(n, device=device)


def digit_code(base: int) -> Tensor:
    """[base, base] float64: row a is the code u_b(a) of anchor a (the plain dimension first, then the P rotary pairs,
    cos before sin). The rows are orthonormal. An even base raises."""
    return _digit_code_cached(int(base)).clone()


def _odd_at_least(n: int) -> int:
    return n if n % 2 == 1 else n + 1


class TokenFrame:
    """The place digits of every token of a grid x grid token picture cut into cells x cells cells (row-major token
    order: token = iy * grid + ix). Attributes, each [T] long: ``column``, ``row`` (0..cells-1), ``inner_x``,
    ``inner_y`` (0..side-1, side = grid / cells) and ``cell`` (= cells * row + column)."""

    def __init__(self, grid: int = TOKEN_GRID, cells: int = GRID_CELLS):
        grid, cells = int(grid), int(cells)
        if grid % cells:
            raise ValueError("the token grid must split into whole cells")
        if cells % 2 == 0:
            raise ValueError("the cell digit's base is odd")
        self.grid, self.cells = grid, cells
        self.side = grid // cells
        self.inner_base = _odd_at_least(self.side)
        self.n_tokens = grid * grid
        idx = torch.arange(self.n_tokens)
        ix, iy = idx % grid, idx // grid
        self.column = ix // self.side
        self.row = iy // self.side
        self.inner_x = ix % self.side
        self.inner_y = iy % self.side
        self.cell = self.row * cells + self.column
        self._dev_cache: dict = {}
        self._cell_place_cache: dict = {}

    def _on(self, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
        cached = self._dev_cache.get(device)
        if cached is None:
            cached = (self.cell.to(device), self.column.to(device), self.row.to(device))
            self._dev_cache[device] = cached
        return cached

    def _cell_on(self, device: torch.device) -> Tensor:
        return self._on(device)[0]

    def _cell_place_on(self, device: torch.device) -> Tensor:
        cached = self._cell_place_cache.get(device)
        if cached is None:
            c = torch.arange(self.cells * self.cells, device=device)
            _, col, row = self._on(device)
            cached = ((col[None] == (c % self.cells)[:, None]) & (row[None] == (c // self.cells)[:, None])).long()
            self._cell_place_cache[device] = cached
        return cached

    def place_code(self) -> Tensor:
        """[T, 2 cells + 2 inner_base] float32 (16 for the default frame): [u_3(column), u_3(row), u_5(inner x),
        u_5(inner y)]."""
        uc = _digit_code_cached(self.cells)
        ui = _digit_code_cached(self.inner_base)
        return torch.cat([uc[self.column], uc[self.row], ui[self.inner_x], ui[self.inner_y]], dim=1).float()

    def cell_place(self) -> Tensor:
        """[n_cells, T] long: 1 where the token lies in the cell, by integers (column and row equal)."""
        return self._cell_place_on(torch.device("cpu")).clone()


_DEFAULT_FRAME: Optional[TokenFrame] = None


def _frame(frame: Optional[TokenFrame]) -> TokenFrame:
    global _DEFAULT_FRAME
    if frame is not None:
        return frame
    if _DEFAULT_FRAME is None:
        _DEFAULT_FRAME = TokenFrame()
    return _DEFAULT_FRAME


def request_of(cell: int, thing: int, color: int, cells: int = GRID_CELLS) -> Tuple[int, ...]:
    """The six digits (column, row, family, member, color high, color low) of a request for ``thing`` in ``color`` at
    ``cell`` (= cells * row + column)."""
    cell, thing, color = int(cell), int(thing), int(color)
    if not 0 <= cell < cells * cells or not 0 <= thing < 9 or not 0 <= color < 9:
        raise ValueError(f"a request is a cell 0..{cells * cells - 1}, a thing 0..8 and a color 0..8")
    return (cell % cells, cell // cells, thing // 3, thing % 3, color // 3, color % 3)


def request_cell(request: Sequence[int], cells: int = GRID_CELLS) -> int:
    return int(request[1]) * cells + int(request[0])


def request_thing(request: Sequence[int]) -> int:
    return 3 * int(request[2]) + int(request[3])


def request_color(request: Sequence[int]) -> int:
    return 3 * int(request[4]) + int(request[5])


def layout_requests(cells_row: Tensor, colors_row: Tensor) -> List[Tuple[int, ...]]:
    """One layout row (cells [9], colors [9]; -1 for none) -> its list of requests, in cell order."""
    cr = [int(v) for v in torch.as_tensor(cells_row).reshape(-1).tolist()]
    co = [int(v) for v in torch.as_tensor(colors_row).reshape(-1).tolist()]
    if len(cr) != GRID_CELLS * GRID_CELLS or len(co) != len(cr):
        raise ValueError("a layout row is cells [9] and colors [9]")
    out = []
    for c, (t, k) in enumerate(zip(cr, co)):
        if (t < 0) != (k < 0):
            raise ValueError(f"cell {c} holds a thing without a color or a color without a thing")
        if t >= 0:
            out.append(request_of(c, t, k))
    return out


def requests_layout(requests: Sequence[Sequence[int]]) -> Tuple[Tensor, Tensor]:
    """A list of requests -> one layout row (cells [9], colors [9] long). Two requests in one cell raise (a layout
    cannot hold them)."""
    cells = torch.full((GRID_CELLS * GRID_CELLS,), -1, dtype=torch.long)
    colors = torch.full_like(cells, -1)
    for r in requests:
        if len(r) != REQUEST_DIGITS or any(not 0 <= int(d) < 3 for d in r):
            raise ValueError(f"a request is six digits of base 3, not {tuple(r)}")
        c = request_cell(r)
        if int(cells[c]) >= 0:
            raise ValueError(f"two requests in cell {c}: places partition the picture")
        cells[c] = request_thing(r)
        colors[c] = request_color(r)
    return cells, colors


def request_code(request: Sequence[int]) -> Tensor:
    """[18] float32: the codes u_3 of the request's six digits, joined in digit order."""
    if len(request) != REQUEST_DIGITS:
        raise ValueError("a request has six digits")
    u = _digit_code_cached(3)
    return torch.cat([u[int(d)] for d in request]).float()


def content_code(thing: int, color: int) -> Tensor:
    """[12] float32: the codes of the last four digits of a request (family, member, color high, color low)."""
    return request_code(request_of(0, thing, color))[6:]


def layout_codes(cells: Tensor, colors: Tensor) -> Tuple[Tensor, Tensor]:
    """A batch of layouts -> (codes [B, 9, 18] float32: the request code of each cell's request, zeros where a cell has
    none; mask [B, 9] bool: the cells with a request)."""
    cells = torch.as_tensor(cells, dtype=torch.long)
    colors = torch.as_tensor(colors, dtype=torch.long)
    u = _digit_code_cached(3, cells.device, torch.float32)
    mask = cells >= 0
    t, k = cells.clamp_min(0), colors.clamp_min(0)
    c = _arange_n_on(GRID_CELLS * GRID_CELLS, cells.device).expand_as(cells)
    digits = [c % GRID_CELLS, c // GRID_CELLS, t // 3, t % 3, k // 3, k % 3]
    codes = torch.cat([u[d] for d in digits], dim=-1)
    return codes * mask[..., None].float(), mask


def _owner_table(owners: Sequence[Sequence[int]]) -> List[int]:
    """The part index owning each thing 0..8 (-1 for none). A thing owned by two parts raises."""
    table = [-1] * 9
    for i, things in enumerate(owners):
        for t in things:
            t = int(t)
            if not 0 <= t < 9:
                raise ValueError(f"a part owns things 0..8, not {t}")
            if table[t] >= 0:
                raise ValueError(f"thing {t} is owned by parts {table[t]} and {i}: every token has one owner")
            table[t] = i
    return table


def _thing_name(t: int) -> str:
    from .scenes import THINGS
    return THINGS[t] if 0 <= t < len(THINGS) else str(t)


def _check_owned(cells: Tensor, table: List[int]) -> None:
    if all(x >= 0 for x in table):
        return
    asked = sorted({int(v) for v in cells[cells >= 0].reshape(-1).tolist()})
    for t in asked:
        if table[t] < 0:
            raise ValueError(f"a request asks for the {_thing_name(t)} (thing {t}) and no part owns it")


def token_owner(cells: Tensor, owners: Sequence[Sequence[int]], frame: Optional[TokenFrame] = None) -> Tensor:
    """[B, T] long: the one owner of every token, by integers: the part owning the thing of the token's cell's request,
    or n = len(owners) (the base) when the cell has no request. A requested thing no part owns raises, naming it."""
    fr = _frame(frame)
    cells = torch.as_tensor(cells, dtype=torch.long)
    table = _owner_table(owners)
    _check_owned(cells, table)
    tbl = torch.tensor(table + [len(owners)], dtype=torch.long, device=cells.device)
    cell_owner = torch.where(cells >= 0, tbl[cells.clamp_min(0)], torch.full_like(cells, len(owners)))
    return cell_owner[:, fr._cell_on(cells.device)]


def token_weights(cells: Tensor, owners: Sequence[Sequence[int]], method: str = "integer",
                  frame: Optional[TokenFrame] = None) -> Tensor:
    """[B, T, n + 1] float64: the gate of every part (column n = the base) at every token. ``method`` "integer": the
    brackets as integer equalities; "kernel": every bracket as the inner product of two digit codes in float64. One
    product a request, summed over the requests in cell order."""
    if method not in ("integer", "kernel"):
        raise ValueError(f"method is integer or kernel, not {method!r}")
    fr = _frame(frame)
    cells = torch.as_tensor(cells, dtype=torch.long)
    table = _owner_table(owners)
    _check_owned(cells, table)
    n = len(owners)
    dev = cells.device
    n_cells = fr.cells * fr.cells
    req = cells >= 0                                                     # [B, 9]
    if method == "integer":
        place = fr._cell_place_on(dev)                                   # [9, T] long
        tbl = torch.tensor(table, dtype=torch.long, device=dev)
        member = (tbl[cells.clamp_min(0)][..., None] == torch.arange(n, device=dev)) & req[..., None]   # [B, 9, n]
        member = member.long()
        reql = req.long()
        w = torch.zeros(cells.shape[0], fr.n_tokens, n, dtype=torch.long, device=dev)
        hit = torch.zeros(cells.shape[0], fr.n_tokens, dtype=torch.long, device=dev)
        for c in range(n_cells):                                         # one product a request, summed
            w = w + member[:, c, None, :] * place[c][None, :, None]
            hit = hit + reql[:, c, None] * place[c][None, :]
        return torch.cat([w, (1 - hit)[..., None]], dim=-1).double()
    u = _digit_code_cached(fr.cells, dev, torch.float64)
    gram = u @ u.T                                                       # [3, 3]: <u(a), u(b)>
    c = _arange_n_on(n_cells, dev)
    col_c, row_c = c % fr.cells, c // fr.cells
    _, fcol, frow = fr._on(dev)
    kp = gram[fcol][:, col_c].T * gram[frow][:, row_c].T                 # [9, T]
    u3 = _digit_code_cached(3, dev, torch.float64)
    g3 = u3 @ u3.T
    things = _arange_n_on(9, dev)
    fam, mem = things // 3, things % 3
    kt = g3[fam][:, fam] * g3[mem][:, mem]                               # [9, 9]: the thing kernel
    own = torch.zeros(9, n, dtype=torch.float64, device=dev)
    for i, ts in enumerate(owners):
        own[[int(t) for t in ts], i] = 1.0
    mk = kt @ own                                                        # [9, n]: sum over owned things of the kernel
    reqd = req.double()
    mrow = mk[cells.clamp_min(0)] * reqd[..., None]                      # [B, 9, n]
    w = torch.zeros(cells.shape[0], fr.n_tokens, n, dtype=torch.float64, device=dev)
    hit = torch.zeros(cells.shape[0], fr.n_tokens, dtype=torch.float64, device=dev)
    for ci in range(n_cells):
        w = w + mrow[:, ci, None, :] * kp[ci][None, :, None]
        hit = hit + reqd[:, ci, None] * kp[ci][None, :]
    return torch.cat([w, (1.0 - hit)[..., None]], dim=-1)


def visible_keys(cells: Tensor, serving: Tensor, hidden: Optional[Tensor] = None,
                 frame: Optional[TokenFrame] = None, *, _validate: bool = True) -> Tensor:
    """[B, T] bool: the tokens a part may read. ``serving`` [B] long: the cell whose request a thing part serves (that
    cell and every cell without a request are visible), or -1 for the base (every cell without a request). ``hidden``
    [B, 9] bool marks cells hidden in training: they count as owned by another request (invisible). The served cell
    must hold a request and must not be hidden."""
    fr = _frame(frame)
    cells = torch.as_tensor(cells, dtype=torch.long)
    dev = cells.device
    serving = torch.as_tensor(serving, dtype=torch.long, device=dev).reshape(-1)
    if serving.shape[0] != cells.shape[0]:
        raise ValueError("serving holds one cell (or -1) a row")
    free = cells < 0
    if hidden is not None:
        hidden = torch.as_tensor(hidden, dtype=torch.bool, device=dev)
        free = free & ~hidden
    if _validate:
        has = serving >= 0
        if bool(has.any()):
            sc = serving[has]
            if bool((sc >= cells.shape[1]).any()):
                raise ValueError("a served cell is 0..8 or -1")
            rows = torch.nonzero(has, as_tuple=True)[0]
            if bool((cells[rows, sc] < 0).any()):
                raise ValueError("a thing part serves a cell that holds a request; this one holds none")
            if hidden is not None and bool(hidden[rows, sc].any()):
                raise ValueError("the served cell cannot be hidden")
    idx = _arange_n_on(cells.shape[1], dev)
    vis = free | (serving[:, None] == idx[None, :])
    return vis[:, fr._cell_on(dev)]
