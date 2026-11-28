"""Members of an addressed collective on the multi-rotary frame: engines that receive the whole written layout as request
tokens and find what is addressed to them by address pairs inside attention, their training rows, request lists that
point, the collective served by the amplitude gate over the same addresses, and the address reads.

THE VARIANTS (``Variant``; one frozen settings object an engine; six as module constants). Every variant uses the
structure ``multirotary.PLACE_MEMBER`` (a head of 64 numbers: relative row 0-9, relative column 10-19, content 20-23,
place column 24, place row 25, member levels 26-27 and 28-29, held 30-31).

| name | told  | see    | identity | turn  | request width |
|------|-------|--------|----------|-------|---------------|
| T1   | own   | masked | both     | True  | 12            |
| T2   | all   | masked | address  | True  | 18            |
| T2K  | all   | masked | name     | False | 18            |
| T3   | all   | open   | address  | True  | 18            |
| C1   | all   | masked | address  | False | 18            |
| C2   | all   | masked | content  | False | 18            |

told "own": only the slot of the one request a row serves is a visible token (no pointers); "all": every request of the
scene. see "masked": an image token is a visible key only when its cell's request is of the engine's class or its cell
holds no request (``frame.visible_keys`` for every cell of the class); "open": every image token. identity: where a
thing's identity is carried, "address" (the member pairs only), "content" (the request token's content only), "both", or
"name" (the key binding on one told head). turn False: every member digit, keys and queries, is the null.

A LAYOUT is three long tensors [B, 9]: ``cells`` (the thing 0..8 of each cell, -1 for none), ``colors`` (the color a
thing is DRAWN in, -1 for none) and ``points`` (-1 when the request states its color, else the class 0..8 of the member
it ASKS). A pointing request is drawn in the asked member's color and its request token carries no color. A valid
layout (``check_layout``): a pointing cell holds a request; the asked class differs from the pointing request's class,
stands exactly once in the scene, and its request states its color (no chains).

MEMBER ADDRESSES. ``address_of(cls)`` = (family, member) = (cls div 3, cls mod 3) for a thing; the null (4, 4) for none.
A mapping {class: (d0, d1)} (digits 0..3) may place a class at another address; every address below is read from it.

THE STREAMS (the same for every engine: they belong to the picture). The sequence is always the 9 request slots (slot s
= cell s) followed by the 144 image tokens: 153 tokens.
    positions       slot s at its cell's center (row 4k + 1.5, column 4j + 1.5; j = s mod 3, k = s div 3); image token
                    (iy, ix) at (iy, ix)
    place digits    keys and queries alike: slot s -> (j, k); an image token -> its cell's (column, row)
    member digits   KEYS: a slot and the 16 image tokens of its cell carry the address of the cell's thing (the null where
                    the cell is empty). QUERIES: the same, except a pointing request: its slot and its cell's 16 image
                    tokens carry the address of the asked class. turn False: the null everywhere.
    the told head   identity "name": the member digits are the null everywhere and block 1, head 0 gets its own angles:
                    the structure's angles with the eight pairs 20-23 and 26-29 replaced by name phases
                    (``multirotary.name_phases`` over the table ``phase_table("member-name|v1", 8, 9)``, a thing's name
                    its English word in bytes). KEYS: a slot and its cell's image tokens take the phases of the cell's
                    thing (zeros where empty). QUERIES: the same, except a pointing request's slot and cell: the phases
                    of the asked thing.
    unturned        a read's switch (``prepare(..., unturned=True)``): the place and member pairs 24-29 are left
                    unturned (angle 0) on every head, the told head's included; the place levels have no null anchor,
                    so their unturned pairs carry the angle of digit 0.

THE REQUEST TOKEN'S CONTENT [B, 9, width]. width 12: the 12 numbers of ``frame.content_code`` (family, member, color
high, color low; the color numbers zero for a pointing request). width 18: [own identity 6 | color 6 | asked identity
6], each a pair of the 3-number codes of ``frame.digit_code(3)``: own identity (family, member) only with identity
"content" or "both"; the stated color (zeros for a pointing request); the asked identity only with identity "content"
and only for a pointing request. A padding slot (an empty cell or a slot not told) has zero content.

AN ENGINE (``MemberEngine``) keeps the stored engine's block, head, time embedding and token sizes. Layers in this order
(the order is part of the seeding): embed Linear(48, dim), req_in Linear(width, dim), ctx_in Linear(18, dim), time
TimeEmbedding(64, dim, 1000), blocks, head ZeroHead(dim, 48); for width 12 these are the names, shapes and order of
``collective.AddressedEngine(read="rotary")``. h = [req_in(content), embed(tokens)]; the context is ctx_in of one null
token, always valid; the key mask is [told slots, the visible image tokens] (a padding slot is an invisible key and
reads only itself); the rotary of every block is ``multirotary.angle_rotary(query angles, key angles)``, block 1 with
the told head's angles when given; the head reads the image tokens (h[:, 9:]). ``forward_tokens`` takes tensors only
and has fixed shapes; ``prepare`` builds its inputs outside it. ``born`` seeds an engine from (seed, its name) by
``units.name_seed`` with the names of ``collective.engine_name`` ("addressed_thing_c<k>", "addressed_base").

THE TRAINING ROWS (``member_batch``), every draw from the CPU generator given, in this order:
    told "own"   exactly ``collective.kind_batch("addressed_thing", cls)`` (``"addressed_base"`` for the base): the
                 scenes (layouts, then the render), then the hidden cells (h, then their keys); points all -1
    told "all", a thing of class cls
                 (1) the own request: ``scenes.draw_layouts("one", n, gen, only_class=cls)`` (its cells [n], then its
                     colors [n]); (2) the clutter: ``scenes.draw_layouts("mix", n, gen)``; the layout = the clutter with
                     the own request written over its cell; a clutter request of the class ``exclude`` is removed;
                 (3) the pointer: u [n] = rand(n), keys [n, 9] = rand(n, 9). The candidates of a scene are its other
                     requests whose class differs from cls, stands exactly once in the scene and is not paired with cls
                     in ``held_pairs`` ((asking, asked) classes). Where u < p_point and a candidate exists, the own
                     request points at the candidate with the smallest key and is drawn in its color;
                 (3b) ONE OTHER POINTER: v [n] = rand(n), keys2 [n, 9, 9] = rand(n, 9, 9). A pair (asking cell a,
                     asked cell q) is valid when both hold requests; a is not the own cell, does not already point and
                     is not asked by the own request's pointer; the asked request states its color; the asked class
                     stands exactly once in the scene and differs from the asking class; the pair of classes is not in
                     ``held_pairs``. Where v < p_point and a valid pair exists, the valid pair with the smallest
                     keys2[a, q] points (the asking request takes the asked color). The asked may be the own request
                     when it states its color; the asking may be another request of class cls;
                 (4) the render of the finished layout (``scenes.render``: shifts, scales, angles, shades, and on bed C
                     the tint and the noise).
                 told: every request; visible ("masked"): the cells whose request is of class cls and the request-free
                 cells ("open": every token); the loss: every cell whose request is of class cls
    told "all", the base
                 a "mix" layout, then one cell [n] (randint 0..8) emptied, then ``exclude`` applied, then step (3b)
                 with every request a possible asking one, then the render; told every request; visible ("masked")
                 and the loss: the request-free cells

REQUEST LISTS THAT POINT (``pointer_lists``): the lists of ``scenes.request_lists(n, count, seed)``; then from a CPU
generator seeded seed + 50000, keys [count, 9, 9] (rand). A pair (asking cell a, asked cell q) is valid when both hold
requests, the asked class stands once in the list and differs from the asking class, and the pair of classes is outside
``held_pairs`` (inside it with ``only_held``); each list's valid pair with the smallest key[a, q] points (the asking
request takes the asked color); a list without a valid pair keeps points -1 and is flagged. With ``distinct``, after
the keys and before the pointer is chosen: perm [count, 9] (rand), and a list's n requests in their drawn order take
the first n colors of argsort(perm) (n different colors).

THE COLLECTIVE (``MemberCollective``). Each engine runs ONE forward over the scenes in which its class is requested
(told "all": every request, its visible keys by the rule above; told "own": one row a (scene, request) pair, each pair's
answer kept on its own cell); the base one forward over the scenes with a request-free cell. The gate (float64, U = 10):

    look(m, r)  = prod over the place levels of k_l(place_l(m), place_l(r))        0 where slot r holds no request
    match(r, i) = prod over the member levels of k_l(member_l(r), g_l(i))           g(i) = member i's address
    u_i(m)      = U sum_r look(m, r) match(r, i)
    rest form   w_i = sinh(u_i) / (1 + sum_j 2 sinh(u_j / 2)^2),    base = 1 - sum_i w_i
    candidate   w_i = sinh(u_i) / (sum_j sinh(u_j) + exp(-sum_j u_j)),    base = exp(-sum_j u_j) / (that denominator)
    serve       v = base v_0 + sum_i w_i v_i                                       (``multirotary.serve``)

The gate reads the declared address digits through the levels' kernel; it never reads an attention weight.

THE ADDRESS READS (no training; FINDING and SERVING are read apart, then together):
    find_read       one attention site of one engine: the input of a block's attention captured once, then the scores
                    q.k / sqrt(64) of a pointing slot's query against two candidate request slots recomputed twice, with
                    the pointing slot's and its cell's query digits at the address of one candidate and of the other;
                    split by pair group, per head, with the softmax weights; and the engine's raw answer on that cell
                    from two full forwards that differ only in those query angles. No gate.
    serve_read      the parts' raw answers given and never recomputed; only the address handed to the gate for one
                    request changes: u, sinh(u), the shared denominator, the weights and the base, before, after and
                    after the reverse edit.
    address_swap    one request's thing changed (the picture not re-rendered): the gate's numbers on its cell; the
                    attention mass of that cell's 16 image-token queries on that cell's request slot, per block and
                    head, for the engines of the old and the new class (the mean over the 16 queries of their softmax
                    weight on the slot); every other member's relative change of contribution on the tokens it owns
                    (its weight above 1/2 before the edit): rms(difference) / rms(contribution before). Controls:
                    turn=False (the engines' streams unturned, the gate as it is) and gate="neutral" (the raw answers
                    on that cell, no weighting).
    address_missing the request removed: the gate's weights on its cell and the base's.
    address_unheld  the request's class placed, through the request side of the address mapping, at an address no
                    member holds (every request of that class in the scene moves with it): the same numbers.
    exchange_read   two collectives that differ in one engine: for every other member, the largest absolute difference
                    of its contribution.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from . import frame, scenes
from . import multirotary as M
from .blocks import Block, TimeEmbedding, ZeroHead
from .collective import CTX_DIM, TOKEN_DIM, engine_name, kind_batch, sample
from .compat import TIME_FREQ
from .scenes import CANVAS, CHANNELS, N_CELLS, PATCH, TOKEN_GRID, ScenesBed
from .spectral import patchify, unpatchify
from .units import name_seed

STRUCTURE = M.PLACE_MEMBER
HEAD_DIM = STRUCTURE.head_dim                 # 64
SLOTS = N_CELLS                               # 9 request slots
IMAGE = TOKEN_GRID * TOKEN_GRID               # 144 image tokens
SEQ = SLOTS + IMAGE                           # 153
NULL = 4                                      # the member levels' null digit
NULL_ADDRESS = (NULL, NULL)
U_GATE = 10.0
PLACE_LEVELS = ("place_column", "place_row")
MEMBER_LEVELS = ("member_0", "member_1")
NAME_TAG = "member-name|v1"
NAME_PAIRS = (20, 21, 22, 23, 26, 27, 28, 29)
TOLD_BLOCK, TOLD_HEAD = 1, 0
UNTURNED_PAIRS = tuple(range(24, 30))         # the place and member pairs
GROUPS = ("row", "column", "content", "place_column", "place_row", "member_0", "member_1", "held")
TOLDS = ("own", "all")
SEES = ("masked", "open")
IDENTITIES = ("address", "content", "both", "name")
FORMS = ("rest", "candidate")
MEMBER_FILES = ("engine.safetensors", "engine_avg.safetensors", "engine.json")


# ------------------------------------------------------------------ variants
@dataclass(frozen=True)
class Variant:
    """The settings of a kind of member engine: see the module's table."""
    name: str
    told: str
    see: str
    identity: str
    turn: bool
    width: int

    def __post_init__(self) -> None:
        if self.told not in TOLDS or self.see not in SEES or self.identity not in IDENTITIES:
            raise ValueError(f"told {TOLDS}, see {SEES}, identity {IDENTITIES}")
        if int(self.width) != (12 if self.told == "own" else 18):
            raise ValueError("the request width is 12 for told 'own' and 18 for told 'all'")

    def settings(self) -> dict:
        return {"name": self.name, "told": self.told, "see": self.see, "identity": self.identity,
                "turn": bool(self.turn), "width": int(self.width)}


T1 = Variant("T1", "own", "masked", "both", True, 12)
T2 = Variant("T2", "all", "masked", "address", True, 18)
T2K = Variant("T2K", "all", "masked", "name", False, 18)
T3 = Variant("T3", "all", "open", "address", True, 18)
C1 = Variant("C1", "all", "masked", "address", False, 18)
C2 = Variant("C2", "all", "masked", "content", False, 18)
VARIANTS: Dict[str, Variant] = {v.name: v for v in (T1, T2, T2K, T3, C1, C2)}


def variant_of(settings: Mapping) -> Variant:
    return Variant(settings["name"], settings["told"], settings["see"], settings["identity"], bool(settings["turn"]),
                   int(settings["width"]))


# ------------------------------------------------------------------ addresses and layouts
def address_of(cls: Optional[int]) -> Tuple[int, int]:
    """(family, member) of a thing 0..8; the null (4, 4) for None or -1."""
    if cls is None or int(cls) < 0:
        return NULL_ADDRESS
    return (scenes.family_of(cls), scenes.member_of(cls))


def address_table(addresses: Optional[Mapping[int, Sequence[int]]] = None, device=None) -> Tensor:
    """[10, 2] long: row c the address of class c (0..8) from ``addresses`` (default ``address_of``), row 9 the null."""
    rows = [address_of(c) for c in range(9)] + [NULL_ADDRESS]
    for c, a in (addresses or {}).items():
        a = tuple(int(d) for d in a)
        if not 0 <= int(c) < 9 or len(a) != 2 or any(not 0 <= d < NULL for d in a):
            raise ValueError(f"an address maps a class 0..8 to two digits 0..3, not {c} -> {a}")
        rows[int(c)] = a
    return torch.tensor(rows, dtype=torch.long, device=device)


def _index(cls: Tensor) -> Tensor:
    """Class tensor -> row of the [10, ...] tables (9 for none)."""
    return torch.where(cls >= 0, cls, torch.full_like(cls, 9))


def no_points(cells: Tensor) -> Tensor:
    return torch.full_like(torch.as_tensor(cells, dtype=torch.long), -1)


def check_layout(cells: Tensor, colors: Tensor, points: Tensor) -> None:
    """Raise unless the layout is valid (see the module's text)."""
    if cells.shape != colors.shape or cells.shape != points.shape or cells.dim() != 2 or cells.shape[1] != N_CELLS:
        raise ValueError("a layout is cells, colors and points, each [B, 9]")
    if bool(((cells >= 0) != (colors >= 0)).any()):
        raise ValueError("a cell holds a color exactly when it holds a thing")
    pointing = points >= 0
    if not bool(pointing.any()):
        return
    if bool((pointing & (cells < 0)).any()):
        raise ValueError("a pointing cell holds no request")
    if bool((pointing & (points == cells)).any()):
        raise ValueError("a request points at its own class")
    same = cells[:, None, :] == points[:, :, None]                       # [B, 9 asking, 9 cells]
    count = same.sum(dim=-1)
    if bool((pointing & (count != 1)).any()):
        raise ValueError("an asked class must stand exactly once in the scene")
    asked_cell = same.long().argmax(dim=-1)
    if bool((pointing & (torch.gather(points, 1, asked_cell) >= 0)).any()):
        raise ValueError("an asked request must state its color (no chains)")


def cell_tokens(per_cell: Tensor) -> Tensor:
    """[B, 9] -> [B, 144]: each image token takes its cell's value."""
    return per_cell[:, _FRAME.cell.to(per_cell.device)]


def visible_for(cells: Tensor, cls: Optional[int]) -> Tensor:
    """[B, 144] bool, the "masked" rule: the tokens of cells whose request is of class ``cls`` and of request-free cells
    (the request-free cells alone for the base, cls None)."""
    free = cells < 0
    keep = free if cls is None else free | (cells == int(cls))
    return cell_tokens(keep)


# ------------------------------------------------------------------ the streams
_FRAME = frame.TokenFrame()
_SEQ_CELL = torch.cat([torch.arange(SLOTS), _FRAME.cell])                # [153]: the cell of each token


def _positions(device) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """[153] each: rows, cols (float64), place column, place row (long) of the sequence."""
    s = torch.arange(SLOTS)
    side = _FRAME.side
    img = torch.arange(IMAGE)
    rows = torch.cat([(s // 3 * side).double() + (side - 1) / 2, (img // TOKEN_GRID).double()])
    cols = torch.cat([(s % 3 * side).double() + (side - 1) / 2, (img % TOKEN_GRID).double()])
    pcol = torch.cat([s % 3, _FRAME.column])
    prow = torch.cat([s // 3, _FRAME.row])
    return rows.to(device), cols.to(device), pcol.to(device), prow.to(device)


def streams(variant: Variant, cells: Tensor, points: Tensor,
            addresses: Optional[Mapping[int, Sequence[int]]] = None) -> dict:
    """The positions and digits of the 153-token sequence: rows, cols [B, 153] float64; key_digits, query_digits
    {level: [B, 153] long}."""
    dev = cells.device
    b = cells.shape[0]
    rows, cols, pcol, prow = _positions(dev)
    seq_cell = _SEQ_CELL.to(dev)
    place = {"place_column": pcol.expand(b, -1), "place_row": prow.expand(b, -1)}
    if variant.turn and variant.identity != "name":
        table = address_table(addresses, dev)
        key_cell = table[_index(cells)]                                   # [B, 9, 2]
        query_cell = table[_index(torch.where(points >= 0, points, cells))]
        key_seq, query_seq = key_cell[:, seq_cell], query_cell[:, seq_cell]       # [B, 153, 2]
    else:
        key_seq = query_seq = torch.full((b, SEQ, 2), NULL, dtype=torch.long, device=dev)
    kd = dict(place, member_0=key_seq[..., 0], member_1=key_seq[..., 1])
    qd = dict(place, member_0=query_seq[..., 0], member_1=query_seq[..., 1])
    return {"rows": rows.expand(b, -1), "cols": cols.expand(b, -1), "key_digits": kd, "query_digits": qd}


_NAME_TABLE: Optional[Tensor] = None


def name_table(device=None) -> Tensor:
    """[10, 8] float64: row c the name phases of thing c (``scenes.THINGS[c]`` in bytes), row 9 zeros."""
    global _NAME_TABLE
    if _NAME_TABLE is None:
        tab = M.phase_table(NAME_TAG, len(NAME_PAIRS), 9)
        rows = [M.name_phases(tab, scenes.THINGS[c].encode()) for c in range(9)]
        _NAME_TABLE = torch.stack(rows + [torch.zeros(len(NAME_PAIRS), dtype=torch.float64)])
    return _NAME_TABLE.to(device)


def request_content(variant: Variant, cells: Tensor, colors: Tensor, points: Tensor, told: Tensor) -> Tensor:
    """[B, 9, width] float32: the request tokens' content (see the module's text)."""
    dev = cells.device
    u = frame.digit_code(3).float().to(dev)
    t, k, a = cells.clamp_min(0), colors.clamp_min(0), points.clamp_min(0)
    stated = (points < 0).float()[..., None]
    pointing = (points >= 0).float()[..., None]
    on = (told & (cells >= 0)).float()[..., None]
    own = torch.cat([u[t // 3], u[t % 3]], dim=-1)
    color = torch.cat([u[k // 3], u[k % 3]], dim=-1) * stated
    if variant.width == 12:
        return torch.cat([own, color], dim=-1) * on
    own = own * float(variant.identity in ("content", "both"))
    asked = torch.cat([u[a // 3], u[a % 3]], dim=-1) * pointing * float(variant.identity == "content")
    return torch.cat([own, color, asked], dim=-1) * on


def prepare(variant: Variant, cells: Tensor, colors: Tensor, points: Optional[Tensor] = None,
            told: Optional[Tensor] = None, visible: Optional[Tensor] = None,
            addresses: Optional[Mapping[int, Sequence[int]]] = None, unturned: bool = False) -> dict:
    """The tensors ``MemberEngine.forward_tokens`` takes (content, query_angles, key_angles, key_mask, told_angles),
    built on the device of ``cells``. ``told`` [B, 9] bool (default every cell holding a request); ``visible``
    [B, 144] bool or None (every image token)."""
    cells = torch.as_tensor(cells, dtype=torch.long)
    dev = cells.device
    colors = torch.as_tensor(colors, dtype=torch.long).to(dev)
    points = no_points(cells) if points is None else torch.as_tensor(points, dtype=torch.long).to(dev)
    check_layout(cells, colors, points)
    b = cells.shape[0]
    told = (cells >= 0) if told is None else (torch.as_tensor(told).to(dev).bool() & (cells >= 0))
    st = streams(variant, cells, points, addresses)
    ka = STRUCTURE.angles(st["rows"], st["cols"], st["key_digits"])
    qa = STRUCTURE.angles(st["rows"], st["cols"], st["query_digits"])
    if unturned:
        ka, qa = ka.clone(), qa.clone()
        ka[..., list(UNTURNED_PAIRS)] = 0.0
        qa[..., list(UNTURNED_PAIRS)] = 0.0
    told_angles = None
    if variant.identity == "name":
        names = name_table(dev)
        seq_cell = _SEQ_CELL.to(dev)
        kp = names[_index(cells)][:, seq_cell]                            # [B, 153, 8]
        qp = names[_index(torch.where(points >= 0, points, cells))][:, seq_cell]
        tk, tq = ka.clone(), qa.clone()
        tk[..., list(NAME_PAIRS)] = kp.float()
        tq[..., list(NAME_PAIRS)] = qp.float()
        told_angles = (tq, tk)
    img = torch.ones(b, IMAGE, dtype=torch.bool, device=dev) if visible is None else \
        torch.as_tensor(visible).to(dev).bool()
    return {"content": request_content(variant, cells, colors, points, told), "query_angles": qa,
            "key_angles": ka, "key_mask": torch.cat([told, img], dim=1), "told_angles": told_angles}


# ------------------------------------------------------------------ the engine
class MemberEngine(nn.Module):
    """See the module's text."""

    def __init__(self, variant: Variant, dim: int = 128, heads: int = 2, depth: int = 3):
        super().__init__()
        if int(dim) // int(heads) != HEAD_DIM or int(dim) % int(heads):
            raise ValueError(f"a member engine needs heads of {HEAD_DIM} numbers")
        if variant.identity == "name" and int(depth) <= TOLD_BLOCK:
            raise ValueError(f"identity 'name' turns block {TOLD_BLOCK}: the depth must exceed it")
        self.variant = variant
        self.settings = {"dim": int(dim), "heads": int(heads), "depth": int(depth), "variant": variant.settings(),
                         "token_dim": TOKEN_DIM, "grid": TOKEN_GRID, "patch": PATCH, "ctx_dim": CTX_DIM,
                         "time_freq": TIME_FREQ, "slots": SLOTS}
        d = int(dim)
        self.embed = nn.Linear(TOKEN_DIM, d)
        self.req_in = nn.Linear(int(variant.width), d)
        self.ctx_in = nn.Linear(CTX_DIM, d)
        self.time = TimeEmbedding(TIME_FREQ, d, 1000.0)
        self.blocks = nn.ModuleList([Block(d, int(heads)) for _ in range(int(depth))])
        self.head = ZeroHead(d, TOKEN_DIM)

    def forward_tokens(self, z: Tensor, t: Tensor, content: Tensor, query_angles: Tensor, key_angles: Tensor,
                       key_mask: Tensor, told_angles: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        """z [B, 3, 48, 48], t [B], content [B, 9, width], angles [B, 153, 32], key_mask [B, 153] bool, told_angles
        None or (query, key) angles [B, 153, 32] for block 1, head 0 -> [B, 3, 48, 48]."""
        b = z.shape[0]
        h = torch.cat([self.req_in(content.to(z.dtype)), self.embed(patchify(z, PATCH))], dim=1)
        e = self.time(t)
        c = self.ctx_in(torch.zeros(b, 1, CTX_DIM, device=z.device, dtype=z.dtype))
        valid = torch.ones(b, 1, dtype=torch.bool, device=z.device)
        rot = M.angle_rotary(query_angles, key_angles)
        rot_told = rot if told_angles is None else M.angle_rotary(query_angles, key_angles, {TOLD_HEAD: told_angles})
        for i, blk in enumerate(self.blocks):
            h = blk(h, c, valid, e, rot_told if i == TOLD_BLOCK else rot, key_mask)
        return unpatchify(self.head(h[:, SLOTS:]), (CHANNELS, CANVAS, CANVAS), PATCH, TOKEN_GRID)

    def forward(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor, points: Optional[Tensor] = None,
                told: Optional[Tensor] = None, visible: Optional[Tensor] = None,
                addresses: Optional[Mapping[int, Sequence[int]]] = None) -> Tensor:
        dev = z.device
        prep = prepare(self.variant, torch.as_tensor(cells).to(dev), torch.as_tensor(colors).to(dev),
                       None if points is None else torch.as_tensor(points).to(dev), told, visible, addresses)
        return self.forward_tokens(z, t, **prep)


def member_name(cls: Optional[int]) -> str:
    """The name of an engine's random stream: the stored engines' names."""
    return engine_name("addressed_base", None) if cls is None else engine_name("addressed_thing", int(cls))


def born(variant: Variant, cls: Optional[int], seed: int, dim: int = 128, heads: int = 2,
         depth: int = 3) -> MemberEngine:
    """A newborn engine whose numbers depend only on (seed, its name) and its shape; the caller's random state is put
    back. A width-12 engine is born equal to the stored rotary engine of the same class and seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(name_seed(int(seed), member_name(cls)))
        eng = MemberEngine(variant, dim, heads, depth)
    eng.cls, eng.seed = (None if cls is None else int(cls)), int(seed)
    return eng


def batch_forward(engine: MemberEngine, batch: dict, z: Tensor, t: Tensor) -> Tensor:
    return engine(z, t, batch["cells"], batch["colors"], batch["points"], batch["told"], batch["visible"])


def save_member(engine: MemberEngine, folder: str, average: Optional[Dict[str, Tensor]] = None,
                card: Optional[dict] = None) -> None:
    """engine.safetensors, engine_avg.safetensors (when given) and engine.json (class, seed, settings with the
    variant, trained numbers and the given card)."""
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in engine.state_dict().items()},
              os.path.join(folder, "engine.safetensors"))
    if average is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in average.items()},
                  os.path.join(folder, "engine_avg.safetensors"))
    info = {"kind": "member", "cls": getattr(engine, "cls", None), "seed": getattr(engine, "seed", None),
            "settings": engine.settings, "trained": int(sum(p.numel() for p in engine.parameters()))}
    info.update(card or {})
    with open(os.path.join(folder, "engine.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(info, fh, indent=1)
        fh.write("\n")


def load_member(folder: str, average: bool = True, device="cpu") -> MemberEngine:
    """An engine written by ``save_member``: the averaged parameters when ``average`` and the file is there."""
    from safetensors.torch import load_file
    with open(os.path.join(folder, "engine.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    s = info["settings"]
    with torch.random.fork_rng(devices=[]):
        eng = MemberEngine(variant_of(s["variant"]), s["dim"], s["heads"], s["depth"])
    state = load_file(os.path.join(folder, "engine.safetensors"))
    avg = os.path.join(folder, "engine_avg.safetensors")
    if average and os.path.isfile(avg):
        state.update(load_file(avg))
    eng.load_state_dict(state, strict=True)
    eng.cls, eng.seed, eng.card = info.get("cls"), info.get("seed"), info
    return eng.to(device).eval()


# ------------------------------------------------------------------ training rows
def _held_asked(cls: int, held_pairs) -> Tensor:
    """[10] bool: the asked classes paired with ``cls`` in ``held_pairs`` (row 9, none, False)."""
    out = torch.zeros(10, dtype=torch.bool)
    for a, q in held_pairs:
        if int(a) == int(cls):
            out[int(q)] = True
    return out


def _held_matrix(held_pairs) -> Tensor:
    """[10, 10] bool: entry (asking class, asked class) True for a held pair (row and column 9, none, False)."""
    held = torch.zeros(10, 10, dtype=torch.bool)
    for a, q in held_pairs:
        held[int(a), int(q)] = True
    return held


def _valid_pairs(cells: Tensor, held: Tensor, only_held: bool = False, asking_ok: Optional[Tensor] = None,
                 asked_ok: Optional[Tensor] = None) -> Tensor:
    """[B, 9, 9] bool: entry (a, q) True when cells a and q both hold requests, the asked class (cell q's) stands
    exactly once in the scene and differs from the asking class (cell a's), the pair of classes is outside ``held``
    (inside it with ``only_held``), and, when given, asking_ok[a] and asked_ok[q] ([B, 9] bool) hold."""
    once = (cells[:, :, None] == cells[:, None, :]).sum(dim=-1) == 1
    a, q = cells[:, :, None], cells[:, None, :]
    pair_held = held[_index(a).expand(-1, -1, N_CELLS), _index(q).expand(-1, N_CELLS, -1)]
    valid = (a >= 0) & (q >= 0) & (a != q) & once[:, None, :] & (pair_held if only_held else ~pair_held)
    if asking_ok is not None:
        valid = valid & asking_ok[:, :, None]
    if asked_ok is not None:
        valid = valid & asked_ok[:, None, :]
    return valid


def _smallest_pair(valid: Tensor, keys: Tensor) -> Tuple[Tensor, Tensor]:
    """(asking cell [B], asked cell [B]): the valid pair with the smallest key (keys [B, 9, 9]; meaningless where no
    pair is valid)."""
    pick = torch.where(valid, keys, torch.full_like(keys, 2.0)).flatten(1).argmin(dim=1)
    return pick // N_CELLS, pick % N_CELLS


def member_batch(variant: Variant, cls: Optional[int], bed: ScenesBed, n: int, gen: torch.Generator,
                 p_point: float = 0.5, exclude: Optional[int] = None, held_pairs=()) -> dict:
    """The training rows of an engine (see the module's text for the draw order) -> x, cells, colors, points, told,
    visible, loss_mask, hidden, params, pointing [n] bool (the rows whose own request points), can_point [n] bool
    (the rows with a candidate for it), other_pointing [n] bool (the rows in which another request points) and
    other_can_point [n] bool (the rows with a valid pair for it); and the layout on the CPU as drawn: cells_cpu,
    colors_cpu, points_cpu, told_cpu and visible_cpu (None for "open"), so that a trainer can prepare its inputs
    without reading anything back from the bed's device (for told "own" these are copies of the stored rows' tensors,
    taken back from the bed's device)."""
    dev = bed.device
    n = int(n)
    if variant.told == "own":
        if exclude is not None or held_pairs:
            raise ValueError("told 'own' rows are the stored rows: no clutter to exclude, no pointers")
        kb = kind_batch("addressed_thing" if cls is not None else "addressed_base", cls, bed, n, gen)
        none = torch.zeros(n, dtype=torch.bool, device=dev)
        cells_cpu = kb["cells"].cpu().clone()                    # the stored path: these are copies back from the bed
        return {"x": kb["x"], "cells": kb["cells"], "colors": kb["colors"], "points": no_points(kb["cells"]),
                "told": kb["told"], "visible": kb["key_mask"], "loss_mask": kb["loss_mask"], "hidden": kb["hidden"],
                "params": kb["params"], "pointing": none, "can_point": none, "other_pointing": none,
                "other_can_point": none, "cells_cpu": cells_cpu, "colors_cpu": kb["colors"].cpu().clone(),
                "points_cpu": no_points(cells_cpu), "told_cpu": kb["told"].cpu().clone(),
                "visible_cpu": kb["key_mask"].cpu().clone()}
    if exclude is not None and cls is not None and int(exclude) == int(cls):
        raise ValueError("an engine's own class cannot be excluded from its clutter")
    r = torch.arange(n)
    if cls is not None:
        oc, ok = scenes.draw_layouts("one", n, gen, only_class=int(cls))                 # (1)
        cells, colors = scenes.draw_layouts("mix", n, gen)                               # (2)
        own = (oc >= 0).long().argmax(dim=1)
        cells[r, own] = int(cls)
        colors[r, own] = ok[r, own]
    else:
        cells, colors = scenes.draw_layouts("mix", n, gen)
        empty = torch.randint(0, N_CELLS, (n,), generator=gen)
        cells[r, empty] = -1
        colors[r, empty] = -1
    if exclude is not None:
        drop = cells == int(exclude)
        cells[drop] = -1
        colors[drop] = -1
    points = no_points(cells)
    pointing = torch.zeros(n, dtype=torch.bool)
    can = torch.zeros(n, dtype=torch.bool)
    if cls is not None:
        u = torch.rand(n, generator=gen)                                                 # (3)
        keys = torch.rand(n, N_CELLS, generator=gen)
        count = (cells[:, :, None] == cells[:, None, :]).sum(dim=-1)
        cand = (cells >= 0) & (cells != int(cls)) & (count == 1) & ~_held_asked(int(cls), held_pairs)[_index(cells)]
        can = cand.any(dim=1)
        pointing = (u < float(p_point)) & can
        pick = torch.where(cand, keys, torch.full_like(keys, 2.0)).argmin(dim=1)
        rp = r[pointing]
        points[rp, own[pointing]] = cells[rp, pick[pointing]]
        colors[rp, own[pointing]] = colors[rp, pick[pointing]]
    asking_ok = (cells >= 0) & (points < 0)                                              # (3b): one other pointer
    if cls is not None:
        asking_ok[r, own] = False
        asking_ok[rp, pick[pointing]] = False                                            # asked by the own pointer
    v = torch.rand(n, generator=gen)
    keys2 = torch.rand(n, N_CELLS, N_CELLS, generator=gen)
    valid = _valid_pairs(cells, _held_matrix(held_pairs), False, asking_ok, points < 0)
    other_can = valid.flatten(1).any(dim=1)
    other = (v < float(p_point)) & other_can
    ask, asked = _smallest_pair(valid, keys2)
    ro = r[other]
    points[ro, ask[other]] = cells[ro, asked[other]]
    colors[ro, ask[other]] = colors[ro, asked[other]]
    pixels, params = scenes.render(cells, colors, bed.bed, gen, dev)                     # (4)
    cpu = {"cells_cpu": cells.clone(), "colors_cpu": colors.clone(), "points_cpu": points.clone(),
           "told_cpu": cells >= 0,
           "visible_cpu": visible_for(cells, cls) if variant.see == "masked" else None}
    cells, colors, points = cells.to(dev), colors.to(dev), points.to(dev)
    told = cells >= 0
    mine = cell_tokens(cells < 0) if cls is None else cell_tokens(cells == int(cls))
    visible = visible_for(cells, cls) if variant.see == "masked" else None
    return dict({"x": scenes.to_model(pixels), "cells": cells, "colors": colors, "points": points, "told": told,
                 "visible": visible, "loss_mask": mine, "hidden": None, "params": params,
                 "pointing": pointing.to(dev), "can_point": can.to(dev), "other_pointing": other.to(dev),
                 "other_can_point": other_can.to(dev)}, **cpu)


# ------------------------------------------------------------------ request lists that point
class PointerLists(NamedTuple):
    cells: Tensor       # [count, 9] long
    colors: Tensor      # [count, 9] long: a pointing request's color is the asked one's
    points: Tensor      # [count, 9] long
    pointed: Tensor     # [count] bool: the lists in which a request points
    order: Tensor       # [count, n] long: the requested cells in the order they were drawn


def pointer_lists(n: int, count: int, seed: int, held_pairs=(), only_held: bool = False,
                  distinct: bool = False) -> PointerLists:
    """See the module's text. ``distinct``: before the pointer is chosen, the n requests of each list are recolored
    with n different colors (after the keys, from the same generator: perm [count, 9] = rand; a list's requests in their
    drawn order take the first n colors of argsort(perm)), so that the other requests all differ in color."""
    rl = scenes.request_lists(int(n), int(count), int(seed))
    gen = torch.Generator().manual_seed(int(seed) + 50000)
    keys = torch.rand(int(count), N_CELLS, N_CELLS, generator=gen)
    cells, colors = rl.cells.clone(), rl.colors.clone()
    if distinct:
        perm = torch.rand(int(count), N_CELLS, generator=gen)
        new = torch.argsort(perm, dim=1, stable=True)[:, :int(n)]                # [count, n] distinct colors
        colors.scatter_(1, rl.order, new)                                          # request j of a list -> new[:, j]
    valid = _valid_pairs(cells, _held_matrix(held_pairs), only_held)
    pointed = valid.flatten(1).any(dim=1)
    ask_cell, asked_cell = _smallest_pair(valid, keys)
    points = no_points(cells)
    rows = torch.nonzero(pointed, as_tuple=True)[0]
    points[rows, ask_cell[rows]] = cells[rows, asked_cell[rows]]
    colors[rows, ask_cell[rows]] = colors[rows, asked_cell[rows]]
    return PointerLists(cells, colors, points, pointed, rl.order)


# ------------------------------------------------------------------ the collective
class MemberCollective:
    """Members {class: engine} of one variant, a base (or None), an address mapping (default ``address_of``) and the
    gate's form ("rest" or "candidate"). See the module's text."""

    def __init__(self, things: Dict[int, MemberEngine], base: Optional[MemberEngine], variant: Variant,
                 addresses: Optional[Mapping[int, Sequence[int]]] = None, form: str = "rest", U: float = U_GATE):
        if form not in FORMS:
            raise ValueError(f"the gate's forms are {FORMS}, not {form!r}")
        self.things = {int(k): things[k] for k in sorted(things)}
        self.classes = sorted(self.things)
        self.base = base
        self.variant = variant
        for e in list(self.things.values()) + ([base] if base is not None else []):
            if e.variant != variant:
                raise ValueError(f"a collective is built from engines of one variant: {e.variant.name} != {variant.name}")
        self.addresses = dict(addresses or {})
        self.form = form
        self.U = float(U)
        self.member_addresses = address_table(self.addresses)[self.classes]           # [M, 2]

    # -- the parts
    def engine_inputs(self, cls: Optional[int], cells: Tensor, colors: Tensor, points: Tensor,
                      unturned: bool = False, cell: Optional[Tensor] = None) -> dict:
        """The inputs of the engine of ``cls`` (None: the base) on these scenes: told "all" every request, the
        "masked" visible keys of its class; told "own" the request of ``cell`` [B] (none for the base) with
        ``frame.visible_keys``."""
        if self.variant.told == "all":
            visible = visible_for(cells, cls) if self.variant.see == "masked" else None
            return prepare(self.variant, cells, colors, points, None, visible, self.addresses, unturned)
        b = cells.shape[0]
        if cls is None:
            told = torch.zeros(b, N_CELLS, dtype=torch.bool, device=cells.device)
            serving = torch.full((b,), -1, dtype=torch.long, device=cells.device)
        else:
            told = torch.nn.functional.one_hot(cell, N_CELLS).bool()
            serving = cell
        visible = frame.visible_keys(cells, serving) if self.variant.see == "masked" else None
        return prepare(self.variant, cells, colors, points, told, visible, self.addresses, unturned)

    def part_answers(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor, points: Optional[Tensor] = None,
                     unturned: bool = False) -> Tuple[Tensor, Tensor]:
        """([M, B, 144, 48] the members' answers in class order, [B, 144, 48] the base's); zeros on the scenes an
        engine is not run on."""
        dev = z.device
        cells, colors = cells.to(dev), colors.to(dev)
        points = no_points(cells) if points is None else points.to(dev)
        b = z.shape[0]
        out = torch.zeros(len(self.classes), b, IMAGE, TOKEN_DIM, device=dev, dtype=z.dtype)
        cell_of = _FRAME.cell.to(dev)
        for i, k in enumerate(self.classes):
            eng = self.things[k]
            if self.variant.told == "all":
                rows = torch.nonzero((cells == k).any(dim=1), as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                prep = self.engine_inputs(k, cells[rows], colors[rows], points[rows], unturned)
                out[i, rows] = patchify(eng.forward_tokens(z[rows], t[rows], **prep), PATCH)
                continue
            rows, cols = torch.nonzero(cells == k, as_tuple=True)
            if rows.numel() == 0:
                continue
            prep = self.engine_inputs(k, cells[rows], colors[rows], points[rows], unturned, cell=cols)
            vt = patchify(eng.forward_tokens(z[rows], t[rows], **prep), PATCH)
            sel = cell_of[None] == cols[:, None]                                # [P, 144]: each pair's own tokens
            p_idx, tok = torch.nonzero(sel, as_tuple=True)
            out[i, rows[p_idx], tok] = vt[p_idx, tok]
        base = torch.zeros(b, IMAGE, TOKEN_DIM, device=dev, dtype=z.dtype)
        rows = torch.nonzero((cells < 0).any(dim=1), as_tuple=True)[0]
        if rows.numel():
            if self.base is None:
                raise ValueError("a scene has cells without a request and the collective has no base")
            prep = self.engine_inputs(None, cells[rows], colors[rows], points[rows], unturned)
            base[rows] = patchify(self.base.forward_tokens(z[rows], t[rows], **prep), PATCH)
        return out, base

    # -- the gate
    def gate_parts(self, cells: Tensor, request_addresses: Optional[Mapping[int, Sequence[int]]] = None,
                   form: Optional[str] = None) -> Dict[str, Tensor]:
        """u [B, 144, M], numerators sinh(u), denominator [B, 144], w [B, 144, M], base [B, 144] (float64). The
        requests' member digits come from ``request_addresses`` (default the collective's mapping); the members keep
        their own addresses."""
        form = self.form if form is None else form
        if form not in FORMS:
            raise ValueError(f"the gate's forms are {FORMS}, not {form!r}")
        cells = torch.as_tensor(cells, dtype=torch.long)
        dev = cells.device
        b = cells.shape[0]
        mapping = self.addresses if request_addresses is None else request_addresses
        req = address_table(mapping, dev)[_index(cells)]                         # [B, 9, 2]
        s = torch.arange(SLOTS, device=dev)
        req_d = {"place_column": (s % 3).expand(b, -1), "place_row": (s // 3).expand(b, -1),
                 "member_0": req[..., 0], "member_1": req[..., 1]}
        tok_d = {"place_column": _FRAME.column.to(dev).expand(b, -1), "place_row": _FRAME.row.to(dev).expand(b, -1)}
        lk = M.look([STRUCTURE.level(n) for n in PLACE_LEVELS], tok_d, req_d, cells >= 0)
        mt = M.member_match([STRUCTURE.level(n) for n in MEMBER_LEVELS], req_d, self.member_addresses.to(dev))
        u = M.arguments(lk, mt, self.U)
        if form == "rest":
            w, base = M.rest_weights(u)
            den = torch.ones(u.shape[:-1], dtype=torch.float64, device=dev)
            for j in range(u.shape[-1]):
                den = den + 2.0 * torch.sinh(u[..., j] / 2.0) ** 2
        else:
            w, base = M.candidate_weights(u)
            den = torch.sinh(u).sum(dim=-1) + torch.exp(-u.sum(dim=-1))
        return {"u": u, "numerators": torch.sinh(u), "denominator": den, "w": w, "base": base}

    def weights(self, cells: Tensor, form: Optional[str] = None,
                request_addresses: Optional[Mapping[int, Sequence[int]]] = None) -> Tuple[Tensor, Tensor]:
        g = self.gate_parts(cells, request_addresses, form)
        return g["w"], g["base"]

    # -- the serve
    def answer(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor, points: Optional[Tensor] = None) -> Tensor:
        answers, base_answer = self.part_answers(z, t, cells, colors, points)
        w, base = self.weights(cells.to(z.device))
        v = M.serve(base_answer, answers, w, base)
        return unpatchify(v, (CHANNELS, CANVAS, CANVAS), PATCH, TOKEN_GRID)

    def packed_answer(self, z: Tensor, t: Tensor, cells: Tensor, packed: Tensor) -> Tensor:
        """``answer`` with colors and points packed as [B, 9, 2] (``pack``): the callable ``collective.sample``
        takes, since it slices its colors argument by rows."""
        return self.answer(z, t, cells, packed[..., 0], packed[..., 1])

    def contributions(self, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor,
                      points: Optional[Tensor] = None, unturned: bool = False) -> Tensor:
        """[M + 1, B, 144, 48]: each member's w_i v_i (class order), the base's base v_0 last."""
        answers, base_answer = self.part_answers(z, t, cells, colors, points, unturned)
        w, base = self.weights(cells.to(z.device))
        parts = [w[..., i].to(answers.dtype)[..., None] * answers[i] for i in range(answers.shape[0])]
        parts.append(base.to(base_answer.dtype)[..., None] * base_answer)
        return torch.stack(parts)


def pack(colors: Tensor, points: Tensor) -> Tensor:
    return torch.stack([torch.as_tensor(colors, dtype=torch.long), torch.as_tensor(points, dtype=torch.long)], dim=-1)


def sample_lists(col: MemberCollective, cells: Tensor, colors: Tensor, points: Tensor, seed: int, steps: int = 50,
                 device="cpu", chunk: int = 0) -> Tensor:
    """``collective.sample`` of request lists with their points."""
    return sample(col.packed_answer, cells, pack(colors, points), seed, steps=steps, device=device, chunk=chunk)


# ------------------------------------------------------------------ the address reads
def _tokens_of(cell: int, device=None) -> Tensor:
    return torch.nonzero(_FRAME.cell == int(cell), as_tuple=True)[0].to(device)


def _the_cell_of(cells_row: Tensor, cls: int) -> int:
    where = torch.nonzero(cells_row == int(cls), as_tuple=True)[0]
    if where.numel() != 1:
        raise ValueError(f"class {cls} must stand exactly once in the scene, not {where.numel()} times")
    return int(where[0])


def block_rotary(prep: dict, block: int):
    """The rotary a block of a member engine uses for these inputs."""
    if prep["told_angles"] is not None and int(block) == TOLD_BLOCK:
        return M.angle_rotary(prep["query_angles"], prep["key_angles"], {TOLD_HEAD: prep["told_angles"]})
    return M.angle_rotary(prep["query_angles"], prep["key_angles"])


def pair_group_scores(q: Tensor, k: Tensor, query_index: int, key_indices: Sequence[int],
                      structure: M.Structure = STRUCTURE) -> Tensor:
    """[B, heads, K, 8] float32: the turned score q.k / sqrt(head_dim) of one query against K keys, summed over the
    pairs of each group of ``GROUPS`` (pair i = numbers i and i + head_dim / 2; the pairs added in increasing order)."""
    half = structure.head_dim // 2
    qq = q[:, :, int(query_index)][:, :, None, :]                          # [B, H, 1, D]
    kk = k[:, :, list(key_indices)]                                         # [B, H, K, D]
    per = (qq[..., :half] * kk[..., :half] + qq[..., half:] * kk[..., half:]) / math.sqrt(structure.head_dim)
    members = {"row": structure.row, "column": structure.column, "content": structure.content, "held": structure.held}
    members.update({lv.name: lv.pairs for lv in structure.levels})
    out = []
    for g in GROUPS:
        s = torch.zeros(per.shape[:-1], dtype=per.dtype, device=per.device)
        for p in members[g]:
            s = s + per[..., p]
        out.append(s)
    return torch.stack(out, dim=-1)


@torch.no_grad()
def find_read(engine: MemberEngine, variant: Variant, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor,
              points: Tensor, scene: int, cell: int, ask_a: int, ask_b: int, block: int,
              told: Optional[Tensor] = None, visible: Optional[Tensor] = None,
              addresses: Optional[Mapping[int, Sequence[int]]] = None, vary: str = "query") -> dict:
    """FIND at one attention site (see the module's text). ``cell`` holds the pointing request; ``ask_a``, ``ask_b``
    stand once each in the scene. ``told`` and ``visible`` are rows for the one scene ([1, 9], [1, 144]) or None
    (every request told; every image token visible).

    ``vary`` "query": the two runs differ ONLY in the query angles of the pointing slot and its cell's image tokens
    (and the told head's queries), the content and every key of ask_a's run kept; the block's input is captured once,
    in ask_a's forward, and both scores are recomputed from it. ``vary`` "content" (only for identity "content"): the
    two runs differ ONLY in the pointing slot's content (its asked-identity numbers), every angle of ask_a's run kept;
    the block's input is captured in each of the two forwards and each run's scores come from its own input, so at a
    block after the first the keys of the two runs differ too (only block 0's keys are the same).

    Returns the groups' names, the candidate slots, scores_a and scores_b [heads, 2, 8], weights_a and weights_b
    [heads, 2] (the softmax weight of the pointing slot's query on each candidate), answer_a and answer_b [16, 48] (the
    raw answers on the cell's tokens) and inputs_a, inputs_b (the two runs' ``forward_tokens`` inputs)."""
    if vary not in ("query", "content"):
        raise ValueError("vary is 'query' or 'content'")
    if vary == "content" and variant.identity != "content":
        raise ValueError(f"vary='content' reads an identity carried in the content; {variant.name} carries it "
                         f"by {variant.identity!r}")
    dev = z.device
    s = slice(int(scene), int(scene) + 1)
    c, k, p = cells[s].to(dev).clone(), colors[s].to(dev).clone(), points[s].to(dev).clone()
    if int(c[0, cell]) < 0:
        raise ValueError("the pointing cell holds no request")
    slot_a, slot_b = _the_cell_of(c[0], ask_a), _the_cell_of(c[0], ask_b)
    pa, pb = p.clone(), p.clone()
    pa[0, cell], pb[0, cell] = int(ask_a), int(ask_b)
    prep_a = prepare(variant, c, k, pa, told, visible, addresses)
    prep_b0 = prepare(variant, c, k, pb, told, visible, addresses)
    if not (torch.equal(prep_a["key_angles"], prep_b0["key_angles"]) and torch.equal(prep_a["key_mask"],
                                                                                   prep_b0["key_mask"])):
        raise RuntimeError("the keys must not depend on the asked address")
    if vary == "query":
        prep_b = dict(prep_a, query_angles=prep_b0["query_angles"])
        if prep_a["told_angles"] is not None:
            prep_b["told_angles"] = (prep_b0["told_angles"][0], prep_a["told_angles"][1])
    else:
        prep_b = dict(prep_a, content=prep_b0["content"])
    attn = engine.blocks[int(block)].attn

    def run(prep: dict) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        captured = {}

        def hook(module, args):
            captured["x"], captured["mask"] = args[0].detach(), (args[2] if len(args) > 2 else None)

        handle = attn.register_forward_pre_hook(hook)
        try:
            out = engine.forward_tokens(z[s], t[s], **prep)
        finally:
            handle.remove()
        return out, captured["x"], captured["mask"]

    out_a, x_a, km = run(prep_a)
    out_b, x_b, _ = run(prep_b)
    if vary == "query":
        x_b = x_a                                       # one captured input; only the query angles differ
    result = {"groups": GROUPS, "candidates": (slot_a, slot_b), "vary": vary, "inputs_a": prep_a, "inputs_b": prep_b}
    for tag, prep, x in (("a", prep_a, x_a), ("b", prep_b, x_b)):
        n = x.shape[1]
        qkv = attn.qkv(x).view(1, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        rot = block_rotary(prep, block)
        q, kk = rot(qkv[0], qkv[1])
        result["scores_" + tag] = pair_group_scores(q, kk, int(cell), (slot_a, slot_b))[0]
        wts = M.attention_weights(attn, x, rot, km)                             # [1, H, N, N]
        result["weights_" + tag] = wts[0, :, int(cell)][:, [slot_a, slot_b]]
    toks = _tokens_of(cell, dev)
    result["answer_a"] = patchify(out_a, PATCH)[0, toks]
    result["answer_b"] = patchify(out_b, PATCH)[0, toks]
    return result


def _gate_on(col: MemberCollective, cells_row: Tensor, toks: Tensor, request_addresses=None) -> Dict[str, Tensor]:
    g = col.gate_parts(cells_row, request_addresses)
    return {k: v[0, toks] for k, v in g.items()}


def serve_read(col: MemberCollective, part_answers: Tensor, base_answer: Tensor, cells: Tensor, scene: int,
               cell: int, new_class: int) -> dict:
    """SERVE (structural): the gate's numbers on ``cell``'s tokens before the edit cells[scene, cell] = new_class,
    after it, and after the reverse edit; the served values from the given answers; the given raw answers on the cell
    (never recomputed)."""
    dev = part_answers.device
    toks = _tokens_of(cell, dev)
    row = cells[int(scene):int(scene) + 1].to(dev).clone()
    old = int(row[0, cell])
    if old < 0:
        raise ValueError("the edited cell holds no request")
    after = row.clone()
    after[0, cell] = int(new_class)
    back = after.clone()
    back[0, cell] = old
    s = slice(int(scene), int(scene) + 1)

    def numbers(c: Tensor) -> Dict[str, Tensor]:
        g = col.gate_parts(c)
        served = M.serve(base_answer[s], part_answers[:, s], g["w"], g["base"])[0, toks]
        out = {k: v[0, toks] for k, v in g.items()}
        out["served"] = served
        return out

    return {"old_class": old, "new_class": int(new_class), "before": numbers(row), "after": numbers(after),
            "reverse": numbers(back), "answers": part_answers[:, int(scene)][:, toks],
            "base_answer": base_answer[int(scene)][toks]}


def _looks(col: MemberCollective, cls: int, z: Tensor, t: Tensor, c: Tensor, k: Tensor, p: Tensor, cell: int,
           unturned: bool) -> Optional[Tensor]:
    """[depth, heads]: the mean over the cell's 16 image-token queries of the softmax weight on the cell's slot."""
    if cls not in col.things:
        return None
    eng = col.things[cls]
    prep = col.engine_inputs(cls, c, k, p, unturned, cell=torch.tensor([int(cell)], device=c.device))
    q_idx = SLOTS + _tokens_of(cell, c.device)
    with M.record_looks(eng.blocks) as looks:
        eng.forward_tokens(z, t, **prep)
    return torch.stack([e["weights"][0][:, q_idx, int(cell)].mean(dim=-1) for e in looks])


@torch.no_grad()
def address_swap(col: MemberCollective, z: Tensor, t: Tensor, cells: Tensor, colors: Tensor, points: Tensor,
                 scene: int, cell: int, new_class: int, turn: bool = True, gate: str = "aleph") -> dict:
    """The swap of one request's thing (see the module's text); ``turn=False`` runs the engines unturned, ``gate=
    "neutral"`` returns the raw answers on the cell in place of the gate's numbers and the contributions. A scene in
    which a pointer asks the old or the new class is refused with a ValueError before any forward (the swap would leave
    that pointer asking a class that stands no longer, or twice, or its own)."""
    if gate not in ("aleph", "neutral"):
        raise ValueError("gate is 'aleph' or 'neutral'")
    dev = z.device
    s = slice(int(scene), int(scene) + 1)
    c, k, p = cells[s].to(dev).clone(), colors[s].to(dev).clone(), points[s].to(dev).clone()
    old = int(c[0, cell])
    if old < 0:
        raise ValueError("the edited cell holds no request")
    asking = [(int(a), int(p[0, a])) for a in range(N_CELLS) if int(p[0, a]) in (old, int(new_class))]
    if asking:
        raise ValueError(f"the swap of cell {cell} from class {old} to class {int(new_class)} would break a pointer: "
                         f"(asking cell, asked class) {asking} asks the old or the new class; pick a scene without one")
    c2 = c.clone()
    c2[0, cell] = int(new_class)
    check_layout(c2, k, p)                                       # before any forward
    toks = _tokens_of(cell, dev)
    unturned = not turn
    out = {"old_class": old, "new_class": int(new_class)}
    states = {}
    for tag, cc in (("before", c), ("after", c2)):
        st = {"looks_old": _looks(col, old, z[s], t[s], cc, k, p, cell, unturned),
              "looks_new": _looks(col, int(new_class), z[s], t[s], cc, k, p, cell, unturned)}
        answers, base_answer = col.part_answers(z[s], t[s], cc, k, p, unturned)
        if gate == "neutral":
            st["answers"] = answers[:, 0, toks]
            st["base_answer"] = base_answer[0, toks]
        else:
            g = col.gate_parts(cc)
            st["gate"] = {kk: v[0, toks] for kk, v in g.items()}
            st["w_all"] = g["w"][0]
            w = g["w"]
            parts = [w[..., i].to(answers.dtype)[..., None] * answers[i] for i in range(answers.shape[0])]
            st["contributions"] = torch.stack(parts)[:, 0]                  # [M, 144, 48]
        states[tag] = st
    out.update(states)
    if gate == "aleph":
        rel = {}
        wb = states["before"]["w_all"]
        for i, kc in enumerate(col.classes):
            if kc in (old, int(new_class)):
                continue
            owned = wb[:, i] > 0.5
            if not bool(owned.any()):
                continue
            a = states["before"]["contributions"][i][owned]
            b = states["after"]["contributions"][i][owned]
            rel[kc] = float((b - a).pow(2).mean().sqrt() / a.pow(2).mean().sqrt().clamp_min(1e-30))
        out["other_relative_change"] = rel
    return out


def address_missing(col: MemberCollective, cells: Tensor, scene: int, cell: int) -> dict:
    """The request of ``cell`` removed: the gate's numbers on its tokens."""
    row = cells[int(scene):int(scene) + 1].clone()
    row[0, cell] = -1
    return _gate_on(col, row, _tokens_of(cell, row.device))


def address_unheld(col: MemberCollective, cells: Tensor, scene: int, cell: int) -> dict:
    """The request's class placed on the request side at the first address (d0, d1), digits 0..3 in increasing order,
    that no member holds: the gate's numbers on the cell's tokens, with the address used."""
    row = cells[int(scene):int(scene) + 1].clone()
    cls = int(row[0, cell])
    if cls < 0:
        raise ValueError("the cell holds no request")
    held = {tuple(int(d) for d in a) for a in col.member_addresses.tolist()}
    free = next((d0, d1) for d0 in range(NULL) for d1 in range(NULL) if (d0, d1) not in held)
    mapping = dict(col.addresses)
    mapping[cls] = free
    out = _gate_on(col, row, _tokens_of(cell, row.device), mapping)
    out["address"] = free
    return out


@torch.no_grad()
def exchange_read(col_a: MemberCollective, col_b: MemberCollective, z: Tensor, t: Tensor, cells: Tensor,
                  colors: Tensor, points: Optional[Tensor] = None) -> dict:
    """Two collectives over the same classes that differ in their engines: the exchanged classes, and for every other
    member (and the base, when it is not exchanged) the largest absolute difference of its contribution."""
    if col_a.classes != col_b.classes:
        raise ValueError("the two collectives hold different classes")
    exchanged = [kc for kc in col_a.classes if col_a.things[kc] is not col_b.things[kc]]
    ca = col_a.contributions(z, t, cells, colors, points)
    cb = col_b.contributions(z, t, cells, colors, points)
    others = {kc: float((ca[i] - cb[i]).abs().max()) for i, kc in enumerate(col_a.classes) if kc not in exchanged}
    base_same = col_a.base is col_b.base
    if base_same:
        others["base"] = float((ca[-1] - cb[-1]).abs().max())
    return {"exchanged": exchanged + ([] if base_same else ["base"]), "others": others}
