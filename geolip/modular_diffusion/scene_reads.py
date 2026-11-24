"""The reads that judge a made scene of the 3 x 3 grid (``scenes``), as functions on tensors (torch only).

Labels: things 0..8 and NOTHING = 9 (what a judge reads in an empty cell); colors 0..8. A layout is cells [B, 9] and
colors [B, 9] (-1 where a cell has no request).

    score_cells        per cell: THING RIGHT and COLOR RIGHT (requested cells), EMPTY RIGHT (cells without a request:
                       the judge reads NOTHING); per scene: the number of right requested cells (on the thing read),
                       SCENE RIGHT (all nine cells right on the thing read), SCENE RIGHT WITH COLORS (and every
                       requested cell's color right)
    summarize          CELL RATE, COLOR RATE (over requested cells), SCENE RIGHT, SCENE RIGHT WITH COLORS (over scenes),
                       GHOSTS (the share of cells without a request not read as NOTHING), the histogram of right
                       requested cells, and BY NEIGHBOR; each rate {"mean", "se", "n"} with se = sqrt(p (1 - p) / n)
    neighbor_groups    a requested cell's group by its edge neighbors (4-neighborhood), the first that applies: "same"
                       (a neighbor of the same class), "family" (a neighbor of another class of its family), "other"
                       (only neighbors of other families), "none" (no neighbor holds a thing)
    paired_difference  two rows' per-list 0/1 results on the same lists: the mean of a - b and its standard error
                       sd / sqrt(n) (sd with ddof 1)
    tint_spread        a cell's tint = the mean of its four 2 x 2 corner patches (16 pixels no thing reaches), per
                       channel; the spread = the population standard deviation across the nine cells, averaged over the
                       channels
    coherence_index    (spread - real) / (shuffled - real)
    shuffle_cells      cell c of scene b taken from scene (b + c) mod B: nine different scenes (B >= 9)
    seam               across each of the 12 inner cell borders, on the two pixels at each end of the border segment
                       (4 pixel rows a border): ACROSS = |p(first pixel past the border) - p(last pixel before it)|,
                       BESIDE = the mean of the two steps one pixel to either side; both averaged over the 4 rows and
                       the channels a border; by border type (base and base, thing and base, thing and thing) the ratio
                       of the pooled means, mean ACROSS / mean BESIDE over every border of that type in the batch
    ghost_ink          the share of pixels of cells without a request more than 0.1 from 0.5 in any channel (pooled)
    corrupt_requests   each request independently with probability e: MOVED to a uniform free cell, its CLASS replaced
                       by a uniform other class, or DROPPED, each a third
    derange_classes    the fixed derangement k -> (k + 4) mod 9 on every request

``corrupt_requests`` draws on the CPU from the generator given, in the order u [B, 9] (rand: corrupted when u < e),
kind [B, 9] (randint 0..2), v [B, 9] (rand: the free cell floor(v * free count) in cell order), other [B, 9]
(randint 0..7: the other class, other + 1 when other >= the class); drawn for all nine cells; requests are treated in
cell order of the given list, a cell being free when it holds no request in the list at that moment.
"""
from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Sequence

import torch
from torch import Tensor

from .scenes import CELL, CELLS, N_CELLS, cell_crops, from_cell_crops

NOTHING = 9
NEIGHBOR_GROUPS = ("same", "family", "other", "none")
BORDER_TYPES = ("base_base", "thing_base", "thing_thing")
DERANGEMENT_SHIFT = 4
CORRUPTIONS = ("kept", "moved", "class", "dropped", "stuck")    # "stuck": chosen to move with no free cell, left as it is
GHOST_TOLERANCE = 0.1


def rate(values: Tensor) -> dict:
    """{"mean", "se", "n"} of a 0/1 tensor (se = sqrt(p (1 - p) / n)); None and n = 0 when it is empty."""
    v = torch.as_tensor(values).reshape(-1).double()
    n = int(v.numel())
    if not n:
        return {"mean": None, "se": None, "n": 0}
    p = float(v.mean())
    return {"mean": p, "se": math.sqrt(max(p * (1 - p), 0.0) / n), "n": n}


def _layout(cells: Tensor, colors: Optional[Tensor] = None):
    cells = torch.as_tensor(cells, dtype=torch.long)
    if cells.dim() != 2 or cells.shape[1] != N_CELLS:
        raise ValueError("a layout is cells [B, 9]")
    if colors is None:
        return cells
    colors = torch.as_tensor(colors, dtype=torch.long, device=cells.device)
    if colors.shape != cells.shape:
        raise ValueError("colors is [B, 9] like cells")
    return cells, colors


# ------------------------------------------------------------------ scores
def score_cells(pred_things: Tensor, pred_colors: Tensor, cells: Tensor, colors: Tensor) -> Dict[str, Tensor]:
    """-> requested, thing_right, color_right, empty_right, cell_right [B, 9] bool; n_requested, n_right [B] long;
    scene_right, scene_right_colors [B] bool. thing_right and color_right are False on cells without a request,
    empty_right is False on requested cells."""
    cells, colors = _layout(cells, colors)
    pt = torch.as_tensor(pred_things, dtype=torch.long, device=cells.device)
    pc = torch.as_tensor(pred_colors, dtype=torch.long, device=cells.device)
    if pt.shape != cells.shape or pc.shape != cells.shape:
        raise ValueError("the judge's reads are [B, 9] like the layout")
    req = cells >= 0
    thing_right = req & (pt == cells)
    color_right = req & (pc == colors)
    empty_right = ~req & (pt == NOTHING)
    cell_right = thing_right | empty_right
    scene_right = cell_right.all(dim=1)
    return {"requested": req, "thing_right": thing_right, "color_right": color_right, "empty_right": empty_right,
            "cell_right": cell_right, "n_requested": req.sum(dim=1), "n_right": thing_right.sum(dim=1),
            "scene_right": scene_right, "scene_right_colors": scene_right & (color_right | ~req).all(dim=1)}


_NEIGHBORS = tuple(tuple(r * CELLS + c for r, c in ((k - 1, j), (k + 1, j), (k, j - 1), (k, j + 1))
                         if 0 <= r < CELLS and 0 <= c < CELLS)
                   for k in range(CELLS) for j in range(CELLS))


def neighbor_groups(cells: Tensor) -> Tensor:
    """[B, 9] long: the index into NEIGHBOR_GROUPS of every requested cell (-1 on cells without a request)."""
    cells = _layout(cells)
    out = torch.full_like(cells, -1)
    for c in range(N_CELLS):
        nb = cells[:, list(_NEIGHBORS[c])]                    # [B, m]
        me = cells[:, c:c + 1]
        has = nb >= 0
        same = (has & (nb == me)).any(dim=1)
        family = (has & (nb != me) & (nb // 3 == me // 3)).any(dim=1)
        any_thing = has.any(dim=1)
        g = torch.where(same, 0, torch.where(family, 1, torch.where(any_thing, 2, 3)))
        out[:, c] = torch.where(cells[:, c] >= 0, g, torch.full_like(g, -1))
    return out


def summarize(pred_things: Tensor, pred_colors: Tensor, cells: Tensor, colors: Tensor) -> dict:
    """The rates of a row (see the module): cell_rate, color_rate, scene_right, scene_right_colors, ghosts (each
    {"mean", "se", "n"}), histogram (a list of 10 counts: scenes with 0..9 right requested cells), by_neighbor (a rate
    of THING RIGHT for each of NEIGHBOR_GROUPS) and scenes (the count)."""
    s = score_cells(pred_things, pred_colors, cells, colors)
    req = s["requested"]
    groups = neighbor_groups(cells)
    hist = torch.bincount(s["n_right"].cpu(), minlength=N_CELLS + 1)
    return {"cell_rate": rate(s["thing_right"][req]), "color_rate": rate(s["color_right"][req]),
            "scene_right": rate(s["scene_right"]), "scene_right_colors": rate(s["scene_right_colors"]),
            "ghosts": rate(~s["empty_right"][~req]), "histogram": [int(v) for v in hist.tolist()],
            "by_neighbor": {name: rate(s["thing_right"][groups == g]) for g, name in enumerate(NEIGHBOR_GROUPS)},
            "scenes": int(req.shape[0])}


def paired_difference(a: Tensor, b: Tensor) -> dict:
    """Two rows' per-list results on the same lists (in the same order) -> {"mean": mean of a - b, "se": sd / sqrt(n)
    (ddof 1; None for one list), "n"}."""
    a = torch.as_tensor(a).reshape(-1).double()
    b = torch.as_tensor(b).reshape(-1).double()
    if a.shape != b.shape:
        raise ValueError("a paired difference needs two rows on the same lists")
    d = a - b
    n = int(d.numel())
    if not n:
        return {"mean": None, "se": None, "n": 0}
    se = float(d.std(correction=1)) / math.sqrt(n) if n > 1 else None
    return {"mean": float(d.mean()), "se": se, "n": n}


# ------------------------------------------------------------------ what the picture shares
def cell_tints(pixels: Tensor) -> Tensor:
    """[B, C, 48, 48] -> [B, 9, C]: each cell's mean over its four 2 x 2 corner patches (16 pixels)."""
    cr = cell_crops(pixels)                                   # [B, 9, C, 16, 16]
    e = CELL - 2
    patches = [cr[..., 0:2, 0:2], cr[..., 0:2, e:CELL], cr[..., e:CELL, 0:2], cr[..., e:CELL, e:CELL]]
    return torch.stack([p.mean(dim=(-2, -1)) for p in patches], dim=0).mean(dim=0)


def tint_spread(pixels: Tensor) -> Tensor:
    """[B]: the population standard deviation of the cell tints across the nine cells, averaged over the channels."""
    return cell_tints(pixels).std(dim=1, correction=0).mean(dim=1)


def coherence_index(spread, real, shuffled):
    """(spread - real) / (shuffled - real): 0 = as coherent as real scenes, 1 = as incoherent as unrelated cells."""
    return (spread - real) / (shuffled - real)


def shuffle_cells(pixels: Tensor, cells: Optional[Tensor] = None):
    """SHUFFLED scenes: cell c of scene b is cell c of scene (b + c) mod B, so the nine cells of a scene come from nine
    different scenes (B >= 9). -> the pixels, or (pixels, cells) when a layout is given."""
    b = pixels.shape[0]
    if b < N_CELLS:
        raise ValueError("shuffled scenes need a batch of at least nine")
    cr = cell_crops(pixels)
    rows = (torch.arange(b, device=pixels.device)[:, None] + torch.arange(N_CELLS, device=pixels.device)[None]) % b
    cols = torch.arange(N_CELLS, device=pixels.device).expand(b, N_CELLS)
    out = from_cell_crops(cr[rows, cols])
    if cells is None:
        return out
    cells = _layout(cells)
    return out, cells[rows.to(cells.device), cols.to(cells.device)]


def _borders():
    """The 12 inner borders: (axis, cells a and b, the border coordinate, the 4 coordinates along it)."""
    out = []
    for k in range(CELLS):                 # vertical borders: between columns j and j + 1 in row k
        for j in range(CELLS - 1):
            along = [CELL * k, CELL * k + 1, CELL * k + CELL - 2, CELL * k + CELL - 1]
            out.append(("v", CELLS * k + j, CELLS * k + j + 1, CELL * (j + 1), along))
    for k in range(CELLS - 1):             # horizontal borders: between rows k and k + 1 in column j
        for j in range(CELLS):
            along = [CELL * j, CELL * j + 1, CELL * j + CELL - 2, CELL * j + CELL - 1]
            out.append(("h", CELLS * k + j, CELLS * (k + 1) + j, CELL * (k + 1), along))
    return tuple(out)


BORDERS = _borders()


def seam(pixels: Tensor, cells: Tensor) -> dict:
    """The seam read: {type: {"ratio", "across", "beside", "n"}} for each of BORDER_TYPES (pooled over every border of
    that type in the batch; ratio None when the type has no border), and "per_border": {"across", "beside" [B, 12],
    "type" [B, 12] long (an index into BORDER_TYPES)}."""
    cells = _layout(cells).to(pixels.device)
    across, beside, kinds = [], [], []
    for axis, a, b, x, along in BORDERS:
        idx = torch.tensor(along, device=pixels.device)
        if axis == "v":
            line = pixels[:, :, idx, :]                       # [B, C, 4, W]
            p = [line[..., x - 2], line[..., x - 1], line[..., x], line[..., x + 1]]
        else:
            line = pixels[:, :, :, idx]                       # [B, C, H, 4]
            p = [line[:, :, x - 2], line[:, :, x - 1], line[:, :, x], line[:, :, x + 1]]
        across.append((p[2] - p[1]).abs().mean(dim=(1, 2)))
        beside.append(0.5 * ((p[1] - p[0]).abs() + (p[3] - p[2]).abs()).mean(dim=(1, 2)))
        kinds.append((cells[:, a] >= 0).long() + (cells[:, b] >= 0).long())
    across_t, beside_t, kind_t = torch.stack(across, 1), torch.stack(beside, 1), torch.stack(kinds, 1)
    out = {}
    for g, name in enumerate(BORDER_TYPES):
        sel = kind_t == g
        n = int(sel.sum())
        if n:
            ac, be = float(across_t[sel].double().mean()), float(beside_t[sel].double().mean())
            out[name] = {"ratio": ac / be if be > 0 else None, "across": ac, "beside": be, "n": n}
        else:
            out[name] = {"ratio": None, "across": None, "beside": None, "n": 0}
    out["per_border"] = {"across": across_t, "beside": beside_t, "type": kind_t}
    return out


def ghost_ink(pixels: Tensor, cells: Tensor, ground: float = 0.5, tolerance: float = GHOST_TOLERANCE) -> dict:
    """{"share": the share of pixels of cells without a request more than ``tolerance`` from ``ground`` in any channel,
    pooled over the batch (None when no cell is free), "n": the pixels counted, "per_scene" [B] (nan where a scene has
    no free cell)}."""
    cells = _layout(cells).to(pixels.device)
    cr = cell_crops(pixels)
    off = ((cr - ground).abs() > tolerance).any(dim=2)        # [B, 9, 16, 16]
    free = cells < 0
    counts = (off & free[..., None, None]).sum(dim=(1, 2, 3)).double()
    pix = free.sum(dim=1).double() * CELL * CELL
    n = int(pix.sum())
    per = counts / pix
    return {"share": float(counts.sum()) / n if n else None, "n": n, "per_scene": per}


# ------------------------------------------------------------------ wrong addresses
class Corruption(NamedTuple):
    cells: Tensor      # [B, 9] long: the corrupted layout
    colors: Tensor     # [B, 9] long
    action: Tensor     # [B, 9] long, by the ORIGINAL cell: an index into CORRUPTIONS, -1 where the list asked nothing
    moved_to: Tensor   # [B, 9] long, by the original cell: the destination of a moved request, else -1


def corrupt_requests(cells: Tensor, colors: Tensor, e: float, gen: torch.Generator) -> Corruption:
    """THE WRONG-ADDRESS corruption of a batch of request lists (see the module for the kinds and the draw order)."""
    if gen.device.type != "cpu":
        raise ValueError("the corruption draws on the CPU: give a CPU generator")
    cells, colors = _layout(cells, colors)
    dev = cells.device
    b = cells.shape[0]
    u = torch.rand(b, N_CELLS, generator=gen)
    kind = torch.randint(0, 3, (b, N_CELLS), generator=gen)
    v = torch.rand(b, N_CELLS, generator=gen)
    other = torch.randint(0, 8, (b, N_CELLS), generator=gen)
    c_in, k_in = cells.cpu().tolist(), colors.cpu().tolist()
    uu, kk, vv, oo = u.tolist(), kind.tolist(), v.tolist(), other.tolist()
    e = float(e)
    out_c, out_k, act, dest = [], [], [], []
    for i in range(b):
        cur_c, cur_k = list(c_in[i]), list(k_in[i])
        a_row, d_row = [-1] * N_CELLS, [-1] * N_CELLS
        for c in range(N_CELLS):
            t = c_in[i][c]
            if t < 0:
                continue
            a_row[c] = 0
            if not uu[i][c] < e:
                continue
            kd = kk[i][c]
            if kd == 0:
                # the request sits at cell c in the current list (requests are moved only when treated, in cell order)
                free = [x for x in range(N_CELLS) if cur_c[x] < 0]
                if not free:
                    a_row[c] = 4
                    continue
                to = free[min(int(vv[i][c] * len(free)), len(free) - 1)]
                cur_c[to], cur_k[to] = cur_c[c], cur_k[c]
                cur_c[c], cur_k[c] = -1, -1
                a_row[c], d_row[c] = 1, to
            elif kd == 1:
                o = oo[i][c]
                cur_c[c] = o + 1 if o >= t else o
                a_row[c] = 2
            else:
                cur_c[c], cur_k[c] = -1, -1
                a_row[c] = 3
        out_c.append(cur_c)
        out_k.append(cur_k)
        act.append(a_row)
        dest.append(d_row)
    mk = lambda rows: torch.tensor(rows, dtype=torch.long, device=dev).reshape(b, N_CELLS)
    return Corruption(mk(out_c), mk(out_k), mk(act), mk(dest))


def derange_classes(cells: Tensor) -> Tensor:
    """Control (c)'s layout: every request's class k becomes (k + 4) mod 9 (no class stays); empty cells stay -1."""
    cells = _layout(cells)
    return torch.where(cells >= 0, (cells + DERANGEMENT_SHIFT) % 9, cells)
