"""Scenes made by code: a 3 x 3 grid of cells, each holding one small thing or nothing, with every layout known exactly.
Torch only.

THE GEOMETRY. A canvas of 48 x 48 pixels and 3 channels, cut into a 3 x 3 grid of cells of 16 pixels. Pixel (iy, ix)
has its center at (ix + 0.5, iy + 0.5) (x to the right, y down). Cell c = 3 * row + column (column 0 = left, row 0 =
top), the order of ``layout.COLUMNS`` and ``layout.ROWS``; the center of cell (column j, row k) is (16 j + 8, 16 k + 8).
With patch 4 the token grid is 12 x 12 and a cell is 4 x 4 tokens.

THE NINE THINGS, index = 3 * family + member: disc, ring, crescent (family 0, round); square, triangle, diamond (family
1, angular); plus, cross, star (family 2, crossing). THE NINE COLORS, index = 3 * high + low (``COLOR_VALUES``).

A LAYOUT is two long tensors: ``cells [B, 9]`` (the thing index of each cell, -1 for none) and ``colors [B, 9]`` (the
color index, -1 where the cell is empty).

THE DRAW OF A THING. Its center is the cell's center plus a shift uniform on -1.5 to 1.5 px on each axis; its scale s is
uniform on 0.85 to 1.15; its angle uniform on -15 to 15 degrees; its color is the table value plus a shade shift uniform
on -0.04 to 0.04 in each channel, clipped to 0..1. A shape is a signed distance in pixels in the thing's own frame
(p = the pixel minus the thing's center, turned by minus the angle: p_x = cos(a) d_x + sin(a) d_y, p_y = -sin(a) d_x +
cos(a) d_y). The outer radius is R = 5.5 s and every point of a thing lies within R of its center:

    disc       |p| - R
    ring       | |p| - (R - 0.8 s) | - 0.8 s                    a band of width 1.6 s whose outer edge is at R
    crescent   max(|p| - R, R - |p - (0.45 R, 0)|)               the disc minus an equal disc shifted along x
    square     corners at distance R, at 45 + 90 k degrees
    triangle   equilateral, corners at distance R, at -90, 30, 150 degrees (one corner up)
    diamond    (0, -R), (R/2, 0), (0, R), (-R/2, 0)              half-height R, half-width R / 2
    plus       two bars of half-width h = 1.1 s; the 12 outline corners (+-L, +-h), (+-h, +-L) with L = sqrt(R^2 - h^2),
               so the far corners lie at distance R
    cross      the plus turned 45 degrees
    star       10 corners at -90 + 36 k degrees, alternating radius R (k even) and 0.45 R (k odd)

The six polygons go through one exact signed-distance routine (``polygon_distance``: the distance to the nearest edge,
negative inside by the crossing-number test). Coverage = clamp(0.5 - distance, 0, 1): one pixel of soft edge.

THE PICTURE. Bed "F": the ground is 0.5 in every channel. Bed "C": a tint (each channel uniform on 0.40 to 0.60, one
draw a scene) plus a texture: white noise (one field a scene, shared by the three channels) blurred by a Gaussian of
standard deviation 6 px (a separable kernel of radius 18, normalized to sum one, reflect padding), then multiplied so
that its own standard deviation over the canvas (population form) is 0.04; no mean is taken off. Pixel = coverage x
color + (1 - coverage) x ground, clipped to 0..1. A thing reaches at most 1.5 + 5.5 x 1.15 + 0.5 = 8.325 px from its
cell's center along an axis, so it never leaves its cell (the nearest pixel center of a neighbor is 8.5 px away), and
at most 8.95 px in any direction, so the four 2 x 2 corner patches of a cell (9.19 px away and more) are always ground.
The coverage of a thing is computed on a window of its cell plus ``margin`` pixels on each side (default 1, the
smallest window that holds every pixel a thing could reach); things of different cells never overlap, so the colored
parts are added up.

Model space = (pixel - 0.5) / 0.25 (``to_model``; ``to_pixels`` goes back).

THE DRAW ORDER (every random number is drawn on the CPU from the generator given, then moved to the device):
    the layout (``draw_layouts``):
        one        the classes [B] (only when no class is given), the cells [B], the colors [B]   (randint 0..8)
        composed   keys [B, 9] (rand; the cells are the first n of argsort(keys), a stable sort), classes [B, 9], colors
                   [B, 9] (randint 0..8; request j takes column j of both)
        mix        n [B] (randint 0..9), then as composed with n per scene
        empty      nothing
    the render (``render``), drawn for all nine cells whether or not they hold a thing:
        shift x [B, 9], shift y [B, 9], scale [B, 9], angle [B, 9] (rand, mapped to their ranges), shade [B, 9, 3];
        then on bed C only: the tint [B, 3] (rand), the noise [B, 48, 48] (randn)
    ``request_lists``: keys [count, 9] (rand; the first n of argsort), classes [count, n], colors [count, n] (randint).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

CANVAS = 48
CHANNELS = 3
CELLS = 3                      # cells on each axis
CELL = 16                      # pixels on a cell's side
N_CELLS = CELLS * CELLS
PATCH = 4
TOKEN_GRID = CANVAS // PATCH   # 12

THINGS = ("disc", "ring", "crescent", "square", "triangle", "diamond", "plus", "cross", "star")
FAMILIES = ("round", "angular", "crossing")
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black")
COLOR_VALUES = ((0.90, 0.10, 0.10), (0.95, 0.55, 0.10), (0.95, 0.90, 0.15), (0.15, 0.70, 0.20), (0.15, 0.30, 0.90),
                (0.55, 0.20, 0.75), (0.95, 0.55, 0.75), (0.50, 0.30, 0.12), (0.05, 0.05, 0.05))

GROUND = 0.5
SHIFT = 1.5
SCALE_RANGE = (0.85, 1.15)
ANGLE = 15.0                   # degrees, either way
SHADE = 0.04
RADIUS = 5.5                   # the outer radius at scale 1
RING_WIDTH = 1.6
CRESCENT_SHIFT = 0.45
BAR_HALF = 1.1
STAR_INNER = 0.45
TINT_RANGE = (0.40, 0.60)
TEXTURE_SIGMA = 6.0
TEXTURE_RADIUS = 18            # the blur kernel runs over -18..18 (three standard deviations)
TEXTURE_STD = 0.04
MARGIN = 1

BEDS = ("F", "C")
KINDS = ("empty", "one", "composed", "mix")


def family_of(thing: int) -> int:
    return int(thing) // 3


def member_of(thing: int) -> int:
    return int(thing) % 3


# ------------------------------------------------------------------ model space
def to_model(pixels: Tensor) -> Tensor:
    """Pixels in 0..1 -> model space (pixel - 0.5) / 0.25."""
    return (pixels - 0.5) / 0.25


def to_pixels(x: Tensor) -> Tensor:
    """Model space -> pixels, x * 0.25 + 0.5 (the inverse of ``to_model``; nothing is clipped)."""
    return x * 0.25 + 0.5


# ------------------------------------------------------------------ cells
def cell_crops(x: Tensor) -> Tensor:
    """[B, C, 48, 48] -> [B, 9, C, 16, 16], the cells in cell order (3 * row + column)."""
    b, c, h, w = x.shape
    if h != CANVAS or w != CANVAS:
        raise ValueError(f"a scene is {CANVAS} x {CANVAS} pixels, not {h} x {w}")
    y = x.reshape(b, c, CELLS, CELL, CELLS, CELL).permute(0, 2, 4, 1, 3, 5)
    return y.reshape(b, N_CELLS, c, CELL, CELL)


def from_cell_crops(crops: Tensor) -> Tensor:
    """[B, 9, C, 16, 16] -> [B, C, 48, 48] (the inverse of ``cell_crops``)."""
    b, n, c, h, w = crops.shape
    if n != N_CELLS or h != CELL or w != CELL:
        raise ValueError("crops are [B, 9, C, 16, 16]")
    y = crops.reshape(b, CELLS, CELLS, c, CELL, CELL).permute(0, 3, 1, 4, 2, 5)
    return y.reshape(b, c, CANVAS, CANVAS)


def cell_centers() -> Tensor:
    """[9, 2] float32: the (x, y) center of every cell in pixels."""
    c = torch.arange(N_CELLS)
    return torch.stack([(c % CELLS) * CELL + CELL / 2, (c // CELLS) * CELL + CELL / 2], dim=1).float()


# ------------------------------------------------------------------ shapes
def polygon_corners(thing: int) -> Tensor:
    """[M, 2] float64: the corners of a polygon thing at R = 1, in outline order (thing 3..8)."""
    t = int(thing)
    if t == 3:
        a = [math.radians(45 + 90 * k) for k in range(4)]
        pts = [(math.cos(x), math.sin(x)) for x in a]
    elif t == 4:
        a = [math.radians(d) for d in (-90, 30, 150)]
        pts = [(math.cos(x), math.sin(x)) for x in a]
    elif t == 5:
        pts = [(0.0, -1.0), (0.5, 0.0), (0.0, 1.0), (-0.5, 0.0)]
    elif t in (6, 7):
        h = BAR_HALF / RADIUS
        L = math.sqrt(1.0 - h * h)
        pts = [(L, h), (h, h), (h, L), (-h, L), (-h, h), (-L, h), (-L, -h), (-h, -h), (-h, -L), (h, -L), (h, -h),
               (L, -h)]
        if t == 7:
            c = s = math.sqrt(0.5)
            pts = [(x * c - y * s, x * s + y * c) for x, y in pts]
    elif t == 8:
        pts = []
        for k in range(10):
            a = math.radians(-90 + 36 * k)
            r = 1.0 if k % 2 == 0 else STAR_INNER
            pts.append((r * math.cos(a), r * math.sin(a)))
    else:
        raise ValueError(f"thing {thing} is not a polygon")
    return torch.tensor(pts, dtype=torch.float64)


def polygon_distance(px: Tensor, py: Tensor, corners: Tensor) -> Tensor:
    """The exact signed distance from points to polygons: px, py [K, ...] (the points of item k), corners [K, M, 2]
    (one polygon an item, in outline order) -> [K, ...]: the distance to the nearest edge, negative inside (an odd
    number of edge crossings of the ray to +x)."""
    k = px.shape[0]
    view = (k,) + (1,) * (px.dim() - 1)
    m = corners.shape[1]
    best = None
    inside = torch.zeros_like(px, dtype=torch.bool)
    for e in range(m):
        ax, ay = corners[:, e, 0].view(view), corners[:, e, 1].view(view)
        bx, by = corners[:, (e + 1) % m, 0].view(view), corners[:, (e + 1) % m, 1].view(view)
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        ee = (ex * ex + ey * ey).clamp_min(1e-30)
        tt = ((wx * ex + wy * ey) / ee).clamp(0.0, 1.0)
        dx, dy = wx - ex * tt, wy - ey * tt
        dd = dx * dx + dy * dy
        best = dd if best is None else torch.minimum(best, dd)
        straddle = (ay > py) != (by > py)
        safe = torch.where(ey == 0, torch.ones_like(ey), ey)
        xcross = ax + ex * (py - ay) / safe
        inside = inside ^ (straddle & (px < xcross))
    dist = best.sqrt()
    return torch.where(inside, -dist, dist)


def thing_distance(thing: int, px: Tensor, py: Tensor, scale: Tensor) -> Tensor:
    """The signed distance in pixels of ``thing`` at points (px, py) [K, ...] of the thing's own frame, scales [K]."""
    t = int(thing)
    view = (px.shape[0],) + (1,) * (px.dim() - 1)
    s = scale.view(view)
    r = RADIUS * s
    if t == 0:
        return torch.sqrt(px * px + py * py) - r
    if t == 1:
        half = 0.5 * RING_WIDTH * s
        return (torch.sqrt(px * px + py * py) - (r - half)).abs() - half
    if t == 2:
        qx = px - CRESCENT_SHIFT * r
        return torch.maximum(torch.sqrt(px * px + py * py) - r, r - torch.sqrt(qx * qx + py * py))
    corners = polygon_corners(t).to(device=px.device, dtype=px.dtype)
    corners = corners[None] * (RADIUS * scale.to(px.dtype)).view(-1, 1, 1)
    return polygon_distance(px, py, corners)


def coverage(distance: Tensor) -> Tensor:
    """clamp(0.5 - distance, 0, 1): one pixel of soft edge."""
    return (0.5 - distance).clamp(0.0, 1.0)


# ------------------------------------------------------------------ layouts
def _check_gen(gen: torch.Generator) -> None:
    if gen.device.type != "cpu":
        raise ValueError("scenes draw every random number on the CPU: give a CPU generator")


def _place(order: Tensor, n_per: Tensor, classes: Tensor, colors: Tensor) -> Tuple[Tensor, Tensor]:
    """Request j < n_per[b] of scene b goes to cell order[b, j] with classes[b, j] and colors[b, j]."""
    b = order.shape[0]
    j = torch.arange(N_CELLS).expand(b, N_CELLS)
    used = j < n_per[:, None]
    cells = torch.full((b, N_CELLS), -1, dtype=torch.long)
    cols = torch.full((b, N_CELLS), -1, dtype=torch.long)
    cells.scatter_(1, order, torch.where(used, classes, torch.full_like(classes, -1)))
    cols.scatter_(1, order, torch.where(used, colors, torch.full_like(colors, -1)))
    return cells, cols


def draw_layouts(kind: str, n: int, gen: torch.Generator, only_class: Optional[int] = None,
                 count: Optional[int] = None, allowed_cells: Optional[Sequence[int]] = None) -> Tuple[Tensor, Tensor]:
    """A batch of n layouts of a scene kind -> (cells [n, 9], colors [n, 9]) long on the CPU.

    empty: no thing. one: one thing (of class ``only_class`` when given, else a uniform class) at a uniform cell in a
    uniform color. composed: ``count`` distinct uniform cells (1..9), each a uniform class and color (repeats allowed).
    mix: n uniform on 0..9 per scene, then composed. The draw order is the module's. ``allowed_cells`` (kind 'one'
    only): the cell is uniform over this list instead (its index drawn by randint over the list's length, in the cell
    draw's place); None is the default path, unchanged."""
    _check_gen(gen)
    if kind not in KINDS:
        raise ValueError(f"the scene kinds are {KINDS}, not {kind!r}")
    if kind != "one" and only_class is not None:
        raise ValueError("only_class applies to the kind 'one'")
    if kind != "one" and allowed_cells is not None:
        raise ValueError("allowed_cells applies to the kind 'one'")
    if kind != "composed" and count is not None:
        raise ValueError("count applies to the kind 'composed'")
    if kind == "empty":
        return (torch.full((n, N_CELLS), -1, dtype=torch.long), torch.full((n, N_CELLS), -1, dtype=torch.long))
    if kind == "one":
        if only_class is None:
            cls = torch.randint(0, len(THINGS), (n,), generator=gen)
        else:
            if not 0 <= int(only_class) < len(THINGS):
                raise ValueError(f"a class is 0..8, not {only_class}")
            cls = torch.full((n,), int(only_class), dtype=torch.long)
        if allowed_cells is None:
            where = torch.randint(0, N_CELLS, (n,), generator=gen)
        else:
            allow = torch.tensor(sorted({int(c) for c in allowed_cells}), dtype=torch.long)
            if allow.numel() == 0 or int(allow.min()) < 0 or int(allow.max()) >= N_CELLS:
                raise ValueError("allowed_cells is a non-empty list of cells 0..8")
            where = allow[torch.randint(0, allow.numel(), (n,), generator=gen)]
        col = torch.randint(0, len(COLORS), (n,), generator=gen)
        cells = torch.full((n, N_CELLS), -1, dtype=torch.long)
        cols = torch.full((n, N_CELLS), -1, dtype=torch.long)
        rows = torch.arange(n)
        cells[rows, where] = cls
        cols[rows, where] = col
        return cells, cols
    if kind == "composed":
        if count is None or not 1 <= int(count) <= N_CELLS:
            raise ValueError("a composed scene needs count in 1..9")
        n_per = torch.full((n,), int(count), dtype=torch.long)
    else:
        n_per = torch.randint(0, N_CELLS + 1, (n,), generator=gen)
    keys = torch.rand(n, N_CELLS, generator=gen)
    order = torch.argsort(keys, dim=1, stable=True)
    cls = torch.randint(0, len(THINGS), (n, N_CELLS), generator=gen)
    col = torch.randint(0, len(COLORS), (n, N_CELLS), generator=gen)
    return _place(order, n_per, cls, col)


class RequestLists(NamedTuple):
    cells: Tensor      # [count, 9] long, -1 where the list asks nothing
    colors: Tensor     # [count, 9] long
    order: Tensor      # [count, n] long: the requested cells in the order they were drawn


def request_lists(n: int, count: int = 500, seed: Optional[int] = None) -> RequestLists:
    """The request lists of the judged rows: for each of ``count`` lists, n distinct cells (the first n of a random
    permutation of the nine), each a uniform class and a uniform color; drawn from a CPU generator seeded ``seed``
    (default 5000 + n) in the order keys [count, 9], classes [count, n], colors [count, n]."""
    n = int(n)
    if not 0 <= n <= N_CELLS:
        raise ValueError("a list asks for 0..9 things")
    gen = torch.Generator().manual_seed(5000 + n if seed is None else int(seed))
    keys = torch.rand(count, N_CELLS, generator=gen)
    order = torch.argsort(keys, dim=1, stable=True)
    cls = torch.randint(0, len(THINGS), (count, n), generator=gen)
    col = torch.randint(0, len(COLORS), (count, n), generator=gen)
    pad = torch.full((count, N_CELLS - n), -1, dtype=torch.long)
    cells, colors = _place(order, torch.full((count,), n, dtype=torch.long), torch.cat([cls, pad], 1),
                           torch.cat([col, pad], 1))
    return RequestLists(cells, colors, order[:, :n].contiguous())


# ------------------------------------------------------------------ rendering
def _blur_kernel(device) -> Tensor:
    x = torch.arange(-TEXTURE_RADIUS, TEXTURE_RADIUS + 1, dtype=torch.float32, device=device)
    k = torch.exp(-x * x / (2 * TEXTURE_SIGMA ** 2))
    return k / k.sum()


def texture(noise: Tensor) -> Tensor:
    """[B, 48, 48] white noise -> the bed's texture: the Gaussian blur (separable, reflect padding), multiplied so that
    each field's own population standard deviation over the canvas is 0.04."""
    k = _blur_kernel(noise.device)
    r = TEXTURE_RADIUS
    z = noise[:, None]
    z = F.conv2d(F.pad(z, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    z = F.conv2d(F.pad(z, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    z = z[:, 0]
    sd = z.std(dim=(1, 2), correction=0, keepdim=True)
    return z * (TEXTURE_STD / sd)


def draw_parameters(b: int, gen: torch.Generator, bed: str) -> Dict[str, Tensor]:
    """The render's random numbers for b scenes, on the CPU, in the module's order."""
    _check_gen(gen)
    if bed not in BEDS:
        raise ValueError(f"the beds are {BEDS}, not {bed!r}")
    f32 = torch.float32
    p = {}
    p["shift_x"] = torch.rand(b, N_CELLS, generator=gen, dtype=f32) * (2 * SHIFT) - SHIFT
    p["shift_y"] = torch.rand(b, N_CELLS, generator=gen, dtype=f32) * (2 * SHIFT) - SHIFT
    lo, hi = SCALE_RANGE
    p["scale"] = torch.rand(b, N_CELLS, generator=gen, dtype=f32) * (hi - lo) + lo
    p["angle"] = torch.rand(b, N_CELLS, generator=gen, dtype=f32) * (2 * ANGLE) - ANGLE
    p["shade"] = torch.rand(b, N_CELLS, 3, generator=gen, dtype=f32) * (2 * SHADE) - SHADE
    if bed == "C":
        lo, hi = TINT_RANGE
        p["tint"] = torch.rand(b, CHANNELS, generator=gen, dtype=f32) * (hi - lo) + lo
        p["noise"] = torch.randn(b, CANVAS, CANVAS, generator=gen, dtype=f32)
    return p


def _render_layout(cells: Tensor, colors: Tensor, margin: int) -> Tuple[Tensor, Tensor, int]:
    """The layout and the margin of a render, checked."""
    cells = torch.as_tensor(cells, dtype=torch.long)
    colors = torch.as_tensor(colors, dtype=torch.long)
    if cells.dim() != 2 or cells.shape[1] != N_CELLS or colors.shape != cells.shape:
        raise ValueError("a layout is cells [B, 9] and colors [B, 9]")
    if bool(((cells >= 0) != (colors >= 0)).any()):
        raise ValueError("a cell holds a color exactly when it holds a thing")
    if bool((cells >= len(THINGS)).any()) or bool((colors >= len(COLORS)).any()):
        raise ValueError("things and colors are 0..8")
    margin = int(margin)
    if not 0 <= margin <= CELL:
        raise ValueError("margin is 0..16")
    return cells, colors, margin


DRAWN = ("shift_x", "shift_y", "scale", "angle", "shade")
DRAWN_C = ("tint", "noise")


def render(cells: Tensor, colors: Tensor, bed: str, gen: torch.Generator, device="cpu",
           margin: int = MARGIN) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Render a batch of layouts -> (pixels [B, 3, 48, 48] in 0..1 on ``device``, the draw: shift_x, shift_y, scale,
    angle (degrees), shade, centers [B, 9, 2] (x, y in pixels), color [B, 9, 3] (the shaded colors), and on bed C tint
    [B, 3], noise and texture [B, 48, 48] (the texture before any clipping), and ground [B, 3, 48, 48]). The random
    numbers are drawn on the CPU (``draw_parameters``) before anything is computed; then ``render_drawn``."""
    cells, colors, margin = _render_layout(cells, colors, margin)
    p = draw_parameters(cells.shape[0], gen, bed)
    return _render_core(cells, colors, p, bed, device, margin)


def render_drawn(cells: Tensor, colors: Tensor, p: Dict[str, Tensor], bed: str, device="cpu",
                 margin: int = MARGIN, parts: Optional[Sequence[int]] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    """``render`` from random numbers already drawn: ``p`` holds what ``draw_parameters`` returns for these B scenes
    (on the CPU or on ``device``; other keys are ignored) -> what ``render`` returns. Rows drawn by different
    generators can be concatenated on their first axis (the layouts and every tensor of ``p``) and rendered in one
    call. Every number of a scene depends on its own row only, EXCEPT bed C's texture: the blur's convolution may round
    differently with the batch it runs in (by a few 1e-8 on a CPU), so ``parts`` (the row counts of the concatenated
    parts, in order, summing to B) computes the texture of each part apart, the numbers of a render of that part
    alone; everything else is computed once for all rows. None: the whole batch is one part."""
    cells, colors, margin = _render_layout(cells, colors, margin)
    if parts is not None:
        parts = [int(k) for k in parts]
        if any(k < 1 for k in parts) or sum(parts) != cells.shape[0]:
            raise ValueError("parts are positive row counts that sum to the number of scenes")
    if bed not in BEDS:
        raise ValueError(f"the beds are {BEDS}, not {bed!r}")
    keys = DRAWN + (DRAWN_C if bed == "C" else ())
    missing = [k for k in keys if k not in p]
    if missing:
        raise ValueError(f"the drawn numbers lack {missing}")
    b = cells.shape[0]
    if any(int(p[k].shape[0]) != b for k in keys):
        raise ValueError("the drawn numbers are for another number of scenes")
    return _render_core(cells, colors, {k: p[k] for k in keys}, bed, device, margin, parts)


def _render_core(cells: Tensor, colors: Tensor, p: Dict[str, Tensor], bed: str, device,
                 margin: int, parts: Optional[Sequence[int]] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
    b = cells.shape[0]
    p = {k: v.to(device) for k, v in p.items()}
    cells_d, colors_d = cells.to(device), colors.to(device)
    centers = cell_centers().to(device)
    p["centers"] = torch.stack([centers[:, 0] + p["shift_x"], centers[:, 1] + p["shift_y"]], dim=-1)
    table = torch.tensor(COLOR_VALUES, dtype=torch.float32, device=device)
    p["color"] = (table[colors_d.clamp_min(0)] + p["shade"]).clamp(0.0, 1.0)
    if bed == "F":
        ground = torch.full((b, CHANNELS, CANVAS, CANVAS), GROUND, dtype=torch.float32, device=device)
    else:
        if parts is None or len(parts) == 1:
            p["texture"] = texture(p["noise"])
        else:
            p["texture"] = torch.cat([texture(x) for x in torch.split(p["noise"], list(parts))])
        ground = p["tint"][:, :, None, None] + p["texture"][:, None]
    p["ground"] = ground

    w = CELL + 2 * margin
    side = CANVAS + 2 * margin
    cov_sum = torch.zeros(b, side, side, dtype=torch.float32, device=device)
    colored = torch.zeros(b, CHANNELS, side, side, dtype=torch.float32, device=device)
    # the indices are taken from the layout where it arrived: a CPU layout gives them without waiting for a card
    host = cells if cells.device.type == "cpu" else cells_d
    item_b_h, item_c_h = torch.nonzero(host >= 0, as_tuple=True)
    things_h = host[item_b_h, item_c_h]
    item_b, item_c = item_b_h.to(device), item_c_h.to(device)
    if item_b.numel():
        cx, cy = p["centers"][item_b, item_c, 0], p["centers"][item_b, item_c, 1]
        ang = torch.deg2rad(p["angle"][item_b, item_c])
        ca, sa = torch.cos(ang), torch.sin(ang)
        scale = p["scale"][item_b, item_c]
        local = torch.arange(w, dtype=torch.float32, device=device) - margin + 0.5
        xs = (item_c % CELLS).float()[:, None] * CELL + local[None]          # [K, w] pixel-center x
        ys = (item_c // CELLS).float()[:, None] * CELL + local[None]
        dx = (xs - cx[:, None])[:, None, :]                                 # [K, 1, w]
        dy = (ys - cy[:, None])[:, :, None]                                 # [K, w, 1]
        px = ca[:, None, None] * dx + sa[:, None, None] * dy                # [K, w, w]
        py = -sa[:, None, None] * dx + ca[:, None, None] * dy
        cov = torch.zeros(item_b.shape[0], w, w, dtype=torch.float32, device=device)
        for t in range(len(THINGS)):
            sel = torch.nonzero(things_h == t, as_tuple=True)[0].to(device)
            if sel.numel():
                cov[sel] = coverage(thing_distance(t, px[sel], py[sel], scale[sel]))
        col = p["color"][item_b, item_c]
        for c in range(N_CELLS):
            sel = torch.nonzero(item_c_h == c, as_tuple=True)[0].to(device)
            if sel.numel():
                bs = item_b[sel]
                y0, x0 = (c // CELLS) * CELL, (c % CELLS) * CELL
                cov_sum[bs, y0:y0 + w, x0:x0 + w] += cov[sel]
                colored[bs, :, y0:y0 + w, x0:x0 + w] += cov[sel][:, None] * col[sel][:, :, None, None]
    m = margin
    cs = cov_sum[:, m:m + CANVAS, m:m + CANVAS]
    pixels = colored[:, :, m:m + CANVAS, m:m + CANVAS] + (1.0 - cs)[:, None] * ground
    return pixels.clamp(0.0, 1.0), p


# ------------------------------------------------------------------ the bed
@dataclass
class SceneBatch:
    x: Tensor                         # [B, 3, 48, 48] in model space
    cells: Tensor                     # [B, 9] long on the bed's device
    colors: Tensor                    # [B, 9] long on the bed's device
    params: Dict[str, Tensor] = field(default_factory=dict)   # the draw (``render``), centers included

    @property
    def pixels(self) -> Tensor:
        return to_pixels(self.x)


class ScenesBed:
    """Scenes of bed "F" (a flat gray ground) or "C" (one tint a scene and a smooth texture across the cells)."""

    def __init__(self, bed: str = "F", device="cpu"):
        if bed not in BEDS:
            raise ValueError(f"the beds are {BEDS}, not {bed!r}")
        self.bed = bed
        self.device = device
        self.field_shape = (CHANNELS, CANVAS, CANVAS)

    def train_batch(self, kind: str, n: int, gen: torch.Generator, only_class: Optional[int] = None,
                    count: Optional[int] = None, allowed_cells: Optional[Sequence[int]] = None) -> SceneBatch:
        """n fresh scenes of a kind from the CPU generator ``gen``: the layouts first, then the render's draws."""
        cells, colors = draw_layouts(kind, n, gen, only_class=only_class, count=count, allowed_cells=allowed_cells)
        pixels, params = render(cells, colors, self.bed, gen, self.device)
        return SceneBatch(to_model(pixels), cells.to(self.device), colors.to(self.device), params)

    def heldout(self, kind: str, n: int, seed: int, only_class: Optional[int] = None,
                count: Optional[int] = None, allowed_cells: Optional[Sequence[int]] = None) -> SceneBatch:
        """The same as ``train_batch`` from a CPU generator seeded ``seed``."""
        gen = torch.Generator().manual_seed(int(seed))
        return self.train_batch(kind, n, gen, only_class=only_class, count=count, allowed_cells=allowed_cells)
