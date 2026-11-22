"""Sampling: Euler steps of the predicted velocity from noise (t = 0) toward the image (t = 1), and a sheet of samples.

    euler(model, z, cls, steps)       steps equal steps: the velocity at t = i / steps
    euler_grid(model, z, cls, ts)     the steps of a time grid 0 = ts[0] < ts[1] < .. < ts[K] = 1
    stage_grid(cuts, counts)          a grid with counts[s] equal steps inside stage s = [cuts[s], cuts[s + 1])
    share_counts(shares, n)           n steps divided among stages in proportion to their shares
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence

import numpy as np
import torch

from .spectral import token_filter

SHEET_SEED = 2026


SYMBOL_OFFSET = 7919                 # sampled pictures' symbols: fixed_symbols(n, the sampling seed + SYMBOL_OFFSET)


def picture_symbols(model, n: int, seed: int, device):
    """The symbols of n sampled pictures for a model with symbol-gated units (one fixed draw per picture from a
    generator seeded from the sampling seed, so the noise of every picture is the noise of a model without them), or
    None (nothing is drawn)."""
    from .model import fixed_symbols, symbol_gated
    return fixed_symbols(n, int(seed) + SYMBOL_OFFSET, device) if symbol_gated(model) else None


def _kw(symbols) -> dict:
    return {} if symbols is None else {"symbols": symbols}


@torch.no_grad()
def euler(model, z: torch.Tensor, cls: torch.Tensor, steps: int = 50, symbols: Optional[torch.Tensor] = None) -> torch.Tensor:
    """z_{i+1} = z_i + v(z_i, i / steps) / steps for i = 0 .. steps - 1 (the last velocity is taken at t = 1 - 1/steps).
    symbols: the rows' symbols for a model with symbol-gated units."""
    for i in range(steps):
        t = torch.full((cls.shape[0],), i / steps, device=z.device)
        z = z + model.velocity(z, t, cls, **_kw(symbols)) / steps
    return z


def check_grid(ts: Sequence[float]) -> List[float]:
    """The grid as a list of Python floats: one dimension, at least two points, strictly increasing from exactly 0.0 to
    exactly 1.0 (ValueError otherwise)."""
    if isinstance(ts, torch.Tensor) or isinstance(ts, np.ndarray):
        if ts.ndim != 1:
            raise ValueError(f"a time grid is one-dimensional, not of shape {tuple(ts.shape)}")
        ts = ts.tolist()
    try:
        pts = [float(x) for x in ts]
    except TypeError:
        raise ValueError("a time grid is a sequence of numbers") from None
    if len(pts) < 2:
        raise ValueError(f"a time grid needs at least two points: {pts}")
    if pts[0] != 0.0 or pts[-1] != 1.0:
        raise ValueError(f"a time grid runs from 0.0 to 1.0: {pts[0]} .. {pts[-1]}")
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("a time grid must be strictly increasing")
    return pts


@torch.no_grad()
def euler_grid(model, z: torch.Tensor, cls: torch.Tensor, ts: Sequence[float],
               symbols: Optional[torch.Tensor] = None) -> torch.Tensor:
    """z <- z + v(z, ts[k]) (ts[k + 1] - ts[k]) for k = 0 .. K - 1 on a grid of K + 1 points (``check_grid``); the
    differences are taken in Python floats."""
    pts = check_grid(ts)
    for k in range(len(pts) - 1):
        t = torch.full((cls.shape[0],), pts[k], device=z.device)
        z = z + model.velocity(z, t, cls, **_kw(symbols)) * (pts[k + 1] - pts[k])
    return z


def uniform_grid(n: int) -> List[float]:
    """The points i / n for i = 0 .. n (``euler_grid`` on it takes the velocities of ``euler(steps=n)``)."""
    n = int(n)
    if n < 1:
        raise ValueError(f"a grid needs at least one step: {n}")
    return [i / n for i in range(n + 1)]


def stage_grid(cuts: Sequence[float], counts: Sequence[int]) -> List[float]:
    """cuts [0, .., 1] and one positive step count per stage -> the grid with counts[s] equal steps inside stage s: the
    points cuts[s] + i (cuts[s + 1] - cuts[s]) / counts[s] for i = 0 .. counts[s] - 1, then exactly 1.0."""
    cuts = [float(c) for c in cuts]
    if len(cuts) < 2 or cuts[0] != 0.0 or cuts[-1] != 1.0 or any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"cuts must rise from 0 to 1: {cuts}")
    if len(counts) != len(cuts) - 1:
        raise ValueError(f"{len(cuts) - 1} stages need {len(cuts) - 1} step counts, not {len(counts)}")
    pts: List[float] = []
    for s, c in enumerate(counts):
        if int(c) != c or int(c) < 1:
            raise ValueError(f"a step count is a positive integer: {c!r}")
        c = int(c)
        lo, hi = cuts[s], cuts[s + 1]
        pts.extend(lo + i * (hi - lo) / c for i in range(c))
    pts.append(1.0)
    return check_grid(pts)


def share_counts(shares: Sequence[float], n: int) -> List[int]:
    """n steps divided among the stages in proportion to ``shares`` by largest remainders, at least 1 per stage, summing
    to exactly n (ValueError when n is below the number of stages).

    The quotas n x share / sum(shares) are computed exactly on the shares' decimal values (``Fraction(str(x))``); each
    stage gets the whole part of its quota, the steps left go one each to the largest remainders, a tie going to the
    earlier stage; a stage left at 0 then takes one step from the stage holding the most (the earlier one on a tie)."""
    n = int(n)
    if not shares:
        raise ValueError("no shares given")
    if n < len(shares):
        raise ValueError(f"{n} steps cannot give each of {len(shares)} stages at least one")
    fr = [Fraction(str(float(x))) for x in shares]
    if any(x < 0 for x in fr) or sum(fr) <= 0:
        raise ValueError(f"shares are non-negative with a positive sum: {list(shares)}")
    quota = [n * x / sum(fr) for x in fr]
    counts = [int(q) for q in quota]                          # quotas are non-negative: int() is the floor
    left = n - sum(counts)
    order = sorted(range(len(fr)), key=lambda s: (-(quota[s] - counts[s]), s))
    for s in order[:left]:
        counts[s] += 1
    for s in range(len(counts)):
        if counts[s] == 0:
            donor = max(range(len(counts)), key=lambda j: (counts[j], -j))
            counts[donor] -= 1
            counts[s] = 1
    return counts


@torch.no_grad()
def sample_sheet(model, bed, path: str, device, steps: int = 50, per_class: int = 4,
                 grid: Optional[Sequence[float]] = None) -> None:
    """A PNG of ``per_class`` samples for each of the ten classes (rows), enlarged three times. With ``grid`` the samples
    take the steps of that time grid (``euler_grid``) and ``steps`` is ignored."""
    from PIL import Image
    model.eval()
    cls = torch.arange(10, device=device).repeat_interleave(per_class)
    gen = torch.Generator(device=device).manual_seed(SHEET_SEED)
    z = torch.randn((cls.shape[0],) + tuple(bed.field_shape), generator=gen, device=device)
    sym = picture_symbols(model, cls.shape[0], SHEET_SEED, device)
    z = euler(model, z, cls, steps, symbols=sym) if grid is None else euler_grid(model, z, cls, grid, symbols=sym)
    if bed.tilt is not None:                 # back to ordinary pictures for viewing
        z = token_filter(z, 1.0 / bed.tilt.to(device), bed.grid)
    x = (z * bed.std.to(device) + bed.mean.to(device)).clamp(0, 1)
    x = (x * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    hh = x.shape[1]
    sheet = np.zeros((10 * hh, per_class * hh, 3), dtype=np.uint8)
    for i in range(x.shape[0]):
        r, c = i // per_class, i % per_class
        sheet[r * hh:(r + 1) * hh, c * hh:(c + 1) * hh] = x[i]
    Image.fromarray(sheet).resize((per_class * hh * 3, 10 * hh * 3), Image.NEAREST).save(path)
