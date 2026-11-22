"""Rotary position codes for a raster-ordered token grid.

A rotary is a callable (q, k) -> (q, k) acting on [B, heads, N, head_dim]. AxialBandCode places all of its rotation rates
inside one band of wavelengths (in tokens): a quarter of the head turns with the row, a quarter with the column, and the
other half repeats those angles (the rotate-half pairing).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .blocks import Rotary

# The three output bands and the wavelength range (in tokens) of each band's position code.
BANDS3 = (("fine", 2.0, 4.0), ("mid", 4.0, 8.0), ("coarse", 8.0, 64.0))
WHOLE = (2.0, 64.0)                                   # the code that spans every band


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def table_rotary(cos: Tensor, sin: Tensor) -> Rotary:
    """A rotary from tables cos, sin [N, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def _rot(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        c, s = cos.to(q.dtype), sin.to(q.dtype)
        return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s

    return _rot


class AxialBandCode(nn.Module):
    """Axial rotary code whose wavelengths run geometrically from ``lo`` to ``hi`` tokens (``hi`` excluded).

    The tables are buffers that are not saved: they follow from (head_dim, grid, lo, hi).
    """

    def __init__(self, head_dim: int, grid: int, lo: float, hi: float):
        super().__init__()
        per_axis = head_dim // 4
        lam = lo * (hi / lo) ** (torch.arange(per_axis, dtype=torch.float64) / per_axis)
        rates = 2 * math.pi / lam
        rows = torch.arange(grid, dtype=torch.float64).repeat_interleave(grid)
        cols = torch.arange(grid, dtype=torch.float64).repeat(grid)
        ang = torch.cat([rows[:, None] * rates[None, :], cols[:, None] * rates[None, :]], dim=1)
        ang = torch.cat([ang, ang], dim=1)
        self.register_buffer("cos", ang.cos().float(), persistent=False)
        self.register_buffer("sin", ang.sin().float(), persistent=False)
        self.band = (float(lo), float(hi))
        self.rates = tuple(float(r) for r in rates)

    def rotary(self) -> Rotary:
        return table_rotary(self.cos, self.sin)
