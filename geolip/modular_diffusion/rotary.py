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


# ------------------------------------------------------------------ the anchored address rotary
# A head of 64 numbers = 32 rotate-half pairs, split by job: pairs 0-9 turn with the token's row and pairs 10-19 with
# its column on the relative band (the axial ladder's geometric wavelengths from lo to hi tokens, 10 a axis), pairs
# 20-23 do not turn (content), pairs 24-27 turn by 2 pi d / 3 with the column digit d of the token's address and pairs
# 28-31 with its row digit (base 3: the kernel cos(2 pi (d - d') / 3) is 1 for the same digit and -1/2 for another).
ANCHORED_PAIRS = {"row": (0, 10), "column": (10, 20), "content": (20, 24), "anchor_column": (24, 28),
                  "anchor_row": (28, 32)}
ANCHORED_HEAD_DIM = 64
ANCHOR_BASE = 3


def anchored_angles(rows: Tensor, cols: Tensor, digit_col: Tensor, digit_row: Tensor, lo: float = WHOLE[0],
                    hi: float = WHOLE[1]) -> Tensor:
    """[B, N, 32] float32: the angle of each pair for tokens at (row, column) on the relative band (floats) with the
    address digits (digit_col, digit_row); every input [B, N]."""
    per = ANCHORED_PAIRS["row"][1] - ANCHORED_PAIRS["row"][0]
    lam = float(lo) * (float(hi) / float(lo)) ** (torch.arange(per, dtype=torch.float64) / per)
    rates = (2 * math.pi / lam).to(rows.device)
    r, c = rows.double(), cols.double()
    rel = torch.cat([r[..., None] * rates, c[..., None] * rates], dim=-1)                  # [B, N, 20]
    zero = torch.zeros(rel.shape[:-1] + (4,), dtype=torch.float64, device=rows.device)
    a_col = (2 * math.pi / ANCHOR_BASE) * digit_col.double()[..., None].expand(*digit_col.shape, 4)
    a_row = (2 * math.pi / ANCHOR_BASE) * digit_row.double()[..., None].expand(*digit_row.shape, 4)
    return torch.cat([rel, zero, a_col, a_row], dim=-1).float()


def anchored_rotary(rows: Tensor, cols: Tensor, digit_col: Tensor, digit_row: Tensor, lo: float = WHOLE[0],
                    hi: float = WHOLE[1]) -> Rotary:
    """The rotary (q, k) -> (q, k) on [B, heads, N, 64] of ``anchored_angles`` (the rotate-half pairing: pair i joins
    the numbers i and i + 32, as ``AxialBandCode`` pairs them)."""
    ang = anchored_angles(rows, cols, digit_col, digit_row, lo, hi)
    ang = torch.cat([ang, ang], dim=-1)[:, None]                                         # [B, 1, N, 64]
    cos, sin = ang.cos(), ang.sin()

    def _rot(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        if q.shape[-1] != ANCHORED_HEAD_DIM:
            raise ValueError(f"the anchored rotary needs a head of {ANCHORED_HEAD_DIM} numbers, not {q.shape[-1]}")
        c, s = cos.to(q.dtype), sin.to(q.dtype)
        return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s

    return _rot
