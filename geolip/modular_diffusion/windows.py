"""Windows of the noise level t (t = 0 pure noise, t = 1 the image): stage edges, the gate ramp and window weights.

A window [lo, hi) weighs 1 inside and exactly 0 outside when hard. With a half-width w > 0 each edge becomes a cosine
ramp from x = -w to x = +w around the edge, so neighboring windows crossfade over a transition 2 w wide.
"""
from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor

# Named partitions of [0, 1] by their inner edges.
STAGES = {"1": [], "3": [0.25, 0.65], "3even": [1 / 3, 2 / 3], "5mid": [0.25, 0.38, 0.52, 0.65],
          "5even": [0.2, 0.4, 0.6, 0.8], "7mid": [0.25, 0.33, 0.41, 0.49, 0.57, 0.65]}


def stage_edges(spec: str) -> List[float]:
    """A preset name, or ``e:a,b,c`` for explicit inner edges."""
    if spec in STAGES:
        return list(STAGES[spec])
    if spec.startswith("e:"):
        return [float(x) for x in spec[2:].split(",")]
    raise ValueError(f"unknown stages {spec!r}; known: {sorted(STAGES)} or e:<edges>")


def stage_cuts(spec: str) -> List[float]:
    """0, the inner edges, 1."""
    return [0.0] + stage_edges(spec) + [1.0]


def ramp(x: Tensor, half_width: float) -> Tensor:
    """0 below -half_width, 1 above +half_width, a cosine ramp between; a hard step (x >= 0) when half_width is 0."""
    if half_width <= 0:
        return (x >= 0).to(x.dtype)
    u = ((x / half_width).clamp(-1, 1) + 1) / 2
    return 0.5 - 0.5 * torch.cos(u * math.pi)


def window_weight(t: Tensor, lo: float, hi: float, soft: float) -> Tensor:
    """[B]: the weight of the window [lo, hi) at t. An edge at 0 or 1 is no edge, so a hard window includes t = 1 when
    hi = 1."""
    w = torch.ones_like(t)
    if lo > 0:
        w = w * ramp(t - lo, soft)
    if hi < 1:
        w = w * (1 - ramp(t - hi, soft))
    return w
