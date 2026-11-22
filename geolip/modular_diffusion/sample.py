"""Sampling: Euler steps of the predicted velocity from noise (t = 0) toward the image (t = 1), and a sheet of samples."""
from __future__ import annotations

import numpy as np
import torch

from .spectral import token_filter

SHEET_SEED = 2026


@torch.no_grad()
def euler(model, z: torch.Tensor, cls: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """z_{i+1} = z_i + v(z_i, i / steps) / steps for i = 0 .. steps - 1 (the last velocity is taken at t = 1 - 1/steps)."""
    for i in range(steps):
        t = torch.full((cls.shape[0],), i / steps, device=z.device)
        z = z + model.velocity(z, t, cls) / steps
    return z


@torch.no_grad()
def sample_sheet(model, bed, path: str, device, steps: int = 50, per_class: int = 4) -> None:
    """A PNG of ``per_class`` samples for each of the ten classes (rows), enlarged three times."""
    from PIL import Image
    model.eval()
    cls = torch.arange(10, device=device).repeat_interleave(per_class)
    gen = torch.Generator(device=device).manual_seed(SHEET_SEED)
    z = torch.randn((cls.shape[0],) + tuple(bed.field_shape), generator=gen, device=device)
    z = euler(model, z, cls, steps)
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
