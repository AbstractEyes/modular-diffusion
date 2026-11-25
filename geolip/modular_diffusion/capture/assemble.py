"""The assembly: band parts blended into one denoiser by the bands' fixed weights, and a deterministic DDIM sampler.

    parts = {"clean": engine_a, "middle": teacher.unet, "noisy": engine_c}
    model = Assembly(parts)
    eps = model(z_t, t, states)          sum over bands of w_band(t) x part_band(z_t, t, states)
    z0 = sample(model, states, empty, seed=0, steps=50, guidance=7.5)

At a step t the three weights come from ``schedule.band_weights`` (they are fixed numbers of the step: nothing learned
chooses among the parts). Only the parts whose weight is above zero are run, each on the rows where it is; a part
that serves two bands (the same object) is run once with the two weights added. The weighted sum is taken in float64
and returned in the parts' dtype, so an assembly whose three parts are one U-Net gives that U-Net's answers to the bit.

Guidance uses the same weights for the prompted and the empty branch (both rows share the step), so guiding the blend
equals blending the guided parts, up to rounding.

The sampler is DDIM with eta 0: 50 steps by default with the step offset 1 (steps 981, 961, .., 1), the teacher's
schedule, the last step to alpha_bar of step 0, guidance 7.5 by default: eps = u + g (c - u).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

from . import BANDS
from .engine import as_part
from .schedule import STEPS, band_weights, sd15_alphas_cumprod


class Assembly(nn.Module):
    """Parts of the three bands blended by the bands' weights. ``parts`` = {band: part} for all three bands; a part is
    an engine, a teacher, a diffusers U-Net, or another assembly."""
    is_part = True

    def __init__(self, parts: Dict[str, object]):
        super().__init__()
        if set(parts) != set(BANDS):
            raise ValueError(f"an assembly has one part for each band {list(BANDS)}, not {sorted(parts)}")
        wrapped: Dict[int, nn.Module] = {}
        self._group: List[int] = []
        for band in BANDS:
            key = id(parts[band])
            if key not in wrapped:
                wrapped[key] = as_part(parts[band])
            self._group.append(key)
        order = list(dict.fromkeys(self._group))
        self.parts = nn.ModuleList([wrapped[k] for k in order])
        self._slot = [order.index(k) for k in self._group]      # band -> its part's slot (a shared part, one slot)

    def part(self, band: str) -> nn.Module:
        return self.parts[self._slot[BANDS.index(band)]]

    def weights(self, t: Tensor) -> Tensor:
        """[B, n_slots] float64: each distinct part's weight (the weights of the bands it serves, added)."""
        w = band_weights(t)
        out = torch.zeros(w.shape[0], len(self.parts), dtype=torch.float64)
        for b, slot in enumerate(self._slot):
            out[:, slot] = out[:, slot] + w[:, b]
        return out

    def forward(self, z: Tensor, t, states: Optional[Tensor] = None) -> Tensor:
        b = z.shape[0]
        if not isinstance(t, Tensor):
            t = torch.full((b,), int(t), dtype=torch.long)
        t = t.reshape(-1)
        if t.numel() == 1 and b > 1:
            t = t.expand(b)
        w = self.weights(t.cpu()).to(z.device)
        acc = None
        dtype = z.dtype
        for slot, part in enumerate(self.parts):
            ws = w[:, slot]
            on = ws > 0
            n_on = int(on.sum())
            if n_on == 0:
                continue
            if n_on == b:
                o = part(z, t.to(z.device), states)
                contrib = ws.reshape(-1, *([1] * (z.dim() - 1))) * o.double()
            else:
                idx = torch.nonzero(on, as_tuple=False).squeeze(1)
                o = part(z[idx], t.to(z.device)[idx], None if states is None else states[idx])
                contrib = torch.zeros((b,) + tuple(o.shape[1:]), dtype=torch.float64, device=z.device)
                contrib[idx] = ws[idx].reshape(-1, *([1] * (z.dim() - 1))) * o.double()
            dtype = o.dtype
            acc = contrib if acc is None else acc + contrib
        return acc.to(dtype)


def ddim_timesteps(steps: int = 50, offset: int = 1) -> List[int]:
    """The DDIM steps, noisiest first: i x (1000 // steps) + offset for i = steps - 1 .. 0."""
    steps = int(steps)
    if not 1 <= steps <= STEPS:
        raise ValueError(f"steps lie in 1 .. {STEPS}")
    ratio = STEPS // steps
    ts = [i * ratio + int(offset) for i in reversed(range(steps))]
    if ts[0] >= STEPS:
        raise ValueError("the offset puts a step past 999")
    return ts


def _default_shape(model) -> Sequence[int]:
    unet = getattr(model, "unet", None)
    if unet is None and hasattr(model, "parts"):
        unet = getattr(model.parts[0], "unet", None)
    cfg = getattr(unet, "config", None)
    if cfg is not None:
        return (int(cfg.in_channels), int(cfg.sample_size), int(cfg.sample_size))
    return (4, 64, 64)


def initial_noise(seed: int, batch: int, latent_shape: Sequence[int]) -> Tensor:
    """The starting noise of ``sample``: a CPU generator seeded with ``seed``, float32."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn((int(batch),) + tuple(int(x) for x in latent_shape), generator=g)


@torch.no_grad()
def guided(part: nn.Module, z: Tensor, t: Tensor, states: Tensor, empty: Optional[Tensor], guidance: float) -> Tensor:
    """u + g (c - u): the empty branch u and the prompted branch c in one batch (empty rows first)."""
    if empty is None or float(guidance) == 1.0:
        return part(z, t, states)
    b = z.shape[0]
    both = part(torch.cat([z, z]), torch.cat([t, t]), torch.cat([empty.expand(b, -1, -1), states]))
    u, c = both[:b], both[b:]
    return u + float(guidance) * (c - u)


@torch.no_grad()
def sample(model, prompts_states: Tensor, empty_states: Optional[Tensor], seed: int, steps: int = 50,
           guidance: float = 7.5, *, latent_shape: Optional[Sequence[int]] = None,
           alphas_cumprod: Optional[Tensor] = None, z: Optional[Tensor] = None, offset: int = 1,
           on_step: Optional[Callable[[int, Tensor, Tensor], None]] = None) -> Tensor:
    """The final latents of DDIM (eta 0) from the seed's noise (or from ``z``), with guidance. ``model``: an assembly,
    an engine, a teacher or a diffusers U-Net. The latents are in the prompt states' dtype and device.

    ``on_step(t, z, eps)``, when given, is called at every step with the step, the state the model was asked on and its
    guided answer as the sampler uses it, before the update; it only watches (the latents are the same with or
    without it)."""
    part = as_part(model)
    dev, dtype = prompts_states.device, prompts_states.dtype
    b = prompts_states.shape[0]
    ac = (alphas_cumprod if alphas_cumprod is not None else sd15_alphas_cumprod()).double().cpu()
    if z is None:
        z = initial_noise(seed, b, latent_shape or _default_shape(model))
    z = z.to(device=dev, dtype=torch.float32)      # the latents stay float32 over all steps; parts cast their inputs
    empty = None if empty_states is None else empty_states.to(device=dev, dtype=dtype)
    ts = ddim_timesteps(steps, offset)
    ratio = STEPS // int(steps)
    for t in ts:
        tt = torch.full((b,), t, dtype=torch.long, device=dev)
        eps = guided(part, z, tt, prompts_states, empty, guidance)
        if on_step is not None:
            on_step(t, z, eps)
        a = float(ac[t])
        prev = t - ratio
        a_prev = float(ac[prev]) if prev >= 0 else float(ac[0])
        x0 = (z - math.sqrt(1.0 - a) * eps) / math.sqrt(a)
        z = math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * eps
    return z
