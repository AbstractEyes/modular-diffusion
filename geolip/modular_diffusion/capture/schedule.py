"""The teacher's noise schedule and the three noise bands.

The teacher (Stable Diffusion 1.5) works on discrete steps t = 0 .. 999, larger is noisier:

    beta_t      = (sqrt(0.00085) + (sqrt(0.012) - sqrt(0.00085)) t / 999)^2
    alpha_bar_t = prod over s <= t of (1 - beta_s)
    z_t         = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps          the teacher predicts eps

The band coordinate is s = t / 1000 (the step itself, never a stand-in for the noise level). Three bands share the
range with edges 0.35 and 0.75 and a crossfade half-width of 0.06:

    ramp(x)  = 0.5 - 0.5 cos(pi u),   u = (clamp(x / 0.06, -1, 1) + 1) / 2
    up1      = ramp(s - 0.35),        up2 = ramp(s - 0.75)
    w_clean  = 1 - up1,   w_middle = up1 (1 - up2),   w_noisy = up1 up2       (they sum to one at every step)

These weights are not written a second time here: each band is a declared address of the library's address frame
(``frame.owner`` with the window [0, 0.35), [0.35, 0.75) or [0.75, 1] on s and the half-width 0.06), and the weights
are the frame's projective gate of those owners (``frame.projective``). On the noisy band the gate computes up2 alone;
up1 is exactly 1.0 wherever up2 is above zero, so this is the product above to the bit.

A band's STEP SET is the steps where its weight is above zero. Its training law draws, with probability 1 - rho, a
uniform step of its step set and, with probability rho (the whole-range share), a uniform step of 0 .. 999.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch
from torch import Tensor

from .. import frame
from . import BANDS

EDGES = (0.35, 0.75)
SOFT = 0.06
STEPS = 1000
BETA_START = 0.00085
BETA_END = 0.012

Steps = Union[int, Sequence[int], Tensor]


def band_index(band: Union[str, int]) -> int:
    """The column of a band in every weight table: clean 0, middle 1, noisy 2 (an index is returned as it is)."""
    if isinstance(band, int) and not isinstance(band, bool):
        if 0 <= band < len(BANDS):
            return band
        raise ValueError(f"a band index is 0, 1 or 2, not {band}")
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}; known: {list(BANDS)}")
    return BANDS.index(band)


def band_owners() -> List[frame.Owner]:
    """The three bands as declared owners of the address frame: windows on s = t / 1000 with the half-width 0.06."""
    cuts = (0.0,) + EDGES + (1.0,)
    return [frame.owner(cuts[i], cuts[i + 1], SOFT, name=name) for i, name in enumerate(BANDS)]


def _steps(t: Steps) -> Tensor:
    if isinstance(t, Tensor):
        x = t.reshape(-1)
    elif isinstance(t, int):
        x = torch.tensor([t])
    else:
        x = torch.as_tensor(list(t))
    if x.is_floating_point() and not bool((x == x.round()).all()):
        raise ValueError("steps are whole numbers 0 .. 999")
    x = x.long()
    if x.numel() and (int(x.min()) < 0 or int(x.max()) >= STEPS):
        raise ValueError("steps lie in 0 .. 999")
    return x


def band_coordinate(t: Steps) -> Tensor:
    """[B] float64: s = t / 1000."""
    return _steps(t).double().cpu() / STEPS


def band_weights(t: Steps) -> Tensor:
    """[B, 3] float64: the weights of the bands (columns clean, middle, noisy) at the steps t, from the frame's gate."""
    return frame.projective(band_owners(), band_coordinate(t))


def band_weight(t: Steps, band: Union[str, int]) -> Tensor:
    """[B] float64: one band's weight at the steps t."""
    return band_weights(t)[:, band_index(band)]


_ALL = torch.arange(STEPS)


def band_steps(band: Union[str, int]) -> Tensor:
    """The band's step set: the steps 0 .. 999 where its weight is above zero, ascending."""
    return _ALL[band_weight(_ALL, band) > 0]


def core_steps(band: Union[str, int]) -> Tensor:
    """The steps where the band's weight is exactly one (no other band is asked there)."""
    return _ALL[band_weight(_ALL, band) == 1.0]


def crossfade_steps() -> Tensor:
    """The steps where two bands are above zero."""
    return _ALL[(band_weights(_ALL) > 0).sum(dim=1) > 1]


def sample_steps(band: Union[str, int], n: int, rho: float, generator: torch.Generator) -> Tensor:
    """[n] long: steps drawn by the band's training law. Each draw is from the whole range 0 .. 999 with probability rho
    and from the band's step set otherwise (both uniform). The draws use the generator's device; the result is on it."""
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError(f"the whole-range share rho lies in [0, 1], not {rho}")
    dev = generator.device
    own = band_steps(band).to(dev)
    whole = torch.rand(n, generator=generator, device=dev) < float(rho)
    pick_own = own[torch.randint(len(own), (n,), generator=generator, device=dev)]
    pick_all = torch.randint(STEPS, (n,), generator=generator, device=dev)
    return torch.where(whole, pick_all, pick_own)


def sd15_betas() -> Tensor:
    """[1000] float64: the teacher's betas (the scaled linear schedule)."""
    t = torch.arange(STEPS, dtype=torch.float64)
    a, b = math.sqrt(BETA_START), math.sqrt(BETA_END)
    return (a + (b - a) * t / (STEPS - 1)) ** 2


def sd15_alphas_cumprod() -> Tensor:
    """[1000] float64: alpha_bar_t, the running product of 1 - beta."""
    return torch.cumprod(1.0 - sd15_betas(), dim=0)


def add_noise(x0: Tensor, eps: Tensor, t: Tensor, alphas_cumprod: Tensor) -> Tensor:
    """z_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps, one step per row, in x0's dtype."""
    a = alphas_cumprod.to(device=x0.device)[t.to(x0.device).long()]
    shape = (-1,) + (1,) * (x0.dim() - 1)
    sa = a.sqrt().to(x0.dtype).reshape(shape)
    sb = (1.0 - a).sqrt().to(x0.dtype).reshape(shape)
    return sa * x0 + sb * eps
