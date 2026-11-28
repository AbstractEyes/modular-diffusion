"""Frequency-domain helpers: token layout, band masks, the band filter, power spectra and the closed-form spectral anchor.

Fourier conventions: torch's default (unnormalized forward FFT). A power spectrum divides the FFT by the side length, so
that its mean over frequencies equals the mean square of the signal; unit white noise then has power 1 at every frequency.

The anchor. For z_t = (1 - t) noise + t x with x of power spectrum S (per frequency) and unit noise, the least-squares
linear predictor of the velocity x - noise from z_t, frequency by frequency, is the gain

    g(t, S) = (t S - (1 - t)) / (t^2 S + (1 - t)^2)

and its error power (the floor no linear predictor can beat) is S / (t^2 S + (1 - t)^2).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor


# ------------------------------------------------------------------ token layout
def patchify(z: Tensor, patch: int) -> Tensor:
    """[B, C, H, W] -> tokens [B, (H/p)(W/p), C p p] in raster order."""
    b, c, hh, ww = z.shape
    p = patch
    x = z.reshape(b, c, hh // p, p, ww // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (hh // p) * (ww // p), c * p * p)


def unpatchify(x: Tensor, field_shape: Sequence[int], patch: int, grid: int) -> Tensor:
    """Tokens [B, N, C p p] -> [B, C, H, W]."""
    b = x.shape[0]
    c, hh, ww = field_shape
    p, g = patch, grid
    z = x.reshape(b, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5)
    return z.reshape(b, c, hh, ww)


def to_grid(x: Tensor, grid: int) -> Tensor:
    """Tokens [B, N, P] -> [B, P, G, G]."""
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], grid, grid)


def from_grid(y: Tensor, grid: int) -> Tensor:
    """[B, P, G, G] -> tokens [B, N, P]."""
    return y.reshape(y.shape[0], y.shape[1], grid * grid).transpose(1, 2)


# ------------------------------------------------------------------ bands
def frequency_radius(grid: int) -> Tensor:
    """|f| per frequency of a grid x grid FFT, in cycles per token (float64)."""
    f = torch.fft.fftfreq(grid, dtype=torch.float64)
    return torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


def band_masks(grid: int, bands: int) -> Tuple[List[Tensor], Tensor]:
    """Boolean masks over the token grid's frequencies, disjoint and covering everything, and the radius table.

    bands 1: one mask of everything. bands 3: fine (wavelength under 4 tokens, |f| >= 1/4), mid (4 to 8 tokens) and
    coarse (8 tokens and longer, the constant included). A wavelength of exactly 4 is fine; exactly 8 is mid.
    """
    r = frequency_radius(grid)
    if bands == 1:
        return [torch.ones_like(r, dtype=torch.bool)], r
    if bands != 3:
        raise ValueError("bands must be 1 or 3")
    return [r >= 1 / 4, (r >= 1 / 8) & (r < 1 / 4), r < 1 / 8], r


def keep_band(tokens: Tensor, mask: Tensor, grid: int) -> Tensor:
    """Keep only the frequencies of ``mask`` (a float [G, G] table) of every token channel; float32 in and out."""
    f = torch.fft.fft2(to_grid(tokens.float(), grid))
    return from_grid(torch.fft.ifft2(f * mask).real, grid)


def band_power(x: Tensor, patch: int, grid: int) -> Tensor:
    """x [B, C, H, W] -> power per token-grid frequency [G, G] (float64), mean over items and token channels.
    Its mean over frequencies is mean(x^2)."""
    y = to_grid(patchify(x.double(), patch), grid)
    f = torch.fft.fft2(y) / grid
    return (f.real ** 2 + f.imag ** 2).mean(dim=(0, 1))


# ------------------------------------------------------------------ spectra
def field_spectrum(grid: int, slope: float, dc_share: float) -> Tensor:
    """A power law |f|^-slope over a grid x grid FFT (float64), scaled so that its mean is 1, with ``dc_share`` of the
    total at the constant."""
    r = frequency_radius(grid)
    s = torch.zeros_like(r)
    m = r > 0
    s[m] = r[m] ** (-slope)
    s = s * ((1 - dc_share) * grid * grid / s.sum())
    s[0, 0] = dc_share * grid * grid
    return s


def measure_spectrum(x: Tensor, chunk: int = 2000) -> Tensor:
    """x [B, C, H, W] -> power per channel and pixel-grid frequency [C, H, W] (float64), mean over items."""
    n = x.shape[0]
    side = x.shape[-1]
    if n <= chunk:
        f = torch.fft.fft2(x.double()) / side
        return (f.real ** 2 + f.imag ** 2).mean(dim=0)
    total = torch.zeros(x.shape[1:], dtype=torch.float64, device=x.device)
    for i in range(0, n, chunk):
        f = torch.fft.fft2(x[i:i + chunk].double()) / side
        total += (f.real ** 2 + f.imag ** 2).sum(dim=0)
    return total / n


def token_filter(x: Tensor, gain: Tensor, grid: int) -> Tensor:
    """x [B, C, H, W] -> the same shape: every token channel multiplied by gain [P, G, G] in the token grid's Fourier
    basis."""
    b, c, hh, ww = x.shape
    p = hh // grid
    t = x.reshape(b, c, grid, p, grid, p).permute(0, 1, 3, 5, 2, 4).reshape(b, c * p * p, grid, grid)
    t = torch.fft.ifft2(torch.fft.fft2(t) * gain).real
    return t.reshape(b, c, p, p, grid, grid).permute(0, 1, 4, 2, 5, 3).reshape(b, c, hh, ww)


def tilt_gain(x: Tensor, grid: int, slope: float, dc_share: float) -> Tensor:
    """The gain per token channel and token-grid frequency that turns the measured spectrum of x into the power law of
    field_spectrum (unit variance per channel): sqrt(target / measured), float32 [P, G, G]."""
    b, c, hh, ww = x.shape
    p = hh // grid
    t = x.double().reshape(b, c, grid, p, grid, p).permute(0, 1, 3, 5, 2, 4).reshape(b, c * p * p, grid, grid)
    f = torch.fft.fft2(t) / grid
    measured = (f.real ** 2 + f.imag ** 2).mean(dim=0)
    target = field_spectrum(grid, slope, dc_share)[None]
    return (target / measured.clamp_min(1e-12)).sqrt().float()


# ------------------------------------------------------------------ the anchor
def anchor_gain(t: Tensor, s: Tensor) -> Tensor:
    """g(t, S) per item and frequency: t [B], s [C, H, W] -> [B, C, H, W]."""
    s = s[None]
    tt = t.view(-1, 1, 1, 1).float()
    return (tt * s - (1 - tt)) / (tt * tt * s + (1 - tt) ** 2).clamp_min(1e-8)


def anchor_velocity(zt: Tensor, t: Tensor, spectrum: Tensor) -> Tensor:
    """The best linear velocity per pixel-grid frequency and channel: ifft2(fft2(z_t) g(t, S)), float32."""
    return torch.fft.ifft2(torch.fft.fft2(zt.float()) * anchor_gain(t, spectrum)).real


def class_anchor_velocity(zt: Tensor, t: Tensor, means: Tensor, spectra: Tensor) -> Tensor:
    """The per-class closed-form velocity with per-row statistics: means and spectra [B, C, H, W] (the row's class),
    a = mu + ifft2(fft2(z_t - t mu) g(t, S)), float32. With mu = 0 and the pooled spectrum it is ``anchor_velocity``
    bit for bit (the same elementwise operations)."""
    tt = t.view(-1, 1, 1, 1).float()
    gain = (tt * spectra - (1 - tt)) / (tt * tt * spectra + (1 - tt) ** 2).clamp_min(1e-8)
    return means + torch.fft.ifft2(torch.fft.fft2((zt.float() - tt * means)) * gain).real


def class_statistics(x: Tensor, y: Tensor, n_classes: int, flips: bool = True, pooled: Tensor = None,
                     chunk: int = 2000):
    """Per class: the mean field and the power spectrum of the centered rows -> (means [K, C, H, W], spectra
    [K, C, H, W]), computed in float64 and returned in float32 on the device of x.

    Over the class's rows of x and, with ``flips``, their mirror images x.flip(-1) as well (a training stream that flips
    half of its pictures sees both). The spectrum is the estimator of ``measure_spectrum`` applied to the rows minus
    the class mean. A class with no rows gets a zero mean and ``pooled`` (required then)."""
    k = int(n_classes)
    shape = tuple(x.shape[1:])
    means = torch.zeros((k,) + shape, dtype=torch.float64, device=x.device)
    spectra = torch.zeros((k,) + shape, dtype=torch.float64, device=x.device)
    side = x.shape[-1]
    y = y.to(x.device)
    for c in range(k):
        rows = torch.nonzero(y == c, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            if pooled is None:
                raise ValueError(f"class {c} has no rows and no pooled spectrum was given")
            spectra[c] = pooled.to(device=x.device, dtype=torch.float64)
            continue
        views = (False, True) if flips else (False,)
        count = rows.numel() * len(views)
        total = torch.zeros(shape, dtype=torch.float64, device=x.device)
        for i in range(0, rows.numel(), chunk):
            xc = x[rows[i:i + chunk]].double()
            for fl in views:
                total += (xc.flip(-1) if fl else xc).sum(dim=0)
        mu = total / count
        power = torch.zeros(shape, dtype=torch.float64, device=x.device)
        for i in range(0, rows.numel(), chunk):
            xc = x[rows[i:i + chunk]].double()
            for fl in views:
                f = torch.fft.fft2((xc.flip(-1) if fl else xc) - mu) / side
                power += (f.real ** 2 + f.imag ** 2).sum(dim=0)
        means[c] = mu
        spectra[c] = power / count
    return means.float(), spectra.float()


def floor(s, t):
    """The anchor's error power per frequency, S / (t^2 S + (1 - t)^2): the least error of any linear predictor."""
    return s / (t * t * s + (1 - t) ** 2)
