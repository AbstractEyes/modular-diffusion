"""Windows of t (test 1), band masks (test 2) and the anchor as the least-squares linear predictor (test 3)."""
import numpy as np
import pytest
import torch

from geolip.modular_diffusion import spectral as S
from geolip.modular_diffusion import windows as W


def _t_grid():
    return torch.linspace(0, 1, 10002, dtype=torch.float32)[:-1]           # 10,001 values in [0, 1)


@pytest.mark.parametrize("preset", sorted(W.STAGES))
def test_1_hard_windows_partition(preset):
    t = _t_grid()
    edges = W.stage_edges(preset)
    t = torch.cat([t, torch.tensor(edges, dtype=torch.float32)])          # every edge itself
    cuts = W.stage_cuts(preset)
    ws = torch.stack([W.window_weight(t, cuts[s], cuts[s + 1], 0.0) for s in range(len(cuts) - 1)])
    assert bool(((ws == 0) | (ws == 1)).all())
    assert torch.equal(ws.sum(0), torch.ones_like(t))


@pytest.mark.parametrize("preset", ["3", "3even", "5mid", "5even"])
def test_1_crossfade_sums_to_one(preset):
    hw = 0.06
    t = _t_grid()
    cuts = W.stage_cuts(preset)
    ws = torch.stack([W.window_weight(t, cuts[s], cuts[s + 1], hw) for s in range(len(cuts) - 1)])
    total = ws.sum(0)
    ulp = float(torch.finfo(torch.float32).eps)
    assert float((total - 1).abs().max()) <= ulp
    edges = torch.tensor(W.stage_edges(preset), dtype=torch.float32)
    outside = ((t[:, None] - edges[None, :]).abs() >= hw).all(1) if len(edges) else torch.ones_like(t, dtype=torch.bool)
    assert bool((total[outside] == 1.0).all())
    assert bool(((ws[:, outside] == 0) | (ws[:, outside] == 1)).all())


def test_1_ramp_is_hard_at_zero_width():
    x = torch.tensor([-1e-7, 0.0, 1e-7])
    assert W.ramp(x, 0.0).tolist() == [0.0, 1.0, 1.0]
    assert W.ramp(torch.tensor([-0.06, 0.0, 0.06]), 0.06).tolist() == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        W.stage_edges("nope")


def test_2_band_masks():
    masks, r = S.band_masks(16, 3)
    stack = torch.stack(masks)
    assert stack.dtype == torch.bool
    assert torch.equal(stack.sum(0), torch.ones(16, 16, dtype=torch.long))      # disjoint, covering everything
    assert [int(m.sum()) for m in masks] == [211, 36, 9]
    assert bool(masks[0][r == 0.25].all())                                      # wavelength exactly 4 is fine
    assert bool(masks[1][r == 0.125].all())                                     # wavelength exactly 8 is mid
    one, _ = S.band_masks(16, 1)
    assert bool(one[0].all())
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3, 256, 12, generator=gen)
    parts = [S.keep_band(x, m.float(), 16) for m in masks]
    assert float((sum(parts) - x).abs().max()) < 1e-6


def test_3_anchor_is_the_least_squares_predictor():
    grid, n = 16, 20000
    s = S.field_spectrum(grid, 1.225, 0.060)                                    # [G, G] float64, mean 1
    gen = torch.Generator().manual_seed(7)
    w = torch.randn(n, 1, grid, grid, generator=gen, dtype=torch.float64)
    x1 = torch.fft.ifft2(torch.fft.fft2(w) * s.sqrt()).real
    z0 = torch.randn(n, 1, grid, grid, generator=gen, dtype=torch.float64)
    spec = s[None]                                                               # [C, H, W]
    for t in (0.2, 0.5, 0.8):
        zt = (1 - t) * z0 + t * x1
        v = x1 - z0
        Z, V = torch.fft.fft2(zt) / grid, torch.fft.fft2(v) / grid
        szz = (Z.real ** 2 + Z.imag ** 2).sum(0)[0]
        fitted = (V * Z.conj()).real.sum(0)[0] / szz
        tt = torch.full((1,), t, dtype=torch.float64)
        g = (tt * spec - (1 - tt)) / (tt * tt * spec + (1 - tt) ** 2)            # the closed form, in float64
        g = g[0]
        se = (S.floor(s, t) / szz).sqrt()
        assert float(((fitted - g).abs() / se).max()) < 5.0
        # the anchor's loss is the floor; scaling the gain by 1 +- 1% raises it
        def loss(gain):
            pred = torch.fft.ifft2(torch.fft.fft2(zt) * gain).real
            return float(((pred - v) ** 2).mean())
        base = loss(g)
        floor = float(S.floor(s, t).mean())
        assert abs(base - floor) / floor < 0.01
        assert loss(g * 1.01) > base and loss(g * 0.99) > base
        # the package's float32 anchor agrees with the float64 closed form
        va = S.anchor_velocity(zt[:64].float(), torch.full((64,), t), spec)
        ref = torch.fft.ifft2(torch.fft.fft2(zt[:64]) * g).real
        assert float((va.double() - ref).abs().max()) < 1e-4
    g0 = S.anchor_gain(torch.zeros(1), spec)
    g1 = S.anchor_gain(torch.ones(1), spec)
    assert bool((g0 == -1).all()) and bool((g1 == 1).all())
