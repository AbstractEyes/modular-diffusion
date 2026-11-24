"""Capture, the bands and the schedule: the address gate gives the band weights of the formulas on every step (test 1),
and the band sampling law keeps its draws in the band at rho 0 and sends the stated share to the whole range (test 8)."""
import math

import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion import frame
from geolip.modular_diffusion.capture import BANDS, schedule


def _formulas():
    out = []
    for t in range(1000):
        s = t / 1000

        def ramp(x):
            u = (min(max(x / 0.06, -1.0), 1.0) + 1) / 2
            return 0.5 - 0.5 * math.cos(math.pi * u)
        up1, up2 = ramp(s - 0.35), ramp(s - 0.75)
        out.append((1 - up1, up1 * (1 - up2), up1 * up2))
    return torch.tensor(out, dtype=torch.float64)


def test_1_the_gate_gives_the_band_weights_on_every_step():
    w = schedule.band_weights(torch.arange(1000))
    assert w.dtype == torch.float64 and w.shape == (1000, 3)
    ref = _formulas()
    assert float((w - ref).abs().max()) <= 1e-12
    assert float((w.sum(dim=1) - 1).abs().max()) <= 1e-12
    # the weights are the frame's projective gate of three declared owners, not a second window
    owners = schedule.band_owners()
    assert [o.name for o in owners] == list(BANDS)
    assert torch.equal(w, frame.projective(owners, torch.arange(1000).double() / 1000))
    assert float(w.min()) >= 0.0


def test_1_step_sets_core_and_crossfade():
    clean, middle, noisy = (schedule.band_steps(b) for b in BANDS)
    assert int(clean[0]) == 0 and int(noisy[-1]) == 999
    # a weight is above zero within 0.06 of a band's window, so the sets overlap only in the crossfades
    assert int(clean[-1]) == 409 and int(middle[0]) == 291 and int(middle[-1]) == 809 and int(noisy[0]) == 691
    cross = set(schedule.crossfade_steps().tolist())
    assert cross == set(range(291, 410)) | set(range(691, 810))
    for b in BANDS:
        core = set(schedule.core_steps(b).tolist())
        assert core == set(schedule.band_steps(b).tolist()) - cross
    assert schedule.band_index("middle") == 1 and schedule.band_index(2) == 2
    with pytest.raises(ValueError):
        schedule.band_index("loud")


def test_8_the_band_sampling_law():
    n = 20000
    for band in BANDS:
        own = set(schedule.band_steps(band).tolist())
        t = schedule.sample_steps(band, n, 0.0, torch.Generator().manual_seed(3))
        assert t.dtype == torch.long and t.shape == (n,)
        assert all(int(x) in own for x in t)
        assert len(set(t.tolist())) > 0.95 * len(own)              # spread over the whole step set
        t = schedule.sample_steps(band, n, 0.3, torch.Generator().manual_seed(4))
        out = sum(1 for x in t.tolist() if x not in own)
        # a whole-range draw lands outside the step set with probability 1 - |own| / 1000
        expect = n * 0.3 * (1 - len(own) / 1000)
        assert abs(out - expect) < 5 * math.sqrt(expect), (band, out, expect)
        assert int(t.min()) >= 0 and int(t.max()) <= 999
    with pytest.raises(ValueError):
        schedule.sample_steps("clean", 4, 1.5, torch.Generator())


def test_the_schedule_is_the_teachers():
    from diffusers import DDIMScheduler
    ref = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear").alphas_cumprod.double()
    ac = schedule.sd15_alphas_cumprod()
    assert ac.shape == (1000,) and float((ac - ref).abs().max()) < 1e-6
    x0, eps = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    t = torch.tensor([0, 500, 999])
    z = schedule.add_noise(x0, eps, t, ac)
    a = ac[999]
    assert torch.allclose(z[2], a.sqrt().float() * x0[2] + (1 - a).sqrt().float() * eps[2])
