"""Capture, the assembly and its sampler: an assembly whose three parts are the teacher samples exactly the teacher's
latents (the seam test, test 4), and guidance with the same weights commutes with the blend (test 5)."""
import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion.capture import BANDS
from geolip.modular_diffusion.capture.assemble import Assembly, ddim_timesteps, guided, sample
from geolip.modular_diffusion.capture.engine import as_part, build_engine, fresh_engine
from geolip.modular_diffusion.capture.schedule import band_weights
from geolip.modular_diffusion.capture.teacher import tiny_teacher


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


@pytest.fixture(scope="module")
def states(teacher):
    return teacher.text_states(teacher.tokenize(["a red house", "two boats"])), teacher.empty_states(1)


def test_ddim_steps():
    ts = ddim_timesteps(50)
    assert ts[0] == 981 and ts[-1] == 1 and len(ts) == 50 and all(a - b == 20 for a, b in zip(ts, ts[1:]))
    assert ddim_timesteps(10)[:3] == [901, 801, 701]


def test_4_the_seam_test(teacher, states):
    cond, empty = states
    for parts in ({b: teacher for b in BANDS}, {b: teacher.unet for b in BANDS}):
        model = Assembly(parts)
        assert len(model.parts) == 1                            # one part serving three bands runs once
        z_t = sample(teacher, cond, empty, seed=5, steps=50, guidance=7.5)
        z_a = sample(model, cond, empty, seed=5, steps=50, guidance=7.5)
        assert z_t.shape == (2, 4, 16, 16)
        assert torch.equal(z_t, z_a)
    # the crossfade steps were visited, so the blend itself was exercised
    w = band_weights(torch.tensor(ddim_timesteps(50)))
    assert int(((w > 0).sum(dim=1) > 1).sum()) >= 4


def test_4_three_separate_copies_of_the_teacher(teacher, states):
    cond, empty = states
    copies = {b: build_engine(teacher, "full") for b in BANDS}     # three distinct objects with the teacher's numbers
    model = Assembly(copies)
    assert len(model.parts) == 3
    z_t = sample(teacher, cond, empty, seed=2, steps=50)
    z_a = sample(model, cond, empty, seed=2, steps=50)
    # in the crossfade two equal answers are blended by weights summing to one, in float64: the teacher's to the bit
    assert torch.equal(z_t, z_a)


def test_only_live_parts_run(teacher):
    calls = {b: 0 for b in BANDS}

    class Counting(torch.nn.Module):
        def __init__(self, band):
            super().__init__()
            self.band, self.inner = band, as_part(teacher)

        def forward(self, z, t, s):
            calls[self.band] += 1
            return self.inner(z, t, s)

    model = Assembly({b: Counting(b) for b in BANDS})
    z = torch.randn(1, 4, 16, 16)
    s = teacher.empty_states(1)
    model(z, torch.tensor([100]), s)
    assert calls == {"clean": 1, "middle": 0, "noisy": 0}
    model(z, torch.tensor([350]), s)
    assert calls == {"clean": 2, "middle": 1, "noisy": 0}
    # rows at different steps: each part runs on its own rows only
    out = model(torch.randn(2, 4, 16, 16), torch.tensor([100, 900]), s.expand(2, -1, -1))
    assert calls == {"clean": 3, "middle": 1, "noisy": 1} and out.shape == (2, 4, 16, 16)


def test_5_guidance_commutes_with_the_blend(teacher, states):
    cond, empty = states
    parts = {"clean": build_engine(teacher, "base"), "middle": fresh_engine(teacher, "small", seed=3),
             "noisy": build_engine(teacher, "tiny")}
    model = Assembly(parts)
    z = torch.randn(2, 4, 16, 16, generator=torch.Generator().manual_seed(9))
    for t in (100, 330, 350, 380, 720, 900):                       # cores and both crossfades
        tt = torch.full((2,), t)
        g_blend = guided(model, z, tt, cond, empty, 7.5)
        w = band_weights(tt)[0]
        g_parts = sum(float(w[i]) * guided(parts[b], z, tt, cond, empty, 7.5).double()
                      for i, b in enumerate(BANDS) if float(w[i]) > 0)
        assert torch.allclose(g_blend.double(), g_parts, rtol=1e-5, atol=1e-5), t


def _diffusers_ddim(unet, cond, empty, z, steps, guidance):
    """The diffusers library's own DDIM loop (eta 0, epsilon prediction) with the same guidance arithmetic."""
    from diffusers import DDIMScheduler
    sch = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                        steps_offset=1, set_alpha_to_one=False, clip_sample=False, prediction_type="epsilon")
    sch.set_timesteps(steps)
    b = z.shape[0]
    states = torch.cat([empty.expand(b, -1, -1), cond])
    ts = []
    with torch.no_grad():
        for t in sch.timesteps:
            ts.append(int(t))
            both = unet(torch.cat([z, z]), t, encoder_hidden_states=states).sample
            u, c = both[:b], both[b:]
            eps = u + guidance * (c - u)
            z = sch.step(eps, t, z, eta=0.0).prev_sample
    return z, ts


@pytest.mark.parametrize("steps", (10, 50))
def test_the_sampler_agrees_with_the_diffusers_ddim_scheduler(teacher, states, steps):
    """The same start noise, prompts and guidance through assemble.sample and through diffusers.DDIMScheduler: the
    final latents agree to 1e-4 relative (the step lists are the same)."""
    from geolip.modular_diffusion.capture.assemble import initial_noise
    cond, empty = states
    z0 = initial_noise(3, cond.shape[0], (4, 16, 16))
    ours = sample(teacher, cond, empty, seed=3, steps=steps, guidance=7.5)
    theirs, ts = _diffusers_ddim(teacher.unet, cond, empty, z0.clone(), steps, 7.5)
    assert ts == ddim_timesteps(steps)
    rel = float((ours.double() - theirs.double()).norm() / theirs.double().norm())
    assert rel < 1e-4, (steps, rel)


def test_an_assembly_needs_every_band(teacher):
    with pytest.raises(ValueError):
        Assembly({"clean": teacher, "middle": teacher})
