"""The process of a source (capture.process): the noise-predicting teacher's and a flow source's formulas against direct
computations; the band weights at a real coordinate; the samplers (DDIM as before, Euler on a shifted grid); a flow
source's unguided side, its latent-scale ratio applied once on each path, and the trainer refusing it. A gated test
compares the library's flow sampler with a second implementation kept outside the repository."""
import importlib.util
import math
import os

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion.capture import BANDS, measure, pictures, reads  # noqa: E402
from geolip.modular_diffusion.capture import train as T  # noqa: E402
from geolip.modular_diffusion.capture.assemble import Assembly, ddim_timesteps, initial_noise, sample  # noqa: E402
from geolip.modular_diffusion.capture.data import synthetic_shards  # noqa: E402
from geolip.modular_diffusion.capture.process import EpsProcess, FlowProcess, make_process, shifted  # noqa: E402
from geolip.modular_diffusion.capture.schedule import (STEPS, add_noise, band_weights, band_weights_at,  # noqa: E402
                                                       sd15_alphas_cumprod)
from geolip.modular_diffusion.capture.teacher import tiny_teacher  # noqa: E402
from geolip.modular_diffusion.experiments import band_needs, capture  # noqa: E402

PROMPTS = ["a red house", "two boats", "a cat"]
SECOND_SAMPLER_SHA256 = "b01de28039b987a7f68445023e93ef58d2a7ba5d4d46091a9ea990466e5ac1cd"


@pytest.fixture(scope="module")
def flow():
    return tiny_teacher(0, kind="flow", shift=2.5)


def _x(seed, shape=(3, 4, 16, 16), dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=dtype)


# ------------------------------------------------------------------ the formulas
def test_eps_noised_and_clean_are_the_teachers_formulas():
    p = EpsProcess()
    ac = sd15_alphas_cumprod()
    x0, eps = _x(1), _x(2)
    for t in (0, 1, 381, 999):
        a = float(ac[t])
        want = math.sqrt(a) * x0 + math.sqrt(1 - a) * eps
        got = p.noised(x0, eps, t)
        assert torch.equal(got, add_noise(x0, eps, torch.full((3,), t), ac))
        assert torch.allclose(got, want, rtol=0, atol=1e-12)
        assert torch.allclose(p.clean(got, eps, t), x0, rtol=0, atol=1e-9)       # the exact answer gives x0 back
        assert float(p.snr(t)[0]) == pytest.approx(a / (1 - a), rel=1e-12)
    per_row = torch.tensor([3, 500, 990])
    z = p.noised(x0, eps, per_row)
    assert torch.allclose(p.clean(z, eps, per_row), x0, rtol=0, atol=1e-9)


def test_flow_noised_clean_and_snr_are_section_formulas():
    p = FlowProcess(2.0)
    x0, eps = _x(3), _x(4)
    for t in (0.0, 1.0, 375.5, 812.25, 1000.0):
        s = t / 1000
        z = p.noised(x0, eps, t)
        assert torch.allclose(z, (1 - s) * x0 + s * eps, rtol=0, atol=1e-15)
        v = eps - x0                                                               # the exact answer
        assert torch.allclose(p.clean(z, v, t), x0, rtol=0, atol=1e-12)
        if 0 < s:
            assert float(p.snr(t, 0.8)[0]) == pytest.approx(((1 - s) * 0.8 / s) ** 2, rel=1e-12)
    with pytest.raises(ValueError):
        p.snr(500.0)
    with pytest.raises(ValueError):
        p.noised(x0, eps, 1000.5)


def test_grids():
    assert EpsProcess().grid(50) == ddim_timesteps(50)
    p = FlowProcess(2.5)
    g = p.grid(30)
    want = [1000 * 2.5 * u / (1 + 1.5 * u) for u in [1 - i / 30 for i in range(30)]]
    assert len(g) == 30 and g[0] == 1000.0 and all(abs(a - b) < 1e-9 for a, b in zip(g, want))
    assert all(a > b for a, b in zip(g, g[1:]))
    assert p.after(g, 29) == 0.0 and abs(p.after(g, 0) - g[1]) < 1e-12
    assert shifted(0.5, 2.5) == pytest.approx(1.25 / 1.75)
    with pytest.raises(ValueError):
        make_process("flow")                                                       # a flow source needs its shift
    with pytest.raises(ValueError):
        make_process("velocity", 2.0)


def test_real_coordinate_weights_equal_the_whole_step_ones():
    t = torch.arange(STEPS)
    assert torch.equal(band_weights_at(t.double() / STEPS), band_weights(t))
    w = band_weights_at(torch.linspace(0, 1, 1001, dtype=torch.float64))
    assert torch.allclose(w.sum(dim=1), torch.ones(1001, dtype=torch.float64), atol=1e-12)
    with pytest.raises(ValueError):
        band_weights_at([1.01])


def test_assembly_weights_at_real_times(flow):
    asm = Assembly({b: flow for b in BANDS})
    t = torch.tensor([0.0, 350.0, 999.0])
    assert torch.equal(asm.weights(t), asm.weights(t.long()))


def test_assembly_at_a_non_integer_time_of_a_flow_source(flow):
    """Assembly.forward at a real scalar time (in the clean / middle crossfade) against a direct computation: the
    stretch weights written out from the ramp, each part's answer at the same real time, the sum in float64."""
    from geolip.modular_diffusion.capture.engine import UNetPart, build_engine
    base, tiny = build_engine(flow, "base").eval(), build_engine(flow, "tiny").eval()
    parts = {"clean": base, "middle": flow, "noisy": tiny}
    asm = Assembly(parts)
    states = flow.text_states(flow.tokenize(PROMPTS))
    z = _x(6, (3, 4, 16, 16), torch.float32)
    t = 372.5
    s = t / 1000

    def ramp(x):
        u = (max(-1.0, min(1.0, x / 0.06)) + 1) / 2
        return 0.5 - 0.5 * math.cos(math.pi * u)
    up1, up2 = ramp(s - 0.35), ramp(s - 0.75)
    w = (1 - up1, up1 * (1 - up2), up1 * up2)
    assert 0 < w[0] < 1 and 0 < w[1] < 1 and w[2] == 0.0
    assert torch.allclose(band_weights_at([s])[0], torch.tensor(w, dtype=torch.float64), rtol=0, atol=1e-12)
    tt = torch.full((3,), t)
    with torch.no_grad():
        want = (w[0] * base(z, tt, states).double() + w[1] * UNetPart(flow.unet)(z, tt, states).double()).float()
        got = asm(z, t, states)
    assert torch.allclose(got, want, rtol=0, atol=1e-6)
    with torch.no_grad():                                   # the real time reaches the parts: not the whole step 372
        assert not torch.equal(asm(z, 372, states), got)


# ------------------------------------------------------------------ the samplers
class Constant(torch.nn.Module):
    is_part = True

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, z, t, states):
        return torch.full_like(z, self.value)


def test_euler_is_exact_on_a_constant_velocity_field():
    states = torch.zeros(2, 77, 8)
    z0 = initial_noise(5, 2, (4, 8, 8))
    for steps, shift in ((1, 1.0), (7, 2.5), (30, 2.0)):
        out = sample(Constant(0.75), states, torch.zeros(1, 77, 8), 0, steps, 4.0, z=z0.clone(),
                     process=FlowProcess(shift))
        assert torch.allclose(out, z0 - 0.75, rtol=0, atol=1e-6)               # s runs from 1 to 0


def test_the_eps_sampler_is_the_ddim_loop_to_the_bit():
    t = tiny_teacher(0)
    states = t.text_states(t.tokenize(PROMPTS))
    empty = t.empty_states(1)
    got = sample(t, states, empty, 3, steps=8, guidance=7.5, latent_shape=(4, 16, 16))
    again = sample(t, states, empty, 3, steps=8, guidance=7.5, latent_shape=(4, 16, 16), process=t.process)
    ac = sd15_alphas_cumprod()
    z = initial_noise(3, 3, (4, 16, 16))
    for k in ddim_timesteps(8):                                                    # the loop as it was written
        tt = torch.full((3,), k, dtype=torch.long)
        both = t.predict(torch.cat([z, z]), torch.cat([tt, tt]), torch.cat([empty.expand(3, -1, -1), states]))
        e = both[:3] + 7.5 * (both[3:] - both[:3])
        a, prev = float(ac[k]), k - 1000 // 8
        a_prev = float(ac[prev]) if prev >= 0 else float(ac[0])
        x0 = (z - math.sqrt(1.0 - a) * e) / math.sqrt(a)
        z = math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * e
    assert torch.equal(got, z) and torch.equal(again, z)


def _second_chain(source, states, z, steps, shift, guidance, first_only=True):
    """Section 2's chain written out here: Euler on s_i = k u_i / (1 + (k - 1) u_i), zero states unguided."""
    x = z.clone()
    sig = [shift * u / (1 + (shift - 1) * u) for u in [1 - i / steps for i in range(steps + 1)]]
    zero = torch.zeros_like(states)
    for i in range(steps):
        t = torch.full((x.shape[0],), 1000.0 * sig[i])
        v_p = source.unet(x, t, encoder_hidden_states=states).sample
        v_n = source.unet(x, t, encoder_hidden_states=zero).sample
        x = x + (sig[i + 1] - sig[i]) * (v_n + guidance * (v_p - v_n))
        if first_only:
            return x
    return x


def test_a_flow_chain_equals_a_second_implementation_at_step_one(flow):
    states = flow.text_states(flow.tokenize(PROMPTS))
    z0 = initial_noise(9, 3, (4, 16, 16))
    seen = []
    sample(flow, states, flow.unguided_states(1), 0, 6, 4.0, z=z0, process=flow.process,
           on_step=lambda t, z, e: seen.append(z.clone()))
    want = _second_chain(flow, states, z0, 6, 2.5, 4.0)
    assert torch.allclose(seen[1], want, rtol=0, atol=2e-6 * float(want.abs().max()))
    full = sample(flow, states, flow.unguided_states(1), 0, 6, 4.0, z=z0, process=flow.process)
    assert torch.allclose(full, _second_chain(flow, states, z0, 6, 2.5, 4.0, first_only=False), rtol=0, atol=1e-4)
    again = sample(flow, states, flow.unguided_states(1), 0, 6, 4.0, z=z0, process=flow.process)
    assert torch.equal(full, again)                                                # the sampler repeats itself


# ------------------------------------------------------------------ the source
def test_the_unguided_side(flow):
    z = flow.unguided_states(3)
    assert z.shape == (3, 77, flow.context_dim) and bool((z == 0).all()) and flow.unguided == "zeros"
    eps = tiny_teacher(0)
    assert eps.unguided == "empty" and torch.equal(eps.unguided_states(2), eps.empty_states(2))
    assert torch.equal(tiny_teacher(0, kind="flow", shift=2.0, unguided="empty").unguided_states(1), eps.empty_states(1))
    with pytest.raises(ValueError):
        tiny_teacher(0, unguided="none")
    assert flow.record() == {"kind": "flow", "shift": 2.5, "unguided": "zeros", "kappa": 1.0}


def test_the_prompt_read_of_a_flow_source_uses_its_sampler_and_zero_states(flow, monkeypatch):
    seen = []
    from geolip.modular_diffusion.capture import assemble
    real = assemble.sample

    def spy(model, states, empty, seed, *args, **kw):
        seen.append((kw.get("process"), empty))
        return real(model, states, empty, seed, *args, **kw)
    monkeypatch.setattr(assemble, "sample", spy)
    r = measure.text_need(flow, PROMPTS, seed=1, steps=4, log=None)
    assert seen and all(p is flow.process and bool((e == 0).all()) for p, e in seen)
    assert r["repeat"] == 0.0


def test_kappa_is_applied_once_on_each_path(tmp_path, monkeypatch):
    k = 0.25
    data = str(tmp_path / "rows")
    synthetic_shards(data, rows=6, shard_size=3, size=16, seed=2)
    from geolip.modular_diffusion.capture.data import rows_in_order
    x0, _ = rows_in_order(data, 0, 4)
    a = capture.parse(capture.DEFAULTS, tiny=1, kind="flow", shift="2.5", kappa=k, probe_data=data, probe_rows=4,
                      out=str(tmp_path / "x"), probe_per_band=2)
    src, _ = capture._teacher(a)
    probe = capture._probe(a, src)
    assert torch.equal(probe.x0, x0 * k)                                         # the capture probe: once
    job = band_needs._Job(band_needs.parse(band_needs.DEFAULTS, tiny=1, kind="flow", shift="2.5", kappa=k, rows=4,
                                           prompts=4, out=str(tmp_path / "bn")), print)
    job.data(data, 4)
    assert torch.equal(band_needs._probe(job, 0, 0).x0, x0 * k)                   # the band-needs probe: once
    got = {}
    real_decode = pictures.decode

    def spy(vae, z, batch=8):
        got.setdefault("z", []).append(z.clone())
        return real_decode(vae, z, batch)
    real_sample = pictures.sample_latents

    def spy_sample(*args, **kw):
        z = real_sample(*args, **kw)
        got.setdefault("sampled", []).append(z.clone())
        return z
    monkeypatch.setattr(pictures, "decode", spy)
    monkeypatch.setattr(pictures, "sample_latents", spy_sample)
    capture.main(variant="pictures", tiny=1, kind="flow", shift="2.5", kappa=k, pic_data=data, pic_n=4, group=2,
                 sample_steps=3, judge="fake", clip="fake", sheet_n=0, out=str(tmp_path / "pic"),
                 parts="clean=teacher,middle=teacher,noisy=teacher")
    assert torch.equal(got["z"][0], got["sampled"][0] / k)                         # decoded after one division
    capture.main(variant="pictures", tiny=1, kind="flow", shift="2.5", kappa=k, pic_data=data, pic_n=4, group=2,
                 source="rows", judge="fake", clip="fake", sheet_n=0, out=str(tmp_path / "rows_pic"))
    assert torch.equal(got["z"][1], x0)                                            # the rows' own: never scaled


def test_a_flow_source_does_not_train_a_band_engine(flow, tmp_path):
    with pytest.raises(ValueError, match="noise-predicting"):
        T.train_band_engine(flow, "tiny", "clean", capture.random_source(0, (4, 16, 16)), steps=1, batch=2, seed=0,
                            out=str(tmp_path / "t"))
    with pytest.raises(ValueError, match="noise-predicting"):
        capture.main(variant="band_engine", tiny=1, kind="flow", shift="2.5", steps=1, out=str(tmp_path / "v"))
    with pytest.raises(ValueError):
        capture.main(variant="born", tiny=1, kind="flow", out=str(tmp_path / "w"))          # no shift


def test_born_reads_of_a_flow_source_use_its_noised_inputs(flow):
    states = flow.text_states(flow.tokenize(PROMPTS))
    probe = reads.make_probe(states, seed=4, per_band=2, latent_shape=(4, 16, 16), process=flow.process)
    t = int(probe.steps["middle"][0])
    assert torch.equal(probe.z(t), (1 - t / 1000) * probe.x0 + (t / 1000) * probe.eps)
    r = reads.born_read(flow, "full", "middle", probe)
    assert r["mse"] == 0.0                                                         # the source's own copy
    assert reads.born_read(flow, "base", "noisy", probe)["mse"] > 0


def test_gated_a_source_from_a_unet_folder_of_its_own():
    """Opt-in (MD_SD15_PATH, a local diffusers folder of the teacher): a source loaded with ``unet=`` reads its U-Net from
    that folder (here the teacher's own unet/, so every tensor equals the teacher's), the rest from the teacher folder."""
    path = os.environ.get("MD_SD15_PATH", "")
    if not path or not os.path.isdir(path):
        pytest.skip("MD_SD15_PATH is not set")
    from geolip.modular_diffusion.capture.teacher import load_teacher
    t = load_teacher(path)
    s = load_teacher(path, unet=os.path.join(path, "unet"), kind="flow", shift=2.5)
    assert s.kind == "flow" and s.unguided == "zeros" and t.kind == "eps"
    ts, ss = t.unet.state_dict(), s.unet.state_dict()
    assert set(ts) == set(ss) and all(torch.equal(ts[k], ss[k]) for k in ts)
    with pytest.raises(FileNotFoundError):
        load_teacher(path, unet=os.path.join(path, "text_encoder"), kind="flow", shift=2.5)


# ------------------------------------------------------------------ the gated comparison
def _second_sampler():
    folder = os.environ.get("MD_FLOW_SAMPLER", "")
    if not folder:
        pytest.skip("MD_FLOW_SAMPLER is not set")
    import hashlib

    def sha256(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    found = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
             if n.endswith(".py") and sha256(os.path.join(folder, n)) == SECOND_SAMPLER_SHA256]
    assert found, "the second flow sampler (by its SHA-256) is not in MD_FLOW_SAMPLER"
    spec = importlib.util.spec_from_file_location("second_flow_sampler", found[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gated_the_flow_chain_against_the_second_sampler(flow):
    mod = _second_sampler()
    states = flow.text_states(flow.tokenize(PROMPTS))
    want = mod.flow_sample(flow.unet, states, n_steps=6, guidance=4.0, shift=2.5, seed=21, device="cpu", latent_hw=16)
    got = sample(flow, states, flow.unguided_states(1), 21, 6, 4.0, latent_shape=(4, 16, 16), process=flow.process)
    assert torch.allclose(got, want, rtol=0, atol=1e-4 * float(want.abs().max()))
    first = mod.flow_sample(flow.unet, states, n_steps=1, guidance=4.0, shift=2.5, seed=21, device="cpu", latent_hw=16)
    one = sample(flow, states, flow.unguided_states(1), 21, 1, 4.0, latent_shape=(4, 16, 16), process=flow.process)
    assert np.allclose(one.numpy(), first.numpy(), rtol=0, atol=2e-6 * float(first.abs().max()))
