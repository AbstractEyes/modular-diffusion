"""Opt-in: the real teacher (Stable Diffusion 1.5) on the CPU. Skipped unless MD_SD15_PATH names a local diffusers
folder of it (for example the snapshot folder in the local hub cache). Nothing is downloaded; no card is used.

Checks: the engines of the named shapes base, small and tiny have the published parameter counts (580M, 483M, 324M)
to within 1%; the engine of shape ``full`` reads exactly 0 on one small probe (one row, 64 x 64 latents, two steps)."""
import os

import pytest
import torch

PUBLISHED = {"base": 580e6, "small": 483e6, "tiny": 324e6}


@pytest.fixture(scope="module")
def sd15():
    path = os.environ.get("MD_SD15_PATH", "")
    if not path:
        pytest.skip("MD_SD15_PATH is not set")
    if not os.path.isdir(path):
        pytest.skip(f"MD_SD15_PATH is not a folder: {path}")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    from geolip.modular_diffusion.capture.teacher import load_teacher
    teacher = load_teacher(path, device="cpu", dtype=torch.float32)
    # Tensors read from a mapped safetensors file can sit at addresses that are not 64-byte aligned, and the CPU
    # kernels then take another path: equal numbers give answers that differ in the last digits (measured: 2e-6 at one
    # step against a fresh copy of the same U-Net). ``load_teacher`` copies every tensor into memory of its own, so
    # the teacher's kernels are the engine's; the test below depends on it.
    assert all(p.data_ptr() % 64 == 0 for p in teacher.unet.parameters())
    return teacher


def test_sd15_named_shapes_have_the_published_sizes(sd15):
    from geolip.modular_diffusion.capture.engine import build_engine, param_count
    assert torch.get_num_threads() <= 8
    assert next(sd15.unet.parameters()).device.type == "cpu"
    counts = {}
    for name, want in PUBLISHED.items():
        engine = build_engine(sd15, name)
        counts[name] = param_count(engine)
        del engine
    for name, want in PUBLISHED.items():
        assert abs(counts[name] - want) / want < 0.01, (name, counts[name], want)


def test_sd15_full_reads_exactly_zero(sd15):
    from geolip.modular_diffusion.capture.engine import build_engine
    from geolip.modular_diffusion.capture.reads import band_read, make_probe
    assert torch.get_num_threads() <= 8
    states = sd15.text_states(sd15.tokenize(["a photograph of a red house by a lake"]))
    probe = make_probe(states, seed=0, per_band=2, latent_shape=(4, 64, 64), alphas_cumprod=sd15.alphas_cumprod)
    engine = build_engine(sd15, "full")
    r = band_read(engine, sd15, "middle", probe)
    assert len(r["steps"]) == 2
    assert r["mse"] == 0.0
