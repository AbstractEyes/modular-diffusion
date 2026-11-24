"""Capture, precision: parts held in different floating-point precisions are read on one probe and sampled together.
A teacher run in a lower precision beside a full-precision engine is the usual case on a card; here the teacher's
U-Net is cast to float64 (every CPU supports it), which exercises the same casts."""
import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion.capture import reads
from geolip.modular_diffusion.capture.assemble import Assembly, sample
from geolip.modular_diffusion.capture.engine import build_engine
from geolip.modular_diffusion.capture.teacher import tiny_teacher


def test_parts_of_different_precision_share_a_probe_and_a_sampler():
    teacher = tiny_teacher(0)
    engine = build_engine(teacher.unet, "base")                     # float32, the teacher's weights inherited
    states = teacher.text_states(teacher.tokenize(["a red house", "two boats"]))
    probe = reads.make_probe(states, seed=5, per_band=2, latent_shape=(4, 16, 16), empty=teacher.empty_states(1))
    before = reads.band_read(engine, teacher, "middle", probe)

    teacher.unet.double()                                           # the teacher now runs in another precision
    assert next(teacher.unet.parameters()).dtype == torch.float64
    assert next(engine.unet.parameters()).dtype == torch.float32
    assert probe.x0.dtype == torch.float32 and probe.eps.dtype == torch.float32

    after = reads.band_read(engine, teacher, "middle", probe)       # must not raise; the read barely moves
    key = [k for k, v in before.items() if isinstance(v, float)][0]
    assert after[key] == pytest.approx(before[key], rel=1e-3)

    mixed = Assembly({"clean": engine, "middle": teacher, "noisy": teacher})
    z = sample(mixed, states, teacher.empty_states(1), seed=3, steps=10, guidance=3.0, latent_shape=(4, 16, 16))
    assert z.dtype == torch.float32 and bool(torch.isfinite(z).all())
