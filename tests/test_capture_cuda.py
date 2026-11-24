"""Capture on a card: the reads, the measurements, the sampler and the trainer with its resume, with every tensor on
the card and the teacher kept in half precision beside full-precision engines. A tensor left behind on the CPU, or a
part fed another precision than its own, is caught only here.

Off by default: set MD_TEST_CUDA=1 on a machine with a card (the suite itself runs with the card hidden). With
MD_TEST_DEVICE=cpu the same logic runs without a card; the teacher then stays in float32 (the half-precision kernels
these tests are about are a card's)."""
import os

import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion.capture import measure, reads  # noqa: E402
from geolip.modular_diffusion.capture import train as T  # noqa: E402
from geolip.modular_diffusion.capture.assemble import Assembly, sample  # noqa: E402
from geolip.modular_diffusion.capture.engine import build_engine, fresh_engine  # noqa: E402
from geolip.modular_diffusion.capture.teacher import tiny_teacher  # noqa: E402
from geolip.modular_diffusion.experiments import capture  # noqa: E402
from geolip.modular_diffusion.experiments.capture import random_source  # noqa: E402

DEV = os.environ.get("MD_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skipif(os.environ.get("MD_TEST_CUDA") != "1" or (DEV == "cuda" and not torch.cuda.is_available()),
                                reason="MD_TEST_CUDA=1 and a visible card are needed")
HALF = torch.float16 if DEV == "cuda" else torch.float32
PROMPTS = ["a red house", "two boats", "a cat on a sofa", "a road in the snow"]


def _teacher(dtype=torch.float32):
    return tiny_teacher(0).to(DEV, dtype)


def _probe(teacher, batch=0):
    states = teacher.text_states(teacher.tokenize(PROMPTS))
    return reads.make_probe(states, seed=5, per_band=2, latent_shape=(4, 16, 16), empty=teacher.empty_states(1),
                            batch=batch)


def test_reads_on_the_card():
    teacher = _teacher()
    probe = _probe(teacher)
    assert probe.x0.device.type == DEV and probe.states.device.type == DEV
    assert reads.band_read(teacher, teacher, "middle", probe)["mse"] == 0.0
    full = build_engine(teacher, "full")
    assert next(full.parameters()).device.type == DEV
    assert reads.band_read(full, teacher, "noisy", probe)["mse"] == 0.0          # the teacher's copy reads exactly 0
    born = reads.born_read(teacher, "base", "clean", probe)
    fresh = reads.band_read(fresh_engine(teacher, "base", seed=1), teacher, "clean", probe)
    assert 0.0 < born["mse"] < fresh["mse"]
    cut = _probe(teacher, batch=3)                                               # the same rows, three a forward
    assert reads.band_read(full, teacher, "noisy", cut)["mse"] == 0.0
    assert reads.born_read(teacher, "base", "clean", cut)["mse"] == pytest.approx(born["mse"], rel=1e-3)


def test_a_half_precision_teacher_beside_full_precision_engines():
    teacher = _teacher(HALF)
    assert next(teacher.unet.parameters()).dtype == HALF
    probe = _probe(teacher)
    assert probe.x0.dtype == torch.float32 and probe.eps.dtype == torch.float32
    engine = build_engine(teacher, "base")
    assert next(engine.unet.parameters()).dtype == torch.float32
    r = reads.band_read(engine, teacher, "middle", probe)
    assert r["mse"] > 0.0 and r["mse"] < float("inf")
    mixed = Assembly({"clean": engine, "middle": teacher, "noisy": teacher})
    z = sample(mixed, probe.states, probe.empty, seed=3, steps=10, guidance=3.0, latent_shape=(4, 16, 16))
    assert z.dtype == torch.float32 and z.device.type == DEV and bool(torch.isfinite(z).all())


def test_measurements_on_the_card():
    full = _teacher()
    scores = measure.removal_scores(full, _probe(full, batch=2), log=None)
    assert len(scores["rows"]) == 18 and all(0.0 < r["mse"] < float("inf") for r in scores["rows"])
    half = _teacher(HALF)
    need = measure.text_need(half, PROMPTS[:3], seed=1, steps=6, guidance=2.0, latent_shape=(4, 16, 16), log=None)
    assert need["repeat"] == 0.0                                                 # the sampler repeats itself
    assert need["empty_everywhere"] > 0.0
    assert all(0.0 <= r["empty"] < float("inf") and 0.0 <= r["shuffled"] < float("inf") for r in need["rows"])
    ids = half.tokenize(PROMPTS[:3])                                             # the same prompts as token ids
    again = measure.text_need(half, ids, seed=1, steps=6, guidance=2.0, latent_shape=(4, 16, 16), log=None)
    assert again["empty_everywhere"] == need["empty_everywhere"]
    c = measure.cost(full, "tiny", batch=2, repeats=1)
    assert c["seconds"] > 0 and c["teacher_seconds"] > 0


def test_the_trainer_and_its_resume_on_the_card(tmp_path):
    teacher = _teacher(HALF)
    source = random_source(0, (4, 16, 16))
    lines = []
    kw = dict(steps=6, batch=4, seed=0, lr=1e-3, log_every=1, log=lines.append, save_every=3)
    one = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "one"), **kw)
    assert one["finished"] and all(r["loss"] == r["loss"] and r["grad_norm"] > 0 for r in one["log"])
    assert next(one["engine"].parameters()).device.type == DEV
    assert next(one["engine"].parameters()).dtype == torch.float32
    part = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "two"), until=3, **kw)
    assert not part["finished"] and part["step"] == 3
    two = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "two"), **kw)
    assert two["finished"] and two["step"] == 6 and any(x.startswith("RESUME") for x in lines)
    a = torch.cat([p.detach().flatten().cpu() for p in one["engine"].parameters()])
    b = torch.cat([p.detach().flatten().cpu() for p in two["engine"].parameters()])
    assert torch.allclose(a, b, rtol=1e-2, atol=1e-3)     # a card does not promise the last digit (the CPU test does)
    read = reads.band_read(one["average"].eval(), teacher, "clean", _probe(teacher))
    assert 0.0 <= read["mse"] < float("inf")


def test_the_experiment_on_the_card(tmp_path):
    kind = "fp16" if DEV == "cuda" else "fp32"
    born = capture.main(variant="born", tiny=1, out=str(tmp_path / "born"), device=DEV, dtype=kind)
    assert born["rows"] and all(r["mse"] == r["mse"] for r in born["rows"])
    job = capture.main(variant="band_engine", tiny=1, out=str(tmp_path / "job"), device=DEV, dtype=kind, steps=4,
                       batch=2, save_every=0, log_every=1)
    assert job["train"]["finished"] and job["rows"]
    parts = f"clean={tmp_path / 'job'},middle=teacher,noisy=teacher"
    whole = capture.main(variant="assembly", tiny=1, out=str(tmp_path / "whole"), device=DEV, dtype=kind, parts=parts,
                         sample_steps=6)
    assert whole["final_latent_change"] == whole["final_latent_change"]
