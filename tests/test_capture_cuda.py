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


def test_the_trainer_from_random_numbers_with_reads_on_the_card(tmp_path):
    """A random start, reads during training and a finished job without its checkpoint, with a half-precision
    teacher: the reads see card tensors, the weights stay float32, and a second call trains nothing."""
    teacher = _teacher(HALF)
    probe = _probe(teacher, batch=2)
    source = random_source(1, (4, 16, 16))
    seen = []

    def on_every(n, engine, averaged):
        seen.append(next(averaged.parameters()).device.type)
        return {"live": reads.band_read(engine, teacher, "clean", probe)["mse"],
                "averaged": reads.band_read(averaged, teacher, "clean", probe)["mse"]}

    kw = dict(steps=4, batch=2, seed=2, lr=1e-3, log_every=0, log=lambda s: None, init="random", every=2,
              on_every=on_every, keep_checkpoint=False)
    one = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "job"), **kw)
    assert one["finished"] and [r["step"] for r in one["reads"]] == [2, 4] and seen == [DEV, DEV]
    assert all(0.0 < r["value"]["live"] < float("inf") for r in one["reads"])
    assert next(one["engine"].parameters()).dtype == torch.float32
    assert not os.path.isfile(tmp_path / "job" / "checkpoint.pt")
    again = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "job"), **kw)
    assert again["restored"] and next(again["engine"].parameters()).device.type == DEV
    a = torch.cat([p.detach().flatten().cpu() for p in one["average"].parameters()])
    b = torch.cat([p.detach().flatten().cpu() for p in again["average"].parameters()])
    assert torch.equal(a, b)                                     # read back from its files, exactly


def test_held_out_rows_a_read_curve_and_the_exchange_on_the_card(tmp_path):
    """The experiment with a probe of held-out rows in file order, a read curve, a random start, then the exchange
    variant sampling in groups: the same part put back reads exactly 0 on the card too."""
    from geolip.modular_diffusion.capture.data import synthetic_shards
    kind = "fp16" if DEV == "cuda" else "fp32"
    rows = str(tmp_path / "rows")
    synthetic_shards(rows, rows=8, shard_size=3, size=16, seed=4)
    common = dict(tiny=1, device=DEV, dtype=kind, probe_data=rows, probe_first=2, probe_rows=5, chunk=2,
                  probe_per_band=2, sample_steps=4, group=2)
    job = tmp_path / "job"
    r = capture.main(variant="band_engine", out=str(job), steps=4, batch=2, init="random", read_every=2,
                     keep_checkpoint=0, save_every=0, log_every=0, **common)
    assert [c["step"] for c in r["curve"]] == [0, 2, 4] and r["probe"]["rows"] == 5
    assert all(0.0 < c["averaged"] < float("inf") for c in r["curve"])
    parts = f"clean={job},middle=teacher,noisy=teacher"
    same = capture.main(variant="exchange", parts=parts, band="clean", other=str(job), out=str(tmp_path / "same"),
                        **common)
    assert all(v["change"] == 0.0 for v in same["exchange"]["bands"].values())
    assert same["exchange"]["final_latent_change"] == 0.0
    copy = capture.main(variant="exchange", parts=parts, band="clean", other="teacher", out=str(tmp_path / "copy"),
                        **common)
    assert all(v["after"] == 0.0 for v in copy["exchange"]["bands"].values())
    assert copy["exchange"]["final_latent_change"] > 0.0


def test_band_needs_on_the_card(tmp_path):
    """The band-needs reads and their timing on the card: the born reads in full precision, the prompt read with the
    teacher cast to half precision; the exact controls hold there too."""
    from geolip.modular_diffusion.experiments import band_needs
    kind = "fp16" if DEV == "cuda" else "fp32"
    small = dict(rows=4, prompts=4, group=2, chunk=2, per_band=2, steps=6, cost_batch=2, cost_repeats=1,
                 sample_dtype=kind)
    r = band_needs.main(tiny=1, device=DEV, out=str(tmp_path / "reads"), **small)
    assert r["sample_dtype_used"] == kind and r["switches"]["cudnn.allow_tf32"] is False
    for s in ("0", "1"):
        assert all(v == 0.0 for v in r["controls"][s]["full"].values())          # the copy reads exactly 0
        assert r["r3"][s]["repeat"] == 0.0                                       # the sampler repeats itself
        assert all(0.0 < v["empty"] < float("inf") for v in r["r3"][s]["table"].values())
    assert all(x["seconds"] > 0 for x in r["r4"])
    again = band_needs.main(tiny=1, device=DEV, out=str(tmp_path / "reads"), **small)
    assert again["units"]["ran"] == [] and again["r3"] == r["r3"]
    t = band_needs.main(variant="timing", tiny=1, device=DEV, out=str(tmp_path / "timing"), **small)
    assert all(v > 0 for v in t["minutes"].values())
