"""Capture, the loss and the trainer (test 7): the loss's two terms, a 5-step run that lowers the loss on the tiny model,
pure Adam, and a resumed job that reaches the same final numbers as one that never stopped; the experiment's three
variants run end to end on the tiny teacher."""
import json
import os

import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion import run
from geolip.modular_diffusion.capture import train as T
from geolip.modular_diffusion.capture.loss import capture_loss, drop_prompts
from geolip.modular_diffusion.capture.schedule import add_noise
from geolip.modular_diffusion.capture.teacher import tiny_teacher
from geolip.modular_diffusion.experiments.capture import random_source


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


def test_7_the_loss_terms():
    g = torch.Generator().manual_seed(0)
    o_s, o_t, eps = (torch.randn(2, 4, 8, 8, generator=g) for _ in range(3))
    want_out, want_data = float((o_s - o_t).pow(2).mean()), float((o_s - eps).pow(2).mean())
    o_s.requires_grad_(True)
    total, terms = capture_loss(o_s, o_t, eps)
    assert terms["out"] == pytest.approx(want_out)
    assert terms["data"] == pytest.approx(want_data)
    assert terms["total"] == pytest.approx(terms["out"])                   # lam_data is 0 by default
    total2, terms2 = capture_loss(o_s, o_t, eps, lam_out=0.5, lam_data=2.0)
    assert terms2["total"] == pytest.approx(0.5 * terms["out"] + 2.0 * terms["data"])
    total.backward()
    assert torch.allclose(o_s.grad, 2 * (o_s - o_t).detach() / o_s.numel())
    # no gradient reaches the teacher's side: the teacher's answer and the noise are made leaves that ask for a
    # gradient, and the loss still leaves them without one (it would fill them, were its stop removed)
    a, b, c = (torch.randn(2, 4, 8, 8, generator=g).requires_grad_(True) for _ in range(3))
    capture_loss(a, b, c, lam_out=1.0, lam_data=1.0)[0].backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0
    assert b.grad is None and c.grad is None
    states, empty = torch.ones(1000, 77, 4), torch.zeros(1, 77, 4)
    out, dropped = drop_prompts(states, empty, 0.1, torch.Generator().manual_seed(1))
    assert 60 < int(dropped.sum()) < 140
    assert bool((out[dropped] == 0).all()) and bool((out[~dropped] == 1).all())


def _fixed_loss(teacher, eng, source):
    g = torch.Generator().manual_seed(99)
    x0, ids = source(10_000, 4)
    t = torch.tensor([20, 120, 220, 320])
    eps = torch.randn(tuple(x0.shape), generator=g)
    z = add_noise(x0, eps, t, teacher.alphas_cumprod)
    s = teacher.text_states(ids)
    with torch.no_grad():
        return capture_loss(eng(z, t, s), teacher.predict(z, t, s), eps)[1]["out"]


def test_7_five_steps_lower_the_loss(teacher, tmp_path):
    source = random_source(0, (4, 16, 16))
    lines = []
    r = T.train_band_engine(teacher, "small", "clean", source, steps=5, batch=4, seed=0, out=str(tmp_path),
                            lr=1e-3, log_every=1, log=lines.append)
    assert r["finished"] and r["step"] == 5 and len(r["log"]) == 5
    born = T.build_engine(teacher, "small")
    before = _fixed_loss(teacher, born, source)
    after = _fixed_loss(teacher, r["engine"], source)
    assert after < before, (before, after)
    assert any(l.startswith("JOB band engine") for l in lines) and any(l.startswith("STEP 1 terms") for l in lines)
    assert any("left" in l for l in lines)
    for name in ("checkpoint.pt", "engine.safetensors", "engine_avg.safetensors", "engine.json", "train.json"):
        assert os.path.isfile(tmp_path / name), name
    assert sorted(os.listdir(tmp_path)) == sorted(["checkpoint.pt", "engine.safetensors", "engine_avg.safetensors",
                                                   "engine.json", "train.json"])
    ck = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    group = ck["adam"]["param_groups"][0]
    assert group["weight_decay"] == 0.0 and tuple(group["betas"]) == (0.9, 0.999) and group["lr"] == 1e-3
    assert "decoupled_weight_decay" not in group or group["decoupled_weight_decay"] is False
    # the averaged copy: decay min(0.9999, (1 + n) / (10 + n)) after update n
    assert T.average_decay_at(1, 0.9999) == pytest.approx(2 / 11)
    for (n, p) in r["engine"].named_parameters():
        q = dict(r["average"].named_parameters())[n]
        assert q.shape == p.shape


def test_7_a_resumed_job_ends_on_the_same_numbers(teacher, tmp_path):
    source = random_source(1, (4, 16, 16))
    kw = dict(steps=6, batch=2, seed=3, lr=1e-3, rho=0.3, save_every=2, log_every=0, log=lambda s: None)
    whole = T.train_band_engine(teacher, "tiny", "middle", source, out=str(tmp_path / "whole"), **kw)
    first = T.train_band_engine(teacher, "tiny", "middle", source, out=str(tmp_path / "parts"), until=3, **kw)
    assert not first["finished"] and first["step"] == 3
    assert not os.path.isfile(tmp_path / "parts" / "engine.safetensors")
    lines = []
    kw["log"] = lines.append
    rest = T.train_band_engine(teacher, "tiny", "middle", source, out=str(tmp_path / "parts"), **kw)
    assert rest["finished"] and any(l.startswith("RESUME from step 3") for l in lines)
    a, b = whole["engine"].state_dict(), rest["engine"].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    a, b = whole["average"].state_dict(), rest["average"].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert whole["log"] == rest["log"]
    with pytest.raises(ValueError):
        T.train_band_engine(teacher, "tiny", "middle", source, out=str(tmp_path / "parts"), **dict(kw, lr=2e-3))


def test_the_experiment_variants_run_on_the_tiny_teacher(tmp_path):
    common = dict(tiny=1, probe_n=2, probe_per_band=2, sample_steps=4, threads=2)
    born = run.main("capture", ["variant=born", "shape=small", "out=" + str(tmp_path / "born")], **common)
    assert {r["part"] for r in born["rows"]} >= {"teacher copy", "small as born"}
    assert all(r["mse"] == 0.0 for r in born["rows"] if r["part"] == "teacher copy")
    job = tmp_path / "band"
    tr = run.main("capture", ["variant=band_engine", "shape=tiny", "band=noisy", "steps=2", "batch=2",
                              "out=" + str(job)], log_every=1, **common)
    assert tr["train"]["finished"] and "trained_clean" in tr["reads"]
    asm = run.main("capture", ["variant=assembly", "out=" + str(tmp_path / "asm")],
                   parts=f"clean=teacher,middle=teacher,noisy={job}", **common)
    assert asm["reads"]["clean"]["mse"] == 0.0 and asm["reads"]["noisy"]["mse"] > 0.0
    assert asm["final_latent_change"] > 0.0
    for d in ("born", "band", "asm"):
        with open(tmp_path / d / "log.txt", encoding="utf-8") as fh:
            assert fh.read().rstrip().endswith("done")
        json.load(open(tmp_path / d / "result.json", encoding="utf-8"))
    from geolip.modular_diffusion.experiments import capture
    table = capture.summarize(str(tmp_path))
    assert "| band |" in table or "band_engine" in table
