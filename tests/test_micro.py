"""A training step in smaller groups of rows (``train_band_engine(..., micro=n)``): the rows of a step go through the
forwards and the backward in groups, the gradients add up, one optimizer step follows. The same update as the
one-forward step up to floating-point rounding (step 1 compared term by term, gradient norm and every parameter; the
same sums in float64 agree to 1e-12); micro = 0 and micro = batch are the one-forward path element by element; a
resumed job equals one that never stopped, and a checkpoint made with another micro resumes; the experiment's
band_engine variant records it."""
import json
import math

import pytest
import torch

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion import run  # noqa: E402
from geolip.modular_diffusion.capture import train as T  # noqa: E402
from geolip.modular_diffusion.capture.engine import build_engine  # noqa: E402
from geolip.modular_diffusion.capture.schedule import add_noise  # noqa: E402
from geolip.modular_diffusion.capture.teacher import tiny_teacher  # noqa: E402
from geolip.modular_diffusion.experiments.capture import random_source  # noqa: E402

SHAPE, BAND, BATCH = "small", "clean", 4
# the bounds asserted at step 1, with margin over what was measured on the CPU (the measured numbers are printed)
TERM_REL = 1e-5
GRAD_REL = 1e-4          # measured: about 9e-6 of a tensor's largest entry (tensors whose entries nearly cancel)
PARAM_ABS = 1e-6


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


def _job(teacher, out, micro=None, steps=1, **kw):
    args = dict(steps=steps, batch=BATCH, seed=0, out=str(out), lr=1e-3, log_every=0, log=lambda s: None)
    args.update(kw)
    if micro is not None:
        args["micro"] = micro
    return T.train_band_engine(teacher, SHAPE, BAND, random_source(0, (4, 16, 16)), **args)


def _params(r, which="engine"):
    return {k: v.detach().clone() for k, v in r[which].state_dict().items()}


def _largest(a, b):
    """The largest absolute difference over all tensors, and the largest relative one (against the larger size)."""
    abs_d, rel_d = 0.0, 0.0
    for k in a:
        d = (a[k].double() - b[k].double()).abs()
        abs_d = max(abs_d, float(d.max()))
        rel_d = max(rel_d, float((d / a[k].double().abs().clamp_min(1e-30)).max()))
    return abs_d, rel_d


def _rel(x, y):
    return abs(x - y) / max(abs(x), abs(y), 1e-30)


def _recording_adam(monkeypatch):
    """Adam's step wrapped so that the gradients it is given are kept (in the order of the parameters)."""
    seen = []
    step = torch.optim.Adam.step

    def recorded(self, *a, **k):
        seen.append([p.grad.detach().clone() for group in self.param_groups for p in group["params"]])
        return step(self, *a, **k)
    monkeypatch.setattr(torch.optim.Adam, "step", recorded)
    return seen


def test_step_1_the_same_update_in_groups(teacher, tmp_path, monkeypatch):
    seen = _recording_adam(monkeypatch)
    lr = 1e-3
    whole = _job(teacher, tmp_path / "m0", micro=0)
    w_row, w_par, w_grad = whole["log"][0], _params(whole), seen[-1]
    names = [n for n, _ in whole["engine"].named_parameters()]
    found = {}
    for micro in (2, 1, 3):                     # 3: groups of 3 and 1 rows
        r = _job(teacher, tmp_path / f"m{micro}", micro=micro)
        assert r["micro"] == micro
        row, grad, par = r["log"][0], seen[-1], _params(r)
        for key in ("loss", "out", "data", "grad_norm"):
            assert _rel(row[key], w_row[key]) <= TERM_REL, (micro, key, row[key], w_row[key])
        assert row["dropped"] == w_row["dropped"]
        # the gradients: each tensor agrees with the one-forward step's to float32 rounding of its size
        g_rel = max(float((a.double() - b.double()).abs().max()) / max(float(a.abs().max()), 1e-30)
                    for a, b in zip(w_grad, grad))
        assert g_rel <= GRAD_REL, (micro, g_rel)
        # the parameters: Adam's first step moves each number by lr * g / (|g| + eps), so a gradient's rounding
        # reaches a parameter divided by (|g| + eps): where |g| is far above eps this is float32 rounding of the step;
        # where |g| is near 0 the same rounding is magnified by 1 / eps. Both are checked: numbers whose gradient is
        # at least 1e-4 agree to PARAM_ABS, and every number within lr * |dg| / eps (with margin) + PARAM_ABS
        abs_d, rel_d = _largest(w_par, par)
        big_abs, worst_ratio = 0.0, 0.0
        for n, a, b in zip(names, w_grad, grad):
            d = (w_par[n].double() - par[n].double()).abs()
            big = a.abs() >= 1e-4
            if bool(big.any()):
                big_abs = max(big_abs, float(d[big].max()))
            allowed = lr * (a.double() - b.double()).abs() / T.EPS
            worst_ratio = max(worst_ratio, float(((d - PARAM_ABS) / allowed.clamp_min(1e-30)).max()))
        assert big_abs <= PARAM_ABS, (micro, big_abs)
        assert worst_ratio <= 1.5, (micro, worst_ratio)
        found[micro] = (max(_rel(row[k], w_row[k]) for k in ("loss", "out", "data", "grad_norm")), g_rel, abs_d, rel_d,
                        big_abs, worst_ratio)
    for micro, (t_rel, g_rel, abs_d, rel_d, big_abs, ratio) in found.items():
        print(f"step 1, micro {micro} against micro 0: terms and gradient norm {t_rel:.3g} relative; gradients {g_rel:.3g} "
              f"of each tensor's size; parameters {abs_d:.3g} absolute, {rel_d:.3g} relative (largest), {big_abs:.3g} "
              f"absolute where |g| >= 1e-4; largest share of lr * |dg| / eps used {ratio:.3g}")


def _gradients(teacher, engine, z, t, states, eps, micro):
    """The gradient of mean((engine - teacher)^2) over the batch, in one piece (micro 0) or in groups weighted by
    their share of the rows, in the precision of the inputs."""
    for p in engine.parameters():
        p.grad = None
    b = z.shape[0]
    parts = [(0, b)] if micro == 0 else [(a, min(a + micro, b)) for a in range(0, b, micro)]
    for a, e in parts:
        o_t = teacher.predict(z[a:e], t[a:e], states[a:e])
        o_s = engine(z[a:e], t[a:e], states[a:e])
        ((o_s - o_t).pow(2).mean() * ((e - a) / b)).backward()
    return {n: p.grad.detach().clone() for n, p in engine.named_parameters() if p.grad is not None}


def test_float64_the_gradients_agree_to_1e_12(teacher):
    t64 = tiny_teacher(0)
    t64.to(dtype=torch.float64)
    engine = build_engine(t64, SHAPE).double().train()
    x0, ids = random_source(0, (4, 16, 16))(0, BATCH)
    g = torch.Generator().manual_seed(5)
    eps = torch.randn(tuple(x0.shape), generator=g, dtype=torch.float64)
    t = torch.tensor([10, 40, 80, 120])
    z = add_noise(x0.double(), eps, t, t64.alphas_cumprod).double()
    states = t64.text_states(ids).double()
    whole = _gradients(t64, engine, z, t, states, eps, 0)
    worst = 0.0
    for micro in (2, 1, 3):
        got = _gradients(t64, engine, z, t, states, eps, micro)
        assert set(got) == set(whole)
        for n in whole:
            scale = float(whole[n].abs().max())
            if scale == 0.0:
                continue
            worst = max(worst, float((got[n] - whole[n]).abs().max()) / scale)
    print(f"float64: the largest gradient difference in groups against one piece, relative to the tensor's size: {worst:.3g}")
    assert worst <= 1e-12, worst


def test_micro_0_and_micro_batch_are_the_one_forward_path(teacher, tmp_path):
    today = _job(teacher, tmp_path / "today", steps=2)          # the call without the argument
    for micro in (0, BATCH, BATCH + 3):
        r = _job(teacher, tmp_path / f"m{micro}", micro=micro, steps=2)
        a, b = _params(today), _params(r)
        assert all(torch.equal(a[k], b[k]) for k in a), micro
        a, b = _params(today, "average"), _params(r, "average")
        assert all(torch.equal(a[k], b[k]) for k in a), micro
        assert r["log"] == today["log"], micro
        assert r["settings"] == today["settings"]


def test_a_resumed_job_in_groups_and_another_micro_on_resume(teacher, tmp_path):
    whole = _job(teacher, tmp_path / "whole", micro=2, steps=3)
    first = _job(teacher, tmp_path / "parts", micro=2, steps=3, until=2)
    assert not first["finished"] and first["step"] == 2
    rest = _job(teacher, tmp_path / "parts", micro=2, steps=3)
    assert rest["finished"]
    a, b = _params(whole), _params(rest)
    assert all(torch.equal(a[k], b[k]) for k in a)
    a, b = _params(whole, "average"), _params(rest, "average")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert whole["log"] == rest["log"]
    # a checkpoint made in one forward resumes in groups: micro is not a setting the checkpoint is checked against
    _job(teacher, tmp_path / "switch", micro=0, steps=3, until=1)
    lines = []
    done = _job(teacher, tmp_path / "switch", micro=2, steps=3, log=lines.append)
    assert done["finished"] and done["micro"] == 2
    assert any(l.startswith("RESUME from step 1") for l in lines)
    assert any("micro 2" in l for l in lines if l.startswith("JOB band engine"))
    with pytest.raises(ValueError):
        _job(teacher, tmp_path / "bad", micro=-1)


def test_ten_steps_in_groups_drift_by_rounding_only(teacher, tmp_path):
    a = _job(teacher, tmp_path / "m0", micro=0, steps=10)
    b = _job(teacher, tmp_path / "m2", micro=2, steps=10)
    abs_d, rel_d = _largest(_params(a), _params(b))
    print(f"ten steps, micro 2 against micro 0: the largest parameter difference {abs_d:.3g} absolute, {rel_d:.3g} relative")
    assert math.isfinite(abs_d) and abs_d < 1e-3


def test_the_experiment_records_micro(tmp_path):
    out = tmp_path / "band"
    r = run.main("capture", ["variant=band_engine", "shape=tiny", "band=noisy", "steps=2", f"batch={BATCH}", "micro=2",
                             "out=" + str(out)], tiny=1, probe_n=2, probe_per_band=2, sample_steps=4, threads=2, log_every=1)
    assert r["train"]["finished"] and r["train"]["micro"] == 2 and r["args"]["micro"] == 2
    stored = json.load(open(out / "result.json", encoding="utf-8"))
    assert stored["train"]["micro"] == 2
    with open(out / "log.txt", encoding="utf-8") as fh:
        assert "micro 2" in fh.read()
    with pytest.raises(ValueError):
        run.main("capture", ["variant=band_engine", "micro=-1", "out=" + str(tmp_path / "bad")], tiny=1)
