"""Averaged weights beside the last weights (T9): the average follows its rule and changes nothing else; use_average puts
the last weights back exactly; frozen units have no average; jobs with avg= write the block and the extra build folder."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.experiments import content, time_split
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.linker import link
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.read import read
from geolip.modular_diffusion.train import fit_units, use_average

FIT = dict(lr=1e-3, warmup=1, batch=8, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)


def _born(bed):
    return Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)


def _params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


@pytest.fixture(scope="module")
def bed(synthetic_npz):
    return PicturesBed(synthetic_npz, "cpu")


def test_T9_no_averages_changes_nothing(bed):
    a, b = _born(bed), _born(bed)
    ra = fit_units(a, bed, 0, steps=4, **FIT)
    rb = fit_units(b, bed, 0, steps=4, averages=(), **FIT)
    assert "averaged" not in ra and "averaged" not in rb
    assert ra["curve"] == rb["curve"] and ra["loss_over_levels"] == rb["loss_over_levels"]
    pa, pb = _params(a), _params(b)
    assert all(torch.equal(pa[n], pb[n]) for n in pa)


def test_T9_the_average_follows_its_rule_and_the_last_weights_are_unchanged(bed, monkeypatch):
    plain = _born(bed)
    r_plain = fit_units(plain, bed, 0, steps=4, **FIT)
    model = _born(bed)
    start = _params(model)                                    # the parameters before the first step
    names = {id(p): n for n, p in model.named_parameters()}
    after = []
    real_step = torch.optim.Adam.step

    def capture(self, *args, **kw):
        out = real_step(self, *args, **kw)
        after.append({names[id(p)]: p.detach().clone() for g in self.param_groups for p in g["params"]})
        return out

    monkeypatch.setattr(torch.optim.Adam, "step", capture)
    res = fit_units(model, bed, 0, steps=4, averages=(0.5,), **FIT)
    monkeypatch.undo()
    assert len(after) == 4
    assert res["curve"] == r_plain["curve"]
    last, ref = _params(model), _params(plain)
    assert all(torch.equal(last[n], ref[n]) for n in last)
    # the rule replayed here from the captured weights: in float32 (the averages' precision) within 1e-7, and in
    # float64 within 1e-6 of the value's size (float32 rounding of values near 1 is itself about 1e-7)
    avg32 = {n: v.clone() for n, v in start.items()}
    avg64 = {n: v.double() for n, v in start.items()}
    for k, w in enumerate(after):
        dk = min(0.5, (1 + k) / (10 + k))
        for n in avg32:
            avg32[n] = dk * avg32[n] + (1 - dk) * w[n]
            avg64[n] = dk * avg64[n] + (1 - dk) * w[n].double()
    got = res["averaged"][0.5]
    assert set(got) == set(start)
    moved = False
    for n in got:
        assert got[n].dtype == torch.float32 and not got[n].requires_grad
        assert float((got[n] - avg32[n]).abs().max()) <= 1e-7, n
        assert float(((got[n].double() - avg64[n]).abs() / avg64[n].abs().clamp_min(1.0)).max()) <= 1e-6, n
        moved = moved or not torch.equal(got[n], start[n])
    assert moved
    # use_average swaps the average in and puts the last weights back exactly, also when the body raises
    with use_average(model, got):
        now = _params(model)
        assert all(torch.equal(now[n], got[n]) for n in got)
    after_exit = _params(model)
    assert all(torch.equal(after_exit[n], last[n]) for n in last)
    with pytest.raises(RuntimeError):
        with use_average(model, got):
            raise RuntimeError("inside")
    again = _params(model)
    assert all(torch.equal(again[n], last[n]) for n in last)


def test_T9_frozen_units_have_no_average(bed):
    model = _born(bed)
    res = fit_units(model, bed, 0, steps=3, trainable=["head"], averages=(0.9, 0.99), **FIT)
    head = {n for n, p in model.named_parameters() if n.startswith("units.2.")}
    assert model.units[2].name == "head"
    for d in (0.9, 0.99):
        assert set(res["averaged"][d]) == head


SMALL = dict(seed=0, device="cpu", scale=0.0005, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=1,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)


def _check_job(out, r, keys, card="head"):
    with open(os.path.join(out, "result.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    assert set(stored["averaged"]) == set(keys)
    for key, folder in keys.items():
        block = stored["averaged"][key]
        assert os.path.exists(os.path.join(out, folder, "build.json"))
        m = link(os.path.join(out, folder))
        assert read(m, r["bed"], 32, "cpu")["loss_over_levels"] == block["loss_over_levels"]
        assert block["score"]["n"] == 32 and len(block["by_stage"]) == 3
        with open(os.path.join(out, folder, card + ".card.json"), encoding="utf-8") as fh:
            assert json.load(fh)["recipe"]["average_decay"] == float(key)
    return stored


def test_T9_jobs_with_averages(synthetic_npz, tmp_path):
    bed = PicturesBed(synthetic_npz, "cpu")
    ts = str(tmp_path / "ts_plain")
    r = time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=ts, avg="0.999,0.5"))
    stored = _check_job(ts, {"bed": bed}, {"0.999": "model_avg999", "0.5": "model_avg5"})
    assert stored["args"]["avg"] == "0.999,0.5"
    assert stored["loss_over_levels"] == r["loss_over_levels"]
    with pytest.raises(ValueError):
        time_split.main(**dict(SMALL, variant="split", data=synthetic_npz, out=str(tmp_path / "split"), avg="0.999"))
    assert not os.path.exists(tmp_path / "split" / "stage_0")
    # content: a one-model variant with the class read, then experts on it (the experts' parameters averaged)
    ref = str(tmp_path / "ref.npz")
    cp = str(tmp_path / "plain")
    content.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=cp, avg="0.9995", reader=1, ref=ref))
    stored = _check_job(cp, {"bed": bed}, {"0.9995": "model_avg9995"})
    block = stored["averaged"]["0.9995"]
    assert len(block["by_class"]) == 10 and len(block["class_worth"]["by_stage"]) == 3
    assert "class_read" in block["score"] and "reader_check" not in block["score"]
    ex = str(tmp_path / "experts_start")
    content.main(**dict(SMALL, variant="experts_start", data=synthetic_npz, out=ex, source=cp, avg="0.999"))
    _check_job(ex, {"bed": bed}, {"0.999": "model_avg999"}, card="expert_c0")
    src = link(os.path.join(cp, "model"))
    avg = link(os.path.join(ex, "model_avg999"))
    for u in src.units:                                       # only the experts were averaged
        assert torch.equal(torch.cat([p.reshape(-1) for p in u.parameters()] or [torch.zeros(0)]),
                           torch.cat([p.reshape(-1) for p in avg.unit(u.name).parameters()] or [torch.zeros(0)]))
    text = content.summarize(str(tmp_path))
    assert "average total" in text and "| plain | 0 |" in text
    # anchors ignores avg=
    an = content.main(**dict(SMALL, variant="anchors", data=synthetic_npz, out=str(tmp_path / "anchors"), avg="0.999"))
    assert "averaged" not in an
