"""Named averages (T2): "g8" is the decay 0.999 to the bit; "g3" follows its rule; "tail<n>" is the mean of the last n
weights; a carried "g8" equals the straight run's; averages never change the last weights; the experiments' avg= takes
names (folders model_avg_<name>, keys the name) and score_last / score_avg choose what is scored."""
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
from geolip.modular_diffusion.train import average_spec, fit_units

FIT = dict(lr=1e-3, warmup=1, batch=8, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)


@pytest.fixture(scope="module")
def bed(synthetic_npz):
    return PicturesBed(synthetic_npz, "cpu")


def _born(bed):
    return Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)


def _params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _capture(model, monkeypatch):
    names = {id(p): n for n, p in model.named_parameters()}
    after = []
    real_step = torch.optim.Adam.step

    def capture(self, *args, **kw):
        out = real_step(self, *args, **kw)
        after.append({names[id(p)]: p.detach().clone() for g in self.param_groups for p in g["params"]})
        return out

    monkeypatch.setattr(torch.optim.Adam, "step", capture)
    return after


def test_T2_specs():
    assert average_spec("g8") == ("g8", "g", 8) and average_spec("tail100") == ("tail100", "tail", 100)
    assert average_spec(0.999) == (0.999, "decay", 0.999) and average_spec("0.5") == (0.5, "decay", 0.5)
    for bad in ("tail0", "1.5", "gx"):
        with pytest.raises(ValueError):
            average_spec(bad)


def test_T2_g8_is_the_decay_0999_to_the_bit(bed):
    m = _born(bed)
    r = fit_units(m, bed, 0, steps=12, averages=(0.999, "g8"), **FIT)
    assert list(r["averaged"]) == [0.999, "g8"]
    a, b = r["averaged"][0.999], r["averaged"]["g8"]
    assert set(a) == set(b)
    for n in a:
        assert torch.equal(a[n], b[n]), n


def test_T2_g3_and_tail_follow_their_rules(bed, monkeypatch):
    plain = _born(bed)
    fit_units(plain, bed, 0, steps=10, **FIT)
    m = _born(bed)
    start = _params(m)
    after = _capture(m, monkeypatch)
    r = fit_units(m, bed, 0, steps=10, averages=("g3", "tail4", "tail50"), **FIT)
    monkeypatch.undo()
    assert len(after) == 10
    last, ref = _params(m), _params(plain)
    for n in last:                                             # averages never change the last weights
        assert torch.equal(last[n], ref[n])
    g3 = {n: v.clone() for n, v in start.items()}
    for k, w in enumerate(after):
        dk = (1 + k) / (k + 3 + 2)
        for n in g3:
            g3[n] = dk * g3[n] + (1 - dk) * w[n]
    for n in g3:
        assert float((r["averaged"]["g3"][n] - g3[n]).abs().max()) <= 1e-7, n
    for key, rows in (("tail4", after[6:]), ("tail50", after)):
        for n in last:
            mean = (sum(w[n].double() for w in rows) / len(rows)).float()
            got = r["averaged"][key][n]
            assert got.dtype == torch.float32 and float((got - mean).abs().max()) <= 1e-7, (key, n)


def test_T2_a_carried_g8_equals_the_straight_run(bed):
    straight = _born(bed)
    r6 = fit_units(straight, bed, 0, steps=6, averages=("g8",), **FIT)
    first = _born(bed)
    r4 = fit_units(first, bed, 0, steps=4, averages=("g8",), keep_state=True, **FIT)
    second = _born(bed)
    with torch.no_grad():
        for n, p in second.named_parameters():
            p.copy_(dict(first.named_parameters())[n])
    r2 = fit_units(second, bed, 0, steps=2, averages=("g8",), start_step=4, optimizer_state=r4["optimizer_state"],
                   average_start={"g8": r4["averaged"]["g8"]}, **FIT)
    for n, v in r6["averaged"]["g8"].items():
        assert torch.equal(r2["averaged"]["g8"][n], v), n
    # without the start state the average starts again (k from 0, from the weights before the first step)
    third = _born(bed)
    with torch.no_grad():
        for n, p in third.named_parameters():
            p.copy_(dict(first.named_parameters())[n])
    r2b = fit_units(third, bed, 0, steps=2, averages=("g8",), start_step=4, optimizer_state=r4["optimizer_state"], **FIT)
    assert any(not torch.equal(r2b["averaged"]["g8"][n], v) for n, v in r6["averaged"]["g8"].items())
    with pytest.raises(ValueError):
        fit_units(_born(bed), bed, 0, steps=1, averages=("tail4",), average_start={"tail4": {}}, **FIT)


SMALL = dict(seed=0, device="cpu", scale=0.0005, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=1,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)


def test_T2_jobs_take_named_averages(synthetic_npz, tmp_path, bed):
    out = str(tmp_path / "plain")
    r = time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=out, avg="0.999,g8,tail2",
                               score_avg="g8"))
    assert list(r["averaged"]) == ["0.999", "g8", "tail2"]
    assert os.path.isdir(os.path.join(out, "model_avg999")) and os.path.isdir(os.path.join(out, "model_avg_g8"))
    assert os.path.isdir(os.path.join(out, "model_avg_tail2"))
    assert r["averaged"]["g8"]["loss_over_levels"] == r["averaged"]["0.999"]["loss_over_levels"]
    assert r["averaged"]["g8"]["score"]["n"] == 32
    assert r["averaged"]["0.999"]["score"] is None and r["averaged"]["tail2"]["score"] is None    # read, not scored
    with open(os.path.join(out, "model_avg_g8", "head.card.json"), encoding="utf-8") as fh:
        assert json.load(fh)["recipe"]["average"] == "g8"
    m = link(os.path.join(out, "model_avg_tail2"))
    assert read(m, bed, 32, "cpu")["loss_over_levels"] == r["averaged"]["tail2"]["loss_over_levels"]
    # score_last=0: the last weights are read and not scored
    r2 = content.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=str(tmp_path / "c"), avg="g8",
                             score_last=0))
    assert r2["score"] is None and r2["averaged"]["g8"]["score"]["n"] == 32
    assert isinstance(r2["loss_over_levels"], float)
    text = content.summarize(str(tmp_path))
    assert "g8" in text
    with pytest.raises(ValueError):
        time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=str(tmp_path / "x"), avg="g8,g8"))
    with pytest.raises(ValueError):
        time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=str(tmp_path / "y"), avg="g8",
                               score_avg="g3"))
