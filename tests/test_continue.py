"""A run that continues another (T1): n + m steps straight equal n steps, the trainer state written and read, and m
steps more from global step n, bit for bit in every weight and in the optimizer's moments; start_step=0 with no state
is the run of earlier versions."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.train import fit_units, read_state, run_loop, write_state

FIT = dict(lr=1e-3, batch=8, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)


@pytest.fixture(scope="module")
def bed(synthetic_npz):
    return PicturesBed(synthetic_npz, "cpu")


def _born(bed):
    return Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)


def _params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _same(a, b):
    assert set(a) == set(b)
    for n in a:
        assert torch.equal(a[n], b[n]), n


@pytest.mark.parametrize("warmup", [1, 3])
def test_T1_continued_run_equals_the_straight_run(bed, tmp_path, warmup):
    straight = _born(bed)
    r6 = fit_units(straight, bed, 0, steps=6, warmup=warmup, keep_state=True, **FIT)
    first = _born(bed)
    r4 = fit_units(first, bed, 0, steps=4, warmup=warmup, keep_state=True, **FIT)
    assert r4["optimizer_state"]["step"] == 4 and set(r4["optimizer_state"]["t"].values()) == {4}
    folder = str(tmp_path / "state")
    write_state(folder, r4["optimizer_state"], extra={"note": "after 4 steps"})
    assert sorted(os.listdir(folder)) == ["adam.safetensors", "state.json"]
    with open(os.path.join(folder, "state.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["step"] == 4 and meta["extra"] == {"note": "after 4 steps"} and meta["betas"] == [0.9, 0.999]
    state = read_state(folder)
    for k in ("m", "v"):
        _same(state[k], r4["optimizer_state"][k])
    # the continuation: a model holding the weights after 4 steps (a fresh object, as a linked folder would be)
    second = _born(bed)
    with torch.no_grad():
        for n, p in second.named_parameters():
            p.copy_(dict(first.named_parameters())[n])
    r2 = fit_units(second, bed, 0, steps=2, warmup=warmup, start_step=4, optimizer_state=state, keep_state=True, **FIT)
    _same(_params(second), _params(straight))
    for k in ("m", "v"):
        _same(r2["optimizer_state"][k], r6["optimizer_state"][k])
    assert r2["optimizer_state"]["t"] == r6["optimizer_state"]["t"] and r2["optimizer_state"]["step"] == 6
    assert r2["loss_over_levels"] == r6["loss_over_levels"]
    assert [c[1] for c in r2["curve"]] == [c[1] for c in r6["curve"]][4:]
    assert [c[0] for c in r2["curve"]] == [0, 1]                   # the curve's step numbers stay local
    assert r2["start_step"] == 4 and r2["recipe"]["start_step"] == 4 and r2["recipe"]["optimizer_state"] == "continued"


def test_T1_the_default_is_the_run_of_earlier_versions(bed):
    a, b = _born(bed), _born(bed)
    ra = fit_units(a, bed, 0, steps=3, warmup=2, **FIT)
    rb = fit_units(b, bed, 0, steps=3, warmup=2, start_step=0, optimizer_state=None, **FIT)
    _same(_params(a), _params(b))
    assert ra["curve"] == rb["curve"] and set(ra) == set(rb) and ra["levels"] == rb["levels"]
    assert ra["recipe"] == rb["recipe"]
    assert "start_step" not in ra and "start_step" not in ra["recipe"] and "optimizer_state" not in ra


def test_T1_the_learning_rate_and_stream_follow_the_global_step(bed, monkeypatch):
    from geolip.modular_diffusion import train
    seen, lrs = [], []
    real = train.step_generator

    def spy(seed, step, device):
        seen.append(step)
        return real(seed, step, device)

    real_step = torch.optim.Adam.step

    def step_spy(self, *a, **k):
        lrs.append(self.param_groups[0]["lr"])
        return real_step(self, *a, **k)

    monkeypatch.setattr(train, "step_generator", spy)
    monkeypatch.setattr(torch.optim.Adam, "step", step_spy)
    m = _born(bed)
    fit_units(m, bed, 0, steps=3, warmup=10, start_step=5, **FIT)
    assert seen == [5, 6, 7]
    assert lrs == pytest.approx([1e-3 * 6 / 10, 1e-3 * 7 / 10, 1e-3 * 8 / 10])


def test_T1_a_state_of_other_parameters_is_refused(bed):
    a = _born(bed)
    r = fit_units(a, bed, 0, steps=2, warmup=1, keep_state=True, **FIT)
    b = _born(bed)
    with pytest.raises(KeyError):
        fit_units(b, bed, 0, steps=1, warmup=1, trainable=["head"], start_step=2, optimizer_state=r["optimizer_state"],
                  **FIT)


def test_T1_run_loop_signature_is_unchanged_for_its_callers(bed):
    m = _born(bed)
    curve, seconds = run_loop(m, bed, 0, 2, 1e-3, 1, 8, 0.1, 0.0, "uniform", (0.0, 1.0), True, "cpu", lambda s: None, 1)
    assert len(curve) == 2 and seconds >= 0
