"""A continued optimizer state that covers only some trainable parameters (T1): a tiny source continued with newborn
parts; before the first step the old parameters' Adam state is the stored one and the new parameters have none; one
step moves both; without the switch the KeyError names the new parameters; a name only in the state still raises; a
carried average that lacks the new parameters starts them at their values (and raises without the switch)."""
import os

import pytest
import torch

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.experiments import stages
from geolip.modular_diffusion.forms import with_parts
from geolip.modular_diffusion.linker import link
from geolip.modular_diffusion.train import BETAS, EPS, fit_units, load_adam_state, read_state, set_trainable_units

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=0,
             judge="fake", sheet=0, threads=4)
FIT = dict(lr=1e-3, warmup=1, batch=8, heldout=16, device="cpu", log=lambda s: None, log_every=100, equalize=False)


@pytest.fixture(scope="module")
def source(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("partial")
    stages.main(**dict(SMALL, data=synthetic_npz, variant="source", out=str(root / "source")))
    return root / "source", PicturesBed(synthetic_npz, "cpu")


def _with_parts(src):
    model, names = with_parts(str(src / "model"), 0, stages.part_groups("experts10"))
    set_trainable_units(model, "all")
    unit_ids = {id(p) for n in names for p in model.unit(n).parameters()}
    new = sorted(n for n, p in model.named_parameters() if id(p) in unit_ids)
    return model, names, new


def test_T1_the_old_state_is_loaded_and_the_new_parameters_have_none(source):
    src, _ = source
    state = read_state(str(src / "state"))
    model, names, new = _with_parts(src)
    assert new and not set(new) & set(state["m"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3, betas=BETAS, eps=EPS, weight_decay=0.0)
    left = load_adam_state(opt, model, state, partial=True)
    assert left == new
    named = dict(model.named_parameters())
    for n, p in named.items():
        if n in state["m"]:
            st = opt.state[p]
            assert torch.equal(st["exp_avg"], state["m"][n]) and torch.equal(st["exp_avg_sq"], state["v"][n]), n
            assert float(st["step"]) == float(state["t"][n])
        else:
            assert p not in opt.state, n


def test_T1_without_the_switch_the_error_names_the_new_parameters(source):
    src, _ = source
    state = read_state(str(src / "state"))
    model, _, new = _with_parts(src)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    with pytest.raises(KeyError) as e:
        load_adam_state(opt, model, state)
    assert new[0] in str(e.value)
    extra = dict(state, m=dict(state["m"], only_in_the_state=state["m"][next(iter(state["m"]))]))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    with pytest.raises(KeyError) as e:
        load_adam_state(opt, model, extra, partial=True)
    assert "only_in_the_state" in str(e.value)


def test_T1_one_step_moves_both(source):
    src, bed = source
    state = read_state(str(src / "state"))
    model, names, new = _with_parts(src)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    res = fit_units(model, bed, 0, steps=1, start_step=state["step"], optimizer_state=state, optimizer_partial=True,
                    keep_state=True, **FIT)
    assert res["recipe"]["optimizer_state"] == "continued, partial" and res["optimizer_new"] == new
    after = dict(model.named_parameters())
    old_moved = [n for n in state["m"] if not torch.equal(before[n], after[n])]
    new_moved = [n for n in new if not torch.equal(before[n], after[n])]
    assert old_moved and new_moved
    # the new parameters that took their first step now have moments (a part whose class was not in the batch took none)
    got = set(new) & set(res["optimizer_state"]["m"])
    assert set(new_moved) <= got and all(res["optimizer_state"]["t"][n] == 1 for n in got)
    model2, _, _ = _with_parts(src)
    with pytest.raises(KeyError):                                      # the default stays strict
        fit_units(model2, bed, 0, steps=1, start_step=state["step"], optimizer_state=state, **FIT)


def test_T1_a_carried_average_starts_the_new_parameters_at_their_values(source):
    src, bed = source
    state = read_state(str(src / "state"))
    carried = {n: p.detach().clone() for n, p in link(str(src / "model_avg_g8")).named_parameters()}
    model, names, new = _with_parts(src)
    res = fit_units(model, bed, 0, steps=2, start_step=state["step"], optimizer_state=state, optimizer_partial=True,
                    averages=("g8",), average_start={"g8": carried}, **FIT)
    assert res["average_new"] == {"g8": new}
    # the new parameters' average is an average that is not carried: it starts at their values, k from 0
    model_b, _, _ = _with_parts(src)
    res_b = fit_units(model_b, bed, 0, steps=2, start_step=state["step"], optimizer_state=state,
                      optimizer_partial=True, averages=("g8",), **FIT)
    for n in new:
        assert torch.equal(res["averaged"]["g8"][n], res_b["averaged"]["g8"][n]), n
    assert any(not torch.equal(res["averaged"]["g8"][n], res_b["averaged"]["g8"][n]) for n in carried)
    model2, _, _ = _with_parts(src)
    with pytest.raises(KeyError):                                      # without the switch: today's error
        fit_units(model2, bed, 0, steps=2, averages=("g8",), average_start={"g8": carried}, **FIT)
    assert os.path.isdir(src / "model_avg_g8")
