"""Rows of a table as the only trainable part of a parameter (T2): after a fit with trainable_rows={"stem":
{"cls.weight": [9]}} only row 9 differs; every other row is bit-equal in the weights and in every average; the hook is
gone afterwards (a later fit of the stem moves other rows); refused together with a state that holds the parameter."""
import pytest
import torch

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.experiments import stages
from geolip.modular_diffusion.linker import link
from geolip.modular_diffusion.train import fit_units, read_state

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=0,
             judge="fake", sheet=0, threads=4)
FIT = dict(lr=1e-2, warmup=1, batch=32, heldout=16, device="cpu", log=lambda s: None, log_every=100)
ROWS = {"stem": {"cls.weight": [9]}}


@pytest.fixture(scope="module")
def source(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("rows")
    stages.main(**dict(SMALL, data=synthetic_npz, variant="source", out=str(root / "source")))
    return root / "source", PicturesBed(synthetic_npz, "cpu")


def test_T2_only_the_listed_row_moves(source):
    src, bed = source
    model = link(str(src / "model"))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    table = before_table = model.unit("stem").cls.weight.detach().clone()
    res = fit_units(model, bed, 0, steps=4, trainable=[], trainable_rows=ROWS, averages=("g8", "tail100", 0.9),
                    **FIT)
    table = model.unit("stem").cls.weight.detach()
    others = [r for r in range(table.shape[0]) if r != 9]
    assert not torch.equal(table[9], before_table[9])
    assert torch.equal(table[others], before_table[others])
    name = next(n for n, p in model.named_parameters() if p is model.unit("stem").cls.weight)
    for n, p in model.named_parameters():                              # nothing else moved
        if n != name:
            assert torch.equal(p.detach(), before[n]), n
    for key, avg in res["averaged"].items():
        assert set(avg) == {name}, key
        assert torch.equal(avg[name][others], before_table[others]), key
        assert not torch.equal(avg[name][9], before_table[9]), key
    assert res["trainable_rows"] == {"stem": {"cls.weight": [9]}} and res["rows_frozen_equal"] is True
    assert res["trained_numbers"] == table.shape[1] and res["frozen_unchanged"] is True
    assert isinstance(res["rows_restored"], bool)


def test_T2_the_hook_is_gone_after_the_fit(source):
    src, bed = source
    model = link(str(src / "model"))
    fit_units(model, bed, 0, steps=2, trainable=[], trainable_rows=ROWS, **FIT)
    mid = model.unit("stem").cls.weight.detach().clone()
    fit_units(model, bed, 0, steps=3, trainable=["stem"], **FIT)
    after = model.unit("stem").cls.weight.detach()
    moved = [r for r in range(after.shape[0]) if r != 9 and not torch.equal(after[r], mid[r])]
    assert moved


def test_T2_refused_with_a_state_that_holds_the_parameter(source):
    src, bed = source
    state = read_state(str(src / "state"))
    model = link(str(src / "model"))
    with pytest.raises(ValueError):
        fit_units(model, bed, 0, steps=1, trainable="all", trainable_rows=ROWS, optimizer_state=state,
                  start_step=state["step"], **FIT)
    with pytest.raises(KeyError):
        fit_units(link(str(src / "model")), bed, 0, steps=1, trainable=[], trainable_rows={"stem": {"nothing": [1]}},
                  **FIT)
    with pytest.raises(ValueError):
        fit_units(link(str(src / "model")), bed, 0, steps=1, trainable=[], trainable_rows={"stem": {"cls.weight": [11]}},
                  **FIT)
