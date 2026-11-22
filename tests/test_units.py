"""Per-name birth (test 10) and initialization parity with the original script (P2)."""
import pytest
import torch

from geolip.modular_diffusion import compat, train
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.units import BandMember, born, name_seed

FIELD = (3, 32, 32)
GEOM = {"field": [3, 32, 32], "grid": 16, "patch": 2, "token_dim": 12, "dim": 64, "heads": 2, "time_freq": 64,
        "classes": 11, "bands": 3}
ADDR = {"site": "out", "band": 1, "window": ["0.0", "1.0"], "soft": "0.0", "classes": None}
SPEC = {"depth": 1, "code": [4.0, 8.0], "keep": True}


def _args(**kw):
    return dict(train.DEFAULTS, bed="pictures", **kw)


def test_10_birth_depends_on_seed_and_name_only():
    torch.manual_seed(99)
    before = torch.get_rng_state()
    a = born(BandMember, 0, "b1_s0", ADDR, SPEC, GEOM)
    assert torch.equal(torch.get_rng_state(), before)                     # the caller's generator is left as found
    b = born(BandMember, 0, "b1_s0", ADDR, SPEC, GEOM)
    for (k, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), k
    c = born(BandMember, 1, "b1_s0", ADDR, SPEC, GEOM)
    d = born(BandMember, 0, "b2_s0", ADDR, SPEC, GEOM)
    w = lambda u: u.blocks[0].attn.qkv.weight
    assert not torch.equal(w(a), w(c)) and not torch.equal(w(a), w(d))
    assert name_seed(0, "x") != name_seed(1, "x")


def test_10_member_independent_of_companions_and_order():
    small = compat.build_from_args(_args(dim=64, extra="1@0.25-0.65"), FIELD)
    big = compat.build_from_args(_args(dim=64, extra="0@0-0.5#q,1@0.25-0.65", stages="3", content="0-1:2"), FIELD)
    big.units = list(reversed(big.units))
    m1, m2 = Model.born(small, 3), Model.born(big, 3)
    u1, u2 = m1.unit("x_b1_0.25-0.65"), m2.unit("x_b1_0.25-0.65")
    for (k, p), (_, q) in zip(u1.named_parameters(), u2.named_parameters()):
        assert torch.equal(p, q), k
    for k, p in m1.stem.named_parameters():
        assert torch.equal(p, dict(m2.stem.named_parameters())[k]), k


CASES = {
    "base": dict(bands=3, keep=1, codes="rung", anchor=1, anchor_in=1),
    "ain0": dict(bands=3, keep=1, codes="rung", anchor=1, anchor_in=0),
    "ain2": dict(bands=3, keep=1, codes="rung", anchor=0, anchor_in=2),
    "body4": dict(body_depth=4, bands=3),
    "member2": dict(member_depth=2, bands=3, stages="3"),
    "extras9": dict(base=1, extra="0@0-0.25,1@0-0.25,2@0-0.25,0@0.25-0.65#1,1@0.25-0.65#1,2@0.25-0.65#1,a@0.65-1,a@0-1#z,1@0.5-1#w"),
    "content": dict(content="0-1:2;0.25-0.65:10", stages="3", soft=0.05),
    "stage": dict(stage_body="3"),
    "stage_zero": dict(stage_body="5mid", stage_zero=1, stage_soft=0.06),
    "plain": dict(arch="plain", plain_depth=2, anchor=1, anchor_in=1),
    "nobase": dict(base=0, extra="2@0-1"),
    "bands1": dict(bands=1, keep=1),
    "nohead": dict(body_out=0, codes="whole"),
}


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("case", sorted(CASES))
def test_p2_init_matches_original(legacy, case, seed):
    a = _args(**CASES[case])
    la = dict(legacy.ARGS, **{k: v for k, v in a.items() if k in legacy.ARGS})
    spectrum = torch.rand(FIELD, generator=torch.Generator().manual_seed(1), dtype=torch.float64) + 0.1
    torch.manual_seed(seed)
    ref = legacy.Collective(la, FIELD, spectrum, seed)
    mine = Model.born(compat.build_from_args(a, FIELD), seed, spectrum)
    want = ref.state_dict()
    got = compat.to_legacy_state(mine)
    assert list(got) == list(want)
    for k in want:
        assert torch.equal(want[k], got[k]), k
    assert [m.name for m in ref.members] == [m.name for m in mine.members]
    assert torch.equal(ref.spectrum, mine.anchor.spectrum)
    assert torch.equal(ref.masks, mine.masks)
    for m_ref, m in zip(ref.members, mine.members):
        assert (m_ref.lo, m_ref.hi, m_ref.classes) == (m.lo, m.hi, m.classes)
        assert m_ref.band == mine.band_index(m)
        assert torch.equal(m_ref.code.cos, m.code.cos)
        assert torch.equal(m_ref.cls_mask, m.cls_mask)
    assert ref.edges == mine.edges


def test_stored_flat_file_loads_into_units_and_back(stored_runs):
    """The stored 6,000-step base weights load into units by name and come back as the same flat state."""
    import json
    import os
    from conftest import find_stored, STORED_BASE_PT
    path = find_stored(stored_runs, "_pictures_b3k_rung_st1_anchor_ain1_s0_steps6000_d128.pt", STORED_BASE_PT)
    with open(path[:-3] + ".json", encoding="utf-8") as fh:
        stored_args = json.load(fh)["args"]
    a = dict(train.DEFAULTS, **{k: v for k, v in stored_args.items() if k in train.DEFAULTS})
    model = Model.empty(compat.build_from_args(a, FIELD))
    info = compat.load_legacy_files(model, [path])
    assert info["members_new"] == [] and info["members_unused"] == []
    state = torch.load(path)
    back = compat.to_legacy_state(model)
    assert len(state) == 79 and sum(v.numel() for v in state.values()) == 1_487_152
    assert list(back) == list(state)
    assert all(torch.equal(state[k], back[k]) for k in state)
