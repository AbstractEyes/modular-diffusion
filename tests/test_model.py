"""The assembled model: training-step parity with the original script (P3), silence off window and class (test 5),
band leak (test 6) and zero birth (test 7)."""
import zlib

import pytest
import torch

from geolip.modular_diffusion import compat, train
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.spectral import band_masks, band_power

FIELD = (3, 32, 32)


def _args(**kw):
    return dict(train.DEFAULTS, bed="pictures", **kw)


def _spectrum():
    return torch.rand(FIELD, generator=torch.Generator().manual_seed(1), dtype=torch.float64) + 0.1


def _live_by_name(model):
    """Zero-born tensors get small random values seeded by (unit name, key), so two builds sharing a unit agree."""
    with torch.no_grad():
        for u in model.units:
            for k, p in u.named_parameters():
                if float(p.abs().max()) == 0.0:
                    g = torch.Generator().manual_seed(zlib.crc32(f"{u.name}/{k}".encode()))
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)


# ------------------------------------------------------------------ P3
P3_CASES = {
    "base": dict(bands=3, keep=1, codes="rung", anchor=1, anchor_in=1),
    "no_anchor": dict(bands=3, keep=1, codes="rung", anchor=0),
    "ain2": dict(bands=3, keep=1, anchor=1, anchor_in=2),
    "soft_windows": dict(stages="3", soft=0.05, anchor=1),
    "stage_crossfade": dict(stage_body="3", stage_soft=0.06, anchor=1),
    "stage_hard_zero": dict(stage_body="5mid", stage_zero=1, anchor=1),
    "member_dropout": dict(mdrop=0.25, keep=0, anchor=1),
    "logit_normal": dict(t_sampling="logit_normal", anchor=1, anchor_in=1),
    "content": dict(content="0.25-0.65:2;0-1:10", anchor=1, extra="a@0.65-1"),
    "plain": dict(arch="plain", plain_depth=2, anchor=1, anchor_in=1),
}


@pytest.mark.parametrize("case", sorted(P3_CASES))
def test_p3_training_steps_match_original(legacy, case):
    a = _args(**P3_CASES[case])
    la = dict(legacy.ARGS, **{k: v for k, v in a.items() if k in legacy.ARGS})
    spectrum = _spectrum()
    ref = legacy.Collective(la, FIELD, spectrum, 0)
    for p in ref.parameters():                         # leave the zero-born state so every part is live
        if float(p.detach().abs().max()) == 0.0:
            torch.manual_seed(int(p.numel()))
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.02)
    mine = Model.born(compat.build_from_args(a, FIELD), 0, spectrum)
    info = compat.load_legacy_state(mine, ref.state_dict(), [m.name for m in ref.members])
    assert info["members_unused"] == [] and len(info["members_loaded"]) == len(mine.members)
    pmap = compat.legacy_parameter_map(mine)
    rmap = dict(ref.named_parameters())
    assert set(pmap) == set(rmap)
    for k in rmap:
        assert torch.equal(rmap[k], pmap[k]), k
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(16, *FIELD, generator=gen)
    cls = torch.randint(0, 10, (16,), generator=gen)
    o_ref = torch.optim.Adam(ref.parameters(), lr=1e-3)
    o_mine = torch.optim.Adam(mine.parameters(), lr=1e-3)
    for step in range(4):
        g1 = torch.Generator().manual_seed(100 + step)
        g2 = torch.Generator().manual_seed(100 + step)
        l1 = ref.loss(x, cls, g1, 0.1)
        l2 = mine.loss(x, cls, g2, 0.1, mdrop=float(a["mdrop"]), t_sampling=a["t_sampling"])
        assert torch.equal(l1, l2), (step, float(l1), float(l2))
        assert torch.equal(g1.get_state(), g2.get_state())           # the same number of draws
        o_ref.zero_grad()
        o_mine.zero_grad()
        l1.backward()
        l2.backward()
        for k in rmap:
            assert (rmap[k].grad is None) == (pmap[k].grad is None), k
            if rmap[k].grad is not None:
                assert torch.equal(rmap[k].grad, pmap[k].grad), (step, k)
        legacy.equalize(ref)
        train.equalize(mine)
        for k in rmap:
            if rmap[k].grad is not None:
                assert torch.equal(rmap[k].grad, pmap[k].grad), ("equalized", step, k)
        o_ref.step()
        o_mine.step()
        for k in rmap:
            assert torch.equal(rmap[k], pmap[k]), ("weights", step, k)


def test_p3_components_and_read_paths_match(legacy):
    """components(only=...), skipping the stage units and the gate agree with the original."""
    a = _args(stage_body="3", anchor=1, stages="3", soft=0.05, content="0-1:2")
    la = dict(legacy.ARGS, **{k: v for k, v in a.items() if k in legacy.ARGS})
    spectrum = _spectrum()
    ref = legacy.Collective(la, FIELD, spectrum, 2)
    for p in ref.parameters():
        if float(p.detach().abs().max()) == 0.0:
            torch.manual_seed(int(p.numel()))
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.02)
    mine = Model.born(compat.build_from_args(a, FIELD), 2, spectrum)
    compat.load_legacy_state(mine, ref.state_dict(), [m.name for m in ref.members])
    gen = torch.Generator().manual_seed(9)
    zt = torch.randn(12, *FIELD, generator=gen)
    t = torch.tensor([0.0, 0.1, 0.24, 0.25, 0.26, 0.3, 0.6, 0.64, 0.65, 0.66, 0.9, 1.0])
    cls = torch.tensor([0, 1, 2, 3, 10, 5, 6, 7, 8, 9, 10, 4])
    with torch.no_grad():
        for kw_ref, kw_mine in (({}, {}), ({"skip_stage": True}, {"skip": [u.name for u in mine.stage_units]}),
                                ({"only": {"b0_s1", "c0_b2_0-1"}}, {"only": {"b0_s1", "c0_b2_0-1"}})):
            p1, h1, c1, e1 = ref.components(zt, t, cls, **kw_ref)
            p2, h2, c2, e2 = mine.components(zt, t, cls, **kw_mine)
            assert list(p1) == list(p2)
            for k in p1:
                assert torch.equal(p1[k], p2[k]), k
            assert torch.equal(h1, h2) and torch.equal(c1, c2) and torch.equal(e1, e2)
        for m_ref, m in zip(ref.members, mine.members):
            assert torch.equal(ref.gate(m_ref, t, cls), mine.gate(m, t, cls))
        assert torch.equal(ref.velocity(zt, t, cls), mine.velocity(zt, t, cls))


# ------------------------------------------------------------------ test 5
def test_5_silent_off_window_and_class():
    base = _args(anchor=1, dim=64)
    more = _args(anchor=1, dim=64, extra="1@0.25-0.65", content="0.25-0.65:2")
    m0 = Model.born(compat.build_from_args(base, FIELD), 0, _spectrum())
    m1 = Model.born(compat.build_from_args(more, FIELD), 0, _spectrum())
    _live_by_name(m0)
    _live_by_name(m1)
    gen = torch.Generator().manual_seed(3)
    zt = torch.randn(10, *FIELD, generator=gen)
    t = torch.tensor([0.0, 0.1, 0.249, 0.25, 0.4, 0.5, 0.6499, 0.65, 0.9, 1.0])
    cls = torch.tensor([2, 0, 3, 8, 10, 4, 1, 9, 5, 10])
    with torch.no_grad():
        p0, _, _, _ = m0.components(zt, t, cls)
        p1, _, _, _ = m1.components(zt, t, cls)
    for k in p0:
        assert torch.equal(p0[k], p1[k]), k
    inside = (t >= 0.25) & (t < 0.65)
    animals = (cls >= 2) & (cls <= 7)
    x = p1["x_b1_0.25-0.65"]
    assert bool((x[~inside] == 0).all()) and bool((x[inside].abs().sum((1, 2, 3)) > 0).all())
    for name, group in (("c0_b0_0.25-0.65", animals), ("c1_b0_0.25-0.65", (cls <= 9) & ~animals)):
        y = p1[name]
        on = inside & group
        assert bool((y[~on] == 0).all()), name
        assert bool((y[on].abs().sum((1, 2, 3)) > 0).all()), name
    assert bool((p1["c0_b0_0.25-0.65"][cls == 10] == 0).all())           # the null class reaches no content expert


# ------------------------------------------------------------------ test 6
def test_6_band_leak():
    m = Model.born(compat.build_from_args(_args(anchor=0, dim=64, stages="3"), FIELD), 0, _spectrum())
    _live_by_name(m)
    gen = torch.Generator().manual_seed(4)
    zt = torch.randn(8, *FIELD, generator=gen)
    t = torch.linspace(0.01, 0.99, 8)
    cls = torch.arange(8)
    masks, _ = band_masks(16, 3)
    with torch.no_grad():
        parts, _, _, _ = m.components(zt, t, cls)
    checked = 0
    for mem in m.members:
        if mem.name not in parts:
            continue
        p = band_power(parts[mem.name], m.patch, m.grid)
        b = m.band_index(mem)
        share_off = float(p[~masks[b]].sum() / p.sum())
        assert share_off < 1e-12, (mem.name, share_off)
        checked += 1
    assert checked >= 3


# ------------------------------------------------------------------ test 7
ZERO_BORN = {
    "head": (dict(anchor=1, body_out=0), dict(anchor=1, body_out=1)),
    "band_member": (dict(anchor=1), dict(anchor=1, extra="2@0.25-0.65")),
    "content_expert": (dict(anchor=1), dict(anchor=1, content="0.25-0.65:2")),
    "growth_hard": (dict(anchor=1), dict(anchor=1, stage_body="3", stage_zero=1)),
    "growth_crossfade": (dict(anchor=1), dict(anchor=1, stage_body="3", stage_zero=1, stage_soft=0.06)),
}


@pytest.mark.parametrize("case", sorted(ZERO_BORN))
def test_7_zero_born_units_are_silent_and_learn(case):
    before, after = (_args(dim=64, **kw) for kw in ZERO_BORN[case])
    old = Model.born(compat.build_from_args(before, FIELD), 0, _spectrum())
    _live_by_name(old)
    new = Model.born(compat.build_from_args(after, FIELD), 0, _spectrum())
    newborn = [u for u in new.units if u.name not in old.build.names()]
    assert newborn
    for u in old.units:
        new.unit(u.name).load_owned(u.owned_state())
    gen = torch.Generator().manual_seed(6)
    zt = torch.randn(10, *FIELD, generator=gen)
    t = torch.tensor([0.05, 0.2, 0.22, 0.25, 0.28, 0.5, 0.62, 0.65, 0.68, 0.95])      # inside both crossfade overlaps
    cls = torch.tensor([2, 3, 4, 5, 6, 7, 0, 1, 8, 10])
    with torch.no_grad():
        assert torch.equal(old.velocity(zt, t, cls), new.velocity(zt, t, cls))
    loss = ((new.velocity(zt, t, cls) - zt) ** 2).mean()
    loss.backward()
    for u in newborn:
        zero_layers = [p for k, p in u.named_parameters() if k.endswith("proj.weight") or k.endswith("out.weight")
                       or k.endswith("mlp.2.weight")]
        assert zero_layers
        assert any(float(p.grad.abs().max()) > 0 for p in zero_layers if p.grad is not None), u.name
