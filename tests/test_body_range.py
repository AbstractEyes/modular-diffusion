"""The noise level's training range and whole-range share (6), and the Body unit (7)."""
import zlib

import pytest
import torch

from geolip.modular_diffusion import compat
from geolip.modular_diffusion.build import Build, UnitEntry
from geolip.modular_diffusion.forms import cluster_build, mixed_tails_build
from geolip.modular_diffusion.linker import probe_batch
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.windows import window_weight

FIELD = (3, 32, 32)
CUTS = (0.0, 0.25, 0.65, 1.0)


def _spectrum():
    return torch.rand(FIELD, generator=torch.Generator().manual_seed(1), dtype=torch.float64) + 0.1


def _live(model, names=None):
    with torch.no_grad():
        for u in model.units:
            if names is not None and u.name not in names:
                continue
            for k, p in u.named_parameters():
                g = torch.Generator().manual_seed(zlib.crc32(f"{u.name}/{k}".encode()))
                p.add_(torch.randn(p.shape, generator=g) * 0.05)


def _draw(n, **kw):
    gen = torch.Generator().manual_seed(77)
    z1 = torch.randn(n, *FIELD, generator=torch.Generator().manual_seed(3))
    cls = torch.zeros(n, dtype=torch.long)
    out = Model.draw(z1, cls, gen, 0.1, **kw)
    return out, gen.get_state()


# ------------------------------------------------------------------ 6
def test_6_whole_range_is_bit_identical_and_a_range_confines_t():
    (zt0, t0, c0, y0), s0 = _draw(64)
    (zt1, t1, c1, y1), s1 = _draw(64, t_range=(0.0, 1.0))
    assert torch.equal(t0, t1) and torch.equal(zt0, zt1) and torch.equal(c0, c1) and torch.equal(s0, s1)
    (_, t2, c2, _), s2 = _draw(64, t_range=(0.19, 0.71))
    assert bool(((t2 >= 0.19) & (t2 < 0.71)).all())
    assert torch.equal(c2, c0) and torch.equal(s2, s0)                    # the same draws in the same order
    assert torch.allclose(t2, 0.19 + 0.52 * t0)
    with pytest.raises(ValueError):
        _draw(4, t_sampling="logit_normal", t_range=(0.0, 0.5))
    with pytest.raises(ValueError):
        _draw(4, t_range=(0.5, 0.5))


def test_6_whole_range_share():
    (_, ta, _, _), sa = _draw(64, t_range=(0.19, 0.71))
    (_, tb, _, _), sb = _draw(64, t_range=(0.19, 0.71), t_mix=0.0)
    (_, tc, _, _), sc = _draw(64, t_mix=0.3)                             # the whole range: nothing extra drawn
    (_, td, _, _), sd = _draw(64)
    assert torch.equal(ta, tb) and torch.equal(sa, sb) and torch.equal(tc, td) and torch.equal(sc, sd)
    for bad in (dict(t_mix=1.0), dict(t_mix=-0.1), dict(t_mix=0.3, t_sampling="logit_normal")):
        with pytest.raises(ValueError):
            _draw(4, **bad)
    n, p = 20000, 0.3
    (_, t, _, _), _ = _draw(n, t_range=(0.19, 0.71), t_mix=p)
    # replay the generator: noise, t, the mask, the whole-range draw
    gen = torch.Generator().manual_seed(77)
    torch.randn((n,) + FIELD, generator=gen)
    t_own = 0.19 + 0.52 * torch.rand(n, generator=gen)
    m = torch.rand(n, generator=gen) < p
    t_whole = torch.rand(n, generator=gen)
    assert torch.equal(t[~m], t_own[~m]) and torch.equal(t[m], t_whole[m])
    outside = (t < 0.19) | (t >= 0.71)
    share = float(outside.float().mean())
    assert 0.27 * 0.48 <= share <= 0.33 * 0.48, share
    assert not bool(outside[~m].any())
    assert compat._range_suffix({"t_mix": 0.3, "t_lo": 0.19, "t_hi": 0.71}) == "_t0.19-0.71_tmix0.3"
    assert compat._range_suffix({"t_mix": 0.0}) == ""


# ------------------------------------------------------------------ 7
def test_7_body_birth_is_its_own():
    a = Model.born(cluster_build(FIELD, stem_depth=2, body_depth=2, dim=32), 0, _spectrum())
    b = Model.born(cluster_build(FIELD, stem_depth=3, body_depth=2, dim=32, anchor_in=0), 0, _spectrum())
    c = Model.born(cluster_build(FIELD, stem_depth=2, body_depth=2, dim=32), 1, _spectrum())
    for k, v in a.body.state_dict().items():
        assert torch.equal(v, b.body.state_dict()[k]), k
    assert any(not torch.equal(v, c.body.state_dict()[k]) for k, v in a.body.state_dict().items())
    tails = Model.born(mixed_tails_build(FIELD, stem_depth=2, tail_depth=2, cuts=CUTS, soft=0.06, dim=32), 0, _spectrum())
    for s in range(3):
        for k, v in a.body.state_dict().items():                      # one shared stream "body": born equal
            assert torch.equal(v, tails.unit(f"body_{s}").state_dict()[k]), (s, k)
    own = Model.born(cluster_build(FIELD, stem_depth=2, body_depth=2, dim=32, body_stream="other"), 0, _spectrum())
    assert not torch.equal(own.body.blocks[0].attn.qkv.weight, a.body.blocks[0].attn.qkv.weight)


def test_7_whole_range_body_runs_on_every_row():
    m = Model.born(cluster_build(FIELD, stem_depth=1, body_depth=1, dim=32), 0, _spectrum())
    _live(m)
    rows = []
    orig = m.body.forward
    m.body.forward = lambda h, ctx, e, rot: (rows.append(h.shape[0]), orig(h, ctx, e, rot))[1]
    assert m.stage_units == [] and m.body is not None
    z = torch.randn(5, *FIELD, generator=torch.Generator().manual_seed(2))
    t = torch.tensor([0.0, 0.3, 0.65, 0.99, 1.0])
    cls = torch.zeros(5, dtype=torch.long)
    with torch.no_grad():
        p_on, _, _, _ = m.components(z, t, cls)
        p_off, _, _, _ = m.components(z, t, cls, skip=["body"])
    assert rows == [5]
    assert bool(((p_on["body"] - p_off["body"]).abs().sum((1, 2, 3)) > 0).all())     # t = 1 included


@pytest.mark.parametrize("soft", [0.0, 0.06])
def test_7_windowed_bodies_mix_as_stage_units(soft):
    m = Model.born(mixed_tails_build(FIELD, stem_depth=1, tail_depth=1, cuts=CUTS, soft=soft, dim=32), 0, _spectrum())
    _live(m)
    assert [u.name for u in m.stage_units] == ["body_0", "body_1", "body_2"] and m.body is None
    z = torch.randn(9, *FIELD, generator=torch.Generator().manual_seed(2))
    t = torch.tensor([0.0, 0.2, 0.24, 0.25, 0.28, 0.5, 0.64, 0.7, 0.99])
    cls = torch.zeros(9, dtype=torch.long)
    with torch.no_grad():
        parts, h, ctx, e = m.components(z, t, cls)
        x = m.patchify(z)
        x = torch.cat([x, m.patchify(m.anchor_velocity(z, t))], dim=-1)
        h0, ctx0, e0, rot = m.stem(x, t, cls)
        want = torch.zeros_like(h0)
        for u in m.stage_units:
            w = window_weight(t, u.lo, u.hi, soft) if soft > 0 else ((t >= u.lo) & (t < u.hi)).float()
            want = want + w.view(-1, 1, 1) * u(h0, ctx0, e0, rot)
    assert float((h - want).abs().max()) < 1e-5


def test_7_mixing_kinds_raises_and_mixed_tails_is_born_the_anchor():
    geo = cluster_build(FIELD, stem_depth=1, dim=32).geometry
    base = cluster_build(FIELD, stem_depth=1, dim=32).units
    whole = UnitEntry("body", "Body", {"site": "trunk", "slot": 1}, {"depth": 1})
    whole2 = UnitEntry("body2", "Body", {"site": "trunk", "slot": 1, "window": ["0", "1"]}, {"depth": 1})
    win = UnitEntry("body_w", "Body", {"site": "trunk", "slot": 1, "window": ["0.0", "0.5"], "soft": "0.0"}, {"depth": 1})
    stage = UnitEntry("stage_0", "StageBody", {"site": "trunk", "slot": 1, "window": ["0.5", "1.0"], "soft": "0.0"},
                      {"index": 0}, stream="stage_blocks")
    for extra in ([whole, whole2], [whole, win], [win, stage], [whole, stage]):
        with pytest.raises(ValueError):
            Model.born(Build("x", geo, base[:2] + extra + base[2:]), 0, _spectrum())
    m = Model.born(mixed_tails_build(FIELD, stem_depth=1, tail_depth=1, cuts=CUTS, soft=0.06, dim=32), 0, _spectrum())
    _live(m, names=["stem", "body_0", "body_1", "body_2"])
    z, t, cls = probe_batch(m.build.geometry)
    with torch.no_grad():
        assert float((m.velocity(z, t, cls) - m.anchor_velocity(z, t)).abs().max()) == 0.0


def test_7_a_body_has_no_flat_layout():
    m = Model.born(cluster_build(FIELD, stem_depth=1, body_depth=1, dim=32), 0, _spectrum())
    with pytest.raises(ValueError, match="no flat layout"):
        compat.to_legacy_state(m)
    plain = Model.born(cluster_build(FIELD, stem_depth=1, dim=32), 0, _spectrum())
    assert len(compat.to_legacy_state(plain)) > 0
