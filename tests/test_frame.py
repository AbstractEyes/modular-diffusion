"""The address frame (T1-T3): the projective gate is the gate every part used before (bit for bit, the reference
expressions written out here); the cosine ramp is sin^2 of the stage digit's angle; stage owners sum to one and cover
the range; the amplitude forms' table and their roster behavior."""
import math

import pytest
import torch

from geolip.modular_diffusion import frame
from geolip.modular_diffusion.forms import add_band_members, add_class_experts, cluster_build
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.staged import Staged
from geolip.modular_diffusion.units import NULL_CLASS
from geolip.modular_diffusion.windows import ramp, window_weight

FIELD = (3, 32, 32)
CUTS = (0.0, 0.25, 0.65, 1.0)
SOFT = 0.06


def _t_grid(n=400, seed=0):
    """Random t with every edge, both ends of every crossfade and the crossfade middles added."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=gen)
    edges = [0.0, 1.0, 0.19, 0.25, 0.31, 0.59, 0.65, 0.71, 0.2, 0.3, 0.6, 0.7, 0.5, 0.0999, 0.1]
    return torch.cat([t, torch.tensor(edges)])


def _reference_gate(t, lo, hi, soft, classes, symbol, cls_idx, symbols):
    """The gate as units.BandMember.gate computed it before this pass."""
    w = torch.ones_like(t)
    if lo > 0:
        w = w * ramp(t - lo, soft)
    if hi < 1:
        w = w * (1 - ramp(t - hi, soft))
    if classes is not None:
        mask = torch.zeros(NULL_CLASS + 1)
        mask[list(classes)] = 1.0
        if symbol:
            if symbols is None:
                return torch.zeros_like(w)
            w = w * mask[symbols]
        else:
            w = w * mask[cls_idx]
    return w


def _model_with_parts():
    b = cluster_build(FIELD, stem_depth=1, dim=32, bands=3)
    add_class_experts(b, [[0], [3, 4]], window=(0.0, 0.25), soft=SOFT, depth=0)
    add_class_experts(b, [[c] for c in range(10)], window=(0.25, 0.65), soft=SOFT, depth=0, prefix="mid")
    add_class_experts(b, [[2], [5]], window=(0.0, 1.0), soft=0.0, depth=0, gate="symbol")
    add_class_experts(b, [[7]], window=(0.65, 1.0), soft=0.0, depth=0, prefix="hard")
    add_band_members(b, (0, 1, 2), depth=0, window=(0.0, 1.0))
    add_band_members(b, (1,), depth=0, window=(0.25, 0.65), soft=SOFT, prefix="win")
    return Model.born(b, 0)


def test_T1_projective_is_every_part_gate_bit_for_bit():
    model = _model_with_parts()
    t = _t_grid()
    gen = torch.Generator().manual_seed(1)
    cls = torch.randint(0, NULL_CLASS + 1, (t.shape[0],), generator=gen)
    cls[:11] = torch.arange(11)                                  # every class and the null class
    sym = torch.randint(0, NULL_CLASS + 1, (t.shape[0],), generator=gen)
    owners = [frame.owner_of(m) for m in model.members]
    for symbols in (sym, None):
        g = frame.projective(owners, t, cls, symbols)
        assert g.shape == (t.shape[0], len(model.members))
        for i, m in enumerate(model.members):
            ref = _reference_gate(t, m.lo, m.hi, m.soft, m.classes, m.symbol_gate, cls, symbols)
            assert torch.equal(g[:, i], ref), m.name
            got = m.gate(t, cls, symbols) if m.symbol_gate else m.gate(t, cls)
            assert torch.equal(got, ref if not m.symbol_gate or symbols is not None else torch.zeros_like(ref)), m.name
    # the null class is off for every class list, on for the members without one
    nul = torch.full_like(cls, NULL_CLASS)
    g = frame.projective(owners, t, nul, None)
    for i, m in enumerate(model.members):
        if m.classes is not None:
            assert float(g[:, i].abs().max()) == 0.0
        else:
            assert torch.equal(g[:, i], window_weight(t, m.lo, m.hi, m.soft))
    # owners given by hand equal owners read from the units
    o = frame.owner(0.0, 0.25, SOFT, classes=[3, 4])
    assert torch.equal(frame.factor(o, t, cls), model.unit("expert_c3-4").gate(t, cls))


def test_T1_stage_weights_hard_and_soft_bit_for_bit():
    t = _t_grid()
    for soft in (0.0, SOFT, 0.03):
        clusters = [Model.born(cluster_build(FIELD, stem_depth=1, dim=32), 0) for _ in range(3)]
        st = Staged(clusters, CUTS, soft)
        ref = torch.stack([window_weight(t, lo, hi, soft) for lo, hi in zip(CUTS, CUTS[1:])])
        assert torch.equal(st.weights(t), ref)
        owners = frame.stage_owners(st)
        assert torch.equal(frame.stage_weights(owners, t), ref)
        assert torch.equal(frame.projective(owners, t), ref.T)
    # the stage units of a model: soft through the window weight, hard through the rows lo <= t < hi
    o = frame.owner(0.25, 1.0, 0.0)
    assert torch.equal(frame.stage_rows(o, t), (t >= 0.25) & (t < 1.0))
    assert bool(frame.stage_rows(o, torch.tensor([1.0]))[0]) is False


def test_T1_the_model_and_its_stage_units_are_unchanged():
    from geolip.modular_diffusion.forms import mixed_tails_build
    for soft in (0.0, SOFT):
        b = mixed_tails_build(FIELD, stem_depth=1, tail_depth=1, cuts=CUTS, soft=soft, dim=32)
        m = Model.born(b, 0)
        with torch.no_grad():
            for u in m.units:                                 # a head that is not zero, so the bodies show
                if u.unit_type == "Head":
                    u.proj.weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(3))
        gen = torch.Generator().manual_seed(2)
        z = torch.randn((8,) + FIELD, generator=gen)
        t = torch.tensor([0.0, 0.2, 0.22, 0.25, 0.3, 0.62, 0.7, 1.0])
        c = torch.arange(8)
        with torch.no_grad():
            v = m.velocity(z, t, c)
            # the reference: the stage mix written out as before this pass
            h0, ctx0, e0, rot = m.stem(torch.cat([m.patchify(z), m.patchify(m.anchor_velocity(z, t, c))], -1), t, c)
            if soft > 0:
                mixed = torch.zeros_like(h0)
                for u in m.stage_units:
                    w = window_weight(t, u.lo, u.hi, u.soft)
                    idx = torch.nonzero(w > 0, as_tuple=False).squeeze(1)
                    y = u(h0[idx], ctx0[idx], e0[idx], rot)
                    mixed = mixed.index_add(0, idx, w[idx].view(-1, 1, 1) * y)
            else:
                mixed = h0.clone()
                for u in m.stage_units:
                    idx = torch.nonzero((t >= u.lo) & (t < u.hi), as_tuple=False).squeeze(1)
                    if idx.numel():
                        mixed[idx] = u(h0[idx], ctx0[idx], e0[idx], rot)
            assert torch.equal(torch.stack([m.anchor_velocity(z, t, c), m.head_part(mixed)]).sum(0), v), soft


def test_T2_the_ramp_is_sin_squared_of_the_stage_angle():
    w = SOFT
    for c in (0.25, 0.65):
        t = torch.linspace(c - w, c + w, 20001, dtype=torch.float64)
        phi = (math.pi / 2) * (t - (c - w)) / (2 * w)
        assert float((ramp(t - c, w) - torch.sin(phi) ** 2).abs().max()) < 1e-7
        assert float((1 - ramp(t - c, w) - torch.cos(phi) ** 2).abs().max()) < 1e-7
    # the frame's own form: the squared projection of the stage digit on owned anchors is the window weight
    t = torch.linspace(0, 1, 20001, dtype=torch.float64)
    a = frame.stage_digit(t, CUTS, w)
    assert float((a.pow(2).sum(1) - 1).abs().max()) < 1e-12          # a unit vector everywhere
    for owned in ([0], [1], [2], [0, 1], [1, 2], [0, 1, 2], [0, 2]):
        lo_hi = [(CUTS[k], CUTS[k + 1]) for k in owned]
        ref = sum(window_weight(t, lo, hi, w) for lo, hi in lo_hi)
        assert float((frame.digit_factor(a, owned) - ref).abs().max()) < 1e-7, owned
    hard = frame.stage_digit(t, CUTS, 0.0)
    assert torch.equal(frame.digit_factor(hard, [1]), window_weight(t, 0.25, 0.65, 0.0))


def test_T2_stage_owners_sum_to_one_and_cover():
    clusters = [Model.born(cluster_build(FIELD, stem_depth=1, dim=32), 0) for _ in range(3)]
    st = Staged(clusters, CUTS, SOFT)
    t = torch.cat([torch.linspace(0, 1, 10001), torch.tensor([0.19, 0.25, 0.31, 0.59, 0.65, 0.71, 1.0])])
    s = frame.stage_weights(frame.stage_owners(st), t).sum(0)
    assert float((s - 1).abs().max()) < 1e-7
    outside = ~(((t > 0.25 - SOFT) & (t < 0.25 + SOFT)) | ((t > 0.65 - SOFT) & (t < 0.65 + SOFT)))
    assert bool((s[outside] == 1.0).all())
    assert frame.check_cover(st) == ([], [])
    assert frame.cover([(0.0, 0.25), (0.65, 1.0)]) == ([(0.25, 0.65)], [])            # a missing stage: a hole
    assert frame.cover([(0.0, 0.25), (0.25, 0.65), (0.25, 0.65), (0.65, 1.0)]) == ([], [(0.25, 0.65)])
    assert frame.cover([(0.0, 0.5), (0.3, 1.0)]) == ([], [(0.3, 0.5)])
    m = Model.born(cluster_build(FIELD, stem_depth=1, dim=32), 0)
    assert frame.check_cover(m) == ([], [])                     # a trunk on every row
    from geolip.modular_diffusion.forms import mixed_tails_build
    mt = Model.born(mixed_tails_build(FIELD, stem_depth=1, tail_depth=1, cuts=CUTS, soft=SOFT, dim=32), 0)
    assert frame.check_cover(mt) == ([], [])


def test_T3_the_amplitude_forms():
    t = torch.tensor([0.1, 0.5, 0.9, 0.25])
    cls = torch.tensor([3, 3, 7, 3])
    owners = [frame.owner(0.0, 0.25, SOFT, classes=[3]), frame.owner(0.25, 0.65, SOFT, classes=[3]),
              frame.owner(0.65, 1.0, SOFT, classes=[3]), frame.owner(0.0, 1.0, 0.0, classes=[5])]
    g = frame.projective(owners, t, cls)
    for form in frame.FORMS:
        w = frame.aleph(owners, t, cls, U=5.0, form=form)
        assert bool((w[g == 0] == 0.0).all()), form                  # exactly silent off the address
        assert bool((w[g > 0] > 0).all())
    table = {16: 0.8318, 64: 0.5408, 256: 0.2254, 1024: 0.0676}
    for n, want in table.items():
        assert abs(frame.own_weight(5.0, n, "plain") - want) < 1e-3, n
        assert abs(frame.own_weight(5.0, n, "discounted") - math.tanh(5.0)) < 1e-3, n
    # two neighbors at mid-crossfade (each gate one half)
    two = owners[:2]
    mid = torch.tensor([0.25])
    c3 = torch.tensor([3])
    assert abs(float(frame.projective(two, mid, c3)[0, 0]) - 0.5) < 1e-7
    assert abs(float(frame.aleph(two, mid, c3, form="plain").sum()) - 0.9983) < 1e-3
    assert abs(float(frame.aleph(two, mid, c3, form="discounted").sum()) - 1.0282) < 1e-3
    # removing silent parts: the discounted weights of the others are bit-identical, the plain ones change
    rows = torch.tensor([0.1, 0.25, 0.3])
    cr = torch.tensor([3, 3, 3])
    silent = [frame.owner(0.0, 1.0, 0.0, classes=[c]) for c in (0, 1, 2)] + [frame.owner(0.65, 1.0, 0.0, classes=[3])]
    roster = [owners[0], silent[0], owners[1], silent[1], silent[2], silent[3]]
    assert float(frame.projective(silent, rows, cr).abs().max()) == 0.0
    full_d = frame.aleph(roster, rows, cr, form="discounted")
    less_d = frame.aleph(two, rows, cr, form="discounted")
    assert torch.equal(full_d[:, [0, 2]], less_d)
    full_p = frame.aleph(roster, rows, cr, form="plain")
    less_p = frame.aleph(two, rows, cr, form="plain")
    assert not torch.equal(full_p[:, [0, 2]], less_p)
    with pytest.raises(ValueError):
        frame.aleph(two, rows, cr, form="nope")
