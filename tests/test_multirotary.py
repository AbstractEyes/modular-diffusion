"""The multi-rotary frame (CPU): the structure reproduces the anchored rotary bit for bit, its levels' kernels, the
rotary against the kernel, a query turned by another address, the gate against the integer owner, silence off the
address, the two-owner arithmetic, the key binding and the read of attention weights."""
import math

import pytest
import torch

from geolip.modular_diffusion import collective as C
from geolip.modular_diffusion import frame, rotary, scenes
from geolip.modular_diffusion import multirotary as M
from geolip.modular_diffusion.blocks import SelfAttention
from geolip.modular_diffusion.spectral import patchify

PLACE = ("place_column", "place_row")
MEMBER = ("member_0", "member_1")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _positions(b, n, g):
    """Random integer and half-integer positions [B, N] (float64)."""
    return torch.randint(0, 25, (b, n), generator=g).double() / 2


def _pair_contributions(q, k, half):
    """[..., pairs]: q_i k_i + q_{i + half} k_{i + half} for every pair."""
    return q[..., :half] * k[..., :half] + q[..., half:] * k[..., half:]


# ------------------------------------------------------------------ 1. the structure is the anchored rotary
def test_place_only_reproduces_the_anchored_rotary_bit_for_bit():
    g = _gen(0)
    b, n = 3, 40
    rows, cols = _positions(b, n, g), _positions(b, n, g)
    dcol, drow = torch.randint(0, 3, (b, n), generator=g), torch.randint(0, 3, (b, n), generator=g)
    digits = {"place_column": dcol, "place_row": drow}
    assert torch.equal(M.PLACE_ONLY.angles(rows, cols, digits), rotary.anchored_angles(rows, cols, dcol, drow))
    q, k = torch.randn(b, 2, n, 64, generator=g), torch.randn(b, 2, n, 64, generator=g)
    q1, k1 = M.PLACE_ONLY.rotary(rows, cols, digits)(q, k)
    q2, k2 = rotary.anchored_rotary(rows, cols, dcol, drow)(q, k)
    assert torch.equal(q1, q2) and torch.equal(k1, k2)


# ------------------------------------------------------------------ 2. validation
def test_overlapping_or_missing_pairs_raise():
    with pytest.raises(ValueError):
        M.Structure(64, range(0, 10), range(9, 20), range(20, 24), (), M.PLACE_ONLY.levels)     # pair 9 twice
    with pytest.raises(ValueError):
        M.Structure(64, range(0, 10), range(10, 20), range(20, 23), (), M.PLACE_ONLY.levels)    # pair 23 missing
    with pytest.raises(ValueError):
        M.Structure(64, range(0, 10), range(10, 20), range(20, 24), (24,), M.PLACE_ONLY.levels)  # 24 held and a level's


# ------------------------------------------------------------------ 3. the simplex levels
def test_every_simplex_level_scores_p_or_minus_half_and_its_kernel_is_one_or_zero():
    for lv in M.PLACE_MEMBER.levels:
        assert lv.is_simplex(), lv.name
        p = len(lv.distinct_ks())
        a = torch.arange(lv.modulus)[:, None].expand(-1, lv.modulus)
        b = torch.arange(lv.modulus)[None, :].expand(lv.modulus, -1)
        eq = a == b
        s = M.pair_cosine_sum(lv, a, b)
        assert float((s[eq] - p).abs().max()) < 1e-12, lv.name
        assert float((s[~eq] + 0.5).abs().max()) < 1e-12, lv.name
        k = M.level_kernel(lv, a, b)
        assert torch.equal(k, eq.double()), lv.name                       # whole-number digits: exactly 1.0 or 0.0


# ------------------------------------------------------------------ 4. the rotary agrees with the kernel
def test_the_turned_dot_product_on_one_level_is_its_pair_cosine_sum_over_p():
    """q = k = (1 / sqrt(P)) on the first number of each of the level's P pairs, 0 elsewhere (unit length). Pair i turns
    as x e^{i a}, so q' . k' = sum_i (1 / P) Re[e^{i (a_q,i - a_k,i)}] = pair_cosine_sum(a, b) / P, where the query
    (token 0) carries the digit a and the key (token 1) the digit b of the level."""
    st = M.PLACE_MEMBER
    zero = torch.zeros(1, 2, dtype=torch.float64)
    for lv in st.levels:
        p = len(lv.pairs)
        v = torch.zeros(1, 1, 2, st.head_dim)
        v[..., list(lv.pairs)] = 1.0 / math.sqrt(p)
        for a in range(lv.modulus):
            for b in range(lv.modulus):
                kd = {n: torch.zeros(1, 2, dtype=torch.long) for n in st.names}
                qd = {n: torch.zeros(1, 2, dtype=torch.long) for n in st.names}
                kd[lv.name] = torch.tensor([[0, b]])
                qd[lv.name] = torch.tensor([[a, 0]])
                q, k = st.rotary(zero, zero, kd, qd)(v, v)
                dot = float((q[0, 0, 0] * k[0, 0, 1]).sum())
                want = float(M.pair_cosine_sum(lv, torch.tensor(a), torch.tensor(b))) / p
                assert abs(dot - want) < 1e-6, (lv.name, a, b)


# ------------------------------------------------------------------ 5. a query turned by another address
def test_a_query_at_another_member_address_changes_only_that_levels_pairs():
    st = M.PLACE_MEMBER
    g = _gen(5)
    b, n = 2, 12
    rows, cols = _positions(b, n, g), _positions(b, n, g)
    kd = {"place_column": torch.randint(0, 3, (b, n), generator=g), "place_row": torch.randint(0, 3, (b, n), generator=g),
          "member_0": torch.randint(0, 5, (b, n), generator=g), "member_1": torch.randint(0, 5, (b, n), generator=g)}
    qd = dict(kd, member_0=(kd["member_0"] + 1) % 5)
    q, k = torch.randn(b, 2, n, 64, generator=g), torch.randn(b, 2, n, 64, generator=g)
    q0, k0 = st.rotary(rows, cols, kd)(q, k)
    q1, k1 = st.rotary(rows, cols, kd, qd)(q, k)
    assert torch.equal(k0, k1)                                            # the keys are the same
    c0 = _pair_contributions(q0[..., :, None, :], k0[..., None, :, :], 32)   # [B, H, N, N, 32]
    c1 = _pair_contributions(q1[..., :, None, :], k1[..., None, :, :], 32)
    lv = list(st.level("member_0").pairs)
    others = [i for i in range(32) if i not in lv]
    assert torch.equal(c0[..., others], c1[..., others])
    assert not torch.equal(c0[..., lv], c1[..., lv])


# ------------------------------------------------------------------ 6. the gate is the integer owner
MEMBERS_9 = torch.tensor([[t // 3, t % 3] for t in range(9)])          # thing t at (family, member)


def _digits_of_layout(cells, colors):
    """The request slots in cell order (one slot a cell) with the library's own request digits, and the tokens' place
    digits."""
    fr = frame.TokenFrame()
    b = cells.shape[0]
    req = torch.zeros(b, 9, 4, dtype=torch.long)
    for i in range(b):
        for c in range(9):
            if int(cells[i, c]) >= 0:
                r = frame.request_of(c, int(cells[i, c]), int(colors[i, c]))
                req[i, c] = torch.tensor(r[:4])
    req_d = {"place_column": torch.arange(9).expand(b, 9) % 3, "place_row": torch.arange(9).expand(b, 9) // 3,
             "member_0": req[..., 2], "member_1": req[..., 3]}
    tok_d = {"place_column": fr.column.expand(b, -1), "place_row": fr.row.expand(b, -1)}
    return tok_d, req_d, cells >= 0


def _gate(cells, colors, addresses=MEMBERS_9, U=10.0, mask=None):
    st = M.PLACE_MEMBER
    tok_d, req_d, m = _digits_of_layout(cells, colors)
    m = m if mask is None else m & mask
    lk = M.look([st.level(n) for n in PLACE], tok_d, req_d, m)
    mt = M.member_match([st.level(n) for n in MEMBER], req_d, addresses)
    return M.arguments(lk, mt, U)


def test_the_rest_gate_at_u10_is_the_integer_owner_and_serve_is_the_selection():
    g = _gen(6)
    cells, colors = scenes.draw_layouts("mix", 64, g)
    u = _gate(cells, colors)
    w, base = M.rest_weights(u)
    owner = frame.token_owner(cells, [[t] for t in range(9)])                   # [B, T], 9 = the base
    one_hot = torch.nn.functional.one_hot(owner, 10).double()
    assert float((w - one_hot[..., :9]).abs().max()) < 1e-8
    assert float((base - one_hot[..., 9]).abs().max()) < 1e-8
    free = owner == 9
    assert bool((base[free] == 1.0).all())                                     # exactly 1 where no request is
    assert bool((w[one_hot[..., :9] == 0] == 0.0).all())                       # exactly 0 off the address
    # serve against the selection the gate join performs (each token takes its owner's answer, the base's when free)
    v0 = torch.randn(64, 144, 48, generator=g)
    vs = torch.randn(9, 64, 144, 48, generator=g)
    got = M.serve(v0, vs, w, base)
    allv = torch.cat([vs, v0[None]], dim=0)                                     # [10, B, T, D]
    sel = allv[owner, torch.arange(64)[:, None], torch.arange(144)[None, :]]
    rel = float((got - sel).abs().max() / sel.abs().max())
    assert rel < 1e-7, rel


# ------------------------------------------------------------------ 7. silence off the address
def test_an_unheld_address_or_a_removed_request_leaves_its_cell_to_the_base():
    g = _gen(7)
    cells, colors = scenes.draw_layouts("composed", 16, g, count=5)
    fr = frame.TokenFrame()
    held = MEMBERS_9[:8]                                                       # nobody holds thing 8 (the star)
    cells = cells.clone()
    cells[:, 0] = 8
    colors[:, 0] = colors[:, 0].clamp_min(0)
    w, base = M.rest_weights(_gate(cells, colors, held))
    cell0 = fr.cell == 0
    assert bool((w[:, cell0] == 0.0).all()) and bool((base[:, cell0] == 1.0).all())      # exactly
    keep = torch.ones(16, 9, dtype=torch.bool)
    keep[:, 0] = False                                                         # the request of cell 0 removed
    w2, base2 = M.rest_weights(_gate(cells, colors, MEMBERS_9, mask=keep))
    assert bool((w2[:, cell0] == 0.0).all()) and bool((base2[:, cell0] == 1.0).all())    # exactly


# ------------------------------------------------------------------ 8. two owners on one token
def test_two_owners_and_one_owner_in_both_forms():
    u = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    w, base = M.rest_weights(u)
    assert round(float(w.sum()), 4) == 1.1267 and float(base[0]) < 0
    wc, bc = M.candidate_weights(u)
    assert abs(float(wc.sum() + bc.sum()) - 1.0) < 1e-12 and float(bc[0]) > 0
    for U in (1.0, 5.0, 10.0):
        one = torch.tensor([[U, 0.0, 0.0, 0.0]], dtype=torch.float64)
        assert abs(float(M.rest_weights(one)[0][0, 0]) - math.tanh(U)) < 1e-12
        assert abs(float(M.candidate_weights(one)[0][0, 0]) - math.tanh(U)) < 1e-12
        assert abs(float(M.rest_weights(one)[0][0, 0]) - frame.own_weight(U, 4, "discounted")) < 1e-12
    # on non-negative arguments the candidate form is the library's overlap form, bit for bit
    r = torch.rand(5, 7, generator=_gen(8), dtype=torch.float64) * 3
    wo, bo = frame.overlap_weights(r)
    wc, bc = M.candidate_weights(r)
    assert torch.equal(wo, wc) and torch.equal(bo, bc)


# ------------------------------------------------------------------ 9. the key binding
TAG = "multirotary|key|v1"


def test_the_key_binding_table_and_kernel():
    t1, t2 = M.phase_table(TAG, 8, 9), M.phase_table(TAG, 8, 9)
    assert t1.dtype == torch.float64 and tuple(t1.shape) == (8, 9, 256) and torch.equal(t1, t2)
    assert not torch.equal(t1, M.phase_table(TAG + "x", 8, 9))
    assert float(t1.min()) >= 0.0 and float(t1.max()) < 2 * math.pi
    names = [s.encode() for s in scenes.THINGS]
    ph = [M.name_phases(t1, n) for n in names]
    for p in ph:
        assert abs(float(M.name_kernel(p, p)) - 8.0) < 1e-12
    worst = max(float(M.name_kernel(ph[i], ph[j])) for i in range(9) for j in range(9) if i != j)
    print("largest name kernel between two of the nine names (8 pairs):", worst)
    assert worst < 8.0
    # the sum runs from the last byte: a name's phase is the hand sum
    n = b"ring"
    hand = (t1[:, 0, n[3]] + t1[:, 1, n[2]] + t1[:, 2, n[1]] + t1[:, 3, n[0]]) % (2 * math.pi)
    assert torch.equal(M.name_phases(t1, n), hand)
    with pytest.raises(ValueError):
        M.name_phases(t1, b"0123456789")


# ------------------------------------------------------------------ 10. the look read
def test_attention_weights_rebuild_the_attention_output_and_record_looks_reads_every_block():
    torch.manual_seed(10)
    attn = SelfAttention(128, 2)
    g = _gen(10)
    b, n = 2, 20
    x = torch.randn(b, n, 128, generator=g)
    rows, cols = _positions(b, n, g), _positions(b, n, g)
    dig = {k: torch.randint(0, M.PLACE_MEMBER.level(k).modulus, (b, n), generator=g) for k in M.PLACE_MEMBER.names}
    rot = M.PLACE_MEMBER.rotary(rows, cols, dig)
    km = torch.rand(b, n, generator=g) > 0.4
    for mask in (None, km):
        wts = M.attention_weights(attn, x, rot, mask)
        assert float((wts.sum(-1) - 1).abs().max()) < 1e-5
        if mask is not None:
            allowed = mask[:, None, None, :] | torch.eye(n, dtype=torch.bool)[None, None]
            assert float(wts[~allowed.expand_as(wts)].abs().max()) == 0.0
        with torch.no_grad():
            qkv = attn.qkv(x).view(b, n, 3, 2, 64).permute(2, 0, 3, 1, 4)
            y = (wts @ qkv[2]).transpose(1, 2).reshape(b, n, 128)
            want = attn(x, rot) if mask is None else attn(x, rot, mask)
            got = attn.out(y)
        assert float((got - want).abs().max()) < 1e-5
    eng = C.born("addressed_thing", 2, 0, read="rotary")
    batch = C.kind_batch("addressed_thing", 2, scenes.ScenesBed("F"), 3, _gen(11))
    z, t, _ = C.noisy(batch["x"], _gen(12))
    with M.record_looks(eng.blocks) as looks, torch.no_grad():
        C.engine_forward(eng, batch, z, t)
    assert [e["block"] for e in looks] == [0, 1, 2]
    for e in looks:
        assert tuple(e["weights"].shape) == (3, 2, 145, 145)                  # one request token + 144 image tokens


# ------------------------------------------------------------------ 11. a head with its own angles
def test_head_angles_replace_one_heads_angles_and_leave_the_others():
    st = M.PLACE_MEMBER
    g = _gen(13)
    b, n = 2, 10
    rows, cols = _positions(b, n, g), _positions(b, n, g)
    dig = {k: torch.randint(0, st.level(k).modulus, (b, n), generator=g) for k in st.names}
    hq = torch.rand(b, n, 32, generator=g) * 6
    hk = torch.rand(b, n, 32, generator=g) * 6
    q, k = torch.randn(b, 3, n, 64, generator=g), torch.randn(b, 3, n, 64, generator=g)
    q0, k0 = st.rotary(rows, cols, dig)(q, k)
    q1, k1 = st.rotary(rows, cols, dig, head_angles={1: (hq, hk)})(q, k)
    for h in (0, 2):
        assert torch.equal(q0[:, h], q1[:, h]) and torch.equal(k0[:, h], k1[:, h])
    aq, ak = torch.cat([hq, hq], -1), torch.cat([hk, hk], -1)
    want_q = q[:, 1] * aq.cos() + rotary.rotate_half(q[:, 1]) * aq.sin()
    want_k = k[:, 1] * ak.cos() + rotary.rotate_half(k[:, 1]) * ak.sin()
    assert torch.equal(q1[:, 1], want_q) and torch.equal(k1[:, 1], want_k)


# ------------------------------------------------------------------ 12. the rotary from angle tensors
def test_angle_rotary_is_the_structures_rotary_and_takes_its_angles_as_tensors():
    st = M.PLACE_MEMBER
    g = _gen(14)
    b, n = 2, 9
    rows, cols = _positions(b, n, g), _positions(b, n, g)
    kd = {k: torch.randint(0, st.level(k).modulus, (b, n), generator=g) for k in st.names}
    qd = dict(kd, member_1=(kd["member_1"] + 2) % 5)
    q, k = torch.randn(b, 2, n, 64, generator=g), torch.randn(b, 2, n, 64, generator=g)
    q0, k0 = st.rotary(rows, cols, kd, qd)(q, k)
    q1, k1 = M.angle_rotary(st.angles(rows, cols, qd), st.angles(rows, cols, kd))(q, k)
    assert torch.equal(q0, q1) and torch.equal(k0, k1)
    with pytest.raises(ValueError):
        M.angle_rotary(st.angles(rows, cols, qd), st.angles(rows, cols, kd))(q[..., :32], k[..., :32])
    with pytest.raises(ValueError):
        ang = st.angles(rows, cols, kd)
        M.angle_rotary(ang, ang, head_angles={5: (ang, ang)})(q, k)


# ------------------------------------------------------------------ 13. the kernel on float digits is not rounded
def test_the_kernel_of_float_digits_is_the_cosine_form_itself():
    lv = M.PLACE_MEMBER.level("member_0")
    a = torch.tensor([0.0, 0.5, 1.0])
    k = M.level_kernel(lv, a, torch.zeros(3))
    want = (1 + 2 * sum(torch.cos(2 * math.pi * j * a.double() / 5) for j in (1, 2))) / 5
    assert torch.equal(k, want) and 0.0 < float(k[1]) < 1.0


# ================================================================== an outside audit's claims, each closed by a test
# ------------------------------------------------------------------ 14. unsigned digits
def test_unsigned_digits_give_the_kernel_of_signed_ones():
    for lv in M.PLACE_MEMBER.levels:
        a = torch.arange(lv.modulus)[:, None].expand(-1, lv.modulus).reshape(-1)
        b = torch.arange(lv.modulus)[None, :].expand(lv.modulus, -1).reshape(-1)
        want = M.level_kernel(lv, a, b)
        got = M.level_kernel(lv, a.to(torch.uint8), b.to(torch.uint8))
        assert torch.equal(got, want), (lv.name, got.tolist(), want.tolist())


# ------------------------------------------------------------------ 15. digits out of range
def test_digits_out_of_range_raise_a_value_error():
    lv = M.PLACE_MEMBER.level("member_0")
    for bad in (torch.tensor([-1]), torch.tensor([5]), torch.tensor([-0.5]), torch.tensor([-0.99]),
                torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.tensor([float("-inf")])):
        with pytest.raises(ValueError):
            lv.check_digits(bad)
    lv.check_digits(torch.tensor([0.0, 3.5, 4.0]))                          # inside 0 .. modulus - 1: accepted


# ------------------------------------------------------------------ 16. the rotary's arguments
def test_angle_rotary_refuses_a_negative_head_and_mismatched_widths():
    g = _gen(16)
    ang = torch.rand(1, 4, 32, generator=g)
    q, k = torch.randn(1, 2, 4, 64, generator=g), torch.randn(1, 2, 4, 64, generator=g)
    with pytest.raises(ValueError):
        M.angle_rotary(ang, ang, head_angles={-1: (ang, ang)})(q, k)            # a negative head
    with pytest.raises(ValueError):
        M.angle_rotary(ang, torch.rand(1, 4, 30, generator=g))                 # query and key angles of two widths
    with pytest.raises(ValueError):
        M.angle_rotary(ang, ang, head_angles={0: (ang[..., :31], ang[..., :31])})   # a head's angles of another width
    with pytest.raises(ValueError):
        M.angle_rotary(ang, ang)(q, k[..., :60])                               # keys of another head size


# ------------------------------------------------------------------ 17. the look read with the mask by keyword
def test_record_looks_reads_the_key_mask_and_the_rotary_given_by_keyword():
    torch.manual_seed(17)
    attn = SelfAttention(128, 2)
    g = _gen(17)
    x = torch.randn(2, 12, 128, generator=g)
    rows, cols = _positions(2, 12, g), _positions(2, 12, g)
    dig = {k: torch.randint(0, M.PLACE_MEMBER.level(k).modulus, (2, 12), generator=g) for k in M.PLACE_MEMBER.names}
    rot = M.PLACE_MEMBER.rotary(rows, cols, dig)
    km = torch.rand(2, 12, generator=g) > 0.5
    want = M.attention_weights(attn, x, rot, km)

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = attn

    with M.record_looks([Holder()]) as looks, torch.no_grad():
        attn(x, rot, key_mask=km)
        attn(x, rotary=rot, key_mask=km)
    assert len(looks) == 2 and all(torch.equal(e["weights"], want) for e in looks)


# ------------------------------------------------------------------ 18. no negative zero from the kernel
def test_the_kernel_of_whole_digits_carries_no_negative_zero():
    for lv in M.PLACE_MEMBER.levels + M.PLACE_ONLY.levels:
        a = torch.arange(lv.modulus)[:, None].expand(-1, lv.modulus)
        b = torch.arange(lv.modulus)[None, :].expand(lv.modulus, -1)
        k = M.level_kernel(lv, a, b)
        assert not bool(torch.signbit(k).any()), (lv.name, int(torch.signbit(k).sum()))


# ------------------------------------------------------------------ 19. the null anchor's offset
def test_the_member_levels_null_digit_turns_by_zero_and_digit_zero_by_minus_four_steps():
    for name in ("member_0", "member_1"):
        lv = M.PLACE_MEMBER.level(name)
        assert torch.equal(lv.angles(torch.tensor(4)), torch.zeros(2, dtype=torch.float64))
        want = torch.tensor([2 * math.pi * k / 5 for k in lv.ks], dtype=torch.float64) * (0.0 - 4.0)
        assert torch.equal(lv.angles(torch.tensor(0)), want)


# ------------------------------------------------------------------ 20. the look read on an engine's masked forward
def test_record_looks_on_a_masked_engine_forward_sums_to_one_and_reads_no_masked_key():
    eng = C.born("addressed_thing", 2, 0, read="rotary")
    batch = C.kind_batch("addressed_thing", 2, scenes.ScenesBed("F"), 3, _gen(20))
    z, t, _ = C.noisy(batch["x"], _gen(21))
    seen = []
    hooks = [blk.attn.register_forward_pre_hook(lambda m, a: seen.append((a[0].detach(), a[1], a[2])))
             for blk in eng.blocks]
    try:
        with M.record_looks(eng.blocks) as looks, torch.no_grad():
            C.engine_forward(eng, batch, z, t)
    finally:
        for h in hooks:
            h.remove()
    assert len(looks) == len(seen) == 3
    for e, (x, rot, km) in zip(looks, seen):
        w = e["weights"]
        assert km is not None and not bool(km.all())
        allowed = km[:, None, None, :] | torch.eye(km.shape[1], dtype=torch.bool)[None, None]
        assert float((w.sum(-1) - 1).abs().max()) < 1e-5
        assert float(w[~allowed.expand_as(w)].abs().max()) == 0.0
        assert torch.equal(w, M.attention_weights(eng.blocks[e["block"]].attn, x, rot, km))


# ------------------------------------------------------------------ 21. the edges the tests did not reach
def test_refusals_and_edges_kernel_name_and_serve():
    with pytest.raises(ValueError):
        M.level_kernel(M.Level("even", (0, 1), (1, 2), 4), torch.tensor(0), torch.tensor(1))      # an even modulus
    with pytest.raises(ValueError):
        M.level_kernel(M.Level("gap", (0, 1), (1, 3), 7), torch.tensor(0), torch.tensor(1))       # multiples 1, 3
    with pytest.raises(ValueError):
        M.name_phases(M.phase_table("edges", 8, 9), b"")
    g = _gen(21)
    v0 = torch.randn(2, 5, 3, generator=g)
    base = torch.rand(2, 5, generator=g, dtype=torch.float64)
    none = M.serve(v0, torch.zeros(0, 2, 5, 3), torch.zeros(2, 5, 0, dtype=torch.float64), base)
    assert torch.equal(none, base.float()[..., None] * v0)                    # no member: the base alone
    vs = torch.randn(3, 2, 5, 3, generator=g).bfloat16()
    w = torch.rand(2, 5, 3, generator=g, dtype=torch.float64)
    got = M.serve(v0.bfloat16(), vs, w, base)
    want = base.bfloat16()[..., None] * v0.bfloat16()
    for i in range(3):
        want = want + w[..., i].bfloat16()[..., None] * vs[i]
    assert got.dtype == torch.bfloat16 and torch.equal(got, want)


# ------------------------------------------------------------------ 22. the candidate form on negative arguments
def test_the_candidate_form_refuses_arguments_below_zero_beyond_rounding():
    with pytest.raises(ValueError):
        M.candidate_weights(torch.tensor([[-10.0, 9.9]], dtype=torch.float64))
    w, base = M.candidate_weights(torch.tensor([[-1e-12, 3.0]], dtype=torch.float64))     # a rounding below zero
    assert abs(float(w.sum() + base.sum()) - 1.0) < 1e-12


# ------------------------------------------------------------------ 23. the look read in half precision
def test_attention_weights_in_bfloat16_take_their_logits_in_float32():
    torch.manual_seed(23)
    attn = SelfAttention(128, 2).bfloat16()
    g = _gen(23)
    x = (torch.randn(1, 40, 128, generator=g) * 3).bfloat16()
    rows, cols = _positions(1, 40, g), _positions(1, 40, g)
    dig = {k: torch.randint(0, M.PLACE_MEMBER.level(k).modulus, (1, 40), generator=g) for k in M.PLACE_MEMBER.names}
    rot = M.PLACE_MEMBER.rotary(rows, cols, dig)
    with torch.no_grad():
        qkv = attn.qkv(x).view(1, 40, 3, 2, 64).permute(2, 0, 3, 1, 4)
        q, k = rot(qkv[0], qkv[1])
        want = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / 8.0, dim=-1)
        got = M.attention_weights(attn, x, rot)
    diff = float((got - want).abs().max())
    print("bfloat16 look read against float32 logits:", diff)
    assert diff < 1e-6


# ------------------------------------------------------------------ 24. the minor guards
def test_look_takes_any_mask_as_a_set_serve_checks_shapes_and_a_name_is_bytes():
    st = M.PLACE_MEMBER
    lv = [st.level("place_column"), st.level("place_row")]
    g = _gen(24)
    tok = {n: torch.randint(0, 3, (2, 7), generator=g) for n in ("place_column", "place_row")}
    req = {n: torch.randint(0, 3, (2, 9), generator=g) for n in ("place_column", "place_row")}
    m = torch.rand(2, 9, generator=g) > 0.5
    assert torch.equal(M.look(lv, tok, req, m.long() * 2), M.look(lv, tok, req, m))
    with pytest.raises(ValueError):
        M.serve(torch.zeros(2, 5, 3), torch.zeros(3, 1, 5, 3), torch.zeros(2, 5, 3, dtype=torch.float64),
                torch.zeros(2, 5, dtype=torch.float64))
    with pytest.raises(TypeError):
        M.name_phases(M.phase_table("edges", 8, 9), 5)


# ------------------------------------------------------------------ 25. the faster forms are the present forms, bit for bit
def _ref_band_rates(n, lo, hi):
    lam = float(lo) * (float(hi) / float(lo)) ** (torch.arange(n, dtype=torch.float64) / n)
    return 2 * math.pi / lam


def _ref_level_angles(lv, digits):
    rates = torch.tensor([2 * math.pi * k / lv.modulus for k in lv.ks], dtype=torch.float64)
    return rates * (digits.double() - lv.offset)[..., None]


def _ref_angles(st, rows, cols, digits):
    """The structure's angles as written before the audit."""
    r, c = rows.double(), cols.double()
    out = torch.zeros(r.shape + (st.n_pairs,), dtype=torch.float64)
    lo, hi = st.band
    if st.row:
        out[..., list(st.row)] = r[..., None] * _ref_band_rates(len(st.row), lo, hi)
    if st.column:
        out[..., list(st.column)] = c[..., None] * _ref_band_rates(len(st.column), lo, hi)
    for lv in st.levels:
        out[..., list(lv.pairs)] = _ref_level_angles(lv, digits[lv.name])
    return out.float()


def _ref_tables(ang):
    a = torch.cat([ang, ang], dim=-1).unsqueeze(-3)
    return a.cos(), a.sin()


def _ref_angle_rotary(query_angles, key_angles, head_angles=None):
    ck, sk = _ref_tables(key_angles.float())
    cq, sq = (ck, sk) if query_angles is key_angles else _ref_tables(query_angles.float())
    heads = {int(h): (_ref_tables(hq.float()), _ref_tables(hk.float())) for h, (hq, hk) in (head_angles or {}).items()}

    def _rot(q, k):
        qc, qs, kc, ks = cq, sq, ck, sk
        if heads:
            n_heads = q.shape[-3]
            index = torch.arange(n_heads).view(n_heads, 1, 1)
            for h, ((hqc, hqs), (hkc, hks)) in heads.items():
                on = index == h
                qc, qs = torch.where(on, hqc, qc), torch.where(on, hqs, qs)
                kc, ks = torch.where(on, hkc, kc), torch.where(on, hks, ks)
        return (q * qc.to(q.dtype) + rotary.rotate_half(q) * qs.to(q.dtype),
                k * kc.to(k.dtype) + rotary.rotate_half(k) * ks.to(k.dtype))

    return _rot


def _ref_rest_weights(u):
    u = u.double()
    den = torch.ones(u.shape[:-1], dtype=torch.float64)
    for j in range(u.shape[-1]):
        den = den + 2.0 * torch.sinh(u[..., j] / 2.0) ** 2
    w = torch.sinh(u) / den[..., None]
    total = torch.zeros_like(den)
    for j in range(u.shape[-1]):
        total = total + w[..., j]
    return w, 1.0 - total


def _ref_serve(base_answer, answers, w, base):
    v = base.to(base_answer.dtype)[..., None] * base_answer
    for i in range(answers.shape[0]):
        v = v + w[..., i].to(answers.dtype)[..., None] * answers[i]
    return v


def test_the_faster_forms_equal_the_present_forms_bit_for_bit():
    g = _gen(25)
    b, n = 3, 40
    rows, cols = _positions(b, n, g), _positions(b, n, g)
    for st in (M.PLACE_ONLY, M.PLACE_MEMBER):
        kd = {k: torch.randint(0, st.level(k).modulus, (b, n), generator=g) for k in st.names}
        qd = dict(kd)
        qd[st.names[-1]] = (kd[st.names[-1]] + 1) % st.level(st.names[-1]).modulus
        ak, aq = st.angles(rows, cols, kd), st.angles(rows, cols, qd)
        assert torch.equal(ak, _ref_angles(st, rows, cols, kd)) and torch.equal(aq, _ref_angles(st, rows, cols, qd))
        assert torch.equal(st.angles(rows, cols, kd, check=False), ak)
        hq, hk = torch.rand(b, n, 32, generator=g) * 6, torch.rand(b, n, 32, generator=g) * 6
        for heads, dtype in ((None, torch.float32), ({1: (hq, hk)}, torch.float32), ({0: (hq, hk)}, torch.bfloat16)):
            q = torch.randn(b, 3, n, 64, generator=g).to(dtype)
            k = torch.randn(b, 3, n, 64, generator=g).to(dtype)
            new, old = st.rotary(rows, cols, kd, qd, heads), _ref_angle_rotary(aq, ak, heads)
            for _ in range(3):                                                   # three blocks: the same tables
                q1, k1 = new(q, k)
                q2, k2 = old(q, k)
                assert torch.equal(q1, q2) and torch.equal(k1, k2), (st.names, heads, dtype)
    u = torch.rand(4, 7, 9, generator=g, dtype=torch.float64) * 4 * (torch.rand(4, 7, 9, generator=g) > 0.6)
    w, base = M.rest_weights(u)
    w0, b0 = _ref_rest_weights(u)
    assert torch.equal(w, w0) and torch.equal(base, b0)
    v0, vs = torch.randn(4, 7, 5, generator=g), torch.randn(9, 4, 7, 5, generator=g)
    assert torch.equal(M.serve(v0, vs, w, base), _ref_serve(v0, vs, w, base))
    lvs = [M.PLACE_MEMBER.level("member_0"), M.PLACE_MEMBER.level("member_1")]
    rd = {"member_0": torch.randint(0, 5, (4, 9), generator=g), "member_1": torch.randint(0, 5, (4, 9), generator=g)}
    adr = torch.randint(0, 5, (6, 2), generator=g)
    assert torch.equal(M.member_match(lvs, rd, adr, check=False), M.member_match(lvs, rd, adr))

