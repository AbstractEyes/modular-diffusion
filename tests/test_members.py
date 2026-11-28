"""Members of the addressed collective (CPU, small batches): their birth and the bridge to the stored rotary engine,
padding slots, the training rows, the streams, the told head, the request content, the gate against the integer owner,
the address reads, request lists that point, and a forward of fixed shape under vmap."""
import copy
import math

import pytest
import torch

from geolip.modular_diffusion import collective as C
from geolip.modular_diffusion import frame, scenes
from geolip.modular_diffusion import members as MB
from geolip.modular_diffusion import multirotary as M
from geolip.modular_diffusion.spectral import patchify, unpatchify

BED = scenes.ScenesBed("F")
CELL_OF = frame.TokenFrame().cell                                            # [144]


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _wake(engine, seed):
    """A nonzero head and modulation, so that an engine's answers carry its inputs."""
    g = _gen(seed)
    with torch.no_grad():
        for p in (engine.head.proj.weight, engine.head.proj.bias):
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        for blk in engine.blocks:
            w = blk.modulation[1].weight
            w.copy_(torch.randn(w.shape, generator=g) * 0.02)
    return engine.eval()


def _engine(variant, cls, seed, depth=1):
    return _wake(MB.born(variant, cls, seed, depth=depth), 1000 + 17 * seed + (cls if cls is not None else 9))


def _collective(variant, seed=0, depth=1, swap=None, classes=range(9)):
    things = {k: _engine(variant, k, seed + (50 if swap == k else 0), depth) for k in classes}
    return MB.MemberCollective(things, _engine(variant, None, seed, depth), variant)


def _place_only_angles(b):
    """[b, 153, 32]: the anchored rotary's angles (``multirotary.PLACE_ONLY``) for 9 request slots at their cells'
    centers with their cells' digits, then the 144 image tokens at (iy, ix) with their cells' digits."""
    s = torch.arange(9)
    idx = torch.arange(144)
    rows = torch.cat([(s // 3 * 4).double() + 1.5, (idx // 12).double()])
    cols = torch.cat([(s % 3 * 4).double() + 1.5, (idx % 12).double()])
    dcol = torch.cat([s % 3, (idx % 12) // 4])
    drow = torch.cat([s // 3, (idx // 12) // 4])
    ang = M.PLACE_ONLY.angles(rows[None], cols[None], {"place_column": dcol[None], "place_row": drow[None]})
    return ang.expand(b, -1, -1).contiguous()


def _pointer_layout():
    """One scene: cell 0 a crescent (2) that points at the square (3, address (1, 0)) of cell 4; a star (8, address
    (2, 2)) in cell 8 also stands once; a diamond (5) in cell 2; the rest empty."""
    cells = torch.full((1, 9), -1, dtype=torch.long)
    colors = torch.full((1, 9), -1, dtype=torch.long)
    for c, (thing, color) in {0: (2, 1), 2: (5, 6), 4: (3, 1), 8: (8, 4)}.items():
        cells[0, c], colors[0, c] = thing, color
    points = MB.no_points(cells)
    points[0, 0] = 3
    return cells, colors, points


# ------------------------------------------------------------------ 1. birth
def test_1_a_t1_engine_is_born_equal_to_the_stored_rotary_engine():
    for cls, kind in ((3, "addressed_thing"), (None, "addressed_base")):
        a = MB.born(MB.T1, cls, 7)
        b = C.born(kind, cls, 7, read="rotary")
        sa, sb = a.state_dict(), b.state_dict()
        assert list(sa) == list(sb)
        for k in sa:
            assert torch.equal(sa[k], sb[k]), k


# ------------------------------------------------------------------ 2. the bridge at step 1
def test_2_the_bridge_a_t1_engine_on_place_only_angles_is_the_stored_rotary_engine():
    """EQUAL: the weights (loaded key for key), the request content (the 12 numbers of frame.content_code on the own
    slot), the request token's place in the rotary (its cell's center, its cell's digits on the anchored rotary), the
    image tokens' rotary and the key mask over the image tokens. DIFFERENT: the new sequence carries eight more slots,
    all invisible keys; a key masked out has softmax weight exactly 0, so nothing reads them, and the head reads the
    image tokens only. What remains is the order of float32 sums over a longer sequence."""
    stored = C.born("addressed_thing", 5, 3, read="rotary")
    g = _gen(20)
    with torch.no_grad():
        stored.head.proj.weight.copy_(torch.randn(stored.head.proj.weight.shape, generator=g) * 0.05)
        stored.head.proj.bias.copy_(torch.randn(stored.head.proj.bias.shape, generator=g) * 0.05)
    eng = MB.MemberEngine(MB.T1)
    eng.load_state_dict(stored.state_dict(), strict=True)
    stored.eval(), eng.eval()
    b = C.kind_batch("addressed_thing", 5, BED, 6, _gen(21))
    z, t, _ = C.noisy(b["x"], _gen(22))
    prep = MB.prepare(MB.T1, b["cells"], b["colors"], None, b["told"], b["key_mask"])
    ang = _place_only_angles(6)
    with torch.no_grad():
        got = eng.forward_tokens(z, t, prep["content"], ang, ang, prep["key_mask"])
        want = stored(z, t, b["cells"], b["colors"], b["told"], b["key_mask"])
    diff = float((got - want).abs().max())
    print("bridge: largest |new - stored| =", diff, "; largest |stored| =", float(want.abs().max()))
    assert diff < 1e-5


# ------------------------------------------------------------------ 3. padding slots
def test_3_a_padding_slot_changes_no_number_of_the_output():
    eng = _engine(MB.T2, 4, 0, depth=2)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(30)), torch.tensor([0.4])
    prep = MB.prepare(MB.T2, cells, colors, points)
    pad = [c for c in range(9) if int(cells[0, c]) < 0]
    other = dict(prep)
    g = _gen(31)
    content = prep["content"].clone()
    content[0, pad] = torch.randn(len(pad), 18, generator=g) * 3
    qa, ka = prep["query_angles"].clone(), prep["key_angles"].clone()
    qa[0, pad] = torch.rand(len(pad), 32, generator=g) * 6
    ka[0, pad] = torch.rand(len(pad), 32, generator=g) * 6
    other.update(content=content, query_angles=qa, key_angles=ka)
    with torch.no_grad():
        a = eng.forward_tokens(z, t, **prep)
        b = eng.forward_tokens(z, t, **other)
    assert torch.equal(a, b)                                                   # bit for bit


# ------------------------------------------------------------------ 4. the rows of T1 are the stored rows
def test_4_member_batch_of_t1_is_kind_batch_for_the_same_generator():
    for cls, kind in ((2, "addressed_thing"), (None, "addressed_base")):
        mb = MB.member_batch(MB.T1, cls, BED, 20, _gen(40))
        kb = C.kind_batch(kind, cls, BED, 20, _gen(40))
        assert (torch.equal(mb["x"], kb["x"]) and torch.equal(mb["cells"], kb["cells"])
                and torch.equal(mb["colors"], kb["colors"]) and torch.equal(mb["visible"], kb["key_mask"])
                and torch.equal(mb["loss_mask"], kb["loss_mask"]) and bool((mb["points"] == -1).all())), kind


# ------------------------------------------------------------------ 5. the rows of the told-all variants
def test_5_member_batch_told_all_keeps_its_rules():
    cls, ex = 4, 7
    held = {(4, 0), (4, 1), (4, 2)}
    n = 600
    mb = MB.member_batch(MB.T2, cls, BED, n, _gen(50), exclude=ex, held_pairs=held)
    cells, colors, points = mb["cells"], mb["colors"], mb["points"]
    assert bool((cells == cls).any(1).all()) and not bool((cells == ex).any())
    assert torch.equal(mb["loss_mask"], (cells == cls)[:, CELL_OF])
    assert torch.equal(mb["visible"], ((cells == cls) | (cells < 0))[:, CELL_OF])
    MB.check_layout(cells, colors, points)                                     # every layout valid
    can = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        row = cells[i].tolist()
        cand = [c for c in range(9) if row[c] >= 0 and row[c] != cls and row.count(row[c]) == 1
                and (cls, row[c]) not in held]
        can[i] = bool(cand)
        _check_pointers(cells[i], colors[i], points[i], held)
        count = int((points[i] >= 0).sum())
        assert count == int(mb["pointing"][i]) + int(mb["other_pointing"][i])
    assert torch.equal(can, mb["can_point"]) and not bool((mb["pointing"] & ~can).any())
    frac = float(mb["pointing"][can].float().mean())
    m = int(can.sum())
    print("rows with a candidate", m, "of", n, "; pointing among them", frac)
    assert abs(frac - 0.5) < 4 * math.sqrt(0.25 / m)
    oc = mb["other_can_point"]
    assert not bool((mb["other_pointing"] & ~oc).any())
    ofrac, om = float(mb["other_pointing"][oc].float().mean()), int(oc.sum())
    print("rows with a valid other pair", om, "of", n, "; another request points in", ofrac)
    assert abs(ofrac - 0.5) < 4 * math.sqrt(0.25 / om)
    base = MB.member_batch(MB.T2, None, BED, 300, _gen(51), exclude=ex, held_pairs=held)
    MB.check_layout(base["cells"], base["colors"], base["points"])
    assert bool((base["cells"] < 0).any(1).all()) and not bool(base["pointing"].any())
    assert not bool((base["cells"] == ex).any())
    assert torch.equal(base["loss_mask"], (base["cells"] < 0)[:, CELL_OF])
    for i in range(300):
        _check_pointers(base["cells"][i], base["colors"][i], base["points"][i], held)
        assert int((base["points"][i] >= 0).sum()) == int(base["other_pointing"][i])
    boc = base["other_can_point"]
    bfrac, bm = float(base["other_pointing"][boc].float().mean()), int(boc.sum())
    print("base: rows with a valid pair", bm, "of 300 ; pointing in", bfrac)
    assert abs(bfrac - 0.5) < 4 * math.sqrt(0.25 / bm)
    again = MB.member_batch(MB.T2, cls, BED, n, _gen(50), exclude=ex, held_pairs=held)
    assert all(torch.equal(mb[k], again[k]) for k in ("x", "cells", "colors", "points", "pointing", "other_pointing"))
    assert MB.member_batch(MB.T3, cls, BED, 4, _gen(52))["visible"] is None


def _check_pointers(cells_row, colors_row, points_row, held):
    """Every pointer of one row: the asked class differs from the asking one and stands once, the asked request states
    its color (an asked request never points), the pair is not held, the asking request is drawn in the asked color."""
    row = cells_row.tolist()
    for a in range(9):
        if int(points_row[a]) < 0:
            continue
        q = row.index(int(points_row[a]))
        assert row[q] != row[a] and row.count(row[q]) == 1 and int(points_row[q]) == -1
        assert (row[a], row[q]) not in held and int(colors_row[a]) == int(colors_row[q])


def test_5c_member_batch_returns_its_layout_on_the_cpu_equal_to_its_device_tensors():
    for v, cls in ((MB.T1, 3), (MB.T1, None), (MB.T2, 3), (MB.T2, None), (MB.T3, 5)):
        mb = MB.member_batch(v, cls, BED, 20, _gen(54))
        for key in ("cells", "colors", "points", "told", "visible"):
            dev, cpu = mb[key], mb[key + "_cpu"]
            if dev is None:
                assert cpu is None, (v.name, key)
                continue
            assert cpu.device.type == "cpu" and torch.equal(cpu, dev.cpu()), (v.name, cls, key)


def test_5b_a_things_row_can_hold_two_pointers_at_once_and_stays_valid():
    mb = MB.member_batch(MB.T2, 4, BED, 600, _gen(53))
    both = mb["pointing"] & mb["other_pointing"]
    rows = torch.nonzero(both, as_tuple=True)[0]
    print("rows with two pointers:", rows.numel(), "of 600")
    assert rows.numel() > 0 and bool(((mb["points"][rows] >= 0).sum(1) == 2).all())
    MB.prepare(MB.T2, mb["cells"][rows], mb["colors"][rows], mb["points"][rows])     # check_layout inside: no raise


# ------------------------------------------------------------------ 6. the streams
def test_6_the_streams_keys_queries_and_their_independence_of_the_engine():
    pl = MB.pointer_lists(4, 40, 600)
    rows = torch.nonzero(pl.pointed, as_tuple=True)[0][:12]
    cells, colors, points = pl.cells[rows], pl.colors[rows], pl.points[rows]
    st = MB.streams(MB.T2, cells, points)
    kd, qd = st["key_digits"], st["query_digits"]
    seq_cell = torch.cat([torch.arange(9), CELL_OF])
    for i in range(cells.shape[0]):
        for tok in range(153):
            c = int(seq_cell[tok])
            want_k = MB.address_of(int(cells[i, c]))
            asked = int(points[i, c])
            want_q = MB.address_of(asked) if asked >= 0 else want_k
            assert (int(kd["member_0"][i, tok]), int(kd["member_1"][i, tok])) == want_k
            assert (int(qd["member_0"][i, tok]), int(qd["member_1"][i, tok])) == want_q
    for v in (MB.C1, MB.C2, MB.T2K):
        s2 = MB.streams(v, cells, points)
        for d in (s2["key_digits"], s2["query_digits"]):
            assert bool((d["member_0"] == 4).all()) and bool((d["member_1"] == 4).all()), v.name
    present = sorted({int(x) for x in cells[0][cells[0] >= 0].tolist()})
    assert len(present) >= 2
    col = _collective(MB.T2, classes=present[:2])
    a, b = (col.engine_inputs(k, cells[:1], colors[:1], points[:1]) for k in col.classes)
    assert all(torch.equal(a[k], b[k]) for k in ("content", "query_angles", "key_angles"))


# ------------------------------------------------------------------ 7. the told head of T2K
def test_7_the_told_head_turns_only_the_name_pairs_and_a_pointer_asks_by_the_asked_name():
    cells, colors, points = _pointer_layout()
    prep = MB.prepare(MB.T2K, cells, colors, points)
    tq, tk = prep["told_angles"]
    names = list(MB.NAME_PAIRS)
    rest = [p for p in range(32) if p not in names]
    for told, plain in ((tq, prep["query_angles"]), (tk, prep["key_angles"])):
        assert torch.equal(told[..., rest], plain[..., rest])
        assert not torch.equal(told[..., names], plain[..., names])
    assert torch.equal(tq[0, 0, names], tk[0, 4, names])                       # the pointer asks the square's name
    for other in (2, 8):
        assert not torch.equal(tq[0, 0, names], tk[0, other, names])
    for tok in range(144):                                                     # the pointing cell's image tokens ask too
        if int(CELL_OF[tok]) == 0:
            assert torch.equal(tq[0, 9 + tok, names], tk[0, 4, names])


# ------------------------------------------------------------------ 8. the request content
def test_8_the_content_no_color_on_a_pointer_identity_only_where_its_variant_puts_it():
    cells, colors, points = _pointer_layout()
    u = frame.digit_code(3).float()
    for v in MB.VARIANTS.values():
        told = cells >= 0                                                      # every request, T1 included
        c = MB.request_content(v, cells, colors, points, told)
        assert float(c[0, 0, 6:12].abs().max()) == 0.0, v.name                # no color on the pointing request
        assert float(c[0, 4, 6:12].abs().max()) > 0.0, v.name                 # a stated color is there
        if v.identity in ("address", "name"):
            assert float(c[..., :6].abs().max()) == 0.0 and float(c[..., 12:].abs().max()) == 0.0, v.name
        if v.identity == "content":
            asked = torch.cat([u[3 // 3], u[3 % 3]])
            assert torch.equal(c[0, 0, 12:], asked)
            assert float(c[0, 1:, 12:].abs().max()) == 0.0                    # only the pointing request
        assert float(c[0, cells[0] < 0].abs().max()) == 0.0, v.name           # padding slots: zero


# ------------------------------------------------------------------ 9. the gate is the integer owner
def test_9_the_weights_are_the_one_hot_owner_and_serve_is_the_selection():
    col = _collective(MB.T2)
    cells, _ = scenes.draw_layouts("mix", 64, _gen(90))
    w, base = col.weights(cells)
    owner = frame.token_owner(cells, [[k] for k in range(9)])
    one = torch.nn.functional.one_hot(owner, 10).double()
    assert torch.equal(w[one[..., :9] == 0], torch.zeros_like(w[one[..., :9] == 0]))
    assert float((w - one[..., :9]).abs().max()) < 1e-8 and float((base - one[..., 9]).abs().max()) < 1e-8
    g = _gen(91)
    v0, vs = torch.randn(64, 144, 48, generator=g), torch.randn(9, 64, 144, 48, generator=g)
    got = M.serve(v0, vs, w, base)
    allv = torch.cat([vs, v0[None]])
    sel = allv[owner, torch.arange(64)[:, None], torch.arange(144)[None]]
    rel = float((got - sel).abs().max() / sel.abs().max())
    print("serve against the selection, relative:", rel)
    assert rel < 1e-7


# ------------------------------------------------------------------ 10. the swap, the two together
def test_10_address_swap_moves_the_gate_on_its_cell_only_and_the_unturned_control_keeps_the_gate():
    col = _collective(MB.T3)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(100)), torch.tensor([0.5])
    cell, old, new = 2, 5, 6                                                   # the diamond becomes a plus
    r = MB.address_swap(col, z, t, cells, colors, points, 0, cell, new)
    i_old, i_new = col.classes.index(old), col.classes.index(new)
    wb, wa = r["before"]["gate"]["w"], r["after"]["gate"]["w"]
    th = math.tanh(10.0)                                                       # one owner at u = U: tanh(U), to rounding
    assert float((wb[:, i_old] - th).abs().max()) < 1e-12 and bool((wb[:, i_new] == 0.0).all())
    assert float((wa[:, i_new] - th).abs().max()) < 1e-12 and bool((wa[:, i_old] == 0.0).all())
    off = CELL_OF != cell
    assert torch.equal(r["before"]["w_all"][off], r["after"]["w_all"][off])
    print("looks of the old engine before / after:", r["before"]["looks_old"].tolist(), r["after"]["looks_old"].tolist())
    print("looks of the new engine before / after:", r["before"]["looks_new"].tolist(), r["after"]["looks_new"].tolist())
    print("other members' relative change:", r["other_relative_change"])
    u = MB.address_swap(col, z, t, cells, colors, points, 0, cell, new, turn=False)
    for tag in ("before", "after"):
        assert torch.equal(u[tag]["w_all"], r[tag]["w_all"])
    n = MB.address_swap(col, z, t, cells, colors, points, 0, cell, new, gate="neutral")
    assert tuple(n["before"]["answers"].shape) == (9, 16, 48) and "gate" not in n["before"]


@pytest.mark.parametrize("variant", [MB.T1, MB.T2, MB.T3])
def test_10c_the_neutral_swap_runs_the_old_and_the_new_engine_on_the_edited_cell_on_both_sides(variant):
    col = _collective(variant)
    cells, colors, points = _pointer_layout()
    if variant.told == "own":
        points = MB.no_points(cells)
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(101)), torch.tensor([0.5])
    cell, old, new = 2, 5, 6                       # the diamond stands once; no plus stands before the swap
    assert int((cells == old).sum()) == 1 and int((cells == new).sum()) == 0
    n = MB.address_swap(col, z, t, cells, colors, points, 0, cell, new, gate="neutral")
    after = cells.clone()
    after[0, cell] = new
    toks = CELL_OF == cell
    for tag, layout in (("before", cells), ("after", after)):
        for cls in (old, new):
            i = col.classes.index(cls)
            prep = col.engine_inputs(cls, layout, colors, points, cell=torch.tensor([cell]))
            own = patchify(col.things[cls].forward_tokens(z, t, **prep), 4)[0, toks]
            got = n[tag]["answers"][i]
            assert bool((got != 0).any()), (variant.name, tag, cls)
            assert torch.equal(got, own), (variant.name, tag, cls)
    # the other members' answers are those of the collective's own parts on each side
    for tag, layout in (("before", cells), ("after", after)):
        answers, _ = col.part_answers(z, t, layout, colors, points)
        for i, cls in enumerate(col.classes):
            if cls not in (old, new):
                assert torch.equal(n[tag]["answers"][i], answers[i, 0, toks]), (variant.name, tag, cls)


def test_10b_address_swap_refuses_a_scene_whose_pointer_asks_the_old_or_the_new_class_before_any_forward():
    col = _collective(MB.T3)
    cells, colors, points = _pointer_layout()                                  # cell 0 asks the square (3) of cell 4
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(101)), torch.tensor([0.5])
    calls = []
    engines = list(col.things.values()) + [col.base]
    hooks = [e.blocks[0].register_forward_pre_hook(lambda m, a: calls.append(1)) for e in engines]   # blocks run
    try:
        for cell, new in ((4, 6), (2, 3)):                                     # asks the old class; asks the new one
            with pytest.raises(ValueError, match="would break a pointer"):
                MB.address_swap(col, z, t, cells, colors, points, 0, cell, new)
    finally:
        for h in hooks:
            h.remove()
    assert calls == []


# ------------------------------------------------------------------ 11. FIND at one site
def test_11_find_read_differs_only_on_the_member_groups_and_unit_vectors_score_the_kernel():
    eng = _engine(MB.T2, 2, 0, depth=2)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(110)), torch.tensor([0.3])
    r = MB.find_read(eng, MB.T2, z, t, cells, colors, points, 0, 0, 3, 8, block=1)
    member = [MB.GROUPS.index("member_0"), MB.GROUPS.index("member_1")]
    others = [i for i in range(len(MB.GROUPS)) if i not in member]
    assert torch.equal(r["scores_a"][..., others], r["scores_b"][..., others])
    assert not torch.equal(r["scores_a"][..., member], r["scores_b"][..., member])
    print("softmax weights on (square, star), asking the square:", r["weights_a"].tolist(), "the star:",
          r["weights_b"].tolist(), "; answers differ by", float((r["answer_a"] - r["answer_b"]).abs().max()))
    # q = k = 1 on the first number of each pair of one member level, 0 elsewhere. A turned pair scores
    # cos(angle_q - angle_k) = cos(w_k (a - b)); over the level's two pairs (k = 1, 2, modulus 5) the sum is 2 when the
    # query's digit a equals the key's digit b and -1/2 otherwise. The pointing slot (0) asks the square, (1, 0); the
    # square's slot (4) holds (1, 0), the star's slot (8) holds (2, 2): on either level, 2 for the square and -1/2 for
    # the star. The groups are reported divided by sqrt(64) = 8.
    prep = MB.prepare(MB.T2, cells, colors, points)
    for name in ("member_0", "member_1"):
        v = torch.zeros(1, 2, 153, 64)
        v[..., list(M.PLACE_MEMBER.level(name).pairs)] = 1.0
        q, k = MB.block_rotary(prep, 1)(v, v)
        sc = MB.pair_group_scores(q, k, 0, (4, 8))[0, :, :, MB.GROUPS.index(name)] * 8
        assert float((sc[:, 0] - 2.0).abs().max()) < 1e-5 and float((sc[:, 1] + 0.5).abs().max()) < 1e-5, name


def test_11b_find_read_vary_content_changes_only_the_pointing_slots_asked_identity():
    eng = _engine(MB.C2, 2, 0, depth=2)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(111)), torch.tensor([0.3])
    r = MB.find_read(eng, MB.C2, z, t, cells, colors, points, 0, 0, 3, 8, block=1, vary="content")
    a, b = r["inputs_a"], r["inputs_b"]
    assert torch.equal(a["query_angles"], b["query_angles"]) and torch.equal(a["key_angles"], b["key_angles"])
    diff = a["content"] != b["content"]
    where = torch.nonzero(diff)
    print("content entries that differ (scene, slot, number):", where.tolist())
    assert bool(diff.any()) and bool((where[:, 1] == 0).all()) and bool((where[:, 2] >= 12).all())
    with pytest.raises(ValueError):
        MB.find_read(eng, MB.T2, z, t, cells, colors, points, 0, 0, 3, 8, block=1, vary="content")


# ------------------------------------------------------------------ 12. SERVE, structural
def test_12_serve_read_keeps_the_given_answers_one_denominator_and_the_reverse_edit_restores_every_number():
    col = _collective(MB.T2)
    cells, _, _ = _pointer_layout()
    g = _gen(120)
    answers, base_answer = torch.randn(9, 1, 144, 48, generator=g), torch.randn(1, 144, 48, generator=g)
    r = MB.serve_read(col, answers, base_answer, cells, 0, 2, 6)
    toks = torch.nonzero(CELL_OF == 2, as_tuple=True)[0]
    assert torch.equal(r["answers"], answers[:, 0, toks]) and torch.equal(r["base_answer"], base_answer[0, toks])
    assert torch.equal(r["before"]["denominator"], r["after"]["denominator"])
    assert all(torch.equal(r["before"][k], r["reverse"][k]) for k in r["before"])


# ------------------------------------------------------------------ 13. one engine exchanged
@pytest.mark.parametrize("variant", [MB.T1, MB.T2, MB.T3])
def test_13_exchanging_one_engine_changes_no_other_members_contribution(variant):
    pl = MB.pointer_lists(4, 6, 130)
    points = pl.points if variant.told == "all" else MB.no_points(pl.cells)
    colors = pl.colors
    k = int(pl.cells[0][pl.cells[0] >= 0][0])
    z, t = torch.randn(6, 3, 48, 48, generator=_gen(131)), torch.full((6,), 0.45)
    col_a = _collective(variant)
    twin = {c: (_engine(variant, k, 50) if c == k else e) for c, e in col_a.things.items()}   # one engine replaced
    col_b = MB.MemberCollective(twin, col_a.base, variant)
    r = MB.exchange_read(col_a, col_b, z, t, pl.cells, colors, points)
    print(variant.name, "exchanged", r["exchanged"], "others", r["others"])
    assert r["exchanged"] == [k] and all(v == 0.0 for v in r["others"].values())


# ------------------------------------------------------------------ 14. silence off the address
def test_14_an_unheld_address_or_a_removed_request_leaves_its_cell_to_the_base():
    col = _collective(MB.T2)
    cells, _, _ = _pointer_layout()
    for r in (MB.address_unheld(col, cells, 0, 2), MB.address_missing(col, cells, 0, 2)):
        assert bool((r["w"] == 0.0).all()) and bool((r["base"] == 1.0).all())


# ------------------------------------------------------------------ 15. request lists that point
def test_15_pointer_lists_obey_the_rules_and_the_held_pairs():
    held = {(a, q) for a in range(9) for q in range(9) if (a + 2 * q) % 4 == 0 and a != q}
    for only in (False, True):
        pl = MB.pointer_lists(5, 200, 150, held_pairs=held, only_held=only)
        base = scenes.request_lists(5, 200, 150)
        assert torch.equal(pl.cells, base.cells)
        for i in range(200):
            row = pl.cells[i].tolist()
            ps = [c for c in range(9) if int(pl.points[i, c]) >= 0]
            assert len(ps) == (1 if bool(pl.pointed[i]) else 0)
            if not ps:
                assert torch.equal(pl.colors[i], base.colors[i])
                continue
            (a,) = ps
            q = row.index(int(pl.points[i, a]))
            assert row[q] != row[a] and row.count(row[q]) == 1 and int(pl.points[i, q]) == -1
            assert ((row[a], row[q]) in held) == only
            assert int(pl.colors[i, a]) == int(base.colors[i, q])
        print("only_held", only, ": lists pointed", int(pl.pointed.sum()), "of 200")


def test_15b_pointer_lists_with_distinct_colors_give_every_request_of_a_list_its_own_color():
    n = 6
    pl = MB.pointer_lists(n, 200, 151, distinct=True)
    plain = MB.pointer_lists(n, 200, 151)
    assert torch.equal(pl.cells, plain.cells)
    stated = []
    for i in range(200):
        req = [c for c in range(9) if int(pl.cells[i, c]) >= 0]
        cols = [int(pl.colors[i, c]) for c in req if int(pl.points[i, c]) < 0]      # the stated colors
        assert len(set(cols)) == len(cols)
        ps = [c for c in req if int(pl.points[i, c]) >= 0]
        for a in ps:                                                               # the pointer takes the asked color
            q = pl.cells[i].tolist().index(int(pl.points[i, a]))
            assert int(pl.colors[i, a]) == int(pl.colors[i, q])
        stated.append(len(cols) == n - len(ps))
    MB.check_layout(pl.cells, pl.colors, pl.points)
    print("lists pointed with distinct colors:", int(pl.pointed.sum()), "of 200")
    assert all(stated)


# ------------------------------------------------------------------ 16. a forward of fixed shape
@pytest.mark.parametrize("variant", [MB.T2, MB.T2K])
def test_16_vmap_over_two_stacked_engines_gives_each_engines_own_output(variant):
    from torch.func import functional_call, stack_module_state, vmap
    e1, e2 = _engine(variant, 1, 0, depth=2), _engine(variant, 1, 1, depth=2)
    class Tokens(torch.nn.Module):                                             # functional_call runs forward()
        def __init__(self, engine):
            super().__init__()
            self.engine = engine

        def forward(self, *args):
            return self.engine.forward_tokens(*args)

    params, buffers = stack_module_state([Tokens(e1), Tokens(e2)])
    shell = Tokens(copy.deepcopy(e1))
    batches = []
    for s in (160, 161):
        mb = MB.member_batch(variant, 1, BED, 3, _gen(s))
        z, t, _ = C.noisy(mb["x"], _gen(s + 10))
        prep = MB.prepare(variant, mb["cells"], mb["colors"], mb["points"], mb["told"], mb["visible"])
        batches.append((z, t, prep))

    def stack(key):
        return torch.stack([b[2][key] for b in batches])

    told = None if batches[0][2]["told_angles"] is None else tuple(
        torch.stack([b[2]["told_angles"][j] for b in batches]) for j in range(2))
    Z, T = torch.stack([b[0] for b in batches]), torch.stack([b[1] for b in batches])

    def one(p, bu, z, t, content, qa, ka, km, ta):
        return functional_call(shell, (p, bu), (z, t, content, qa, ka, km, ta))

    with torch.no_grad():
        dims = (0,) * 8 + ((None if told is None else 0),)
        out = vmap(one, in_dims=dims)(params, buffers, Z, T, stack("content"), stack("query_angles"), stack("key_angles"),
                        stack("key_mask"), told)
        want = [e.forward_tokens(b[0], b[1], **b[2]) for e, b in zip((e1, e2), batches)]
    diffs = [float((out[i] - want[i]).abs().max()) for i in range(2)]
    print(variant.name, "vmap against each engine's own forward:", diffs)
    assert max(diffs) < 1e-5


# ------------------------------------------------------------------ 17. the unturned streams carry no identity
def _relabeled_pointer_layout():
    """``_pointer_layout`` with other classes in the same cells (2 -> 0, 5 -> 1, 3 -> 4, 8 -> 6), the same colors, the
    pointer asking the class that now stands where the square stood."""
    cells, colors, points = _pointer_layout()
    rename = {2: 0, 5: 1, 3: 4, 8: 6}
    c2 = cells.clone()
    for old, new in rename.items():
        c2[cells == old] = new
    p2 = points.clone()
    p2[0, 0] = rename[int(points[0, 0])]
    return c2, colors.clone(), p2


@pytest.mark.parametrize("name", list(MB.VARIANTS))
def test_17_unturned_angles_do_not_depend_on_which_classes_stand_in_the_cells(name):
    v = MB.VARIANTS[name]
    one, two = _pointer_layout(), _relabeled_pointer_layout()
    a, b = MB.prepare(v, *one, unturned=True), MB.prepare(v, *two, unturned=True)
    assert torch.equal(a["query_angles"], b["query_angles"]) and torch.equal(a["key_angles"], b["key_angles"])
    if a["told_angles"] is not None or b["told_angles"] is not None:
        assert all(torch.equal(x, y) for x, y in zip(a["told_angles"], b["told_angles"])), f"{name}: told angles"
    assert a["told_angles"] is None and b["told_angles"] is None, name
    ta, tb = MB.prepare(v, *one), MB.prepare(v, *two)
    differ = not (torch.equal(ta["query_angles"], tb["query_angles"]) and torch.equal(ta["key_angles"], tb["key_angles"]))
    if ta["told_angles"] is not None:
        differ = differ or not all(torch.equal(x, y) for x, y in zip(ta["told_angles"], tb["told_angles"]))
    assert differ == (v.turn or v.identity == "name"), name


# ------------------------------------------------------------------ 18. values out of range
def test_18_check_layout_bounds_every_value():
    cells, colors, points = _pointer_layout()
    MB.check_layout(cells, colors, points)
    for which, value in (("cells", 9), ("colors", 9), ("points", 9), ("cells", -2), ("colors", -2), ("points", -2)):
        c, k, p = cells.clone(), colors.clone(), points.clone()
        t = {"cells": c, "colors": k, "points": p}[which]
        if which == "points":
            t[0, 2] = value                                                    # the diamond's request
        else:
            t[0, 8] = value                                                    # the star's cell
            if which == "cells" and value == -2:
                k[0, 8] = -1
        with pytest.raises(ValueError):
            MB.check_layout(c, k, p)


def test_18b_the_reads_refuse_a_class_or_cell_outside_0_to_8():
    col = _collective(MB.T2)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(180)), torch.tensor([0.5])
    g = _gen(181)
    answers, base_answer = torch.randn(9, 1, 144, 48, generator=g), torch.randn(1, 144, 48, generator=g)
    for bad in (9, -1):
        with pytest.raises(ValueError):
            MB.serve_read(col, answers, base_answer, cells, 0, 2, bad)
        with pytest.raises(ValueError):
            MB.address_swap(col, z, t, cells, colors, points, 0, 2, bad)
    with pytest.raises(ValueError):
        MB.serve_read(col, answers, base_answer, cells, 0, 9, 6)
    with pytest.raises(ValueError):
        MB.address_missing(col, cells, 0, 9)
    with pytest.raises(ValueError):
        MB.address_unheld(col, cells, 0, 9)
    bad_cells = cells.clone()
    bad_cells[0, 1] = 9
    with pytest.raises(ValueError):
        MB.exchange_read(col, col, z, t, bad_cells, colors, points)


# ------------------------------------------------------------------ 19. find_read refuses what it cannot read
def test_19_find_read_refuses_another_variant_a_cell_that_does_not_point_and_one_ask_twice():
    eng = _engine(MB.T2, 2, 0, depth=2)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(190)), torch.tensor([0.3])
    with pytest.raises(ValueError):
        MB.find_read(eng, MB.T3, z, t, cells, colors, points, 0, 0, 3, 8, block=1)
    with pytest.raises(ValueError):
        MB.find_read(eng, MB.T2, z, t, cells, colors, points, 0, 2, 3, 8, block=1)      # cell 2 states its color
    with pytest.raises(ValueError):
        MB.find_read(eng, MB.T2, z, t, cells, colors, points, 0, 0, 3, 3, block=1)


def test_19b_pair_group_scores_are_float32():
    g = _gen(191)
    q, k = torch.randn(1, 2, 153, 64, generator=g).double(), torch.randn(1, 2, 153, 64, generator=g).double()
    assert MB.pair_group_scores(q, k, 0, (4, 8)).dtype == torch.float32


# ------------------------------------------------------------------ 20. the tests that were missing
def test_20_save_and_load_keep_the_weights_the_average_the_variant_and_the_birth(tmp_path):
    for v, cls in ((MB.T2K, 3), (MB.T2, None)):
        eng = _engine(v, cls, 4, depth=2)
        avg = {k: x + 0.25 for k, x in eng.state_dict().items()}
        folder = str(tmp_path / f"{v.name}_{cls}")
        MB.save_member(eng, folder, average=avg, card={"bed": "F"})
        raw, mean = MB.load_member(folder, average=False), MB.load_member(folder, average=True)
        for k, x in eng.state_dict().items():
            assert torch.equal(raw.state_dict()[k], x) and torch.equal(mean.state_dict()[k], avg[k]), k
        for e in (raw, mean):
            assert e.variant == v and e.cls == cls and e.seed == 4 and e.card["bed"] == "F"
            assert e.settings == eng.settings
        MB.save_member(eng, folder + "_plain")
        assert all(torch.equal(MB.load_member(folder + "_plain").state_dict()[k], x) for k, x in eng.state_dict().items())


def test_20b_pack_and_sample_lists_shape_seed_and_chunks():
    col = _collective(MB.T2, classes=range(9))
    pl = MB.pointer_lists(4, 6, 200, distinct=True)
    packed = MB.pack(pl.colors, pl.points)
    assert tuple(packed.shape) == (6, 9, 2) and torch.equal(packed[..., 0], pl.colors)
    assert torch.equal(packed[..., 1], pl.points) and packed.dtype == torch.long
    a = MB.sample_lists(col, pl.cells, pl.colors, pl.points, 11, steps=3)
    b = MB.sample_lists(col, pl.cells, pl.colors, pl.points, 11, steps=3)
    c = MB.sample_lists(col, pl.cells, pl.colors, pl.points, 11, steps=3, chunk=4)
    d = MB.sample_lists(col, pl.cells, pl.colors, pl.points, 12, steps=3)
    assert tuple(a.shape) == (6, 3, 48, 48) and torch.equal(a, b) and not torch.equal(a, d)
    diff = float((a - c).abs().max())
    print("sample_lists chunked against unchunked: largest difference", diff, "; equal", torch.equal(a, c))
    assert diff < 1e-5


def test_20c_a_collective_with_its_own_addresses_moves_the_streams_and_the_gate_together():
    mapping = {0: (0, 1), 1: (0, 0), 2: (3, 3)}                                # 0 and 1 trade; 2 to a free address
    things = {k: _engine(MB.T2, k, 0) for k in range(9)}
    base = _engine(MB.T2, None, 0)
    col = MB.MemberCollective(things, base, MB.T2, addresses=mapping)
    plain = MB.MemberCollective(things, base, MB.T2)
    assert col.member_addresses[[0, 1, 2]].tolist() == [[0, 1], [0, 0], [3, 3]]
    cells, _ = scenes.draw_layouts("mix", 32, _gen(201))
    w, b = col.weights(cells)
    w0, b0 = plain.weights(cells)
    assert torch.equal(w, w0) and torch.equal(b, b0)                           # the owner is the same thing
    st = MB.streams(MB.T2, cells, MB.no_points(cells), mapping)
    seq_cell = torch.cat([torch.arange(9), CELL_OF])
    for i in range(4):
        for tok in range(153):
            c = int(cells[i, seq_cell[tok]])
            want = tuple(mapping.get(c, MB.address_of(c))) if c >= 0 else MB.NULL_ADDRESS
            assert (int(st["key_digits"]["member_0"][i, tok]), int(st["key_digits"]["member_1"][i, tok])) == want
    other = dict(mapping)
    other[2] = (2, 0)                                                          # the requests of 2 at the plus's address
    wq, _ = col.weights(cells, request_addresses=other)
    has2 = (cells == 2).any(dim=1)
    toks2 = cells[:, CELL_OF] == 2
    assert bool((wq[..., 2][toks2] == 0.0).all()) and bool((wq[..., 6][toks2] > 0.99).all()) and bool(has2.any())


def test_20d_the_candidate_form_sums_to_one_and_is_silent_off_the_address():
    col = MB.MemberCollective(_collective(MB.T2).things, None, MB.T2, form="candidate")
    cells, _ = scenes.draw_layouts("mix", 64, _gen(210))
    w, base = col.weights(cells)
    total = w.sum(dim=-1) + base
    assert float((total - 1.0).abs().max()) < 1e-12
    owner = frame.token_owner(cells, [[k] for k in range(9)])
    off = torch.nn.functional.one_hot(owner, 10)[..., :9] == 0
    assert bool((w[off] == 0.0).all())
    assert float((w[~off] - math.sinh(10.0) / (math.sinh(10.0) + math.exp(-10.0))).abs().max()) < 1e-12


def test_20f_eligible_donors_are_counted_on_hand_built_lists():
    def lists(*rows):
        out = torch.full((len(rows), 9), -1, dtype=torch.long)
        for i, r in enumerate(rows):
            out[i, :len(r)] = torch.tensor(r)
        return out

    cells = lists([0, 1, 1, 2],        # a duplicated class: the two 1s are no donors -> k = 1 (the 2); held: 0
                  [0, 0, 1, 2],        # the asker's own class twice: the other 0 is no donor -> k = 2; held: 1
                  [0, 1, 2],           # k = 2; the pair (0, 2) held -> k = 1 (the 1); only held -> 1 (the 2)
                  [3, 4, 5, 6, 7])     # four donors
    asking = torch.tensor([0, 0, 0, 0])
    held = {(0, 2)}
    assert MB.eligible_donors(cells, asking).tolist() == [1, 2, 2, 4]
    assert MB.eligible_donors(cells, asking, held).tolist() == [0, 1, 1, 4]
    assert MB.eligible_donors(cells, asking, held, only_held=True).tolist() == [1, 1, 1, 0]
    assert MB.eligible_donors(cells, torch.tensor([-1, 1, 2, 4])).tolist() == [0, 2, 2, 4]
    pl = MB.pointer_lists(4, 300, 230, distinct=True)
    for i in range(300):
        if not bool(pl.pointed[i]):
            assert int(pl.asking[i]) == -1 and int(pl.eligible[i]) == 0
            continue
        a = int(pl.asking[i])
        assert int(pl.points[i, a]) >= 0
        row = pl.cells[i].tolist()
        want = sum(1 for q in range(9) if row[q] >= 0 and row[q] != row[a] and row.count(row[q]) == 1)
        assert int(pl.eligible[i]) == want >= 1
    assert tuple(pl[:5]) == tuple(MB.PointerLists(*pl)[:5])                   # the first five fields keep their places


# ------------------------------------------------------------------ 21. the hoists give the same numbers
def _reference_streams(variant, cells, points, addresses=None):
    """``members.streams`` as written before its tables were kept."""
    b = cells.shape[0]
    rows, cols, pcol, prow = MB._positions(cells.device)
    seq_cell = torch.cat([torch.arange(9), CELL_OF])
    place = {"place_column": pcol.expand(b, -1), "place_row": prow.expand(b, -1)}
    if variant.turn and variant.identity != "name":
        table = MB.address_table(addresses)
        key_seq = table[MB._index(cells)][:, seq_cell]
        query_seq = table[MB._index(torch.where(points >= 0, points, cells))][:, seq_cell]
    else:
        key_seq = query_seq = torch.full((b, 153, 2), 4, dtype=torch.long)
    kd = dict(place, member_0=key_seq[..., 0], member_1=key_seq[..., 1])
    qd = dict(place, member_0=query_seq[..., 0], member_1=query_seq[..., 1])
    return {"rows": rows.expand(b, -1), "cols": cols.expand(b, -1), "key_digits": kd, "query_digits": qd}


def _reference_prepare(variant, cells, colors, points=None, told=None, visible=None, addresses=None, unturned=False):
    """``members.prepare`` as written before the hoists: both angle sets from the full computation with its digit
    checks, every table built again (and, unturned, no told head: the rule of the module's text)."""
    points = MB.no_points(cells) if points is None else points
    MB.check_layout(cells, colors, points)
    told = (cells >= 0) if told is None else (told.bool() & (cells >= 0))
    st = _reference_streams(variant, cells, points, addresses)
    ka = M.PLACE_MEMBER.angles(st["rows"], st["cols"], st["key_digits"])
    qa = M.PLACE_MEMBER.angles(st["rows"], st["cols"], st["query_digits"])
    if unturned:
        ka, qa = ka.clone(), qa.clone()
        ka[..., list(MB.UNTURNED_PAIRS)] = 0.0
        qa[..., list(MB.UNTURNED_PAIRS)] = 0.0
    told_angles = None
    if variant.identity == "name" and not unturned:
        names = MB.name_table()
        seq_cell = torch.cat([torch.arange(9), CELL_OF])
        kp = names[MB._index(cells)][:, seq_cell]
        qp = names[MB._index(torch.where(points >= 0, points, cells))][:, seq_cell]
        tk, tq = ka.clone(), qa.clone()
        tk[..., list(MB.NAME_PAIRS)] = kp.float()
        tq[..., list(MB.NAME_PAIRS)] = qp.float()
        told_angles = (tq, tk)
    img = torch.ones(cells.shape[0], 144, dtype=torch.bool) if visible is None else visible.bool()
    return {"content": MB.request_content(variant, cells, colors, points, told), "query_angles": qa,
            "key_angles": ka, "key_mask": torch.cat([told, img], dim=1), "told_angles": told_angles}


def _reference_part_answers(col, z, t, cells, colors, points, unturned=False):
    """``MemberCollective.part_answers`` as written before the plan: every engine prepared from its own rows."""
    v = col.variant
    out = torch.zeros(len(col.classes), z.shape[0], 144, 48)
    for i, k in enumerate(col.classes):
        eng = col.things[k]
        if v.told == "all":
            rows = torch.nonzero((cells == k).any(dim=1), as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            vis = MB.visible_for(cells[rows], k) if v.see == "masked" else None
            prep = _reference_prepare(v, cells[rows], colors[rows], points[rows], None, vis, col.addresses, unturned)
            out[i, rows] = patchify(eng.forward_tokens(z[rows], t[rows], **prep), 4)
            continue
        rows, cols = torch.nonzero(cells == k, as_tuple=True)
        if rows.numel() == 0:
            continue
        vis = frame.visible_keys(cells[rows], cols) if v.see == "masked" else None
        told = torch.nn.functional.one_hot(cols, 9).bool()
        prep = _reference_prepare(v, cells[rows], colors[rows], points[rows], told, vis, col.addresses, unturned)
        vt = patchify(eng.forward_tokens(z[rows], t[rows], **prep), 4)
        p_idx, tok = torch.nonzero(CELL_OF[None] == cols[:, None], as_tuple=True)
        out[i, rows[p_idx], tok] = vt[p_idx, tok]
    base = torch.zeros(z.shape[0], 144, 48)
    rows = torch.nonzero((cells < 0).any(dim=1), as_tuple=True)[0]
    if rows.numel():
        if v.told == "all":
            vis = MB.visible_for(cells[rows], None) if v.see == "masked" else None
            prep = _reference_prepare(v, cells[rows], colors[rows], points[rows], None, vis, col.addresses, unturned)
        else:
            vis = frame.visible_keys(cells[rows], torch.full((rows.numel(),), -1)) if v.see == "masked" else None
            prep = _reference_prepare(v, cells[rows], colors[rows], points[rows], torch.zeros(rows.numel(), 9,
                                      dtype=torch.bool), vis, col.addresses, unturned)
        base[rows] = patchify(col.base.forward_tokens(z[rows], t[rows], **prep), 4)
    return out, base


def _reference_answer(col, z, t, cells, colors, points):
    answers, base_answer = _reference_part_answers(col, z, t, cells, colors, points)
    w, base = col.weights(cells)
    return unpatchify(M.serve(base_answer, answers, w, base), (3, 48, 48), 4, 12)


def _same_prep(a, b):
    for key in ("content", "query_angles", "key_angles", "key_mask"):
        if not torch.equal(a[key], b[key]):
            return False
    if (a["told_angles"] is None) != (b["told_angles"] is None):
        return False
    return a["told_angles"] is None or all(torch.equal(x, y) for x, y in zip(a["told_angles"], b["told_angles"]))


@pytest.mark.parametrize("name", list(MB.VARIANTS))
def test_21_prepare_gives_the_numbers_of_the_full_computation(name):
    v = MB.VARIANTS[name]
    pl = MB.pointer_lists(5, 24, 240, distinct=True)
    points = pl.points if v.told == "all" else MB.no_points(pl.cells)
    told = None if v.told == "all" else torch.nn.functional.one_hot(pl.order[:, 0], 9).bool()
    vis = MB.visible_for(pl.cells, 3)
    mapping = {0: (0, 1), 1: (0, 0), 2: (3, 3)}
    for unturned in (False, True):
        for addresses in (None, mapping):
            got = MB.prepare(v, pl.cells, pl.colors, points, told, vis, addresses, unturned)
            want = _reference_prepare(v, pl.cells, pl.colors, points, told, vis, addresses, unturned)
            assert _same_prep(got, want), (name, unturned, addresses)


@pytest.mark.parametrize("name,depth", [("T1", 1), ("T2", 1), ("T2K", 2), ("T3", 1), ("C2", 1)])
def test_21b_a_plan_gives_the_answers_of_preparing_every_engine_every_step(name, depth):
    v = MB.VARIANTS[name]
    col = _collective(v, depth=depth)
    pl = MB.pointer_lists(4, 10, 241, distinct=True)
    points = pl.points if v.told == "all" else MB.no_points(pl.cells)
    z, t = torch.randn(10, 3, 48, 48, generator=_gen(242)), torch.full((10,), 0.4)
    with torch.no_grad():
        for unturned in (False, True):
            plan = col.plan(pl.cells, pl.colors, points, unturned)
            got = col.part_answers(z, t, pl.cells, pl.colors, points, unturned, plan=plan)
            fresh = col.part_answers(z, t, pl.cells, pl.colors, points, unturned)
            want = _reference_part_answers(col, z, t, pl.cells, pl.colors, points, unturned)
            for g, f, w in zip(got, fresh, want):
                assert torch.equal(g, w) and torch.equal(f, w), (name, unturned)
            c = col.contributions(z, t, pl.cells, pl.colors, points, unturned, plan=plan)
            w, base = col.weights(pl.cells)
            parts = [w[..., i].float()[..., None] * want[0][i] for i in range(9)] + [base.float()[..., None] * want[1]]
            assert torch.equal(c, torch.stack(parts)), (name, unturned)
        plan = col.plan(pl.cells, pl.colors, points)
        assert torch.equal(col.answer(z, t, pl.cells, pl.colors, points, plan=plan),
                           _reference_answer(col, z, t, pl.cells, pl.colors, points))
        with pytest.raises(ValueError):
            col.part_answers(z, t, pl.cells, pl.colors, points, True, plan=plan)       # built with unturned=False


@pytest.mark.parametrize("name,depth", [("T1", 1), ("T2", 1), ("T2K", 2)])
def test_21c_sample_lists_with_one_plan_a_chunk_is_the_sampler_preparing_every_step(name, depth, monkeypatch):
    v = MB.VARIANTS[name]
    col = _collective(v, depth=depth)
    pl = MB.pointer_lists(4, 6, 243, distinct=True)
    points = pl.points if v.told == "all" else MB.no_points(pl.cells)
    calls = []
    real = MB.prepare
    monkeypatch.setattr(MB, "prepare", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = MB.sample_lists(col, pl.cells, pl.colors, points, 21, steps=3, chunk=4)
    monkeypatch.setattr(MB, "prepare", real)
    want = C.sample(lambda z, t, c, k: _reference_answer(col, z, t, c, k[..., 0], k[..., 1]), pl.cells,
                    MB.pack(pl.colors, points), 21, steps=3, chunk=4)
    assert torch.equal(got, want)
    engines_run = 2 * 10 if v.told == "own" else 2                         # a prepare an engine a chunk, or one a chunk
    print(name, "prepare calls over 2 chunks of 3 steps:", len(calls))
    assert len(calls) <= engines_run


@pytest.mark.parametrize("variant,depth", [(MB.T2K, 2), (MB.C1, 1)])
def test_20e_find_read_on_the_name_kind_and_on_the_kind_whose_pairs_are_not_turned(variant, depth):
    eng = _engine(variant, 2, 0, depth=depth)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(220)), torch.tensor([0.3])
    for block in range(depth):
        r = MB.find_read(eng, variant, z, t, cells, colors, points, 0, 0, 3, 8, block=block)
        same = torch.equal(r["weights_a"], r["weights_b"]) and torch.equal(r["answer_a"], r["answer_b"])
        if variant is MB.C1:
            assert same and torch.equal(r["scores_a"], r["scores_b"])
        elif block == MB.TOLD_BLOCK:                                           # the told head reads by name
            assert not torch.equal(r["scores_a"][MB.TOLD_HEAD], r["scores_b"][MB.TOLD_HEAD])
            assert torch.equal(r["scores_a"][1], r["scores_b"][1])             # the other head: nothing turned
        print(variant.name, "block", block, "weights a", r["weights_a"].tolist(), "b", r["weights_b"].tolist())


@pytest.mark.parametrize("variant,vary", [(MB.T2, "query"), (MB.T2K, "query"), (MB.C2, "content")])
def test_20g_find_read_of_every_block_at_once_is_the_reads_of_one_block_at_a_time(variant, vary):
    eng = _engine(variant, 2, 0, depth=3)
    cells, colors, points = _pointer_layout()
    z, t = torch.randn(1, 3, 48, 48, generator=_gen(230)), torch.tensor([0.3])
    whole = MB.find_read(eng, variant, z, t, cells, colors, points, 0, 0, 3, 8, None, vary=vary)
    assert sorted(whole["by_block"]) == [0, 1, 2]
    for block in range(3):
        one = MB.find_read(eng, variant, z, t, cells, colors, points, 0, 0, 3, 8, block, vary=vary)
        for key in ("scores_a", "scores_b", "weights_a", "weights_b"):
            assert torch.equal(whole["by_block"][block][key], one[key]), (variant.name, block, key)
        assert torch.equal(whole["answer_a"], one["answer_a"]) and torch.equal(whole["answer_b"], one["answer_b"])


# ------------------------------------------------------------------ the rows split into the CPU layout and the render
def _reference_member_batch(variant, cls, bed, n, gen, p_point=0.5, exclude=None, held_pairs=()):
    """A copy of ``members.member_batch`` as it was before the split into ``member_layout`` and the render."""
    dev = bed.device
    n = int(n)
    if variant.told == "own":
        if exclude is not None or held_pairs:
            raise ValueError("told 'own' rows are the stored rows: no clutter to exclude, no pointers")
        kb = C.kind_batch("addressed_thing" if cls is not None else "addressed_base", cls, bed, n, gen)
        none = torch.zeros(n, dtype=torch.bool, device=dev)
        cells_cpu = kb["cells"].cpu().clone()
        return {"x": kb["x"], "cells": kb["cells"], "colors": kb["colors"], "points": MB.no_points(kb["cells"]),
                "told": kb["told"], "visible": kb["key_mask"], "loss_mask": kb["loss_mask"], "hidden": kb["hidden"],
                "params": kb["params"], "pointing": none, "can_point": none, "other_pointing": none,
                "other_can_point": none, "cells_cpu": cells_cpu, "colors_cpu": kb["colors"].cpu().clone(),
                "points_cpu": MB.no_points(cells_cpu), "told_cpu": kb["told"].cpu().clone(),
                "visible_cpu": kb["key_mask"].cpu().clone()}
    if exclude is not None and cls is not None and int(exclude) == int(cls):
        raise ValueError("an engine's own class cannot be excluded from its clutter")
    N = MB.N_CELLS
    r = torch.arange(n)
    if cls is not None:
        oc, ok = scenes.draw_layouts("one", n, gen, only_class=int(cls))
        cells, colors = scenes.draw_layouts("mix", n, gen)
        own = (oc >= 0).long().argmax(dim=1)
        cells[r, own] = int(cls)
        colors[r, own] = ok[r, own]
    else:
        cells, colors = scenes.draw_layouts("mix", n, gen)
        empty = torch.randint(0, N, (n,), generator=gen)
        cells[r, empty] = -1
        colors[r, empty] = -1
    if exclude is not None:
        drop = cells == int(exclude)
        cells[drop] = -1
        colors[drop] = -1
    points = MB.no_points(cells)
    pointing = torch.zeros(n, dtype=torch.bool)
    can = torch.zeros(n, dtype=torch.bool)
    if cls is not None:
        u = torch.rand(n, generator=gen)
        keys = torch.rand(n, N, generator=gen)
        count = (cells[:, :, None] == cells[:, None, :]).sum(dim=-1)
        cand = (cells >= 0) & (cells != int(cls)) & (count == 1) & \
            ~MB._held_asked(int(cls), held_pairs)[MB._index(cells)]
        can = cand.any(dim=1)
        pointing = (u < float(p_point)) & can
        pick = torch.where(cand, keys, torch.full_like(keys, 2.0)).argmin(dim=1)
        rp = r[pointing]
        points[rp, own[pointing]] = cells[rp, pick[pointing]]
        colors[rp, own[pointing]] = colors[rp, pick[pointing]]
    asking_ok = (cells >= 0) & (points < 0)
    if cls is not None:
        asking_ok[r, own] = False
        asking_ok[rp, pick[pointing]] = False
    v = torch.rand(n, generator=gen)
    keys2 = torch.rand(n, N, N, generator=gen)
    valid = MB._valid_pairs(cells, MB._held_matrix(held_pairs), False, asking_ok, points < 0)
    other_can = valid.flatten(1).any(dim=1)
    other = (v < float(p_point)) & other_can
    ask, asked = MB._smallest_pair(valid, keys2)
    ro = r[other]
    points[ro, ask[other]] = cells[ro, asked[other]]
    colors[ro, ask[other]] = colors[ro, asked[other]]
    pixels, params = scenes.render(cells, colors, bed.bed, gen, dev)
    cpu = {"cells_cpu": cells.clone(), "colors_cpu": colors.clone(), "points_cpu": points.clone(),
           "told_cpu": cells >= 0,
           "visible_cpu": MB.visible_for(cells, cls) if variant.see == "masked" else None}
    cells, colors, points = cells.to(dev), colors.to(dev), points.to(dev)
    told = cells >= 0
    mine = MB.cell_tokens(cells < 0) if cls is None else MB.cell_tokens(cells == int(cls))
    visible = MB.visible_for(cells, cls) if variant.see == "masked" else None
    return dict({"x": scenes.to_model(pixels), "cells": cells, "colors": colors, "points": points, "told": told,
                 "visible": visible, "loss_mask": mine, "hidden": None, "params": params,
                 "pointing": pointing.to(dev), "can_point": can.to(dev), "other_pointing": other.to(dev),
                 "other_can_point": other_can.to(dev)}, **cpu)


def _same_rows(got, want, what):
    assert sorted(got) == sorted(want), what
    for k in want:
        if k == "params":
            assert sorted(got[k]) == sorted(want[k]), what
            for q in want[k]:
                assert torch.equal(got[k][q], want[k][q]), (what, k, q)
        elif want[k] is None:
            assert got[k] is None, (what, k)
        else:
            assert torch.equal(got[k], want[k]), (what, k)


@pytest.mark.parametrize("name", list(MB.VARIANTS))
@pytest.mark.parametrize("bed_name", scenes.BEDS)
def test_22_member_batch_is_bit_for_bit_the_rows_before_the_split(name, bed_name):
    variant, bed = MB.VARIANTS[name], scenes.ScenesBed(bed_name)
    settings = [dict()] if variant.told == "own" else \
        [dict(), dict(p_point=1.0, exclude=7, held_pairs=((3, 5), (5, 3), (0, 1))), dict(p_point=0.0)]
    for cls in (3, 0, None):
        for i, kw in enumerate(settings):
            seed = 2200 + 10 * i + (9 if cls is None else cls)
            got = MB.member_batch(variant, cls, bed, 19, _gen(seed), **kw)
            gen = _gen(seed)
            want = _reference_member_batch(variant, cls, bed, 19, gen, **kw)
            _same_rows(got, want, (name, bed_name, cls, i))
            # the layout alone: CPU tensors, and the generator left where the rows leave it
            g2 = _gen(seed)
            lay = MB.member_layout(variant, cls, bed, 19, g2, **kw)
            assert torch.equal(torch.rand(4, generator=g2), torch.rand(4, generator=gen))
            for k, v in lay.items():
                for x in (v.values() if isinstance(v, dict) else [v]):
                    assert x is None or x.device.type == "cpu", k
            assert torch.equal(lay["loss_mask"], want["loss_mask"].cpu())
