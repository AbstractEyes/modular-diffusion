"""The block's key mask, the engines of both read forms, their batches, loss and trainer, the collective's joins and
the exactness of the address gate (CPU, tiny engines)."""
import math
import os

import pytest
import torch
import torch.nn.functional as F

from geolip.modular_diffusion import collective as C
from geolip.modular_diffusion import frame, rotary, scenes
from geolip.modular_diffusion.blocks import Block

BED = scenes.ScenesBed("F")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _wake(engine, seed):
    """Give a newborn engine a nonzero head (and modulation) so that its answers carry its inputs."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (engine.head.proj.weight, engine.head.proj.bias):
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        for blk in engine.blocks:
            w = blk.modulation[1].weight
            w.copy_(torch.randn(w.shape, generator=g) * 0.02)
    return engine.eval()


def _tiny(kind, cls, seed, read="embedding"):
    dim, heads = (128, 2) if read == "rotary" else (32, 2)
    return _wake(C.born(kind, cls, seed, dim, heads, 1, read), 1000 + seed * 17 + (cls or 0))


# ------------------------------------------------------------------ the block
def _old_block_forward(blk, x, ctx, ctx_mask, e, rot):
    """The block's forward as it was before the key mask (the reference expressions written out)."""
    ds1, dh1, ds2, dh2, ds3, dh3 = blk.modulation(e).unsqueeze(1).chunk(6, dim=-1)
    y = blk.norm1(x) * (1 + ds1) + dh1
    b, n, d = y.shape
    a = blk.attn
    qkv = a.qkv(y).view(b, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
    q, k = rot(qkv[0], qkv[1])
    att = F.scaled_dot_product_attention(q, k, qkv[2])
    x = x + a.out(att.transpose(1, 2).reshape(b, n, d))
    x = x + blk.cross(blk.norm2(x) * (1 + ds2) + dh2, ctx, ctx_mask)
    return x + blk.mlp(blk.norm3(x) * (1 + ds3) + dh3)


def test_the_block_without_a_key_mask_is_the_old_block_bit_for_bit():
    torch.manual_seed(0)
    blk = Block(32, 2)
    with torch.no_grad():
        blk.modulation[1].weight.normal_(0, 0.1)
    x, ctx, e = torch.randn(3, 20, 32), torch.randn(3, 2, 32), torch.randn(3, 32)
    code = rotary.AxialBandCode(16, 4, 2.0, 64.0)
    x16 = x[:, :16]
    r = code.rotary()
    assert torch.equal(blk(x16, ctx, None, e, r), _old_block_forward(blk, x16, ctx, None, e, r))
    assert torch.equal(blk(x16, ctx, None, e, r, None), _old_block_forward(blk, x16, ctx, None, e, r))


def test_a_masked_token_changes_no_number_of_any_other_token():
    torch.manual_seed(1)
    blk = Block(32, 2)
    with torch.no_grad():
        blk.modulation[1].weight.normal_(0, 0.1)
    code = rotary.AxialBandCode(16, 4, 2.0, 64.0).rotary()
    x, ctx, e = torch.randn(2, 16, 32), torch.randn(2, 1, 32), torch.randn(2, 32)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, [3, 7, 12]] = False
    y = x.clone()
    y[:, [3, 7, 12]] += torch.randn(2, 3, 32) * 5
    a, b = blk(x, ctx, None, e, code, mask), blk(y, ctx, None, e, code, mask)
    keep = [i for i in range(16) if i not in (3, 7, 12)]
    assert torch.equal(a[:, keep], b[:, keep])
    assert not torch.equal(a[:, [3, 7, 12]], b[:, [3, 7, 12]])          # a masked token still reads itself
    w = x.clone()
    w[:, 5] += 1.0                                                     # a visible token changes the others
    c = blk(w, ctx, None, e, code, mask)
    assert not torch.equal(a[:, 0], c[:, 0])
    # a row with every key masked still reads itself (no empty attention row, no NaN)
    none = torch.zeros(2, 16, dtype=torch.bool)
    assert torch.isfinite(blk(x, ctx, None, e, code, none)).all()


# ------------------------------------------------------------------ the anchored rotary
def test_the_anchored_rotary_reproduces_the_axial_ladder_and_its_anchor_kernel():
    g = 12
    idx = torch.arange(g * g)
    rows, cols = (idx // g)[None].double(), (idx % g)[None].double()
    dcol, drow = ((idx % g) // 4)[None], ((idx // g) // 4)[None]
    ang = rotary.anchored_angles(rows, cols, dcol, drow)[0]               # [T, 32]
    ladder = rotary.AxialBandCode(40, g, *rotary.WHOLE)                   # 10 rates an axis
    assert float((torch.cos(ang[:, :20]) - ladder.cos[:, :20]).abs().max()) < 1e-5
    assert float((torch.sin(ang[:, :20]) - ladder.sin[:, :20]).abs().max()) < 1e-5
    assert float(ang[:, 20:24].abs().max()) == 0.0
    # the anchor kernel cos(2 pi (d - d') / 3): 1 for the same digit, -1/2 for another
    for d in range(3):
        for e in range(3):
            a1 = 2 * math.pi * d / 3
            a2 = 2 * math.pi * e / 3
            k = math.cos(a1 - a2)
            assert abs(k - (1.0 if d == e else -0.5)) < 1e-12
    q = torch.zeros(1, 1, 2, 64)
    q[..., 24] = 1.0                                                     # one anchor pair, unit length
    rot = rotary.anchored_rotary(torch.zeros(1, 2).double(), torch.zeros(1, 2).double(), torch.tensor([[0, 1]]),
                                 torch.tensor([[0, 0]]))
    qq, kk = rot(q, q)
    assert abs(float((qq[0, 0, 0] * kk[0, 0, 0]).sum()) - 1.0) < 1e-6
    assert abs(float((qq[0, 0, 0] * kk[0, 0, 1]).sum()) + 0.5) < 1e-6


# ------------------------------------------------------------------ engines
@pytest.mark.parametrize("read", ["embedding", "rotary"])
def test_a_newborn_engine_answers_exactly_zero_and_its_birth_depends_on_seed_and_name(read):
    b = C.kind_batch("addressed_thing", 4, BED, 3, _gen(0))
    z, t, _ = C.noisy(b["x"], _gen(1))
    e = C.born("addressed_thing", 4, 0, read=read)
    with torch.no_grad():
        assert float(C.engine_forward(e, b, z, t).abs().max()) == 0.0
    torch.manual_seed(123)
    other = C.born("whole_empty", None, 5, read=read)                       # another engine first: no effect
    e2 = C.born("addressed_thing", 4, 0, read=read)
    after = torch.rand(1)
    torch.manual_seed(123)
    C.born("whole_empty", None, 5, read=read)
    C.born("addressed_thing", 4, 0, read=read)
    assert torch.equal(after, torch.rand(1))                                # the caller's random state is put back
    for (n1, p1), (n2, p2) in zip(e.state_dict().items(), e2.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)
    e3 = C.born("addressed_thing", 5, 0, read=read)
    assert not torch.equal(e.embed.weight, e3.embed.weight)
    e4 = C.born("addressed_thing", 4, 1, read=read)
    assert not torch.equal(e.embed.weight, e4.embed.weight)
    assert other is not None


def test_the_masked_loss_equals_a_hand_computation_and_refuses_an_empty_row():
    g = _gen(3)
    v, tg = torch.randn(2, 3, 48, 48, generator=g), torch.randn(2, 3, 48, 48, generator=g)
    mask = torch.zeros(2, 144, dtype=torch.bool)
    mask[0, [0, 5, 77]] = True
    mask[1, 10:30] = True
    from geolip.modular_diffusion.spectral import patchify
    d = patchify(v - tg, 4)
    want = 0.0
    for r in range(2):
        toks = torch.nonzero(mask[r], as_tuple=True)[0]
        want += float((d[r, toks] ** 2).sum()) / (48 * len(toks))
    want /= 2
    assert abs(float(C.masked_loss(v, tg, mask)) - want) < 1e-6
    full = torch.ones(2, 144, dtype=torch.bool)
    assert abs(float(C.masked_loss(v, tg, full)) - float(((v - tg) ** 2).mean())) < 1e-6
    mask[1] = False
    with pytest.raises(ValueError):
        C.masked_loss(v, tg, mask)


def _cells_of_tokens():
    return frame.TokenFrame().cell


def test_every_kind_batch_has_the_told_cells_key_mask_and_loss_mask_of_the_table():
    cell = _cells_of_tokens()
    for kind, cls in (("addressed_thing", 2), ("addressed_base", None), ("whole_thing", 6), ("whole_empty", None),
                      ("whole_unconditional", None), ("model_one", None), ("model_mix", None)):
        b = C.kind_batch(kind, cls, BED, 40, _gen(7))
        cells = b["cells"]
        for i in range(40):
            req = [c for c in range(9) if int(cells[i, c]) >= 0]
            told = [c for c in range(9) if bool(b["told"][i, c])]
            if kind in ("addressed_thing", "whole_thing"):
                assert len(req) == 1 and int(cells[i, req[0]]) == cls and told == req
            elif kind in ("addressed_base", "whole_empty"):
                assert req == [] and told == []
            elif kind == "whole_unconditional":
                assert len(req) == 1 and told == []
            elif kind == "model_one":
                assert len(req) == 1 and told == req
            else:
                assert told == req
            if kind == "addressed_thing":
                own = req[0]
                hid = [c for c in range(9) if bool(b["hidden"][i, c])]
                assert own not in hid and len(hid) <= 8
                vis = {int(cell[t]) for t in range(144) if bool(b["key_mask"][i, t])}
                assert vis == ({c for c in range(9) if c not in hid and c not in req} | {own})
                assert {int(cell[t]) for t in range(144) if bool(b["loss_mask"][i, t])} == {own}
                assert int(b["loss_mask"][i].sum()) == 16
            elif kind == "addressed_base":
                hid = {c for c in range(9) if bool(b["hidden"][i, c])}
                vis = {int(cell[t]) for t in range(144) if bool(b["key_mask"][i, t])}
                assert vis == set(range(9)) - hid and len(hid) <= 8
                assert torch.equal(b["loss_mask"][i], b["key_mask"][i])
            else:
                assert b["key_mask"] is None and bool(b["loss_mask"].all())
    # h covers 0..8 for the thing parts
    b = C.kind_batch("addressed_thing", 0, BED, 900, _gen(8))
    counts = torch.bincount(b["hidden"].sum(1), minlength=9)
    assert counts.numel() == 9 and int(counts.min()) > 50
    # the embedding read's codes: place everywhere, content on the told cells only
    b = C.kind_batch("model_mix", None, BED, 6, _gen(9))
    code = C.token_codes(b["cells"], b["colors"], b["told"])
    place = frame.TokenFrame().place_code()
    assert torch.equal(code[..., :16], place.expand(6, -1, -1))
    for i in range(6):
        for t in range(144):
            c = int(cell[t])
            if int(b["cells"][i, c]) >= 0:
                assert torch.equal(code[i, t, 16:], frame.content_code(int(b["cells"][i, c]), int(b["colors"][i, c])))
            else:
                assert float(code[i, t, 16:].abs().max()) == 0.0


def test_allowed_cells_keep_the_center_out():
    b = C.kind_batch("addressed_thing", 3, BED, 400, _gen(10), allowed_cells=[0, 1, 2, 3, 5, 6, 7, 8])
    where = (b["cells"] >= 0).long().argmax(1)
    assert int((where == 4).sum()) == 0 and len(set(where.tolist())) == 8


@pytest.mark.parametrize("read", ["embedding", "rotary"])
def test_a_rotary_or_embedding_engine_reads_its_request_and_never_a_masked_token(read):
    e = _tiny("addressed_thing", 3, 0, read)
    b = C.kind_batch("addressed_thing", 3, BED, 4, _gen(11))
    z, t, _ = C.noisy(b["x"], _gen(12))
    with torch.no_grad():
        v = C.engine_forward(e, b, z, t)
        own = b["loss_mask"]
        # another color in the request changes the answer on the own cell
        b2 = dict(b, colors=torch.where(b["cells"] >= 0, (b["colors"] + 1) % 9, b["colors"]))
        v2 = C.engine_forward(e, b2, z, t)
        from geolip.modular_diffusion.spectral import patchify
        assert not torch.equal(patchify(v, 4)[own], patchify(v2, 4)[own])
        # the content of a masked token changes no visible token's answer
        hid_tok = ~b["key_mask"]
        z3 = z.clone()
        zt = patchify(z3, 4)
        zt[hid_tok] += 3.0
        from geolip.modular_diffusion.spectral import unpatchify
        z3 = unpatchify(zt, (3, 48, 48), 4, 12)
        v3 = C.engine_forward(e, b, z3, t)
        vis = b["key_mask"]
        assert torch.equal(patchify(v, 4)[vis], patchify(v3, 4)[vis])
    if read == "rotary":
        # the request token's cell digits matter: the same thing told at another cell gives another answer
        moved = b["cells"].clone()
        own_c = (moved >= 0).long().argmax(1)
        tgt = (own_c + 1) % 9
        cells2 = torch.full_like(moved, -1)
        cols2 = torch.full_like(moved, -1)
        r = torch.arange(4)
        cells2[r, tgt] = moved[r, own_c]
        cols2[r, tgt] = b["colors"][r, own_c]
        told2 = F.one_hot(tgt, 9).bool()
        with torch.no_grad():
            v4 = e(z, t, cells2, cols2, told2, None)
            v5 = e(z, t, b["cells"], b["colors"], b["told"], None)
        assert not torch.equal(v4, v5)


# ------------------------------------------------------------------ the collective
def _collective(seed=0, read="embedding", addressed=True, classes=range(9), swap=None):
    kinds = ("addressed_thing", "addressed_base") if addressed else ("whole_thing", "whole_empty")
    things = {k: _tiny(kinds[0], k, seed if swap != k else seed + 50, read) for k in classes}
    return C.Collective(things, _tiny(kinds[1], None, seed, read), addressed=addressed,
                        unconditional=None if addressed else _tiny("whole_unconditional", None, seed, read))


def test_the_gate_join_equals_a_slow_per_token_reference():
    col = _collective()
    rl = scenes.request_lists(3, count=5, seed=77)
    cells, colors = rl.cells, rl.colors
    z = torch.randn(5, 3, 48, 48, generator=_gen(13))
    t = torch.full((5,), 0.4)
    from geolip.modular_diffusion.spectral import patchify
    with torch.no_grad():
        got = patchify(col.answer(z, t, cells, colors, "gate"), 4)
        own = frame.token_owner(cells, col.owners())
        fr = frame.TokenFrame()
        for b in range(5):
            for tok in range(144):
                o = int(own[b, tok])
                c = int(fr.cell[tok])
                if o == 9:
                    km = frame.visible_keys(cells[b:b + 1], torch.tensor([-1]))
                    ref = col.base(z[b:b + 1], t[b:b + 1], cells[b:b + 1], colors[b:b + 1],
                                   torch.zeros(1, 9, dtype=torch.bool), km)
                else:
                    km = frame.visible_keys(cells[b:b + 1], torch.tensor([c]))
                    ref = col.things[o](z[b:b + 1], t[b:b + 1], cells[b:b + 1], colors[b:b + 1],
                                        F.one_hot(torch.tensor([c]), 9).bool(), km)
                assert torch.allclose(got[b, tok], patchify(ref, 4)[0, tok], atol=1e-6)


def test_the_sum_with_one_request_is_the_part_and_with_none_the_empty_engine():
    col = _collective(addressed=False)
    rl = scenes.request_lists(1, count=6, seed=78)
    z = torch.randn(6, 3, 48, 48, generator=_gen(14))
    t = torch.full((6,), 0.6)
    with torch.no_grad():
        s = col.answer(z, t, rl.cells, rl.colors, "sum_empty")
        own = torch.zeros_like(s)
        for k, (rows, cols, v) in col.pair_answers(z, t, rl.cells, rl.colors).items():
            own[rows] = v
        assert float((s - own).abs().max()) < 1e-5
        empty = torch.full((6, 9), -1, dtype=torch.long)
        s0 = col.answer(z, t, empty, empty, "sum_empty")
        assert torch.equal(s0, col.background(col.base, z, t, empty, empty))
        u = col.answer(z, t, rl.cells, rl.colors, "sum_unconditional")
        assert not torch.equal(u, s)


def _sample(col, cells, colors, join="gate", steps=5):
    return C.sample(lambda z, t, c, k: col.answer(z, t, c, k, join), cells, colors, seed=7004, steps=steps)


@pytest.mark.parametrize("read", ["embedding", "rotary"])
def test_exchanging_one_class_changes_no_pixel_outside_its_cells(read):
    rl = scenes.request_lists(4, count=12, seed=79)
    cells, colors = rl.cells, rl.colors
    k = int(cells.gather(1, rl.order[:, :1])[0, 0])
    a = _sample(_collective(read=read), cells, colors)
    b = _sample(_collective(read=read, swap=k), cells, colors)
    inside = (cells == k)
    from geolip.modular_diffusion.scenes import cell_crops
    d = cell_crops((a - b).abs())
    assert float(d[~inside].max()) == 0.0
    assert float(d[inside].max()) > 0.0
    # adding an engine nobody asks for changes nothing
    asked = sorted({int(x) for x in cells[cells >= 0].tolist()})
    missing = [c for c in range(9) if c not in asked]
    if missing:
        fewer = _collective(read=read, classes=[c for c in range(9) if c != missing[0]])
        assert torch.equal(_sample(fewer, cells, colors), a)
    # the sampler repeats bit for bit
    assert torch.equal(_sample(_collective(read=read), cells, colors), a)


def test_the_same_exchange_on_whole_picture_parts_changes_pixels_outside():
    rl = scenes.request_lists(4, count=12, seed=79)
    cells, colors = rl.cells, rl.colors
    k = int(cells.gather(1, rl.order[:, :1])[0, 0])
    for join in ("gate", "sum_empty"):
        a = _sample(_collective(addressed=False), cells, colors, join)
        b = _sample(_collective(addressed=False, swap=k), cells, colors, join)
        from geolip.modular_diffusion.scenes import cell_crops
        d = cell_crops((a - b).abs())
        assert float(d[cells != k].max()) > 0.0, join


def test_a_collective_of_mixed_read_forms_is_refused():
    with pytest.raises(ValueError):
        C.Collective({0: _tiny("addressed_thing", 0, 0, "rotary")}, _tiny("addressed_base", None, 0), addressed=True)


# ------------------------------------------------------------------ the trainer
@pytest.mark.parametrize("read", ["embedding", "rotary"])
def test_a_30_step_training_lowers_the_loss_on_its_own_fixed_batch(read):
    e = C.born("whole_thing", 1, 0, 128 if read == "rotary" else 32, 2, 1, read)
    b = C.kind_batch("whole_thing", 1, BED, 16, _gen(15))
    z, t, tg = C.noisy(b["x"], _gen(16))
    opt = torch.optim.Adam(e.parameters(), lr=1e-3)
    first = None
    for s in range(30):
        loss = C.masked_loss(C.engine_forward(e, b, z, t), tg, b["loss_mask"])
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first


def test_train_engine_writes_reads_resumes_bit_for_bit_and_round_trips(tmp_path):
    kw = dict(steps=6, batch=4, heldout=8, every=3, dim=32, heads=2, depth=1, save_every=0, log=lambda s: None)
    whole = C.train_engine("addressed_thing", 2, BED, 0, str(tmp_path / "whole"), **kw)
    assert whole["finished"] and len(whole["curve"]) == 3 and whole["curve"][0]["step"] == 0
    assert isinstance(whole["descended"], bool) and whole["trained"] == C.shape_numbers(32, 2, 1)
    part = C.train_engine("addressed_thing", 2, BED, 0, str(tmp_path / "part"), until=4, **kw)
    assert not part["finished"]
    rest = C.train_engine("addressed_thing", 2, BED, 0, str(tmp_path / "part"), **kw)
    assert rest["finished"]
    a = C.load_engine(str(tmp_path / "whole"))
    b = C.load_engine(str(tmp_path / "part"))
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n
    last_a = C.load_engine(str(tmp_path / "whole"), average=False)
    assert torch.equal(last_a.embed.weight, whole["engine"].embed.weight)
    assert torch.equal(a.embed.weight, whole["average"]["embed.weight"])
    again = C.train_engine("addressed_thing", 2, BED, 0, str(tmp_path / "whole"), **kw)
    assert again.get("restored") and again["last_read"] == whole["last_read"]
    assert [x["own_read"] for x in rest["curve"]] == [x["own_read"] for x in whole["curve"]]


def test_the_depth_that_reaches_the_ten_parts():
    ten = 10 * C.shape_numbers(128, 2, 3)
    d = C.depth_to_reach(ten, 256, 4)
    assert C.shape_numbers(256, 4, d) >= ten > C.shape_numbers(256, 4, d - 1)
    assert d == 8
