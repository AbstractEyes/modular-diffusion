"""The address frame on image tokens: the digit codes are orthonormal, the token frame's digits, requests both ways,
the gate by integers equals the gate by code kernels, one owner a token, the ghost cells a product of per-digit sums
would light, the base's cells, the keys a part may read."""
import pytest
import torch

from geolip.modular_diffusion import frame, scenes
from geolip.modular_diffusion.spectral import patchify

PARTS = [[t] for t in range(9)]                 # one part a thing, the base last (index 9)


def _layouts(n, seed):
    cells, colors = scenes.draw_layouts("mix", n, torch.Generator().manual_seed(seed))
    return cells, colors


def _cell_of_tokens(cell):
    fr = frame.TokenFrame()
    return fr.cell == cell


def test_digit_codes_are_orthonormal_and_an_even_base_raises():
    for b in (1, 3, 5, 7):
        u = frame.digit_code(b)
        assert u.dtype == torch.float64 and u.shape == (b, b)
        assert float((u @ u.T - torch.eye(b, dtype=torch.float64)).abs().max()) < 1e-12
        assert float((u.T @ u - torch.eye(b, dtype=torch.float64)).abs().max()) < 1e-12
    u3 = frame.digit_code(3)
    assert abs(float(u3[0, 0]) - 3 ** -0.5) < 1e-15 and abs(float(u3[1, 1]) - (2 / 3) ** 0.5 * -0.5) < 1e-15
    for b in (2, 4, 0, -3):
        with pytest.raises(ValueError):
            frame.digit_code(b)


def test_the_token_frame_agrees_with_integer_division():
    fr = frame.TokenFrame(12, 3)
    assert fr.n_tokens == 144 and fr.side == 4 and fr.inner_base == 5
    for t in range(144):
        iy, ix = divmod(t, 12)
        assert int(fr.column[t]) == ix // 4 and int(fr.row[t]) == iy // 4
        assert int(fr.inner_x[t]) == ix % 4 and int(fr.inner_y[t]) == iy % 4
        assert int(fr.cell[t]) == 3 * (iy // 4) + ix // 4
    code = fr.place_code()
    assert code.shape == (144, 16) and code.dtype == torch.float32
    u3, u5 = frame.digit_code(3).float(), frame.digit_code(5).float()
    t = 12 * 7 + 9                                                       # iy 7, ix 9: column 2, row 1, inner 1, 3
    assert torch.equal(code[t], torch.cat([u3[2], u3[1], u5[1], u5[3]]))
    assert len({tuple(r) for r in code.tolist()}) == 144                 # every token has its own place
    # the token order is patchify's: a picture whose pixels carry their cell index reads its cell on every token
    x = torch.zeros(1, 1, 48, 48)
    for c in range(9):
        x[0, 0, 16 * (c // 3):16 * (c // 3) + 16, 16 * (c % 3):16 * (c % 3) + 16] = c
    tok = patchify(x, 4)[0, :, 0]
    assert torch.equal(tok.long(), fr.cell)
    with pytest.raises(ValueError):
        frame.TokenFrame(12, 4)
    with pytest.raises(ValueError):
        frame.TokenFrame(10, 3)


def test_requests_both_ways():
    cells, colors = _layouts(300, 1)
    for i in range(300):
        reqs = frame.layout_requests(cells[i], colors[i])
        assert len(reqs) == int((cells[i] >= 0).sum())
        c2, k2 = frame.requests_layout(reqs)
        assert torch.equal(c2, cells[i]) and torch.equal(k2, colors[i])
    r = frame.request_of(5, 7, 4)                     # cell 5 = column 2, row 1; cross = family 2 member 1; blue 1, 1
    assert r == (2, 1, 2, 1, 1, 1)
    with pytest.raises(ValueError):
        frame.requests_layout([(0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0)])
    with pytest.raises(ValueError):
        frame.requests_layout([(0, 0, 3, 0, 0, 0)])


def test_request_codes():
    u3 = frame.digit_code(3).float()
    r = frame.request_of(5, 7, 4)
    code = frame.request_code(r)
    assert code.shape == (18,) and code.dtype == torch.float32
    assert torch.equal(code, torch.cat([u3[d] for d in r]))
    assert torch.equal(frame.content_code(7, 4), code[6:]) and frame.content_code(7, 4).shape == (12,)
    assert torch.equal(frame.request_code(frame.request_of(5, 7, 4)), code)          # equal requests agree
    seen = {}
    for c in range(9):
        for t in range(9):
            for k in range(9):
                key = tuple(frame.request_code(frame.request_of(c, t, k)).tolist())
                assert key not in seen, (c, t, k, seen.get(key))                     # different requests differ
                seen[key] = (c, t, k)
    assert len(seen) == 729
    cells, colors = _layouts(50, 2)
    codes, mask = frame.layout_codes(cells, colors)
    assert codes.shape == (50, 9, 18) and torch.equal(mask, cells >= 0)
    for i in range(50):
        for c in range(9):
            if cells[i, c] >= 0:
                assert torch.equal(codes[i, c], frame.request_code(frame.request_of(c, cells[i, c], colors[i, c])))
            else:
                assert float(codes[i, c].abs().max()) == 0.0


def test_the_kernel_gate_equals_the_integer_gate_on_1000_layouts():
    cells, _ = _layouts(1000, 3)
    wi = frame.token_weights(cells, PARTS, "integer")
    wk = frame.token_weights(cells, PARTS, "kernel")
    assert wi.dtype == torch.float64 and wk.dtype == torch.float64 and wi.shape == (1000, 144, 10)
    err = float((wi - wk).abs().max())
    assert err < 1e-12, err
    # a coarser roster (three family parts) as well
    fam = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert float((frame.token_weights(cells, fam, "integer") - frame.token_weights(cells, fam, "kernel")).abs().max()) < 1e-12


def test_every_token_has_exactly_one_owner_at_weight_exactly_one():
    cells, _ = _layouts(1000, 4)
    w = frame.token_weights(cells, PARTS, "integer")
    own = frame.token_owner(cells, PARTS)
    assert own.shape == (1000, 144) and own.dtype == torch.long
    assert bool(((w == 1.0).sum(dim=-1) == 1).all())
    assert bool(((w == 0.0) | (w == 1.0)).all())                           # a part off its address: exactly 0.0
    assert torch.equal(w, torch.nn.functional.one_hot(own, 10).double())
    fr = frame.TokenFrame()
    for i in range(50):
        for t in range(144):
            c = int(fr.cell[t])
            want = int(cells[i, c]) if int(cells[i, c]) >= 0 else 9
            assert int(own[i, t]) == want


def test_two_requests_of_one_class_light_their_two_cells_and_no_ghost():
    cells = torch.full((1, 9), -1, dtype=torch.long)
    cells[0, 0] = 6                                  # a plus at column 0, row 0
    cells[0, 8] = 6                                  # a plus at column 2, row 2
    w = frame.token_weights(cells, PARTS, "integer")[0]
    lit = {int(c) for c in frame.TokenFrame().cell[w[:, 6] == 1.0].tolist()}
    assert lit == {0, 8}
    assert float(w[:, [i for i in range(9) if i != 6]].abs().max()) == 0.0
    assert {int(c) for c in frame.TokenFrame().cell[w[:, 9] == 1.0].tolist()} == {1, 2, 3, 4, 5, 6, 7}
    # THE WRONG FORM: a product of per-digit sums over the two requests also lights cells (column 2, row 0) and
    # (column 0, row 2), the two ghost cells
    fr = frame.TokenFrame()
    u = frame.digit_code(3)
    reqs = frame.layout_requests(cells[0], torch.tensor([0, -1, -1, -1, -1, -1, -1, -1, 0]))
    col_sum = sum(u[fr.column] @ u[r[0]] for r in reqs)
    row_sum = sum(u[fr.row] @ u[r[1]] for r in reqs)
    wrong = col_sum * row_sum                                                   # [T]
    ghost = {int(c) for c in fr.cell[wrong > 0.5].tolist()}
    assert ghost == {0, 2, 6, 8}
    right = sum((u[fr.column] @ u[r[0]]) * (u[fr.row] @ u[r[1]]) for r in reqs)
    assert {int(c) for c in fr.cell[right > 0.5].tolist()} == {0, 8}


def test_the_base_owns_exactly_the_cells_without_a_request():
    cells, _ = _layouts(500, 5)
    own = frame.token_owner(cells, PARTS)
    fr = frame.TokenFrame()
    base = own == 9
    free_tok = (cells < 0)[:, fr.cell]
    assert torch.equal(base, free_tok)
    empty = torch.full((3, 9), -1, dtype=torch.long)
    assert bool((frame.token_owner(empty, PARTS) == 9).all())
    full = torch.arange(9).repeat(2, 1)
    assert not bool((frame.token_owner(full, PARTS) == 9).any())


def test_an_unowned_thing_raises_with_its_name_and_a_shared_thing_raises():
    cells = torch.full((2, 9), -1, dtype=torch.long)
    cells[1, 3] = 8
    without_star = [[t] for t in range(8)]
    with pytest.raises(ValueError, match="star"):
        frame.token_owner(cells, without_star)
    with pytest.raises(ValueError, match="star"):
        frame.token_weights(cells, without_star, "kernel")
    no_star_cells = cells.clone()
    no_star_cells[1, 3] = 2
    own = frame.token_owner(no_star_cells, without_star)                     # no star asked: fine
    assert int(own.max()) == 8                                               # the base is index 8 of eight parts
    with pytest.raises(ValueError):
        frame.token_owner(cells, [[0, 1], [1, 2]] + [[t] for t in range(3, 9)])


def test_visible_keys():
    fr = frame.TokenFrame()
    cells, _ = _layouts(400, 6)
    for i in range(400):
        req = [c for c in range(9) if int(cells[i, c]) >= 0]
        free = [c for c in range(9) if int(cells[i, c]) < 0]
        for c in req:
            vis = frame.visible_keys(cells[i:i + 1], torch.tensor([c]))[0]
            seen = {int(x) for x in fr.cell[vis].tolist()}
            assert seen == set(free) | {c}
            assert int(vis.sum()) == 16 * (len(free) + 1)
        vb = frame.visible_keys(cells[i:i + 1], torch.tensor([-1]))[0]
        assert {int(x) for x in fr.cell[vb].tolist()} == set(free)
    # hidden cells of training are invisible: a ONE scene with the thing at cell 4 and cells 0, 2, 7 hidden
    cells = torch.full((1, 9), -1, dtype=torch.long)
    cells[0, 4] = 3
    hidden = torch.zeros(1, 9, dtype=torch.bool)
    hidden[0, [0, 2, 7]] = True
    vis = frame.visible_keys(cells, torch.tensor([4]), hidden)[0]
    assert {int(x) for x in fr.cell[vis].tolist()} == {1, 3, 4, 5, 6, 8}
    vb = frame.visible_keys(torch.full((1, 9), -1, dtype=torch.long), torch.tensor([-1]), hidden)[0]
    assert {int(x) for x in fr.cell[vb].tolist()} == {1, 3, 4, 5, 6, 8}
    with pytest.raises(ValueError):
        frame.visible_keys(cells, torch.tensor([0]))                      # cell 0 holds no request
    hidden[0, 4] = True
    with pytest.raises(ValueError):
        frame.visible_keys(cells, torch.tensor([4]), hidden)
