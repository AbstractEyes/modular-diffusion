"""The scene reads on hand-made cases and on real scenes of the bed: the scores, the neighbor groups, the paired
difference, the tint spread and the coherence index, the seam, the ghost ink, the wrong-address corruption and the
derangement."""
import math

import pytest
import torch

from geolip.modular_diffusion import scene_reads as sr
from geolip.modular_diffusion import scenes


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _layout(pairs):
    """{cell: (thing, color)} -> cells [1, 9], colors [1, 9]."""
    cells = torch.full((1, 9), -1, dtype=torch.long)
    colors = torch.full((1, 9), -1, dtype=torch.long)
    for c, (t, k) in pairs.items():
        cells[0, c], colors[0, c] = t, k
    return cells, colors


def _perfect(cells, colors):
    return torch.where(cells >= 0, cells, torch.full_like(cells, sr.NOTHING)), colors.clamp_min(0)


def test_scores_on_hand_made_cases():
    cells, colors = _layout({0: (1, 2), 4: (5, 0), 8: (8, 8)})
    pt, pc = _perfect(cells, colors)
    s = sr.score_cells(pt, pc, cells, colors)
    assert int(s["n_right"]) == 3 and bool(s["scene_right"]) and bool(s["scene_right_colors"])
    assert int(s["empty_right"].sum()) == 6 and int(s["thing_right"].sum()) == 3
    # one wrong thing
    w = pt.clone(); w[0, 4] = 6
    s = sr.score_cells(w, pc, cells, colors)
    assert int(s["n_right"]) == 2 and not bool(s["scene_right"]) and not bool(s["scene_right_colors"])
    assert bool(s["color_right"][0, 4])                                    # the color read is separate
    # one ghost: an empty cell read as a thing
    g = pt.clone(); g[0, 1] = 0
    s = sr.score_cells(g, pc, cells, colors)
    assert int(s["n_right"]) == 3 and not bool(s["scene_right"]) and not bool(s["empty_right"][0, 1])
    summ = sr.summarize(g, pc, cells, colors)
    assert summ["ghosts"]["mean"] == pytest.approx(1 / 6) and summ["ghosts"]["n"] == 6
    assert summ["cell_rate"]["mean"] == 1.0 and summ["cell_rate"]["n"] == 3 and summ["cell_rate"]["se"] == 0.0
    # a wrong color
    k = pc.clone(); k[0, 8] = 3
    s = sr.score_cells(pt, k, cells, colors)
    assert bool(s["scene_right"]) and not bool(s["scene_right_colors"])
    summ = sr.summarize(pt, k, cells, colors)
    assert summ["color_rate"]["mean"] == pytest.approx(2 / 3)
    assert summ["color_rate"]["se"] == pytest.approx(math.sqrt((2 / 3) * (1 / 3) / 3))
    # nine things
    cells9, colors9 = _layout({c: (c, 8 - c) for c in range(9)})
    pt9, pc9 = _perfect(cells9, colors9)
    s = sr.score_cells(pt9, pc9, cells9, colors9)
    assert int(s["n_right"]) == 9 and bool(s["scene_right_colors"])
    pt9[0, 3] = sr.NOTHING
    s = sr.score_cells(pt9, pc9, cells9, colors9)
    assert int(s["n_right"]) == 8 and not bool(s["scene_right"])
    summ = sr.summarize(pt9, pc9, cells9, colors9)
    assert summ["ghosts"]["n"] == 0 and summ["ghosts"]["mean"] is None
    assert summ["histogram"] == [0] * 8 + [1, 0]
    # an empty scene is right when every cell reads NOTHING
    e_cells, e_colors = _layout({})
    s = sr.score_cells(torch.full((1, 9), 9), torch.zeros(1, 9, dtype=torch.long), e_cells, e_colors)
    assert bool(s["scene_right"]) and int(s["n_requested"]) == 0


def test_summarize_rates_and_histogram_on_a_batch():
    lists = scenes.request_lists(4, count=200)
    pt, pc = _perfect(lists.cells, lists.colors)
    pt = pt.clone()
    pt[:100, :] = torch.where(lists.cells[:100] >= 0, (lists.cells[:100] + 1) % 9, pt[:100])   # half the scenes all wrong
    summ = sr.summarize(pt, pc, lists.cells, lists.colors)
    assert summ["cell_rate"]["mean"] == 0.5 and summ["cell_rate"]["n"] == 800
    assert summ["cell_rate"]["se"] == pytest.approx(math.sqrt(0.25 / 800))
    assert summ["scene_right"]["mean"] == 0.5 and summ["scenes"] == 200
    assert summ["histogram"][0] == 100 and summ["histogram"][4] == 100 and sum(summ["histogram"]) == 200


def test_neighbor_groups_on_hand_made_layouts():
    # cell 4 (the center) has edge neighbors 1, 3, 5, 7
    cases = {
        "same": {4: (6, 0), 1: (6, 1), 5: (0, 0)},
        "family": {4: (6, 0), 3: (7, 1), 7: (0, 0)},
        "other": {4: (6, 0), 5: (0, 1), 7: (3, 0), 0: (6, 2)},      # cell 0 is a corner neighbor only: not counted
        "none": {4: (6, 0), 0: (6, 1), 8: (7, 0)},
    }
    for name, pairs in cases.items():
        cells, colors = _layout(pairs)
        g = sr.neighbor_groups(cells)
        assert sr.NEIGHBOR_GROUPS[int(g[0, 4])] == name, (name, g)
        assert int(g[0, 2]) == -1
    cells, colors = _layout(cases["family"])
    pt, pc = _perfect(cells, colors)
    pt[0, 4] = 0
    summ = sr.summarize(pt, pc, cells, colors)["by_neighbor"]
    # cell 4 (wrong) and cell 3 (a cross beside the plus) are "family"; cell 7 (a disc beside the plus) is "other"
    assert summ["family"]["n"] == 2 and summ["family"]["mean"] == 0.5
    assert summ["other"]["n"] == 1 and summ["other"]["mean"] == 1.0
    assert summ["same"]["n"] == 0 and summ["same"]["mean"] is None


def test_paired_difference_on_a_hand_made_pair():
    a = torch.tensor([1, 1, 0, 1, 0, 1])
    b = torch.tensor([1, 0, 0, 0, 0, 1])
    d = sr.paired_difference(a, b)
    diffs = [0, 1, 0, 1, 0, 0]
    m = sum(diffs) / 6
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / 5)
    assert d["mean"] == pytest.approx(1 / 3) and d["se"] == pytest.approx(sd / math.sqrt(6)) and d["n"] == 6
    same = sr.paired_difference(a, a)
    assert same["mean"] == 0.0 and same["se"] == 0.0
    with pytest.raises(ValueError):
        sr.paired_difference(a, b[:3])


def test_tint_spread_and_the_coherence_index():
    f = scenes.ScenesBed("F").train_batch("mix", 64, _gen(1)).pixels
    assert float(sr.tint_spread(f).abs().max()) == 0.0                        # corner patches are the gray ground
    cb = scenes.ScenesBed("C").train_batch("mix", 256, _gen(2))
    real = sr.tint_spread(cb.pixels)
    shuffled = sr.tint_spread(sr.shuffle_cells(cb.pixels))
    r, s = float(real.mean()), float(shuffled.mean())
    assert r < 0.03 and s > 1.5 * r, (r, s)                                  # small within a scene, larger across scenes
    assert sr.coherence_index(r, r, s) == 0.0 and sr.coherence_index(s, r, s) == 1.0
    # the shuffle: cell c of scene b is cell c of scene (b + c) mod B
    x = torch.arange(10).float().view(10, 1, 1, 1).expand(10, 3, 48, 48).contiguous()
    y, cells = sr.shuffle_cells(x, torch.arange(9).repeat(10, 1))
    cr = scenes.cell_crops(y)
    for b in range(10):
        assert [int(cr[b, c, 0, 0, 0]) for c in range(9)] == [(b + c) % 10 for c in range(9)]
    with pytest.raises(ValueError):
        sr.shuffle_cells(x[:8])


def test_seam_reads_one_on_real_textured_scenes_and_high_on_a_planted_step():
    cb = scenes.ScenesBed("C").train_batch("empty", 256, _gen(3))
    pix = cb.pixels
    s = sr.seam(pix, cb.cells)
    assert s["base_base"]["n"] == 256 * 12 and s["thing_base"]["n"] == 0
    assert abs(s["base_base"]["ratio"] - 1.0) < 0.2, s["base_base"]
    # plant a step of 0.2 at the border between cells 0 and 1, marked as two things' cells
    planted = pix.clone()
    planted[:, :, 0:16, 0:16] += 0.2
    cells = cb.cells.clone()
    cells[:, 0] = 0
    cells[:, 1] = 0
    s2 = sr.seam(planted, cells)
    assert s2["thing_thing"]["n"] == 256 and s2["thing_thing"]["ratio"] > 2.0, s2["thing_thing"]
    assert s2["base_base"]["ratio"] == pytest.approx(s["base_base"]["ratio"], rel=0.2)
    # on real scenes with things, the seam pixels are ground: every border type reads near one
    mb = scenes.ScenesBed("C").train_batch("mix", 512, _gen(4))
    s3 = sr.seam(mb.pixels, mb.cells)
    for name in sr.BORDER_TYPES:
        assert abs(s3[name]["ratio"] - 1.0) < 0.2, (name, s3[name])


def test_ghost_ink():
    fb = scenes.ScenesBed("F").train_batch("mix", 128, _gen(5))
    g = sr.ghost_ink(fb.pixels, fb.cells)
    assert g["share"] == 0.0 and g["n"] == int((fb.cells < 0).sum()) * 256
    pix = fb.pixels.clone()
    free = torch.nonzero(fb.cells[0] < 0, as_tuple=True)[0]
    c = int(free[0])
    y0, x0 = 16 * (c // 3), 16 * (c % 3)
    pix[0, 2, y0 + 6:y0 + 10, x0 + 6:x0 + 10] = 0.75                         # a 4 x 4 blob in one channel
    g2 = sr.ghost_ink(pix, fb.cells)
    assert g2["share"] == pytest.approx(16 / g["n"]) and float(g2["per_scene"][0]) > 0
    # ink in a requested cell is not a ghost
    pix2 = fb.pixels.clone()
    req = torch.nonzero(fb.cells[0] >= 0, as_tuple=True)[0]
    if req.numel():
        c = int(req[0])
        pix2[0, :, 16 * (c // 3):16 * (c // 3) + 16, 16 * (c % 3):16 * (c % 3) + 16] = 0.0
        assert sr.ghost_ink(pix2, fb.cells)["share"] == 0.0


def test_corrupt_requests():
    lists = [scenes.request_lists(n, count=1000) for n in range(1, 10)]
    cells = torch.cat([l.cells for l in lists])
    colors = torch.cat([l.colors for l in lists])
    out = sr.corrupt_requests(cells, colors, 0.0, _gen(1))
    assert torch.equal(out.cells, cells) and torch.equal(out.colors, colors)
    assert bool((out.action[cells >= 0] == 0).all()) and bool((out.action[cells < 0] == -1).all())
    out = sr.corrupt_requests(cells, colors, 1.0, _gen(2))
    req = cells >= 0
    acts = out.action[req]
    stuck = int((acts == 4).sum())
    assert int((acts == 0).sum()) == 0
    n9 = (req.sum(dim=1) == 9)
    assert bool((out.action[~n9][req[~n9]] != 4).all())                       # with a free cell, every request changes
    counts = torch.bincount(acts, minlength=5).tolist()
    n_req = int(req.sum())
    moved_like = counts[1] + counts[4]
    for share in (moved_like, counts[2], counts[3]):
        assert abs(share / n_req - 1 / 3) < 0.02, counts
    assert n_req >= 9000
    # what every action did, request by request
    for i in range(cells.shape[0]):
        cur = out.cells[i].tolist()
        for c in range(9):
            a = int(out.action[i, c])
            if a == 2:
                assert cur[c] >= 0 and cur[c] != int(cells[i, c]) and int(out.colors[i, c]) == int(colors[i, c])
            elif a == 1:
                d = int(out.moved_to[i, c])
                # it landed on a cell free in the list at that moment: empty from the start, or emptied by an earlier
                # request's drop or move (requests are treated in cell order)
                assert d != c
                assert int(cells[i, d]) < 0 or (d < c and int(out.action[i, d]) in (1, 3)), (i, c, d)
                assert int(out.cells[i, d]) == int(cells[i, c]) and int(out.colors[i, d]) == int(colors[i, c])
    assert int((out.cells >= 0).sum()) == n_req - counts[3]
    # a stuck move leaves the request as it is
    full_rows = torch.nonzero(n9, as_tuple=True)[0]
    for i in full_rows[:200].tolist():
        for c in range(9):
            if int(out.action[i, c]) == 4:
                assert int(out.cells[i, c]) == int(cells[i, c])
    assert stuck > 0
    # the corruption repeats from its generator, and an intermediate rate corrupts about that share
    a = sr.corrupt_requests(cells, colors, 0.2, _gen(3))
    b = sr.corrupt_requests(cells, colors, 0.2, _gen(3))
    assert torch.equal(a.cells, b.cells) and torch.equal(a.action, b.action)
    share = float((a.action[req] != 0).double().mean())
    assert abs(share - 0.2) < 0.02


def test_derange_classes_moves_every_class():
    cells = torch.tensor([list(range(9)), [-1] * 9])
    d = sr.derange_classes(cells)
    assert d[0].tolist() == [(k + 4) % 9 for k in range(9)]
    assert bool((d[0] != cells[0]).all()) and d[1].tolist() == [-1] * 9
    assert sorted(d[0].tolist()) == list(range(9))
