"""The scene bed: draws repeat, things stay inside their cells and off the corner and seam pixels, shapes keep their
radius and their area, the classes differ, colors land, bed C's texture and tint, the kinds' statistics, model space,
the request lists."""
import math

import pytest
import torch

from geolip.modular_diffusion import scenes
from geolip.modular_diffusion.scene_reads import BORDERS


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _outside_own_cell(cell):
    """[48, 48] bool: True outside cell ``cell``."""
    m = torch.ones(48, 48, dtype=torch.bool)
    k, j = cell // 3, cell % 3
    m[16 * k:16 * k + 16, 16 * j:16 * j + 16] = False
    return m


def _reserved_pixels():
    """[48, 48] bool: every cell's four 2 x 2 corner patches and every pixel the seam read uses."""
    m = torch.zeros(48, 48, dtype=torch.bool)
    for k in range(3):
        for j in range(3):
            y0, x0 = 16 * k, 16 * j
            for dy in (0, 14):
                for dx in (0, 14):
                    m[y0 + dy:y0 + dy + 2, x0 + dx:x0 + dx + 2] = True
    for axis, a, b, x, along in BORDERS:
        for q in along:
            for d in (-2, -1, 0, 1):
                if axis == "v":
                    m[q, x + d] = True
                else:
                    m[x + d, q] = True
    return m


def test_same_generator_state_gives_the_same_batch():
    for bed in ("F", "C"):
        sb = scenes.ScenesBed(bed)
        a = sb.train_batch("mix", 64, _gen(3))
        b = sb.train_batch("mix", 64, _gen(3))
        assert torch.equal(a.x, b.x) and torch.equal(a.cells, b.cells) and torch.equal(a.colors, b.colors)
        for k in a.params:
            assert torch.equal(a.params[k], b.params[k]), k
        c = sb.train_batch("mix", 64, _gen(4))
        assert not torch.equal(a.x, c.x)
        h1, h2 = sb.heldout("composed", 16, 77, count=5), sb.heldout("composed", 16, 77, count=5)
        assert torch.equal(h1.x, h2.x)


def test_a_cuda_generator_is_refused():
    class OnCard:
        device = torch.device("cuda")
    with pytest.raises(ValueError):
        scenes.draw_layouts("mix", 4, OnCard())
    with pytest.raises(ValueError):
        scenes.draw_parameters(4, OnCard(), "F")


def test_things_never_leave_their_cell_nor_touch_corners_or_seam_pixels():
    reserved = _reserved_pixels()
    # 9 cells x 4 patches x 4 pixels; the seam read's pixels (4 rows x 4 pixels at 12 borders) all lie in them
    assert int(reserved.sum()) == 144
    total = 0
    for k in range(9):
        cells, colors = scenes.draw_layouts("one", 2000, _gen(100 + k), only_class=k)
        pix, p = scenes.render(cells, colors, "F", _gen(200 + k))
        own = (cells >= 0).long().argmax(dim=1)
        outside = torch.stack([_outside_own_cell(c) for c in range(9)])[own]      # [B, 48, 48]
        assert bool((pix.permute(1, 0, 2, 3)[:, outside] == 0.5).all()), k
        assert bool((pix[:, :, reserved] == 0.5).all()), k
        # and the thing is there: some ink inside its own cell in every scene
        assert bool(((pix - 0.5).abs().amax(dim=1)[~outside].reshape(2000, -1) > 0.05).any(dim=1).all()), k
        total += cells.shape[0]
    assert total == 18000


def test_a_full_canvas_window_changes_no_pixel_and_things_never_overlap():
    """The render on a window of the cell plus 16 px (the whole canvas for every thing) equals the default window bit
    for bit, and no pixel is covered by two things."""
    cells, colors = scenes.draw_layouts("composed", 300, _gen(5), count=9)
    a, _ = scenes.render(cells, colors, "C", _gen(6))
    b, _ = scenes.render(cells, colors, "C", _gen(6), margin=16)
    assert torch.equal(a, b)
    _, p = scenes.render(cells, colors, "F", _gen(6))
    cov = _coverage_full(cells, p)                               # [B, 9, 48, 48]
    assert int(((cov > 0).sum(dim=1) > 1).sum()) == 0
    for c in range(9):
        assert float(cov[:, c][:, _outside_own_cell(c)].abs().max()) == 0.0


def _coverage_full(cells, p):
    """Each thing's coverage over the whole canvas, computed apart from ``render`` from its draw."""
    b = cells.shape[0]
    out = torch.zeros(b, 9, 48, 48)
    xs = torch.arange(48).float() + 0.5
    for c in range(9):
        for t in range(9):
            rows = torch.nonzero(cells[:, c] == t, as_tuple=True)[0]
            if not rows.numel():
                continue
            cx, cy = p["centers"][rows, c, 0], p["centers"][rows, c, 1]
            a = torch.deg2rad(p["angle"][rows, c])
            dx = (xs[None] - cx[:, None])[:, None, :]
            dy = (xs[None] - cy[:, None])[:, :, None]
            px = torch.cos(a)[:, None, None] * dx + torch.sin(a)[:, None, None] * dy
            py = -torch.sin(a)[:, None, None] * dx + torch.cos(a)[:, None, None] * dy
            out[rows, c] = scenes.coverage(scenes.thing_distance(t, px, py, p["scale"][rows, c]))
    return out


def test_every_point_of_a_thing_lies_within_R_plus_a_half_of_its_center():
    worst = 0.0
    for k in range(9):
        cells, colors = scenes.draw_layouts("one", 300, _gen(300 + k), only_class=k)
        _, p = scenes.render(cells, colors, "F", _gen(400 + k))
        cov = _coverage_full(cells, p).sum(dim=1)              # one thing a scene
        cell = (cells >= 0).long().argmax(dim=1)
        rows = torch.arange(cells.shape[0])
        cx, cy = p["centers"][rows, cell, 0], p["centers"][rows, cell, 1]
        r = scenes.RADIUS * p["scale"][rows, cell]
        xs = torch.arange(48).float() + 0.5
        d = torch.sqrt((xs[None, None, :] - cx[:, None, None]) ** 2 + (xs[None, :, None] - cy[:, None, None]) ** 2)
        excess = torch.where(cov > 0, d - r[:, None, None], torch.full_like(d, -1e9))
        worst = max(worst, float(excess.max()))
    assert worst < 0.5 + 1e-4, worst
    for t in range(3, 9):                                       # every polygon corner lies at distance R or less
        assert float(scenes.polygon_corners(t).norm(dim=1).max()) <= 1.0 + 1e-12


def _one_at_center(t, scale, angle, color=4, cell=4, bed="F"):
    """One thing at a cell's center with given scale and angle (the render's draw replaced by hand)."""
    cells = torch.full((1, 9), -1, dtype=torch.long)
    colors = torch.full((1, 9), -1, dtype=torch.long)
    cells[0, cell], colors[0, cell] = t, color
    xs = torch.arange(48).float() + 0.5
    cx, cy = scenes.cell_centers()[cell]
    a = torch.deg2rad(torch.tensor([float(angle)]))
    dx = (xs - cx)[None, None, :]
    dy = (xs - cy)[None, :, None]
    px = torch.cos(a)[:, None, None] * dx + torch.sin(a)[:, None, None] * dy
    py = -torch.sin(a)[:, None, None] * dx + torch.cos(a)[:, None, None] * dy
    return scenes.coverage(scenes.thing_distance(t, px, py, torch.tensor([float(scale)])))[0]


def test_every_class_draws_at_least_12_pixels_at_the_smallest_scale():
    areas = [float(_one_at_center(t, 0.85, a).sum()) for t in range(9) for a in (-15, 0, 15)]
    assert min(areas) >= 12.0, areas
    # the analytic areas at R = 5.5 x 0.85: the disc pi R^2 within one pixel's rim
    r = 5.5 * 0.85
    assert abs(float(_one_at_center(0, 0.85, 0).sum()) - math.pi * r * r) < 2 * math.pi * r * 0.1


def test_the_nine_classes_are_pairwise_different_pictures():
    floor = 0.04                                                # mean absolute coverage difference over the cell
    for s in (0.85, 1.0, 1.15):
        for a in (-15.0, 0.0, 15.0):
            cov = torch.stack([_one_at_center(t, s, a)[16:32, 16:32] for t in range(9)])
            for i in range(9):
                for j in range(i + 1, 9):
                    d = float((cov[i] - cov[j]).abs().mean())
                    assert d > floor, (s, a, scenes.THINGS[i], scenes.THINGS[j], d)
    # the plus and the cross at the extreme angles (the cross turned back by 15 is 15 degrees from the plus at +15)
    for s in (0.85, 1.15):
        for ap, ac in ((15.0, -15.0), (-15.0, 15.0), (15.0, 15.0), (-15.0, -15.0)):
            d = float((_one_at_center(6, s, ap) - _one_at_center(7, s, ac)).abs().mean() * 9)
            assert d > floor, (s, ap, ac, d)


def test_colors_land_within_0_04_of_their_table_values_at_full_coverage():
    cells, colors = scenes.draw_layouts("composed", 400, _gen(8), count=9)
    pix, p = scenes.render(cells, colors, "F", _gen(9))
    cov = _coverage_full(cells, p)
    table = torch.tensor(scenes.COLOR_VALUES)
    worst, seen = 0.0, 0
    for c in range(9):
        full = cov[:, c] == 1.0                                  # [B, 48, 48]
        for i in torch.nonzero(full.any(dim=(1, 2)), as_tuple=True)[0].tolist():
            vals = pix[i][:, full[i]]                            # [3, m]
            want = table[colors[i, c]][:, None]
            worst = max(worst, float((vals - want).abs().max()))
            seen += 1
            assert torch.allclose(vals, p["color"][i, c][:, None].expand_as(vals), atol=0, rtol=0)
    assert seen > 3000 and worst <= 0.04 + 1e-6, (seen, worst)


def test_bed_C_texture_and_tint():
    sb = scenes.ScenesBed("C")
    batch = sb.train_batch("empty", 64, _gen(10))
    tex = batch.params["texture"]
    sd = tex.std(dim=(1, 2), correction=0)
    assert float((sd - 0.04).abs().max()) < 1e-4
    assert batch.params["tint"].shape == (64, 3)
    t = batch.params["tint"]
    assert float(t.min()) >= 0.40 and float(t.max()) < 0.60
    ground = batch.params["ground"]
    # one tint a scene, shared by every cell: ground minus the texture is the tint at every pixel
    assert torch.allclose(ground - tex[:, None], t[:, :, None, None].expand_as(ground), atol=1e-6)
    # the texture is one field for the three channels
    assert torch.equal(ground[:, 0] - t[:, 0, None, None], ground[:, 0] - t[:, 0, None, None])
    d01 = (ground[:, 0] - t[:, 0, None, None]) - (ground[:, 1] - t[:, 1, None, None])
    assert float(d01.abs().max()) < 1e-6
    # smooth: neighbors differ far less than the field's spread
    step = float((tex[:, :, 1:] - tex[:, :, :-1]).abs().mean())
    assert step < 0.25 * 0.04, step
    # empty scenes are the clipped ground
    assert torch.allclose(batch.pixels, ground.clamp(0, 1), atol=1e-6)


def test_kind_statistics():
    cells, _ = scenes.draw_layouts("mix", 9000, _gen(11))
    n = (cells >= 0).sum(dim=1)
    counts = torch.bincount(n, minlength=10)
    assert counts.numel() == 10 and int(counts.min()) > 0
    assert float((counts.float() / 9000 - 0.1).abs().max()) < 0.015, counts.tolist()
    for k in (1, 4, 9):
        cells, colors = scenes.draw_layouts("composed", 500, _gen(12 + k), count=k)
        assert bool(((cells >= 0).sum(dim=1) == k).all())
        assert bool(((cells >= 0) == (colors >= 0)).all())
    for k in range(9):
        cells, colors = scenes.draw_layouts("one", 200, _gen(30 + k), only_class=k)
        assert bool(((cells >= 0).sum(dim=1) == 1).all()) and bool((cells[cells >= 0] == k).all())
    cells, colors = scenes.draw_layouts("one", 9000, _gen(40))
    things = cells[cells >= 0]
    cnt = torch.bincount(things, minlength=9).float() / 9000
    assert float((cnt - 1 / 9).abs().max()) < 0.015
    where = (cells >= 0).long().argmax(dim=1)
    assert float((torch.bincount(where, minlength=9).float() / 9000 - 1 / 9).abs().max()) < 0.015
    cells, _ = scenes.draw_layouts("empty", 10, _gen(0))
    assert bool((cells == -1).all())
    with pytest.raises(ValueError):
        scenes.draw_layouts("composed", 4, _gen(0))
    with pytest.raises(ValueError):
        scenes.draw_layouts("one", 4, _gen(0), only_class=9)


def test_allowed_cells_restrict_one_and_leave_the_default_path_unchanged():
    # the default path draws exactly what part A drew: classes, cells (randint 0..8), colors
    g = _gen(60)
    cls = torch.randint(0, 9, (50,), generator=g)
    where = torch.randint(0, 9, (50,), generator=g)
    col = torch.randint(0, 9, (50,), generator=g)
    cells, colors = scenes.draw_layouts("one", 50, _gen(60))
    assert torch.equal(cells[torch.arange(50), where], cls) and torch.equal(colors[torch.arange(50), where], col)
    allowed = [0, 1, 2, 3, 5, 6, 7, 8]
    cells, colors = scenes.draw_layouts("one", 900, _gen(61), allowed_cells=allowed)
    w = (cells >= 0).long().argmax(1)
    assert int((w == 4).sum()) == 0 and sorted(set(w.tolist())) == allowed
    with pytest.raises(ValueError):
        scenes.draw_layouts("mix", 4, _gen(0), allowed_cells=allowed)
    with pytest.raises(ValueError):
        scenes.draw_layouts("one", 4, _gen(0), allowed_cells=[9])


def test_model_space_and_back_are_inverse():
    x = torch.rand(4, 3, 48, 48)
    assert torch.allclose(scenes.to_pixels(scenes.to_model(x)), x, atol=1e-7)
    assert float(scenes.to_model(torch.tensor(0.5))) == 0.0 and float(scenes.to_model(torch.tensor(0.75))) == 1.0
    b = scenes.ScenesBed("F").train_batch("mix", 8, _gen(1))
    assert torch.allclose(scenes.to_model(b.pixels), b.x, atol=1e-6)


def test_cell_crops_follow_cell_order_and_invert():
    x = torch.zeros(1, 1, 48, 48)
    for c in range(9):
        k, j = c // 3, c % 3
        x[0, 0, 16 * k:16 * k + 16, 16 * j:16 * j + 16] = c
    cr = scenes.cell_crops(x)
    assert cr.shape == (1, 9, 1, 16, 16)
    for c in range(9):
        assert bool((cr[0, c] == c).all())
    y = torch.rand(3, 3, 48, 48)
    assert torch.equal(scenes.from_cell_crops(scenes.cell_crops(y)), y)


def test_a_thing_is_drawn_where_its_layout_says():
    cells, colors = scenes.draw_layouts("composed", 200, _gen(50), count=3)
    pix, _ = scenes.render(cells, colors, "F", _gen(51))
    inked = ((scenes.cell_crops(pix) - 0.5).abs() > 0.05).any(dim=2).any(dim=(-2, -1))   # [B, 9]
    assert torch.equal(inked, cells >= 0)


def test_request_lists_are_fixed_by_their_seed():
    for n in (1, 2, 3, 4, 6, 9):
        a = scenes.request_lists(n)
        b = scenes.request_lists(n, seed=5000 + n)
        assert torch.equal(a.cells, b.cells) and torch.equal(a.colors, b.colors) and torch.equal(a.order, b.order)
        assert a.cells.shape == (500, 9) and a.colors.shape == (500, 9) and a.order.shape == (500, n)
        assert bool(((a.cells >= 0).sum(dim=1) == n).all())
        assert bool(((a.cells >= 0) == (a.colors >= 0)).all())
        assert all(len(set(r)) == n for r in a.order.tolist())
        assert bool((torch.gather(a.cells, 1, a.order) >= 0).all())
        c = scenes.request_lists(n, seed=1)
        assert not torch.equal(a.cells, c.cells)
    assert not torch.equal(scenes.request_lists(4).cells, scenes.request_lists(4, seed=5003).cells)


# ------------------------------------------------------------------ the render split into its draws and the drawing
def _reference_render(cells, colors, bed, gen, device="cpu", margin=scenes.MARGIN):
    """A copy of ``scenes.render`` as it was before the split into ``draw_parameters`` and ``render_drawn``."""
    S = scenes
    cells = torch.as_tensor(cells, dtype=torch.long)
    colors = torch.as_tensor(colors, dtype=torch.long)
    if cells.dim() != 2 or cells.shape[1] != S.N_CELLS or colors.shape != cells.shape:
        raise ValueError("a layout is cells [B, 9] and colors [B, 9]")
    if bool(((cells >= 0) != (colors >= 0)).any()):
        raise ValueError("a cell holds a color exactly when it holds a thing")
    if bool((cells >= len(S.THINGS)).any()) or bool((colors >= len(S.COLORS)).any()):
        raise ValueError("things and colors are 0..8")
    margin = int(margin)
    if not 0 <= margin <= S.CELL:
        raise ValueError("margin is 0..16")
    b = cells.shape[0]
    p = S.draw_parameters(b, gen, bed)
    p = {k: v.to(device) for k, v in p.items()}
    cells_d, colors_d = cells.to(device), colors.to(device)
    centers = S.cell_centers().to(device)
    p["centers"] = torch.stack([centers[:, 0] + p["shift_x"], centers[:, 1] + p["shift_y"]], dim=-1)
    table = torch.tensor(S.COLOR_VALUES, dtype=torch.float32, device=device)
    p["color"] = (table[colors_d.clamp_min(0)] + p["shade"]).clamp(0.0, 1.0)
    if bed == "F":
        ground = torch.full((b, S.CHANNELS, S.CANVAS, S.CANVAS), S.GROUND, dtype=torch.float32, device=device)
    else:
        p["texture"] = S.texture(p["noise"])
        ground = p["tint"][:, :, None, None] + p["texture"][:, None]
    p["ground"] = ground
    w = S.CELL + 2 * margin
    side = S.CANVAS + 2 * margin
    cov_sum = torch.zeros(b, side, side, dtype=torch.float32, device=device)
    colored = torch.zeros(b, S.CHANNELS, side, side, dtype=torch.float32, device=device)
    item_b, item_c = torch.nonzero(cells_d >= 0, as_tuple=True)
    if item_b.numel():
        things = cells_d[item_b, item_c]
        cx, cy = p["centers"][item_b, item_c, 0], p["centers"][item_b, item_c, 1]
        ang = torch.deg2rad(p["angle"][item_b, item_c])
        ca, sa = torch.cos(ang), torch.sin(ang)
        scale = p["scale"][item_b, item_c]
        local = torch.arange(w, dtype=torch.float32, device=device) - margin + 0.5
        xs = (item_c % S.CELLS).float()[:, None] * S.CELL + local[None]
        ys = (item_c // S.CELLS).float()[:, None] * S.CELL + local[None]
        dx = (xs - cx[:, None])[:, None, :]
        dy = (ys - cy[:, None])[:, :, None]
        px = ca[:, None, None] * dx + sa[:, None, None] * dy
        py = -sa[:, None, None] * dx + ca[:, None, None] * dy
        cov = torch.zeros(item_b.shape[0], w, w, dtype=torch.float32, device=device)
        for t in range(len(S.THINGS)):
            sel = torch.nonzero(things == t, as_tuple=True)[0]
            if sel.numel():
                cov[sel] = S.coverage(S.thing_distance(t, px[sel], py[sel], scale[sel]))
        col = p["color"][item_b, item_c]
        for c in range(S.N_CELLS):
            sel = torch.nonzero(item_c == c, as_tuple=True)[0]
            if sel.numel():
                bs = item_b[sel]
                y0, x0 = (c // S.CELLS) * S.CELL, (c % S.CELLS) * S.CELL
                cov_sum[bs, y0:y0 + w, x0:x0 + w] += cov[sel]
                colored[bs, :, y0:y0 + w, x0:x0 + w] += cov[sel][:, None] * col[sel][:, :, None, None]
    m = margin
    cs = cov_sum[:, m:m + S.CANVAS, m:m + S.CANVAS]
    pixels = colored[:, :, m:m + S.CANVAS, m:m + S.CANVAS] + (1.0 - cs)[:, None] * ground
    return pixels.clamp(0.0, 1.0), p


def _same_render(a, b):
    (pa, qa), (pb, qb) = a, b
    assert torch.equal(pa, pb)
    assert sorted(qa) == sorted(qb)
    for k in qa:
        assert torch.equal(qa[k], qb[k]), k


_LAYOUTS = (("empty", {}), ("one", {}), ("mix", {}), ("composed", {"count": 9}), ("composed", {"count": 2}))


@pytest.mark.parametrize("bed", scenes.BEDS)
def test_render_is_bit_for_bit_the_render_before_the_split(bed):
    for i, (kind, kw) in enumerate(_LAYOUTS):
        for margin in (scenes.MARGIN, 3):
            cells, colors = scenes.draw_layouts(kind, 37, _gen(900 + i), **kw)
            got = scenes.render(cells, colors, bed, _gen(950 + i), margin=margin)
            want = _reference_render(cells, colors, bed, _gen(950 + i), margin=margin)
            _same_render(got, want)
            # render_drawn of the drawn numbers, on the CPU or already moved: the same
            p = scenes.draw_parameters(37, _gen(950 + i), bed)
            _same_render(scenes.render_drawn(cells, colors, p, bed, margin=margin), want)
            _same_render(scenes.render_drawn(cells, colors, {k: v.clone() for k, v in p.items()}, bed,
                                             margin=margin), want)
    # the bed's batches, which the stored experiments use, are the old ones
    sb = scenes.ScenesBed(bed)
    a = sb.train_batch("mix", 23, _gen(5))
    gen = _gen(5)
    cells, colors = scenes.draw_layouts("mix", 23, gen)
    pix, par = _reference_render(cells, colors, bed, gen)
    assert torch.equal(a.x, scenes.to_model(pix)) and all(torch.equal(a.params[k], par[k]) for k in par)


@pytest.mark.parametrize("bed", scenes.BEDS)
def test_render_drawn_of_concatenated_rows_is_the_concatenation_of_each_parts_render(bed):
    parts = []
    for i, (n, (kind, kw)) in enumerate(zip((5, 1, 17, 64, 3), _LAYOUTS)):
        gen = _gen(1200 + i)
        cells, colors = scenes.draw_layouts(kind, n, gen, **kw)
        parts.append((cells, colors, scenes.draw_parameters(n, gen, bed)))
    each = [scenes.render_drawn(c, k, p, bed) for c, k, p in parts]
    cells = torch.cat([c for c, _, _ in parts])
    colors = torch.cat([k for _, k, _ in parts])
    p = {key: torch.cat([q[key] for _, _, q in parts]) for key in parts[0][2]}
    # bed C's texture (a convolution) may round differently with its batch: it is computed part by part
    sizes = [c.shape[0] for c, _, _ in parts]
    for kw in ([{"parts": sizes}, {}] if bed == "F" else [{"parts": sizes}]):
        whole_pixels, whole_params = scenes.render_drawn(cells, colors, p, bed, **kw)
        assert torch.equal(whole_pixels, torch.cat([x for x, _ in each])), kw
        assert sorted(whole_params) == sorted(each[0][1])
        for key in whole_params:
            assert torch.equal(whole_params[key], torch.cat([q[key] for _, q in each])), (key, kw)


def test_render_drawn_refuses_numbers_that_do_not_fit():
    cells, colors = scenes.draw_layouts("one", 4, _gen(1))
    p = scenes.draw_parameters(4, _gen(2), "F")
    with pytest.raises(ValueError):
        scenes.render_drawn(cells, colors, p, "C")                         # bed C needs the tint and the noise
    with pytest.raises(ValueError):
        scenes.render_drawn(cells[:3], colors[:3], p, "F")
    with pytest.raises(ValueError):
        scenes.render_drawn(cells, colors, p, "G")
    with pytest.raises(ValueError):
        scenes.render_drawn(cells, colors, p, "F", parts=[2, 1])


@pytest.mark.parametrize("bed", scenes.BEDS)
def test_render_drawn_of_a_cpu_layout_with_a_device_given_is_the_reference_render(bed):
    """The layout on the CPU, the device named (here the CPU: the card's path is not run by this test)."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for i, (kind, kw) in enumerate(_LAYOUTS):
        cells, colors = scenes.draw_layouts(kind, 29, _gen(1300 + i), **kw)
        want = _reference_render(cells, colors, bed, _gen(1350 + i), device=device)
        p = scenes.draw_parameters(29, _gen(1350 + i), bed)
        got = scenes.render_drawn(cells, colors, p, bed, device=device)
        assert got[0].device.type == device.type
        _same_render(got, want)
        moved = scenes.render_drawn(cells.to(device), colors.to(device), {k: v.to(device) for k, v in p.items()},
                                    bed, device=device)
        _same_render(moved, want)
