"""Stage clusters: the blend weights and where a cluster runs (1), exactness at birth (2), agreement with one model (3),
shared units through training, files and links (4), and the read of a staged model (5)."""
import os
import zlib

import numpy as np
import pytest
import torch

from geolip.modular_diffusion import read as read_mod
from geolip.modular_diffusion.beds import FieldsBed
from geolip.modular_diffusion.build import read_build
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.linker import export_build, probe_batch
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.staged import Staged, link_staged, share, staged_stale_list, write_staged
from geolip.modular_diffusion.train import fit_units, run_loop

FIELD = (4, 16, 16)
CUTS = (0.0, 0.25, 0.65, 1.0)
SOFT = 0.06


def _spectrum():
    return torch.rand(FIELD, generator=torch.Generator().manual_seed(1), dtype=torch.float64) + 0.1


def _cluster(seed=0, body=0, dim=32, stem=2):
    return Model.born(cluster_build(FIELD, stem_depth=stem, body_depth=body, dim=dim, grid=8), seed, _spectrum())


def _live(model, salt=""):
    """Small seeded values on every parameter, so zero-born heads speak."""
    with torch.no_grad():
        for u in model.units:
            for k, p in u.named_parameters():
                g = torch.Generator().manual_seed(zlib.crc32(f"{salt}{u.name}/{k}".encode()))
                p.add_(torch.randn(p.shape, generator=g) * 0.05)


def _probe(n=24, seed=4):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, *FIELD, generator=gen)
    t = torch.linspace(0.0, 1.0, n)
    cls = torch.arange(n) % 11
    return z, t, cls


# ------------------------------------------------------------------ 1
def test_1_weights_sum_to_one_and_vanish_outside_the_widened_window():
    staged = Staged([_cluster() for _ in range(3)], CUTS, SOFT)
    t = torch.linspace(0.0, 1.0, 1001)
    w = staged.weights(t)
    assert w.shape == (3, 1001)
    assert float((w.sum(0) - 1).abs().max()) < 1e-6
    for s, (lo, hi) in enumerate(staged.windows()):
        outside = (t < lo - SOFT) | (t > hi + SOFT)
        assert bool((w[s][outside] == 0).all()), s


def test_1_a_cluster_runs_only_inside_its_widened_window():
    clusters = [_cluster() for _ in range(3)]
    seen = {s: [] for s in range(3)}
    for s, c in enumerate(clusters):
        orig = c.components

        def spy(zt, t, cls_idx, *args, _orig=orig, _s=s, **kw):
            seen[_s].append(t.clone())
            return _orig(zt, t, cls_idx, *args, **kw)
        c.components = spy
    staged = Staged(clusters, CUTS, SOFT)
    z, _, cls = _probe(201)
    t = torch.linspace(0.0, 1.0, 201)
    with torch.no_grad():
        staged.velocity(z, t, cls)
    for s, (lo, hi) in enumerate(staged.windows()):
        got = torch.cat(seen[s])
        assert got.numel() > 0
        assert float(got.min()) >= lo - SOFT - 1e-9 and float(got.max()) <= hi + SOFT + 1e-9, s
        assert bool((staged.weights(got)[s] > 0).all())


# ------------------------------------------------------------------ 2
def test_2_newborn_staged_is_exactly_the_anchor():
    staged = Staged([_cluster(body=1), _cluster(body=1, dim=48), _cluster(body=2)], CUTS, SOFT)
    for c in staged.clusters:                                   # live trunks, zero-born heads
        with torch.no_grad():
            for u in c.units:
                if u.unit_type != "Head":
                    for p in u.parameters():
                        p.add_(0.01)
    z, t, cls = probe_batch(staged.clusters[0].build.geometry)
    with torch.no_grad():
        gap = float((staged.velocity(z, t, cls) - staged.anchor_velocity(z, t)).abs().max())
    assert gap == 0.0


# ------------------------------------------------------------------ 3
def test_3_three_equal_clusters_equal_the_single_model():
    single = _cluster()
    _live(single)
    copies = []
    for _ in range(3):
        c = _cluster()
        for u in single.units:
            c.unit(u.name).load_owned(u.owned_state())
        copies.append(c)
    z, t, cls = _probe()
    with torch.no_grad():
        v1 = single.velocity(z, t, cls)
        soft = Staged(copies, CUTS, SOFT).velocity(z, t, cls)
    assert float((soft - v1).abs().max()) < 1e-6


def test_3_hard_windows_give_each_stage_its_cluster():
    clusters = [_cluster() for _ in range(3)]
    for s, c in enumerate(clusters):
        _live(c, salt=str(s))
    staged = Staged(clusters, CUTS, 0.0)
    z, t, cls = _probe()
    with torch.no_grad():
        v = staged.velocity(z, t, cls)
        for s, (lo, hi) in enumerate(staged.windows()):
            rows = (t >= lo) & ((t < hi) | (t == 1.0) & (hi == 1.0))
            own = clusters[s].velocity(z[rows], t[rows], cls[rows])
            assert float((v[rows] - own).abs().max()) < 1e-6, s


# ------------------------------------------------------------------ 4
def test_4_a_shared_stem_through_training_files_and_links(tmp_path):
    clusters = share([_cluster(body=1) for _ in range(3)], ["anchor", "stem"])
    staged = Staged(clusters, CUTS, SOFT)
    stem_ids = {id(c.stem) for c in staged.clusters}
    assert len(stem_ids) == 1
    n_stem = sum(p.numel() for p in clusters[0].stem.parameters())
    n_all = sum(p.numel() for p in staged.parameters())
    per_cluster = sum(p.numel() for p in clusters[0].parameters())
    assert n_all == per_cluster + 2 * (per_cluster - n_stem)          # the stem counted once
    assert staged.trained_numbers() == n_all
    before = {k: v.clone() for k, v in clusters[0].stem.state_dict().items()}
    bed = FieldsBed(16, 4, 1.225, 0.06)
    run_loop(staged, bed, 0, 3, 1e-3, 1, 8, 0.0, 0.0, "uniform", (0.0, 1.0), True, "cpu", lambda s: None, 1)
    assert any(not torch.equal(before[k], v) for k, v in clusters[0].stem.state_dict().items())
    folders = []
    for s, c in enumerate(clusters):
        f = str(tmp_path / f"stage_{s}")
        export_build(c, f, seed=0)
        folders.append(f"stage_{s}")
    d = write_staged(str(tmp_path), folders, CUTS, SOFT)
    assert set(d["shared"]) == {"anchor", "stem"}
    again = link_staged(str(tmp_path))
    assert len({id(c.stem) for c in again.clusters}) == 1
    assert len({id(c.head) for c in again.clusters}) == 3
    z, t, cls = _probe()
    with torch.no_grad():
        assert torch.equal(again.velocity(z, t, cls), staged.velocity(z, t, cls))
    stale = staged_stale_list(str(tmp_path), "stem")
    assert stale == {f: ["body", "head"] for f in folders}
    with open(tmp_path / "staged.json", encoding="utf-8") as fh:
        text = fh.read()
    assert '"cuts": [\n  "0.0",\n  "0.25",\n  "0.65",\n  "1.0"\n ]' in text and '"soft": "0.06"' in text


def test_4_share_refuses_different_units():
    a, b = _cluster(), _cluster()
    _live(b, salt="x")
    with pytest.raises(ValueError):
        share([a, b], ["stem"])


# ------------------------------------------------------------------ 5
def test_5_read_of_a_staged_model(tmp_path):
    bed = FieldsBed(16, 4, 1.225, 0.06)
    clusters = [Model.born(cluster_build(FIELD, stem_depth=1, dim=32, grid=8), 0, bed.spectrum) for _ in range(3)]
    ranges = ((0.0, 0.31), (0.19, 0.71), (0.59, 1.0))
    for c, r in zip(clusters, ranges):
        fit_units(c, bed, 0, steps=6, t_range=r, batch=16, heldout=32, warmup=1, lr=1e-2, log=lambda s: None)
    staged = Staged(clusters, CUTS, SOFT)
    res = read_mod.read(staged, bed, 64, "cpu")
    # the levels' error by direct computation on the read's own noise
    z1, cls = bed.heldout(64)
    gen = torch.Generator().manual_seed(read_mod.READ_SEED)
    z0 = torch.randn(z1.shape, generator=gen)
    errs = []
    with torch.no_grad():
        for t in read_mod.T_GRID:
            tt = torch.full((64,), t)
            v = staged.velocity((1 - t) * z0 + t * z1, tt, cls)
            errs.append(float(((v - (z1 - z0)) ** 2).double().mean()))
    assert abs(res["loss_over_levels"] - float(np.mean(errs))) < 1e-9
    groups = [[k for k in res["levels"] if lo <= float(k) < hi] for lo, hi in zip(CUTS, CUTS[1:])]
    assert [len(g) for g in groups] == [2, 4, 4]
    bs = read_mod.by_stage(res, CUTS)
    for g, x in zip(groups, bs):
        assert abs(x - float(np.mean([res["levels"][k]["error"]["all"] for k in g]))) < 1e-15
    wrong = read_mod.read(staged.rotated(1), bed, 64, "cpu")
    assert wrong["loss_over_levels"] != res["loss_over_levels"]
    assert staged.rotated(1).clusters[0] is clusters[1]
