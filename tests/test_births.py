"""Identity-born and solve-born parts (T4), class-and-stage experts (T5), restricted solves and the class snapped in by
a solve (T8), and the summaries of the new content and closed-forms variants (T9)."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion import closed
from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import content_hash, read_build
from geolip.modular_diffusion.experiments import closed_forms, content
from geolip.modular_diffusion.forms import add_class_experts, cluster_build, with_parts
from geolip.modular_diffusion.linker import born_gap, export_build, link, probe_batch
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.read import read
from geolip.modular_diffusion.train import fit_units
from geolip.modular_diffusion.units import NULL_CLASS

FIT = dict(lr=1e-3, warmup=1, batch=16, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)
BATCHES = 3
SMALL = dict(seed=0, device="cpu", heldout=16, batch=8, log_every=1, warmup=1, score=1, judge="fake", score_n=16,
             score_steps=2, sheet=0, threads=4, dim=32)
TEN = [[c] for c in range(10)]


@pytest.fixture(scope="module")
def blind(synthetic_npz, tmp_path_factory):
    """A small trained blind model exported as a build folder."""
    bed = PicturesBed(synthetic_npz, "cpu")
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32, class_window=("0", "0")), 0, bed.spectrum)
    res = fit_units(model, bed, 0, steps=5, expect_anchor_at_birth=True, **FIT)
    folder = str(tmp_path_factory.mktemp("blind") / "model")
    export_build(model, folder, seed=0, trained=["stem", "head"], recipe=res["recipe"],
                 marks={"loss_over_levels": res["loss_over_levels"], "frozen_unchanged": True})
    return bed, folder


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    """The blind and blind9 sources (with their g8 averages), then every new content variant and the restricted
    closed-form solves on them."""
    root = tmp_path_factory.mktemp("births")
    ref = str(root / "reference.npz")
    d = dict(SMALL, data=synthetic_npz, ref=ref, scale=0.0005)
    r = {}
    for v in ("blind", "blind9"):                               # the sources are not scored (time)
        r[v] = content.main(**dict(d, variant=v, out=str(root / v), avg="g8", score=0))
    for v in content.PASS7_VARIANTS:
        src = content.ALL_SPECS[v]["source"]
        r[v] = content.main(**dict(d, variant=v, out=str(root / v), source=str(root / src), source_model="model_avg_g8",
                                   avg="g8", batches=BATCHES, reader=1))
    r["snap9_avg"] = content.main(**dict(d, variant="snap9", out=str(root / "snap9_avg"), source=str(root / "blind9"),
                                         source_model="model_avg_g8", avg="g8"))
    for name, v, kw in (("heads10", "class_heads", {}), ("heads30", "class_stage_heads", {}),
                        ("head9", "class_heads", dict(classes="9", only_classes="9")),
                        ("heads9_3", "class_stage_heads", dict(classes="9", only_classes="9"))):
        src = "blind9" if "9" in name else "blind"
        r[name] = closed_forms.main(**dict(d, variant=v, out=str(root / name), source=str(root / src),
                                           source_model="model_avg_g8", batches=BATCHES, score=0, **kw))
    return root, r


def _probe(model, bed):
    z, t, _ = probe_batch(model.build.geometry)
    _, cls = bed.heldout(z.shape[0])
    return z, t, cls


# ------------------------------------------------------------------ T4
def test_T4_without_born_nothing_changes():
    b = cluster_build((3, 32, 32), stem_depth=1, dim=32)
    add_class_experts(b, [[3]])
    assert b.entry("expert_c3").spec == {"depth": 1, "code": [2.0, 64.0], "keep": False}
    b2 = cluster_build((3, 32, 32), stem_depth=1, dim=32)
    add_class_experts(b2, [[3]], born="identity")
    assert b2.entry("expert_c3").spec == {"depth": 1, "code": [2.0, 64.0], "keep": False, "born": "identity"}
    m, m2 = Model.born(b, 0), Model.born(b2, 0)
    blk, blk2 = m.unit("expert_c3").blocks[0], m2.unit("expert_c3").blocks[0]
    assert float(blk.mlp[2].weight.detach().abs().max()) > 0              # a random-born block, as before
    for lin, lin2 in ((blk.attn.out, blk2.attn.out), (blk.cross.out, blk2.cross.out), (blk.mlp[2], blk2.mlp[2])):
        assert float(lin2.weight.detach().abs().max()) == 0.0 and float(lin2.bias.detach().abs().max()) == 0.0
        assert lin.weight.shape == lin2.weight.shape
    # every other number of the identity-born unit is the random-born unit's (the same draws)
    sd, sd2 = blk.state_dict(), blk2.state_dict()
    zeroed = {"attn.out.weight", "attn.out.bias", "cross.out.weight", "cross.out.bias", "mlp.2.weight", "mlp.2.bias"}
    for k in sd:
        if k not in zeroed:
            assert torch.equal(sd[k], sd2[k]), k
    for n in ("anchor", "stem", "head"):
        assert content_hash(m.unit(n)) == content_hash(m2.unit(n))
    with pytest.raises(ValueError):
        add_class_experts(cluster_build((3, 32, 32), stem_depth=1, dim=32), [[3]], born="random")


def test_T4_identity_born_part_equals_the_depth0_part_with_its_head(blind):
    bed, folder = blind
    ident, n1 = with_parts(folder, 0, [dict(classes=TEN, depth=1, born="identity")])
    flat, n0 = with_parts(folder, 0, [dict(classes=TEN, depth=0, prefix="class_head")])
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for a, b in zip(n1, n0):
            w = torch.randn(ident.unit(a).head.proj.weight.shape, generator=gen) * 0.1
            bias = torch.randn(ident.unit(a).head.proj.bias.shape, generator=gen) * 0.1
            for m, n in ((ident, a), (flat, b)):
                m.unit(n).head.proj.weight.copy_(w)
                m.unit(n).head.proj.bias.copy_(bias)
    probe = _probe(ident, bed)
    with torch.no_grad():
        assert torch.equal(ident.velocity(*probe), flat.velocity(*probe))
    assert closed.blocks_at_identity(ident.unit(n1[0])) and not closed.blocks_at_identity(
        with_parts(folder, 0, [dict(classes=[[1]], depth=1)])[0].unit("expert_c1"))


def test_T4_solve_born_parts_are_the_solved_depth0_heads_at_birth(blind):
    bed, folder = blind
    ident, n1 = with_parts(folder, 0, [dict(classes=TEN, depth=1, born="identity")])
    flat, n0 = with_parts(folder, 0, [dict(classes=TEN, depth=0, prefix="class_head")])
    e1 = closed.part_equations(ident, bed, 0, BATCHES, n1, batch=16)
    e0 = closed.part_equations(flat, bed, 0, BATCHES, n0, batch=16)
    for g1, g0 in zip(e1["G"], e0["G"]):
        assert torch.equal(g1, g0)
    p1 = closed.install_parts(ident, closed.solve_parts(e1), e1)
    p0 = closed.install_parts(flat, closed.solve_parts(e0), e0)
    assert p1["all"] == p0["all"]
    probe = _probe(ident, bed)
    with torch.no_grad():
        assert torch.equal(ident.velocity(*probe), flat.velocity(*probe))
    # a part with a block that is not the identity is refused
    rand, nr = with_parts(folder, 0, [dict(classes=[[1]], depth=1)])
    with pytest.raises(ValueError):
        closed.part_equations(rand, bed, 0, 1, nr, batch=16)


def test_T4_solve_born_jobs(jobs):
    root, r = jobs
    for v, heads in (("blind_experts10_solved", "heads10"), ("blind_experts10_solved_1500", "heads10"),
                     ("blind_experts30_solved", "heads30"), ("snap9_solved", "head9")):
        res = r[v]
        assert res["at_birth"]["loss_over_levels"] == r[heads]["loss_over_levels"], v
        assert res["at_birth"]["by_level"] == r[heads]["by_level"], v
        assert res["solve"]["all"]["mse_after"] == r[heads]["solve"]["all"]["mse_after"], v
        assert res["solve"]["all"]["mse_after"] <= res["solve"]["all"]["mse_before"]
        assert res["exact"]["others_max_abs"] == 0.0, v
        assert res["parts"][0]["frozen_unchanged"] is True
        b = read_build(str(root / v / "model"))
        trained = [e for e in b.units if e.type == "ContentExpert"]
        born, _ = with_parts(str(root / content.ALL_SPECS[v]["source"] / "model_avg_g8"), 0,
                             content.expert_groups(content.ALL_SPECS[v]))
        m = link(str(root / v / "model"))
        moved = any(not torch.equal(m.unit(e.name).blocks[0].mlp[2].weight, born.unit(e.name).blocks[0].mlp[2].weight)
                    for e in trained)
        assert moved, v                                           # the identity-born blocks trained
    for v in ("blind_experts10_id", "blind_experts10_id_1500", "blind_experts30_id"):
        assert "at_birth" not in r[v] and "solve" not in r[v]
        assert r[v]["exact"]["others_max_abs"] == 0.0
        assert read_build(str(root / v / "model")).entry(
            "expert_c0" if "10" in v else "expert_s0_c0").spec["born"] == "identity"
    for v in content.PASS7_VARIANTS:
        assert r[v]["as_source"]["equals_stored"] is True and "g8" in r[v]["averaged"], v
        assert "class_read" in r[v]["score"], v
        srcb = read_build(str(root / content.ALL_SPECS[v]["source"] / "model_avg_g8"))
        b = read_build(str(root / v / "model"))
        for e in srcb.units:                                      # the averaged base is carried unchanged
            assert b.entry(e.name).hash == e.hash, (v, e.name)
    assert r["blind_experts10_id"]["parts"][0]["steps"] == 3 and r["blind_experts10_id_1500"]["parts"][0]["steps"] == 2


# ------------------------------------------------------------------ T5
def test_T5_class_and_stage_experts(jobs):
    root, r = jobs
    b = read_build(str(root / "blind_experts30_id" / "model"))
    experts = [e for e in b.units if e.type == "ContentExpert"]
    assert [e.name for e in experts] == [f"expert_s{s}_c{c}" for s in range(3) for c in range(10)]
    wins = [["0.0", "0.25"], ["0.25", "0.65"], ["0.65", "1.0"]]
    for e in experts:
        s, c = int(e.name[8]), int(e.name.split("_c")[1])
        assert e.address["classes"] == [c] and e.address["window"] == wins[s] and e.address["soft"] == "0.06"
    m = link(str(root / "blind_experts30_id" / "model"))
    t = torch.cat([torch.linspace(0, 1, 2001), torch.tensor([0.19, 0.25, 0.31, 0.59, 0.65, 0.71, 1.0])])
    for c in list(range(10)) + [NULL_CLASS]:
        cls = torch.full_like(t, c, dtype=torch.long)
        own = [u for u in m.members if c in u.classes]
        other = [u for u in m.members if c not in u.classes]
        if own:
            s = sum(u.gate(t, cls) for u in own)
            assert float((s - 1).abs().max()) < 1e-7, c
        for u in other:
            assert float(u.gate(t, cls).abs().max()) == 0.0
    assert r["blind_experts30_id"]["parts"][0]["t_range"] == [0.0, 1.0]
    assert len(r["blind_experts30_id"]["parts"][0]["trainable"]) == 30
    assert isinstance(r["blind_experts30_id"]["loss_over_levels"], float)


# ------------------------------------------------------------------ T8
def test_T8_restricted_solves(jobs):
    root, r = jobs
    h9 = r["head9"]
    b = read_build(str(root / "head9" / "model"))
    assert [e.name for e in b.units if e.type == "ContentExpert"] == ["class_head_c9"]
    assert h9["exact"]["others_max_abs"] == 0.0 and h9["exact"]["expert"] == "class_head_c9"
    assert h9["solve"]["all"]["mse_after"] < h9["solve"]["all"]["mse_before"]
    assert h9["args"]["classes"] == "9" and h9["args"]["only_classes"] == "9"
    h93 = read_build(str(root / "heads9_3" / "model"))
    assert [e.name for e in h93.units if e.type == "ContentExpert"] == [f"class_head_s{s}_c9" for s in range(3)]
    assert "classes" not in r["heads10"]["args"]
    # every other class's rows are unchanged bit for bit: the heads change nothing on the rows of classes 0 .. 8
    bed = PicturesBed(str(r["head9"]["args"]["data"]), "cpu")
    src = link(str(root / "blind9" / "model_avg_g8"))
    m = link(str(root / "head9" / "model"))
    z, t, _ = probe_batch(m.build.geometry)
    cls = torch.arange(z.shape[0]) % 9
    with torch.no_grad():
        assert torch.equal(m.velocity(z, t, cls), src.velocity(z, t, cls))
    s9 = r["snap9_solved"]
    assert s9["parts"][0]["trainable"] == ["expert_c9"] and s9["parts"][0]["frozen_unchanged"] is True
    assert s9["exact"]["expert"] == "expert_c9"
    assert r["snap9_avg"]["as_source"]["equals_stored"] is True        # snap9 on an averaged folder


# ------------------------------------------------------------------ T9
def test_T9_summaries(jobs):
    root, r = jobs
    text = content.summarize(str(root))
    for v in content.PASS7_VARIANTS:
        assert any(x.startswith(f"| {v} | 0 |") for x in text.splitlines()), v
    text = closed_forms.summarize(str(root))
    assert "| class_heads | 0 |" in text and "| class_stage_heads | 0 |" in text
