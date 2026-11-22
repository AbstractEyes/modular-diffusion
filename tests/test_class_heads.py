"""Class heads by a solve (T6): a part with no block is silent at birth; the joint solve lowers the error over the
equations' rows (from the equations and computed directly, which agree); a second solve changes nothing; one class's head
left out leaves the other classes' rows exactly unchanged; stage heads per class solve jointly and read no worse; the
closed-forms variants run end to end."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion import closed
from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import read_build
from geolip.modular_diffusion.experiments import closed_forms, content, time_split
from geolip.modular_diffusion.forms import add_class_experts, cluster_build, with_parts, without_units
from geolip.modular_diffusion.linker import born_gap, export_build, link
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.read import read
from geolip.modular_diffusion.train import fit_units, step_generator

FIT = dict(lr=1e-3, warmup=1, batch=16, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)
BATCHES = 6
BATCH = 16


@pytest.fixture(scope="module")
def blind(synthetic_npz, tmp_path_factory):
    """A small trained blind model (the stem never reads the class), exported as a build folder."""
    bed = PicturesBed(synthetic_npz, "cpu")
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32, class_window=("0", "0")), 0, bed.spectrum)
    res = fit_units(model, bed, 0, steps=5, expect_anchor_at_birth=True, **FIT)
    folder = str(tmp_path_factory.mktemp("blind") / "model")
    export_build(model, folder, seed=0, trained=["stem", "head"], recipe=res["recipe"],
                 marks={"loss_over_levels": res["loss_over_levels"], "frozen_unchanged": True})
    return bed, folder


def _groups(variant):
    return closed_forms.class_head_groups(variant)


@torch.no_grad()
def _direct_mse(model, bed, names, batches=BATCHES, batch=BATCH):
    """The training error over the equations' rows (samples where a named part's gate is on), from the velocity."""
    model.eval()
    sse, n = 0.0, 0
    units = [model.unit(x) for x in names]
    for k in range(batches):
        gen = step_generator(0, k, "cpu")
        x, cls = bed.train_batch(batch, gen)
        zt, t, c, target = model.draw(x, cls, gen, 0.1, "uniform", (0.0, 1.0))
        on = torch.zeros(c.shape[0], dtype=torch.bool)
        for u in units:
            on |= u.gate(t, c) > 0
        v = model.velocity(zt, t, c)
        e = (v.double() - target.double()) ** 2
        sse += float(e[on].sum())
        n += int(e[on].numel())
    return sse / n


def test_T6_a_part_with_no_block_is_silent_at_birth(blind):
    bed, folder = blind
    src = link(folder)
    model, names = with_parts(folder, 0, _groups("class_heads"))
    assert names == [f"class_head_c{c}" for c in range(10)]
    assert all(len(model.unit(n).blocks) == 0 for n in names)
    assert born_gap(model, src) == 0.0
    b = cluster_build(bed.field_shape, stem_depth=1, dim=32)
    add_class_experts(b, [[0]], depth=0, prefix="class_head")
    assert b.entry("class_head_c0").spec["depth"] == 0
    assert len(read_build(folder).units) == 3                     # the folder itself is not changed


def test_T6_class_heads_solve(blind):
    bed, folder = blind
    model, names = with_parts(folder, 0, _groups("class_heads"))
    eq = closed.part_equations(model, bed, 0, BATCHES, names, batch=BATCH)
    assert len(eq["groups"]) == 10 and all(len(g) == 1 for g in eq["groups"])
    direct_before = _direct_mse(model, bed, names)
    before = closed.parts_mse(eq, model)["all"]
    assert abs(before - direct_before) / direct_before < 1e-6
    sol = closed.solve_parts(eq, ridge=1e-6)
    prov = closed.install_parts(model, sol, eq)
    direct_after = _direct_mse(model, bed, names)
    assert prov["all"]["mse_after"] <= prov["all"]["mse_before"] and direct_after <= direct_before
    assert abs(prov["all"]["mse_after"] - direct_after) / direct_after < 1e-6
    assert all(prov[n]["solved"] and prov[n]["weight_norm_after"] > 0 for n in names)
    # a second solve on the same rows leaves the weights where they are
    w1 = {n: model.unit(n).head.proj.weight.detach().clone() for n in names}
    eq2 = closed.part_equations(model, bed, 0, BATCHES, names, batch=BATCH)
    closed.install_parts(model, closed.solve_parts(eq2, ridge=1e-6), eq2)
    for n in names:
        assert float((model.unit(n).head.proj.weight.detach() - w1[n]).abs().max()) <= 1e-5
    # one class's head left out: the other classes' rows are unchanged exactly
    ex = content.exactness(model, bed, 32, "cpu", "class_head_c3")
    assert ex["others_max_abs"] == 0.0 and ex["expert"] == "class_head_c3"
    # every head left out gives the source model back
    back = without_units(model, names)
    assert born_gap(model.__class__(back.build, list(back.units)), link(folder)) == 0.0


def test_T6_class_stage_heads_solve_jointly(blind):
    bed, folder = blind
    one, n1 = with_parts(folder, 0, _groups("class_heads"))
    eq1 = closed.part_equations(one, bed, 0, BATCHES, n1, batch=BATCH)
    p1 = closed.install_parts(one, closed.solve_parts(eq1), eq1)
    three, n3 = with_parts(folder, 0, _groups("class_stage_heads"))
    assert len(n3) == 30 and "class_head_s1_c4" in n3
    eq3 = closed.part_equations(three, bed, 0, BATCHES, n3, batch=BATCH)
    assert len(eq3["groups"]) == 10 and all(len(g) == 3 for g in eq3["groups"])
    assert sum(eq3["rows"]) == sum(eq1["rows"])                     # the same rows
    p3 = closed.install_parts(three, closed.solve_parts(eq3), eq3)
    assert p3["all"]["mse_after"] <= p1["all"]["mse_after"] + 1e-3
    ex = content.exactness(three, bed, 32, "cpu", "class_head_s0_c2")
    assert ex["others_max_abs"] == 0.0


SMALL = dict(seed=0, device="cpu", heldout=32, batch=8, log_every=1, warmup=1, score=1, judge="fake", score_n=32,
             score_steps=2, sheet=0, threads=4)


def test_T6_the_closed_forms_variants_end_to_end(synthetic_npz, tmp_path):
    src = str(tmp_path / "blind")
    content.main(**dict(SMALL, variant="blind", data=synthetic_npz, out=src, scale=0.0005, dim=32, avg="g8"))
    ref = str(tmp_path / "ref.npz")
    bed = PicturesBed(synthetic_npz, "cpu")
    out = {}
    for v in ("class_heads", "class_stage_heads"):
        o = str(tmp_path / v)
        out[v] = closed_forms.main(**dict(SMALL, variant=v, data=synthetic_npz, out=o, source=src, batches=3,
                                          source_model="model_avg_g8", reader=1, ref=ref))
        with open(os.path.join(o, "result.json"), encoding="utf-8") as fh:
            stored = json.load(fh)
        want = {"experiment", "variant", "seed", "kind", "cuts", "soft", "parts", "numbers", "train_seconds", "read",
                "loss_over_levels", "by_level", "by_stage", "wrong_assembly", "alone", "score", "args", "torch",
                "device", "version", "solve", "as_trained", "curve", "seconds_per_step", "cache", "as_source",
                "by_class", "class_worth", "exact"}
        assert set(stored) == want, set(stored) ^ want
        assert stored["exact"]["others_max_abs"] == 0.0
        assert stored["as_source"]["equals_stored"] is True and stored["as_source"]["folder"] == "model_avg_g8"
        assert "class_read" in stored["score"] and len(stored["by_class"]) == 10
        m = link(os.path.join(o, "model"))
        assert read(m, bed, 32, "cpu")["loss_over_levels"] == stored["loss_over_levels"]
        srcb = read_build(os.path.join(src, "model_avg_g8"))
        b = read_build(os.path.join(o, "model"))
        for e in srcb.units:                                      # the averaged base is carried unchanged
            assert b.entry(e.name).hash == e.hash
    assert out["class_stage_heads"]["solve"]["all"]["mse_after"] <= out["class_heads"]["solve"]["all"]["mse_after"] + 1e-3
    text = closed_forms.summarize(str(tmp_path))
    assert "| class_heads | 0 |" in text and "| class_stage_heads | 0 |" in text
