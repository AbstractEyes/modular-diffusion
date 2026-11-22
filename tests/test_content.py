"""Class experts on a finished model (T4) and the content experiment end to end (T8): every variant on the synthetic
pictures with tiny step counts, a narrow width, a small held-out set and the test judge."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion import run
from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import content_hash, read_build, read_weights, unit_files
from geolip.modular_diffusion.experiments import content, time_split
from geolip.modular_diffusion.forms import add_class_experts, cluster_build, expert_name, with_class_experts, without_units
from geolip.modular_diffusion.linker import export_build, link, probe_batch
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.train import fit_units

FIT = dict(lr=1e-3, warmup=1, batch=16, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)


# ------------------------------------------------------------------ T4: class experts
@pytest.fixture(scope="module")
def source(synthetic_npz, tmp_path_factory):
    """A small trained model (anchor, stem of 1, head) exported as a build folder."""
    bed = PicturesBed(synthetic_npz, "cpu")
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)
    res = fit_units(model, bed, 0, steps=4, expect_anchor_at_birth=True, **FIT)
    folder = str(tmp_path_factory.mktemp("source") / "model")
    export_build(model, folder, seed=0, trained=["stem", "head"], recipe=res["recipe"],
                 marks={"loss_over_levels": res["loss_over_levels"], "frozen_unchanged": res["frozen_unchanged"]})
    return bed, folder


def _velocity(model, bed):
    z, t, _ = probe_batch(model.build.geometry)
    _, cls = bed.heldout(z.shape[0])
    with torch.no_grad():
        return model.velocity(z, t, cls)


def _weights(model, names):
    return {f"{n}/{k}": v.detach().clone() for n in names for k, v in model.unit(n).state_dict().items()}


def test_T4_names_and_entries():
    b = cluster_build((3, 32, 32), stem_depth=1, dim=32)
    names = add_class_experts(b, [[3], [0, 1]], window=(0.0, 0.25), soft=0.06)
    assert names == ["expert_c3", "expert_c0-1"] and expert_name([0, 1]) == "expert_c0-1"
    e = b.entry("expert_c0-1")
    assert e.type == "ContentExpert" and e.stream is None
    assert e.address == {"site": "out", "band": "all", "window": ["0.0", "0.25"], "soft": "0.06", "classes": [0, 1]}
    assert e.spec == {"depth": 1, "code": [2.0, 64.0], "keep": False}


def test_T4_class_experts(source, tmp_path):
    bed, folder = source
    src = link(folder)
    v_src = _velocity(src, bed)
    src_hashes = {u.name: content_hash(u) for u in src.units}
    # newborn experts change nothing, to the last digit
    model, names = with_class_experts(folder, 0, [[c] for c in range(10)], window=(0.0, 1.0), soft=0.0)
    assert names == [f"expert_c{c}" for c in range(10)]
    assert torch.equal(_velocity(model, bed), v_src)
    res = fit_units(model, bed, 0, steps=6, trainable=names, mdrop=0.0, equalize=False, **FIT)
    assert res["frozen_unchanged"]
    assert {u.name: content_hash(u) for u in model.units if u.name in src_hashes} == src_hashes
    # one expert left out: the other classes' rows are unchanged (0.0), its own rows change
    ex = content.exactness(model, bed, 32, "cpu", "expert_c0")
    assert ex["others_max_abs"] == 0.0 and ex["expert"] == "expert_c0"
    z, cls = bed.heldout(32)
    t = torch.full((32,), 0.4)
    with torch.no_grad():
        p_in, _, _, _ = model.components(z, t, cls)
        p_out, _, _, _ = model.components(z, t, cls, skip=["expert_c0"])
    v_in = sum(p_in.values())
    v_out = sum(p_out.values())
    other = cls != 0
    assert torch.equal(v_in[other], v_out[other])
    if bool((cls == 0).any()):
        assert not torch.equal(v_in[~other], v_out[~other])
    # removing every expert gives the source model back exactly
    back = without_units(model, names)
    assert torch.equal(_velocity(back, bed), v_src)
    # exported with the source as base: the frozen units keep their cards, the link holds
    out = str(tmp_path / "with_experts")
    export_build(model, out, seed=0, base=folder, trained=names, recipe=res["recipe"],
                 marks={"loss_over_levels": res["loss_over_levels"], "frozen_unchanged": True})
    again = link(out)
    for n in src_hashes:
        assert open(unit_files(out, n)[1], "rb").read() == open(unit_files(folder, n)[1], "rb").read()
    assert torch.equal(_velocity(again, bed), _velocity(model, bed))


def test_T4_two_experts_together_equal_one_at_a_time(source):
    bed, folder = source
    together, names = with_class_experts(folder, 0, [[0], [1]])
    fit_units(together, bed, 0, steps=8, trainable=names, mdrop=0.0, equalize=False, **FIT)
    for n in names:
        alone, one = with_class_experts(folder, 0, [[int(n[-1])]])
        assert one == [n]
        fit_units(alone, bed, 0, steps=8, trainable=one, mdrop=0.0, equalize=False, **FIT)
        a, b = _weights(together, [n]), _weights(alone, [n])
        assert list(a) == list(b)
        moved = False
        for k in a:
            assert float((a[k] - b[k]).abs().max()) <= 1e-6, k
            born = _weights(with_class_experts(folder, 0, [[int(n[-1])]])[0], [n])[k]
            moved = moved or not torch.equal(a[k], born)
        assert moved, n


# ------------------------------------------------------------------ T8: the experiment end to end
SMALL = dict(seed=0, device="cpu", scale=0.0005, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=1,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)
REGISTERED = {"experiment", "variant", "seed", "kind", "cuts", "soft", "parts", "numbers", "train_seconds", "read",
              "loss_over_levels", "by_level", "by_stage", "wrong_assembly", "alone", "score", "args", "torch", "device",
              "version", "by_level_no_class", "class_worth", "by_class", "anchor_by_level", "anchor_pooled_by_level",
              "exact"}


@pytest.fixture(scope="module")
def runs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("content")
    ref = str(root / "reference.npz")
    order = ["plain", "blind", "blind9"] + [v for v in content.VARIANTS if v not in ("plain", "blind", "blind9")]
    results = {}
    for v in order:
        kw = dict(SMALL, variant=v, data=synthetic_npz, out=str(root / v), reader=1, ref=ref)
        need = content.SPECS[v].get("source")
        if need:
            kw["source"] = str(root / need)
        if v == "class_start":
            results[v] = run.main("content", [f"{k}={val}" for k, val in kw.items()])
        else:
            results[v] = content.main(**kw)
    twin = time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=str(root / "time_split_plain")))
    return root, results, twin


def test_T8_every_variant_writes_its_keys(runs):
    root, results, _ = runs
    assert set(results) == set(content.VARIANTS)
    for v, r in results.items():
        with open(root / v / "result.json", encoding="utf-8") as fh:
            stored = json.load(fh)
        assert set(stored) == REGISTERED, (v, set(stored) ^ REGISTERED)
        assert stored["experiment"] == "content" and stored["variant"] == v and stored["kind"] == "model"
        with open(root / v / "log.txt", encoding="utf-8") as fh:
            assert fh.read().splitlines()[-1] == "done"
        assert os.path.exists(root / v / "model" / "build.json")
        assert len(stored["by_stage"]) == 3 and all(isinstance(x, float) for x in stored["by_stage"])
        assert len(stored["by_class"]) == 10 and set(stored["by_level_no_class"]) == set(stored["by_level"])
        assert len(stored["class_worth"]["by_stage"]) == 3
        for lv, w in stored["class_worth"]["by_level"].items():
            assert w == stored["by_level_no_class"][lv] - stored["read"]["levels"][lv]["error"]["all"]
        assert set(stored["anchor_by_level"]) == set(stored["anchor_pooled_by_level"])
        sc = stored["score"]
        assert sc["n"] == 32 and sc["judge"].startswith("fake") and sc["class_read"]["n"] == 32
        if v in content.EXPERT_VARIANTS:
            assert stored["exact"]["others_max_abs"] == 0.0, v
            assert stored["parts"][0]["frozen_unchanged"] is True
        else:
            assert stored["exact"] is None
    # the first job that computed the reference file also checked the reader on the test pictures
    first = results["plain"]["score"]
    assert first["reader_check"]["n"] == 300
    assert all("reader_check" not in results[v]["score"] for v in content.VARIANTS if v != "plain")


def test_T8_variant_details(runs, synthetic_npz):
    root, results, _ = runs
    b = read_build(str(root / "class_start" / "model"))
    assert b.entry("stem").spec["class_window"] == ["0", "0.25"]
    assert read_build(str(root / "blind" / "model")).entry("stem").spec["class_window"] == ["0", "0"]
    assert read_build(str(root / "canchor" / "model")).entry("anchor").spec["symbols"] == 10
    assert "class_window" not in read_build(str(root / "canchor" / "model")).entry("stem").spec
    assert "class_window" not in read_build(str(root / "plain" / "model")).entry("stem").spec
    for v, n_exp, src in (("experts_start", 10, "plain"), ("blind_experts_start", 10, "blind"),
                          ("blind_experts_all", 10, "blind"), ("snap9", 1, "blind9")):
        bb = read_build(str(root / v / "model"))
        experts = [e for e in bb.units if e.type == "ContentExpert"]
        assert len(experts) == n_exp, v
        assert results[v]["parts"][0]["trainable"] == sorted(e.name for e in experts)
        srcb = read_build(str(root / src / "model"))
        for e in srcb.units:                                    # the source's units are carried unchanged
            assert bb.entry(e.name).hash == e.hash, (v, e.name)
    assert read_build(str(root / "experts_start" / "model")).entry("expert_c0").address["window"] == ["0.0", "0.25"]
    assert results["experts_start"]["parts"][0]["t_range"] == [0.0, 0.31]
    assert results["snap9"]["exact"]["expert"] == "expert_c9"
    assert results["snap9"]["parts"][0]["steps"] == time_split.steps_of(1500, SMALL["scale"])
    # anchors: no training; the class anchor's mean is the result
    an = results["anchors"]
    assert an["parts"] == [] and an["train_seconds"] == 0
    assert an["loss_over_levels"] == pytest.approx(sum(an["anchor_by_level"].values()) / 10, abs=1e-12)
    pooled = sum(an["anchor_pooled_by_level"].values()) / 10
    assert an["anchor_by_level"] != an["anchor_pooled_by_level"] and pooled > 0
    # every variant without a class anchor reads the same anchor with and without the class
    assert results["plain"]["anchor_by_level"] == results["plain"]["anchor_pooled_by_level"]
    # blind9 never saw class 9 in training
    assert results["blind9"]["parts"][0]["steps"] == results["blind"]["parts"][0]["steps"]


def test_T8_plain_equals_the_time_split_plain_bit_for_bit(runs):
    root, results, twin = runs
    assert results["plain"]["loss_over_levels"] == twin["loss_over_levels"]
    assert results["plain"]["by_level"] == twin["by_level"]
    a, b = read_build(str(root / "plain" / "model")), read_build(str(root / "time_split_plain" / "model"))
    assert a.names() == b.names()
    for e in a.units:
        assert e.hash == b.entry(e.name).hash
        wa = read_weights(os.path.join(a.folder, e.file))
        wb = read_weights(os.path.join(b.folder, b.entry(e.name).file))
        assert sorted(wa) == sorted(wb)
        for k in wa:
            assert torch.equal(wa[k], wb[k]), (e.name, k)


def test_T8_summarize_and_refusals(runs, synthetic_npz, tmp_path):
    root, results, _ = runs
    text = content.summarize(str(root))
    lines = [x for x in text.splitlines() if x.startswith("| ") and not x.startswith("| variant")]
    assert len(lines) == len(content.VARIANTS)
    assert any(x.startswith("| blind | 0 |") for x in lines)
    with pytest.raises(ValueError):
        content.main(variant="experts_start", data=synthetic_npz, out=str(tmp_path / "a"), device="cpu")
    with pytest.raises(ValueError):                          # a source of the wrong variant is refused
        content.main(variant="experts_start", data=synthetic_npz, out=str(tmp_path / "b"), device="cpu",
                     source=str(root / "blind"))
    with pytest.raises(ValueError):
        content.main(variant="plain", data=synthetic_npz, out=str(tmp_path / "c"), device="cpu", score=1, reader=1)
    with pytest.raises(ValueError):
        content.main(variant="nope", data=synthetic_npz, out=str(tmp_path / "d"), device="cpu")
