"""Symbol-gated parts (T5) and a source's averaged weights as the frozen base (T7).

T5: a model without symbol-gated parts draws nothing more; a symbol-gated part is on exactly for its symbols' rows and
never for the null symbol; drawing the symbols in training leaves the pictures, the noise, the levels and the class drops
of every step unchanged; the new content variants run end to end with their exact blocks at 0.0.
T7: an expert job on source_model=model_avg999 freezes exactly the averaged values and records as_source; with
source_model=model the numbers are those of a job without it."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion import model as model_mod
from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import read_build
from geolip.modular_diffusion.experiments import content
from geolip.modular_diffusion.forms import add_class_experts, cluster_build, with_class_experts
from geolip.modular_diffusion.linker import export_build, link
from geolip.modular_diffusion.model import Model, NULL_CLASS, draw_symbols, fixed_symbols, symbol_gated
from geolip.modular_diffusion.read import read
from geolip.modular_diffusion.train import fit_units, step_generator

FIT = dict(lr=1e-3, warmup=1, batch=16, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)


@pytest.fixture(scope="module")
def source(synthetic_npz, tmp_path_factory):
    bed = PicturesBed(synthetic_npz, "cpu")
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)
    res = fit_units(model, bed, 0, steps=4, expect_anchor_at_birth=True, **FIT)
    folder = str(tmp_path_factory.mktemp("source") / "model")
    export_build(model, folder, seed=0, trained=["stem", "head"], recipe=res["recipe"],
                 marks={"loss_over_levels": res["loss_over_levels"], "frozen_unchanged": True})
    return bed, folder


def test_T5_entries_and_names():
    b = cluster_build((3, 32, 32), stem_depth=1, dim=32)
    names = add_class_experts(b, [[3], [0, 1]], gate="symbol")
    assert names == ["expert_s3", "expert_s0-1"]
    assert b.entry("expert_s3").spec == {"depth": 1, "code": [2.0, 64.0], "keep": False, "gate": "symbol"}
    c = add_class_experts(b, [[3]])
    assert "gate" not in b.entry(c[0]).spec                       # a class gate writes the entry of earlier versions
    with pytest.raises(ValueError):
        add_class_experts(b, [[4]], gate="nope")


def test_T5_a_symbol_gated_part_is_on_for_its_symbols_only(source):
    bed, folder = source
    model, names = with_class_experts(folder, 0, [[3], [5, 7]], gate="symbol")
    assert symbol_gated(model) and not symbol_gated(link(folder))
    t = torch.full((12,), 0.4)
    cls = torch.arange(12) % 11
    sym = torch.tensor([3, 5, 7, 0, 3, NULL_CLASS, 9, 7, 3, 1, 5, NULL_CLASS])
    u3, u57 = model.unit("expert_s3"), model.unit("expert_s5-7")
    assert torch.equal(u3.gate(t, cls, sym) > 0, sym == 3)
    assert torch.equal(u57.gate(t, cls, sym) > 0, (sym == 5) | (sym == 7))
    assert float(u3.gate(t, cls).abs().max()) == 0.0                # no symbols: off for every row
    assert float(u3.gate(t, cls, torch.full_like(sym, NULL_CLASS)).abs().max()) == 0.0
    # trained a little: the part changes exactly its symbols' rows
    fit_units(model, bed, 0, steps=4, trainable=names, mdrop=0.0, equalize=False, **FIT)
    z, c = bed.heldout(12)
    with torch.no_grad():
        p_in, _, _, _ = model.components(z, t, c, symbols=sym)
        p_out, _, _, _ = model.components(z, t, c, symbols=sym, skip=["expert_s3"])
    off = sym != 3
    v_in = sum(p_in.values())
    v_out = sum(p_out.values())
    assert torch.equal(v_in[off], v_out[off]) and not torch.equal(v_in[~off], v_out[~off])
    # the symbols' draw: null where the class was dropped
    gen = torch.Generator().manual_seed(0)
    cc = torch.tensor([1, NULL_CLASS, 4, NULL_CLASS])
    s = draw_symbols(cc, gen)
    assert int(s[1]) == NULL_CLASS and int(s[3]) == NULL_CLASS and 0 <= int(s[0]) < 10
    assert torch.equal(fixed_symbols(50, 7), fixed_symbols(50, 7)) and int(fixed_symbols(500, 7).max()) <= 9


def test_T5_no_symbol_part_draws_nothing_more(source):
    bed, folder = source
    model, _ = with_class_experts(folder, 0, [[c] for c in range(10)])
    model.train()
    x, cls = bed.train_batch(8, step_generator(0, 0, "cpu"))
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    model.loss(x, cls, g1, 0.1)
    Model.draw(x, cls, g2, 0.1)
    assert torch.equal(g1.get_state(), g2.get_state())


def test_T5_the_symbols_draw_leaves_every_other_draw_unchanged(source, monkeypatch):
    bed, folder = source
    seen = {}
    real = Model.draw

    def spy(*args, **kw):
        out = real(*args, **kw)
        seen.setdefault(spy.tag, []).append([o.detach().clone() for o in out])
        return out

    monkeypatch.setattr(model_mod.Model, "draw", staticmethod(spy))
    for tag, gate in (("class", "class"), ("symbol", "symbol")):
        spy.tag = tag
        m, names = with_class_experts(folder, 0, [[c] for c in range(10)], window=(0.0, 1.0), gate=gate)
        fit_units(m, bed, 0, steps=3, trainable=names, mdrop=0.0, equalize=False, **FIT)
    assert len(seen["class"]) == len(seen["symbol"]) == 3
    for a, b in zip(seen["class"], seen["symbol"]):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


SMALL = dict(seed=0, device="cpu", scale=0.0005, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=1,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("symbols")
    r = {}
    for v in ("plain", "blind"):
        r[v] = content.main(**dict(SMALL, variant=v, data=synthetic_npz, out=str(root / v), avg="0.999"))
    for v in ("class_to_mid", "rand_start", "shared_start", "blind_rand_all", "blind_shared_all"):
        src = content.SPECS[v].get("source")
        kw = dict(SMALL, variant=v, data=synthetic_npz, out=str(root / v))
        if src:
            kw["source"] = str(root / src)
            kw["source_model"] = "model_avg999"
        r[v] = content.main(**kw)
    r["experts_default"] = content.main(**dict(SMALL, variant="experts_start", data=synthetic_npz,
                                               out=str(root / "experts_default"), source=str(root / "plain")))
    r["experts_model"] = content.main(**dict(SMALL, variant="experts_start", data=synthetic_npz,
                                             out=str(root / "experts_model"), source=str(root / "plain"),
                                             source_model="model"))
    return root, r


def test_T5_new_variants_end_to_end(jobs):
    root, r = jobs
    assert read_build(str(root / "class_to_mid" / "model")).entry("stem").spec["class_window"] == ["0", "0.65"]
    for v, n_parts, gate in (("rand_start", 10, "symbol"), ("shared_start", 1, "class"),
                             ("blind_rand_all", 10, "symbol"), ("blind_shared_all", 1, "class")):
        b = read_build(str(root / v / "model"))
        parts = [e for e in b.units if e.type == "ContentExpert"]
        assert len(parts) == n_parts, v
        assert all(e.spec.get("gate", "class") == gate for e in parts), v
        if n_parts == 1:
            assert parts[0].address["classes"] == list(range(10))
        res = r[v]
        assert res["exact"]["others_max_abs"] == 0.0, v
        assert res["parts"][0]["frozen_unchanged"] is True and res["as_source"]["equals_stored"] is True
        with open(root / v / "log.txt", encoding="utf-8") as fh:
            assert fh.read().splitlines()[-1] == "done"
        m = link(str(root / v / "model"))
        bed = PicturesBed(res["args"]["data"], "cpu")
        assert read(m, bed, 32, "cpu")["loss_over_levels"] == res["loss_over_levels"]
    assert r["rand_start"]["exact"]["expert"] == "expert_s0"
    assert read_build(str(root / "rand_start" / "model")).entry("expert_s0").address["window"] == ["0.0", "0.25"]
    assert r["rand_start"]["parts"][0]["t_range"] == [0.0, 0.31]
    assert r["blind_rand_all"]["parts"][0]["t_range"] == [0.0, 1.0]
    text = content.summarize(str(root))
    for v in ("class_to_mid", "rand_start", "shared_start", "blind_rand_all", "blind_shared_all"):
        assert f"| {v} | 0 |" in text


def test_T7_an_averaged_base_is_frozen_exactly(jobs):
    root, r = jobs
    src = read_build(str(root / "plain" / "model_avg999"))
    last = read_build(str(root / "plain" / "model"))
    b = read_build(str(root / "rand_start" / "model"))
    for e in src.units:
        assert b.entry(e.name).hash == e.hash
    assert any(src.entry(e.name).hash != e.hash for e in last.units if e.type != "Anchor")
    a = r["rand_start"]["as_source"]
    assert a["folder"] == "model_avg999" and a["equals_stored"] is True
    assert a["loss_over_levels"] == r["plain"]["averaged"]["0.999"]["loss_over_levels"]
    with open(root / "rand_start" / "result.json", encoding="utf-8") as fh:
        assert json.load(fh)["args"]["source_model"] == "model_avg999"
    # source_model=model: the numbers of a job without it, and the source re-read recorded
    d, m = r["experts_default"], r["experts_model"]
    assert d["loss_over_levels"] == m["loss_over_levels"] and d["by_level"] == m["by_level"]
    assert d["score"]["score"] == m["score"]["score"]
    hd = {e.name: e.hash for e in read_build(str(root / "experts_default" / "model")).units}
    hm = {e.name: e.hash for e in read_build(str(root / "experts_model" / "model")).units}
    assert hd == hm
    assert "as_source" not in d and m["as_source"]["equals_stored"] is True
    assert "source_model" not in d["args"]
