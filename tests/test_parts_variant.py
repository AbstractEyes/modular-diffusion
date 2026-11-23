"""Parts added to a finished tree of stage copies (T7: the five parts specs, every base unit bit-unchanged, the parts
left out equal the base, the rows they are not on unchanged), chain reads on a tree's averaged folder (T9), and the new
result keys in json and in the summaries (T10). (Grafts: test_graft.py.)"""
import json
import os

import pytest
import torch

from geolip.modular_diffusion.build import read_build, read_weights
from geolip.modular_diffusion.experiments import chain_reads, stages

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=16, batch=8, log_every=100, warmup=1, score=0,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)
CUTS = "0.25,0.65"
SPECS = ("experts10", "shared1", "expert:9", "row:9", "row+expert:9")


def _tag(spec):
    return "p_" + spec.replace(":", "_").replace("+", "_")


def _tensors(folder):
    b = read_build(folder)
    return {e.name: read_weights(os.path.join(b.folder, e.file)) for e in b.units}


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("parts_variant")
    d = dict(SMALL, data=synthetic_npz, ref=str(root / "reference.npz"))
    j = lambda n: str(root / n)
    t = dict(d, avg="tail100")                    # the bases: only the tree the parts stand on (keeps the file short)
    r = {}
    r["src"] = stages.main(**dict(d, variant="source", out=j("src")))
    r["cop"] = stages.main(**dict(t, variant="copies", cuts=CUTS, out=j("cop"), source=j("src")))
    r["one"] = stages.main(**dict(t, variant="copies", out=j("one"), source=j("src")))
    for spec in SPECS:
        kw = dict(only_classes="9") if ":" in spec else {}
        r[_tag(spec)] = stages.main(**dict(d, variant="parts", out=j(_tag(spec)), base=j("cop"), parts=spec, **kw))
    r["src1"] = stages.main(**dict(d, variant="source", seed=1, out=j("src1")))
    r["cop1"] = stages.main(**dict(t, variant="copies", seed=1, cuts=CUTS, out=j("cop1"), source=j("src1")))
    return root, r


def test_T7_parts_on_a_finished_tree(jobs):
    root, r = jobs
    base = {rel: _tensors(str(root / "cop" / "avg_tail100" / rel)) for rel in ("stage_0", "stage_1", "stage_2")}
    for spec in SPECS:
        res = r[_tag(spec)]
        assert res["as_source"]["equals_stored"] is True and res["as_source"]["folder"] == "avg_tail100", spec
        assert res["parts_off"]["equals_base"] is True and res["averaged"]["g8"]["parts_off"]["equals_base"], spec
        assert res["exact"]["others_max_abs"] == 0.0 and res["exact"]["reading"] == "g8", spec
        assert res["parts_spec"] == spec and res["variant"] == "parts" and res["kind"] == "clusters"
        assert res["score"] is None and list(res["averaged"]) == ["g8"]
        for p in res["parts"]:
            assert p["frozen_unchanged"] and p["seconds_per_step"] >= 0 and "peak_memory_mib" in p
            assert p["stream_seed"] == stages.GROW_STREAM * stages.PARTS_STREAM and p["start_step"] == 0
        for top in ("", "avg_g8"):
            for rel, units in base.items():
                now = _tensors(str(root / _tag(spec) / top / rel))
                for u, ts in units.items():
                    for k, v in ts.items():
                        if u == "stem" and k == "cls.weight" and spec.startswith("row"):
                            keep = torch.ones(v.shape[0], dtype=torch.bool)
                            keep[9] = False
                            assert torch.equal(v[keep], now[u][k][keep]), (spec, rel)
                            assert not torch.equal(v[9], now[u][k][9]), (spec, rel)
                        else:
                            assert torch.equal(v, now[u][k]), (spec, rel, u, k)
    assert r["p_row_9"]["row_birth_gap"] == 0.0 and r["p_row_9"]["part_units"] == []
    assert r["p_row_expert_9"]["row_birth_gap"] == 0.0 and r["p_row_expert_9"]["birth_gap"] == 0.0
    assert r["p_row_9"]["parts"][0]["trainable_rows"] == {"stem": {"cls.weight": [9]}}
    assert r["p_row_9"]["parts"][0]["rows_frozen_equal"] is True
    assert len(r["p_experts10"]["part_units"]) == 10 and r["p_expert_9"]["part_units"] == ["expert_c9"]
    assert r["p_expert_9"]["data"]["training_stream"] == {"only_classes": [9]}
    assert "data" not in r["p_experts10"]                                # the base's restriction is not inherited
    # a row part trains one row of numbers per stage copy
    assert r["p_row_9"]["parts"][0]["trained_numbers"] == SMALL["dim"]


def test_T7_refusals(jobs, synthetic_npz, tmp_path):
    root, _ = jobs
    d = dict(SMALL, data=synthetic_npz, variant="parts")
    for i, kw in enumerate((dict(base=str(root / "cop")),                                    # no parts spec
                            dict(base=str(root / "cop"), parts="row:10"),
                            dict(base=str(root / "src"), parts="experts10"),                 # not a copies job
                            dict(base=str(root / "cop"), parts="experts10", cuts="0.5"),
                            dict(base=str(root / "cop"), parts="experts10", init_stage=0),   # no init_from
                            dict(parts="experts10"))):                                       # no base
        with pytest.raises(ValueError):
            stages.main(**dict(d, out=str(tmp_path / f"r{i}"), **kw))


def test_T9_chain_reads_on_averaged_trees(jobs, synthetic_npz, tmp_path):
    root, _ = jobs
    a, b = str(root / "cop" / "avg_tail100"), str(root / "cop1" / "avg_tail100")
    common = dict(data=synthetic_npz, device="cpu", heldout=SMALL["heldout"], score=0, judge="fake")
    twin = chain_reads.main(**dict(common, variant="twin", base=a, donor=b, stage=1, out=str(tmp_path / "twin")))
    blend = chain_reads.main(**dict(common, variant="blend", base=a, soft=0.0, out=str(tmp_path / "blend")))
    steps = chain_reads.main(**dict(common, variant="steps", base=a, plan="start", n=6, out=str(tmp_path / "steps")))
    for res in (twin, blend, steps):
        assert isinstance(res["loss_over_levels"], float) and len(res["by_stage"]) == 3
    with open(root / "cop" / "result.json", encoding="utf-8") as fh:
        stored = json.load(fh)
    assert steps["loss_over_levels"] == stored["averaged"]["tail100"]["loss_over_levels"]
    assert twin["loss_over_levels"] != steps["loss_over_levels"]
    with pytest.raises(ValueError):                                       # a tree of other cuts
        chain_reads.main(**dict(common, variant="steps", base=str(root / "one" / "avg_tail100"), plan="start", n=6,
                                out=str(tmp_path / "bad")))
    assert "| twin |" in chain_reads.summarize(str(tmp_path))


def test_T10_the_new_keys_are_json_and_the_summaries_hold_the_jobs(jobs):
    root, r = jobs
    for name, res in r.items():
        json.dumps(res)
        with open(root / name / "result.json", encoding="utf-8") as fh:
            assert json.load(fh)["variant"] == res["variant"], name
    text = stages.summarize(str(root))
    rows = [x for x in text.splitlines() if x.startswith("| parts |")]
    assert len(rows) == 2 * len(SPECS)                                   # the last weights and g8 of each parts job
