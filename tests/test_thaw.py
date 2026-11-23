"""The thaw pass (T6): a finished tails or grow build trained on with the tail only, the tail and the stem's last one
or two blocks, or everything; every frozen tensor bit-identical; the base re-read; both averages read and the tail
average scored; a stream of its own. And the summary of the stages experiment on the thaw jobs (T9)."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion.build import read_build, read_weights
from geolip.modular_diffusion.experiments import stages, time_split
from geolip.modular_diffusion.linker import link
from geolip.modular_diffusion.staged import link_staged, read_staged

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=1,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)


def _tensors(folder):
    """{unit name: {tensor name: tensor}} of a build folder."""
    b = read_build(folder)
    return {e.name: read_weights(os.path.join(b.folder, e.file)) for e in b.units}


def _staged_tensors(folder):
    d = read_staged(folder)
    return {rel: _tensors(os.path.join(folder, rel)) for rel in d["clusters"]}


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("thaw")
    d = dict(SMALL, data=synthetic_npz, ref=str(root / "reference.npz"))
    r = {}
    time_split.main(**dict(d, variant="plain_d4", out=str(root / "plain_d4"), avg="g8"))
    pd4 = str(root / "plain_d4")
    r["tails"] = stages.main(**dict(d, variant="tails", out=str(root / "tails"), stem_from=pd4, stem_model="model_avg_g8"))
    r["tails3"] = stages.main(**dict(d, variant="tails", out=str(root / "tails3"), stem_from=pd4,
                                     stem_model="model_avg_g8", cuts="0.25,0.65"))
    r["grow"] = stages.main(**dict(d, variant="grow", out=str(root / "grow"), stem_from=pd4, stem_model="model_avg_g8",
                                   blocks=2, steps_each=3000))
    for level in ("tail", "1", "2", "all"):
        r[f"thaw_{level}"] = stages.main(**dict(d, variant="thaw", out=str(root / f"thaw_{level}"),
                                                base=str(root / "tails"), thaw=level))
    r["thaw3_2"] = stages.main(**dict(d, variant="thaw", out=str(root / "thaw3_2"), base=str(root / "tails3"), thaw="2"))
    r["thaw_grown"] = stages.main(**dict(d, variant="thaw", out=str(root / "thaw_grown"), base=str(root / "grow"),
                                         thaw="all"))
    r["ref"] = stages.main(**dict(d, variant="source", out=str(root / "ref"), total=6500, avg="g8,tail100",
                                  score_avg="tail100"))
    return root, r


def _changed(before, after):
    """{unit: [tensor names that differ]} between two {unit: {tensor: value}} maps."""
    out = {}
    for u, ts in before.items():
        diff = [k for k, v in ts.items() if not torch.equal(v, after[u][k])]
        if diff:
            out[u] = diff
    return out


def test_T6_thaw_trains_exactly_the_named_numbers(jobs):
    root, r = jobs
    base = _staged_tensors(str(root / "tails" / "avg_g8"))
    blocks = {k for k in base["stage_0"]["stem"] if k.startswith("blocks.")}
    depth = len({k.split(".")[1] for k in blocks})
    assert depth == 4
    for level in ("tail", "1", "2", "all"):
        after = _staged_tensors(str(root / f"thaw_{level}"))
        changed = {}
        for rel in base:
            changed.update(_changed(base[rel], after[rel]))
        assert set(changed) >= {"body", "head"}, level
        assert "anchor" not in changed
        stem = set(changed.get("stem", []))
        if level == "tail":
            assert set(changed) == {"body", "head"}
        elif level in ("1", "2"):
            k = int(level)
            want = {f"blocks.{i}." for i in range(depth - k, depth)}
            assert stem and all(any(n.startswith(p) for p in want) for n in stem), (level, stem)
            for p in want:                                       # every named block moved
                assert any(n.startswith(p) for n in stem), (level, p)
        else:
            assert any(n.startswith("embed.") for n in stem) and any(n.startswith("blocks.0.") for n in stem)
            assert any(n.startswith("time.") for n in stem) and any(n.startswith("cls.") for n in stem)
        res = r[f"thaw_{level}"]
        assert res["parts"][0]["frozen_unchanged"] is True
        assert res["as_source"]["equals_stored"] is True and res["as_source"]["folder"] == "avg_g8"
        assert set(res["averaged"]) == {"g8", "tail100"}
        assert res["averaged"]["tail100"]["score"] is not None and res["averaged"]["g8"]["score"] is None
        assert res["parts"][0]["seconds_per_step"] > 0 and "peak_memory_mib" in res["parts"][0]
        assert res["thaw"]["level"] == level and res["thaw"]["base_variant"] == "tails"
        assert res["kind"] == "clusters"
    t1 = r["thaw_1"]
    assert t1["thaw"]["trainable_params"] == {"stem": ["blocks.3."]}
    assert r["thaw_2"]["thaw"]["trainable_params"] == {"stem": ["blocks.2.", "blocks.3."]}
    assert r["thaw_tail"]["parts"][0]["trained_numbers"] < t1["parts"][0]["trained_numbers"] < \
        r["thaw_2"]["parts"][0]["trained_numbers"] < r["thaw_all"]["parts"][0]["trained_numbers"]
    # the trained numbers of thaw=all are every number of the model
    m = link_staged(str(root / "thaw_all"))
    assert r["thaw_all"]["parts"][0]["trained_numbers"] == sum(p.numel() for p in m.parameters())


def test_T6_the_stream_the_folders_and_the_grown_base(jobs):
    root, r = jobs
    seeds = {r["thaw_tail"]["thaw"]["stream_seed"]}
    assert seeds == {0 + stages.GROW_STREAM * 7}
    built = {0} | {p["stream_seed"] for p in r["grow"]["steps"]}
    assert not seeds & built
    for level in ("tail", "all"):
        for f in ("stage_0", os.path.join("avg_g8", "stage_0"), os.path.join("avg_tail100", "stage_0")):
            assert os.path.exists(root / f"thaw_{level}" / f / "build.json"), f
        assert read_staged(str(root / f"thaw_{level}"))["cuts"] == read_staged(str(root / "tails" / "avg_g8"))["cuts"]
    three = r["thaw3_2"]
    assert len(three["cuts"]) == 4 and three["kind"] == "clusters"
    st = link_staged(str(root / "thaw3_2"))
    stems = {id(c.unit("stem")) for c in st.clusters}
    assert len(stems) == 1                                        # the stem is still one unit, shared
    g = r["thaw_grown"]
    assert g["kind"] == "model" and g["thaw"]["base_variant"] == "grow" and g["as_source"]["folder"] == "model"
    assert g["as_source"]["equals_stored"] is True
    before = _tensors(str(root / "grow" / "model"))
    after = _tensors(str(root / "thaw_grown" / "model"))
    changed = _changed(before, after)
    assert set(changed) == {"stem", "block_0", "block_1", "head"}
    assert os.path.exists(root / "thaw_grown" / "model_avg_tail100" / "build.json")
    with open(root / "thaw_grown" / "result.json", encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored["thaw"]["trainable"] == "all"


def test_T6_the_reference_scores_its_tail_average(jobs):
    root, r = jobs
    ref = r["ref"]
    assert set(ref["averaged"]) == {"g8", "tail100"}
    assert ref["averaged"]["tail100"]["score"]["n"] == 32 and ref["averaged"]["g8"]["score"] is None
    assert ref["parts"][0]["steps"] == time_split.steps_of(6500, SMALL["scale"])


def test_T6_refusals(jobs, synthetic_npz, tmp_path):
    root, _ = jobs
    d = dict(SMALL, data=synthetic_npz)
    with pytest.raises(ValueError):
        stages.main(**dict(d, variant="thaw", out=str(tmp_path / "a")))                      # no base
    with pytest.raises(ValueError):
        stages.main(**dict(d, variant="thaw", out=str(tmp_path / "b"), base=str(root / "tails"), thaw="3"))
    with pytest.raises(ValueError):
        stages.main(**dict(d, variant="thaw", out=str(tmp_path / "c"), base=str(root / "ref"), thaw="tail"))


def test_T9_the_stages_summary_holds_the_thaw_jobs(jobs):
    root, _ = jobs
    text = stages.summarize(str(root))
    rows = [x for x in text.splitlines() if x.startswith("| thaw |")]
    assert len(rows) == 6 * 3                                     # six thaw jobs, three readings each
