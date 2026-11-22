"""The stages experiment (T3, T4, P8): a source that is the time split's plain to the bit; copies that continue it (to the
bit against one straight run, last weights and the carried g8 average); fresh copies that are the time split's
copies_4k; K-stage staged folders; tails on a frozen shared stem; a model grown block by block with silent births; and
the registered three stages read on any model."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion import run
from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import read_build
from geolip.modular_diffusion.experiments import chain_reads, stages, time_split
from geolip.modular_diffusion.linker import link
from geolip.modular_diffusion.read import by_stage, read
from geolip.modular_diffusion.staged import link_staged, read_staged

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=1,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)
KEYS = {"experiment", "variant", "seed", "kind", "cuts", "soft", "total", "parts", "numbers", "train_seconds", "read",
        "loss_over_levels", "by_level", "by_stage", "by_class", "class_worth", "score", "averaged", "as_source", "args",
        "torch", "device", "version"}


def _hashes(folder):
    return {e.name: e.hash for e in read_build(folder).units}


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("stages")
    d = dict(SMALL, data=synthetic_npz)
    r = {}
    r["ts_plain_4k"] = time_split.main(**dict(d, variant="plain_4k", out=str(root / "ts_plain_4k")))
    r["ts_plain"] = time_split.main(**dict(d, variant="plain", out=str(root / "ts_plain"), avg="g8"))
    r["ts_copies_4k"] = time_split.main(**dict(d, variant="copies_4k", out=str(root / "ts_copies_4k"),
                                               source=str(root / "ts_plain_4k")))
    r["ts_plain_d4"] = time_split.main(**dict(d, variant="plain_d4", out=str(root / "ts_plain_d4")))
    r["source"] = run.main("stages", [f"{k}={v}" for k, v in dict(d, variant="source", out=str(root / "source")).items()])
    src = str(root / "source")
    r["copies_1"] = stages.main(**dict(d, variant="copies", out=str(root / "copies_1"), source=src))
    r["copies_fresh"] = stages.main(**dict(d, variant="copies", out=str(root / "copies_fresh"), source=src, fresh=1,
                                           cuts="0.25,0.65", total=2000))
    for k, cuts in ((2, "0.25"), (3, "0.25,0.65"), (5, "0.25,0.38,0.52,0.65"), (8, "0.1,0.2,0.3,0.4,0.5,0.65,0.8")):
        r[f"copies_{k}"] = stages.main(**dict(d, variant="copies", out=str(root / f"copies_{k}"), source=src, cuts=cuts,
                                              total=8000))
    pd4 = str(root / "ts_plain_d4")
    r["tails_1"] = stages.main(**dict(d, variant="tails", out=str(root / "tails_1"), stem_from=pd4))
    r["tails_3"] = stages.main(**dict(d, variant="tails", out=str(root / "tails_3"), stem_from=pd4, cuts="0.25,0.65"))
    r["grow"] = stages.main(**dict(d, variant="grow", out=str(root / "grow"), stem_from=pd4, blocks=3, steps_each=3000))
    return root, r


def test_T3_every_job_writes_its_keys(jobs):
    root, r = jobs
    for name in ("source", "copies_1", "copies_fresh", "copies_2", "copies_3", "copies_5", "copies_8", "tails_1",
                 "tails_3", "grow"):
        with open(root / name / "result.json", encoding="utf-8") as fh:
            stored = json.load(fh)
        want = KEYS | ({"steps"} if name == "grow" else set())
        assert set(stored) == want, (name, set(stored) ^ want)
        assert stored["experiment"] == "stages"
        assert len(stored["by_stage"]) == 3 and all(isinstance(x, float) for x in stored["by_stage"])
        assert len(stored["by_class"]) == 10 and stored["score"]["n"] == 32
        with open(root / name / "log.txt", encoding="utf-8") as fh:
            assert fh.read().splitlines()[-1] == "done"
        for p in stored["parts"]:
            assert "start_step" in p and p["steps"] >= 2


def test_T3_source_is_the_time_split_plain_4k(jobs):
    root, r = jobs
    assert _hashes(str(root / "source" / "model")) == _hashes(str(root / "ts_plain_4k" / "model"))
    assert r["source"]["loss_over_levels"] == r["ts_plain_4k"]["loss_over_levels"]
    assert os.path.exists(root / "source" / "state" / "state.json")
    assert os.path.isdir(root / "source" / "model_avg_g8") and list(r["source"]["averaged"]) == ["g8"]
    with open(root / "source" / "state" / "state.json", encoding="utf-8") as fh:
        assert json.load(fh)["step"] == r["source"]["parts"][0]["steps"]


def test_T3_a_continued_copy_is_one_straight_run(jobs):
    root, r = jobs
    c = r["copies_1"]
    assert c["parts"][0]["start_step"] == r["source"]["parts"][0]["steps"]
    assert c["parts"][0]["start_step"] + c["parts"][0]["steps"] == r["ts_plain"]["parts"][0]["steps"]
    assert _hashes(str(root / "copies_1" / "stage_0")) == _hashes(str(root / "ts_plain" / "model"))
    assert _hashes(str(root / "copies_1" / "avg_g8" / "stage_0")) == _hashes(str(root / "ts_plain" / "model_avg_g8"))
    assert c["loss_over_levels"] == r["ts_plain"]["loss_over_levels"]
    assert c["averaged"]["g8"]["loss_over_levels"] == r["ts_plain"]["averaged"]["g8"]["loss_over_levels"]
    assert set(c["averaged"]) == {"g8", "tail100"} and c["parts"][0]["averages_carried"] == ["g8"]
    assert c["as_source"]["equals_stored"] is True and c["as_source"]["folder"] == "model"
    with open(root / "copies_1" / "avg_g8" / "stage_0" / "head.card.json", encoding="utf-8") as fh:
        rec = json.load(fh)["recipe"]
    assert rec["average"] == "g8" and rec["average_carried_from"]["k_from"] == c["parts"][0]["start_step"]


def test_T3_fresh_copies_are_the_time_split_copies_4k(jobs):
    root, r = jobs
    for s in range(3):
        assert _hashes(str(root / "copies_fresh" / f"stage_{s}")) == _hashes(str(root / "ts_copies_4k" / f"stage_{s}"))
    assert r["copies_fresh"]["loss_over_levels"] == r["ts_copies_4k"]["loss_over_levels"]
    assert r["copies_fresh"]["by_stage"] == r["ts_copies_4k"]["by_stage"]
    assert [p["t_range"] for p in r["copies_fresh"]["parts"]] == [list(x) for x in time_split.TRAIN_RANGES]
    assert all(p["start_step"] == 0 for p in r["copies_fresh"]["parts"])


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_T3_k_stage_folders(jobs, k):
    root, r = jobs
    res = r[f"copies_{k}"]
    folder = str(root / f"copies_{k}")
    d = read_staged(folder)
    assert d["clusters"] == [f"stage_{s}" for s in range(k)] and len(d["cuts"]) == k + 1
    model, kind = time_split.load_job(folder)
    assert kind == "clusters" and len(model.clusters) == k
    t = torch.rand(200, generator=torch.Generator().manual_seed(k))
    assert float((model.weights(t).sum(0) - 1).abs().max()) < 1e-6
    assert len(res["by_stage"]) == 3 and set(res["averaged"]) == {"g8", "tail100"}
    for key in ("g8", "tail100"):
        m, kk = time_split.load_job(os.path.join(folder, "avg_" + key))
        assert kk == "clusters" and len(m.clusters) == k
    assert res["soft"] == stages.default_soft(res["cuts"])
    assert [p["steps"] for p in res["parts"]] == stages.stage_steps(res["cuts"], 8000, SMALL["scale"])
    assert read(model, PicturesBed(res["args"]["data"], "cpu"), 32, "cpu")["loss_over_levels"] == res["loss_over_levels"]


def test_T3_default_soft_and_ranges():
    assert stages.default_soft([0.0, 0.25, 0.65, 1.0]) == 0.06
    assert stages.default_soft([0.0, 0.1, 0.2, 1.0]) == 0.025
    assert stages.default_soft([0.0, 1.0]) == 0.06
    assert stages.train_ranges([0.0, 0.25, 0.65, 1.0], 0.06) == [tuple(x) for x in time_split.TRAIN_RANGES]
    assert stages.stage_steps([0.0, 0.25, 0.65, 1.0], 2000, 1.0) == time_split.stage_steps(2000, 1.0)
    with pytest.raises(ValueError):
        stages.parse_cuts("0.65,0.25")


def test_T4_tails_share_the_frozen_stem(jobs):
    root, r = jobs
    src = _hashes(str(root / "ts_plain_d4" / "model"))
    for name, k in (("tails_1", 1), ("tails_3", 3)):
        folder = str(root / name)
        for s in range(k):
            h = _hashes(os.path.join(folder, f"stage_{s}"))
            assert h["stem"] == src["stem"] and h["anchor"] == src["anchor"]
            assert os.path.exists(os.path.join(folder, "avg_g8", f"stage_{s}", "build.json"))
            assert _hashes(os.path.join(folder, "avg_g8", f"stage_{s}"))["stem"] == src["stem"]
        staged = link_staged(folder)
        assert len({id(c.stem) for c in staged.clusters}) == 1
        assert all(p["trainable"] == ["body", "head"] and p["frozen_unchanged"] for p in r[name]["parts"])
        assert list(r[name]["averaged"]) == ["g8"] and r[name]["as_source"]["equals_stored"] is True


def test_T4_grow(jobs):
    root, r = jobs
    g = r["grow"]
    assert [s["birth_gap"] for s in g["steps"]] == [0.0, 0.0, 0.0]
    assert all(s["earlier_unchanged"] for s in g["steps"])
    for i, s in enumerate(g["steps"]):
        for later in g["steps"][i + 1:]:
            assert later["hashes"][s["block"]] == s["hashes"][s["block"]]          # a block frozen after its step
    names = [e.name for e in read_build(str(root / "grow" / "model")).units]
    assert names == ["anchor", "stem", "block_0", "block_1", "block_2", "head"]
    assert not os.path.exists(root / "grow" / "staged.json")
    final = link(str(root / "grow" / "model"))
    bed = PicturesBed(g["args"]["data"], "cpu")
    assert read(final, bed, 32, "cpu")["loss_over_levels"] == g["loss_over_levels"]
    assert g["loss_over_levels"] == g["steps"][-1]["averaged"]["loss_over_levels"]
    src = _hashes(str(root / "ts_plain_d4" / "model"))
    h = _hashes(str(root / "grow" / "model"))
    assert h["stem"] == src["stem"] and h["anchor"] == src["anchor"]
    assert [p["trainable"] for p in g["parts"]] == [sorted([f"block_{i}", "head"]) for i in range(3)]


def test_P8_registered_stages_on_any_model(jobs):
    root, r = jobs
    res = r["copies_5"]
    assert res["by_stage"] == by_stage(res["read"], time_split.CUTS)
    assert len(by_stage(res["read"], res["cuts"])) == 5


def test_chain_reads_and_summary_on_stage_folders(jobs, synthetic_npz, tmp_path):
    root, r = jobs
    for v, kw in (("wrong", {}), ("steps", dict(plan="width", n=6))):
        out = chain_reads.main(variant=v, base=str(root / "copies_3"), data=synthetic_npz, out=str(tmp_path / v),
                               device="cpu", heldout=32, score_n=32, score_steps=2, judge="fake", **kw)
        assert isinstance(out["loss_over_levels"], float)
    text = stages.summarize(str(root))
    lines = [x for x in text.splitlines() if x.startswith("| ") and not x.startswith("| variant")]
    jobs_n = sum(1 for n in r if not n.startswith("ts_"))
    readings = sum(1 + len(v["averaged"]) for n, v in r.items() if not n.startswith("ts_"))
    assert len(lines) == readings and jobs_n == 10
    assert any(x.startswith("| grow | 0 | 1 |") for x in lines)
    with pytest.raises(ValueError):
        stages.main(variant="copies", data=synthetic_npz, out=str(tmp_path / "x"), device="cpu")
    with pytest.raises(ValueError):
        stages.main(variant="source", data=synthetic_npz, out=str(tmp_path / "y"), device="cpu", cuts="0.5")
