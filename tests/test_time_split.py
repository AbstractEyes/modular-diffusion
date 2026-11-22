"""The time-split experiment end to end (test 11): every variant on the synthetic pictures with tiny step counts, a
narrow width, a small held-out set and the test judge on 32 pictures."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion import run
from geolip.modular_diffusion.experiments import time_split
from geolip.modular_diffusion.staged import link_staged, read_staged

SMALL = dict(seed=0, device="cpu", scale=0.0005, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=1,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)
SOURCES = {"tails_alone": ("stem_from", "plain_d4"), "copies_4k": ("source", "plain_4k"), "copies": ("source", "plain"),
           "copies_soft": ("source", "plain")}
RESULT_KEYS = {"experiment", "variant", "seed", "kind", "cuts", "soft", "parts", "numbers", "train_seconds", "read",
               "loss_over_levels", "by_level", "by_stage", "wrong_assembly", "alone", "score", "args", "torch", "device",
               "version"}


def _snapshot(folder):
    out = {}
    for dirpath, _, files in os.walk(folder):
        for n in files:
            p = os.path.join(dirpath, n)
            out[os.path.relpath(p, folder)] = open(p, "rb").read()
    return out


@pytest.fixture(scope="module")
def runs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("time_split")
    order = ["plain_d4", "plain_4k", "plain"] + [v for v in time_split.VARIANTS if v not in ("plain_d4", "plain_4k", "plain")]
    results, untouched = {}, {}
    for v in order:
        kw = dict(SMALL, variant=v, data=synthetic_npz, out=str(root / v))
        if v in SOURCES:
            key, src = SOURCES[v]
            kw[key] = str(root / src)
            before = _snapshot(str(root / src / "model"))
        if v == "split":
            kw["sheet"] = 1
            results[v] = run.main("time_split", [f"{k}={val}" for k, val in kw.items()])
        else:
            results[v] = time_split.main(**kw)
        if v in SOURCES:
            untouched[v] = _snapshot(str(root / SOURCES[v][1] / "model")) == before
    return root, results, untouched


def test_11_every_variant_writes_its_result(runs):
    root, results, _ = runs
    assert set(results) == set(time_split.VARIANTS)
    for v, r in results.items():
        folder = root / v
        with open(folder / "result.json", encoding="utf-8") as fh:
            stored = json.load(fh)
        assert set(stored) == RESULT_KEYS, v
        assert stored["variant"] == v and stored["kind"] == time_split.kind_of(v)
        assert len(stored["by_stage"]) == 3 and all(isinstance(x, float) for x in stored["by_stage"])
        assert stored["score"]["n"] == 32 and stored["score"]["judge"].startswith("fake")
        assert stored["numbers"]["all"] >= stored["numbers"]["trained"] > 0
        with open(folder / "log.txt", encoding="utf-8") as fh:
            assert fh.read().splitlines()[-1] == "done"
        for p in stored["parts"]:
            assert p["steps"] >= 2 and "t_mix" in p
        if stored["kind"] == "model":
            assert os.path.exists(folder / "model" / "build.json") and stored["wrong_assembly"] is None
        else:
            d = read_staged(str(folder))
            assert d["clusters"] == ["stage_0", "stage_1", "stage_2"] and d["cuts"] == ["0.0", "0.25", "0.65", "1.0"]
            staged = link_staged(str(folder))
            assert len(staged.clusters) == 3
            assert set(stored["wrong_assembly"]) == {"loss_over_levels", "by_stage"}
            assert len(stored["alone"]) == 3 and all(set(a) == {"loss_over_levels", "by_level"} for a in stored["alone"])
    assert os.path.exists(root / "split" / "samples.png")


def test_11_variant_details(runs):
    root, results, untouched = runs
    assert untouched == {v: True for v in SOURCES}                      # sources left byte-identical
    assert [p["t_range"] for p in results["split"]["parts"]] == [list(r) for r in time_split.TRAIN_RANGES]
    assert [p["t_mix"] for p in results["split_soft"]["parts"]] == [time_split.SOFT_MIX] * 3
    assert [p["t_mix"] for p in results["copies_soft"]["parts"]] == [time_split.SOFT_MIX] * 3
    assert [p["t_mix"] for p in results["split"]["parts"]] == [0.0] * 3
    assert all(p["trainable"] == ["body", "head"] and p["frozen_unchanged"] for p in results["tails_alone"]["parts"])
    joint = link_staged(str(root / "tails_joint"))
    assert len({id(c.stem) for c in joint.clusters}) == 1 and len({id(c.body) for c in joint.clusters}) == 3
    alone = link_staged(str(root / "tails_alone"))
    assert len({id(c.stem) for c in alone.clusters}) == 1
    plain = results["plain"]
    assert results["plain_wide"]["numbers"]["all"] > plain["numbers"]["all"]
    split = results["split"]
    assert split["numbers"]["all"] == 3 * plain["numbers"]["all"]
    assert results["plain_mid"]["parts"][0]["steps"] == plain["parts"][0]["steps"]
    assert time_split.stage_steps(6000, 1.0) == [1500, 2400, 2100]
    assert time_split.stage_steps(2000, 1.0) == [500, 800, 700]


def test_11_summarize(runs):
    root, results, _ = runs
    text = time_split.summarize(str(root))
    lines = [x for x in text.splitlines() if x.startswith("| ") and not x.startswith("| variant")]
    assert len(lines) == len(results)
    assert any(x.startswith("| split | 0 |") for x in lines)


def test_11_missing_source_and_unknown_variant(synthetic_npz, tmp_path):
    with pytest.raises(ValueError):
        time_split.main(variant="copies", data=synthetic_npz, out=str(tmp_path / "x"), device="cpu")
    with pytest.raises(ValueError):
        time_split.main(variant="nope", data=synthetic_npz, out=str(tmp_path / "y"), device="cpu")
    with pytest.raises(ValueError):
        run.main("no_such_experiment")
    with pytest.raises(ValueError):                                     # an unknown judge is refused before any training
        time_split.main(variant="plain", data=synthetic_npz, out=str(tmp_path / "z"), device="cpu", score=1, judge="nope")
    assert not os.path.exists(tmp_path / "z" / "model")


def test_11_a_failing_judge_is_recorded_and_the_job_still_ends(synthetic_npz, tmp_path, monkeypatch):
    def broken(kind, device="cpu"):
        raise RuntimeError("the judge's weights could not be fetched")
    monkeypatch.setattr(time_split, "make_judge", broken)
    out = tmp_path / "plain"
    r = time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=str(out)))
    assert set(r["score"]) == {"error"} and "could not be fetched" in r["score"]["error"]
    assert isinstance(r["loss_over_levels"], float)
    with open(out / "log.txt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines[-1] == "done" and any(x.startswith("SCORE not computed") for x in lines)
    assert "| - |" in time_split.summarize(str(tmp_path))               # the summary shows no score for it


def test_reference_cache_is_written_whole(synthetic_npz, tmp_path):
    from geolip.modular_diffusion import score
    from geolip.modular_diffusion.beds import PicturesBed
    bed = PicturesBed(synthetic_npz, "cpu")
    path = str(tmp_path / "ref" / "reference.npz")
    a = score.reference_stats(bed, score.FakeJudge(), path, batch=256)
    assert sorted(os.listdir(tmp_path / "ref")) == ["reference.npz"]    # no temporary file is left beside it
    b = score.reference_stats(bed, score.FakeJudge(), path, batch=256)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all() and a[2] == b[2] == 600
