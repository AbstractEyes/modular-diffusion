"""The chain-reads experiment end to end: finished time-split jobs (one model, clusters, clusters with whole-range draws;
two seeds) at a tiny scale, then every variant read from them with the test judge."""
import json
import os

import pytest

from geolip.modular_diffusion import run
from geolip.modular_diffusion.experiments import chain_reads, time_split
from geolip.modular_diffusion.read import T_GRID
from geolip.modular_diffusion.sample import share_counts, stage_grid, uniform_grid

JOB = dict(device="cpu", scale=0.002, dim=32, heldout=32, batch=8, log_every=50, warmup=1, score=1, judge="fake",
           score_n=32, score_steps=50, sheet=0, threads=4)
READ = dict(device="cpu", heldout=32, score=1, judge="fake", score_n=32, score_steps=50, threads=4)
RESULT_KEYS = {"experiment", "variant", "seed_label", "base", "donor", "stage", "soft", "plan", "n", "grid", "numbers",
               "loss_over_levels", "by_level", "by_stage", "score", "seconds", "args", "torch", "device", "version"}
CUT_LEVELS = ("0.25", "0.65")


def _snapshot(folder):
    out = {}
    for dirpath, _, files in os.walk(folder):
        for n in files:
            p = os.path.join(dirpath, n)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, folder)] = fh.read()
    return out


def _stored(folder):
    with open(os.path.join(folder, "result.json"), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("jobs")
    for seed in (0, 1):
        for v in ("plain", "split", "split_soft"):
            time_split.main(**dict(JOB, variant=v, seed=seed, data=synthetic_npz, out=str(root / f"seed{seed}" / v)))
    return root


@pytest.fixture(scope="module")
def reads(jobs, synthetic_npz, tmp_path_factory):
    out_root = tmp_path_factory.mktemp("reads")
    before = _snapshot(str(jobs))
    j = lambda seed, v: str(jobs / f"seed{seed}" / v).replace("\\", "/")
    plan = {
        "twin_self": dict(variant="twin", base=j(0, "split"), donor=j(0, "split"), stage=1),
        "twin_seed1": dict(variant="twin", base=j(0, "split"), donor=j(1, "split"), stage=1),
        "twin_soft": dict(variant="twin", base=j(0, "split"), donor=j(0, "split_soft"), stage=2),
        "wrong": dict(variant="wrong", base=j(0, "split")),
        "blend_006": dict(variant="blend", base=j(0, "split"), soft=0.06),
        "blend_0": dict(variant="blend", base=j(0, "split"), soft=0.0),
        "blend_012": dict(variant="blend", base=j(0, "split"), soft=0.12),
        "steps_uniform": dict(variant="steps", base=j(0, "plain"), plan="uniform", n=50),
        "steps_width": dict(variant="steps", base=j(0, "split"), plan="width", n=50),
        "steps_finish": dict(variant="steps", base=j(1, "plain"), plan="finish", n=10),
    }
    for k in range(3):
        plan[f"swap_{k}"] = dict(variant="swap", base=j(0, "plain"), donor=j(0, "split"), stage=k)
    results = {}
    for name, kw in plan.items():
        kw = dict(READ, data=synthetic_npz, out=str(out_root / name), seed_label="seed 0", **kw)
        if name == "wrong":                                      # once through the entry point with key=value items
            results[name] = run.main("chain_reads", [f"{k}={v}" for k, v in kw.items()])
        else:
            results[name] = chain_reads.main(**kw)
    untouched = _snapshot(str(jobs)) == before
    return jobs, out_root, results, untouched


def test_every_read_writes_its_result_and_writes_no_weights(reads):
    jobs, out_root, results, untouched = reads
    assert untouched                                             # the job folders read are byte-identical
    for name, r in results.items():
        folder = out_root / name
        stored = _stored(folder)
        assert set(stored) == RESULT_KEYS, name
        assert stored["experiment"] == "chain_reads" and stored["seed_label"] == "seed 0"
        assert stored["loss_over_levels"] == r["loss_over_levels"]
        assert set(stored["by_level"]) == {f"{t:.2f}" for t in T_GRID}
        assert len(stored["by_stage"]) == 3 and all(isinstance(x, float) for x in stored["by_stage"])
        assert stored["score"]["n"] == 32 and stored["score"]["judge"].startswith("fake"), name
        assert stored["numbers"] > 0
        with open(folder / "log.txt", encoding="utf-8") as fh:
            assert fh.read().splitlines()[-1] == "done"
        assert sorted(os.listdir(folder)) == ["log.txt", "result.json"], name
    assert results["steps_width"]["grid"] is not None and results["twin_self"]["grid"] is None
    assert results["blend_0"]["soft"] == 0.0 and results["swap_1"]["stage"] == 1 and results["wrong"]["stage"] is None


def test_twin_of_itself_blend_at_the_trained_width_and_wrong_reproduce_the_job(reads):
    jobs, _, results, _ = reads
    base = _stored(jobs / "seed0" / "split")
    for name in ("twin_self", "blend_006"):
        r = results[name]
        assert r["loss_over_levels"] == base["loss_over_levels"], name
        assert r["by_stage"] == base["by_stage"], name
        assert r["by_level"] == base["by_level"], name
    for name, level in (("blend_0", "0.25"), ("blend_012", "0.15")):   # a cut weighs 0.5 / 0.5 at any half-width
        r = results[name]
        assert r["loss_over_levels"] != base["loss_over_levels"], name
        assert r["by_level"][level] != base["by_level"][level], name
    assert results["blend_012"]["by_level"]["0.05"] == base["by_level"]["0.05"]    # outside the wider crossfade
    assert results["wrong"]["loss_over_levels"] == base["wrong_assembly"]["loss_over_levels"]
    assert results["wrong"]["by_stage"] == base["wrong_assembly"]["by_stage"]
    assert results["twin_seed1"]["loss_over_levels"] != base["loss_over_levels"]
    assert results["twin_self"]["numbers"] == base["numbers"]["all"]


def test_twin_changes_only_its_stage(reads):
    jobs, _, results, _ = reads
    base = _stored(jobs / "seed0" / "split")
    donor = _stored(jobs / "seed1" / "split")
    r = results["twin_seed1"]
    for t in chain_reads.levels_outside(1):
        if f"{t:.2f}" not in CUT_LEVELS:
            assert abs(r["by_level"][f"{t:.2f}"] - base["by_level"][f"{t:.2f}"]) <= 1e-6, t
    for t in chain_reads.levels_inside(1):
        assert abs(r["by_level"][f"{t:.2f}"] - donor["by_level"][f"{t:.2f}"]) <= 1e-6, t


def test_swap_by_level(reads):
    jobs, _, results, _ = reads
    plain = _stored(jobs / "seed0" / "plain")
    donor = _stored(jobs / "seed0" / "split")
    assert chain_reads.levels_inside(0) == [0.05, 0.15]
    assert chain_reads.levels_inside(1) == [0.35, 0.45, 0.55]
    assert chain_reads.levels_inside(2) == [0.75, 0.85, 0.95]
    assert chain_reads.levels_outside(0) == [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    assert chain_reads.levels_outside(1) == [0.05, 0.15, 0.75, 0.85, 0.95]
    assert chain_reads.levels_outside(2) == [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    for k in range(3):
        r = results[f"swap_{k}"]
        compared = 0
        for t in chain_reads.levels_outside(k):
            key = f"{t:.2f}"
            if key not in CUT_LEVELS:
                assert abs(r["by_level"][key] - plain["by_level"][key]) <= 1e-6, (k, key)
                compared += 1
        for t in chain_reads.levels_inside(k):
            key = f"{t:.2f}"
            assert abs(r["by_level"][key] - donor["by_level"][key]) <= 1e-6, (k, key)
            compared += 1
        assert compared == 8                                     # every fixed level but the two cuts
        assert r["numbers"] == plain["numbers"]["all"] + donor["numbers"]["all"] // 3   # the base model counted once


def test_steps(reads):
    jobs, _, results, _ = reads
    plain = _stored(jobs / "seed0" / "plain")
    r = results["steps_uniform"]
    assert r["grid"] == uniform_grid(50) and r["score"]["steps"] == 50
    assert r["loss_over_levels"] == plain["loss_over_levels"] and r["by_level"] == plain["by_level"]
    assert plain["score"]["steps"] == 50
    assert abs(r["score"]["score"] - plain["score"]["score"]) <= 1e-4 * abs(plain["score"]["score"])
    # 50 x (0.25, 0.40, 0.35) = 12.5 / 20 / 17.5: the tie of the remainders goes to the earlier stage
    assert share_counts(chain_reads.PLANS["width"], 50) == [13, 20, 17]
    w = results["steps_width"]
    assert w["grid"] == stage_grid(time_split.CUTS, [13, 20, 17]) and w["score"]["grid"] == w["grid"]
    assert w["loss_over_levels"] == _stored(jobs / "seed0" / "split")["loss_over_levels"]
    f = results["steps_finish"]
    assert f["grid"] == stage_grid(time_split.CUTS, [2, 3, 5]) and f["score"]["steps"] == 10


def test_summarize_one_row_per_read(reads):
    _, out_root, results, _ = reads
    text = chain_reads.summarize(str(out_root))
    lines = [x for x in text.splitlines() if x.startswith("| ") and not x.startswith("| variant")]
    assert len(lines) == len(results)
    assert any(x.startswith("| swap | seed0/plain | seed0/split | 2 |") for x in lines)
    assert any(x.startswith("| blend | seed0/split | - | - | 0.12 |") for x in lines)
    assert chain_reads.summarize(str(reads[0])).count("\n") == 2     # time-split results are not rows of this table


def test_a_failing_judge_is_recorded_and_the_read_still_ends(jobs, synthetic_npz, tmp_path, monkeypatch):
    def broken(kind, device="cpu"):
        raise RuntimeError("the judge's weights could not be fetched")
    monkeypatch.setattr(chain_reads, "make_judge", broken)
    out = tmp_path / "wrong"
    r = chain_reads.main(**dict(READ, variant="wrong", base=str(jobs / "seed0" / "split"), data=synthetic_npz,
                                out=str(out)))
    assert set(r["score"]) == {"error"} and "could not be fetched" in r["score"]["error"]
    assert isinstance(r["loss_over_levels"], float)
    with open(out / "log.txt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines[-1] == "done" and any(x.startswith("SCORE not computed") for x in lines)
    assert "| - |" in chain_reads.summarize(str(tmp_path))


def test_refusals(jobs, synthetic_npz, tmp_path):
    split, plain = str(jobs / "seed0" / "split"), str(jobs / "seed0" / "plain")
    common = dict(READ, data=synthetic_npz)
    bad = [dict(variant="nope", base=split), dict(variant="twin", base=split, stage=1),
           dict(variant="twin", base=split, donor=split), dict(variant="swap", base=plain, donor=split, stage=3),
           dict(variant="blend", base=split), dict(variant="steps", base=split, plan="sideways", n=10),
           dict(variant="steps", base=split, plan="width", n=2), dict(variant="wrong", base=split, judge="nope"),
           dict(variant="wrong")]
    for i, kw in enumerate(bad):
        with pytest.raises(ValueError):
            chain_reads.main(**dict(common, out=str(tmp_path / f"refused_{i}"), **kw))
        assert not os.path.exists(tmp_path / f"refused_{i}"), kw   # refused before anything is written
    for i, kw in enumerate([dict(variant="swap", base=split, donor=split, stage=0),
                            dict(variant="swap", base=plain, donor=plain, stage=0),
                            dict(variant="wrong", base=plain), dict(variant="steps", base=str(tmp_path), plan="uniform", n=5)]):
        with pytest.raises(ValueError):                          # the wrong kind of job
            chain_reads.main(**dict(common, out=str(tmp_path / f"kind_{i}"), **kw))
    with pytest.raises(ValueError):
        time_split.load_job(str(jobs / "seed0" / "split"), kind="other")
    model, kind = time_split.load_job(str(jobs / "seed0" / "plain"))
    assert kind == "model" and len(model.units) == 3


def test_the_sampling_seed_is_an_argument(jobs, synthetic_npz, tmp_path):
    """The same model scored with two noise seeds: the seed is recorded and the pictures (so the score) differ; the
    default seed is the library's."""
    from geolip.modular_diffusion.score import SCORE_SEED
    plain = str(jobs / "seed0" / "plain").replace("\\", "/")
    kw = dict(READ, variant="steps", base=plain, plan="uniform", n=4, data=synthetic_npz)
    a = chain_reads.main(**dict(kw, out=str(tmp_path / "a")))
    b = chain_reads.main(**dict(kw, out=str(tmp_path / "b"), score_seed=7))
    c = chain_reads.main(**dict(kw, out=str(tmp_path / "c"), score_seed=7))
    assert a["score"]["seed"] == SCORE_SEED and b["score"]["seed"] == 7
    assert b["score"]["score"] == c["score"]["score"] != a["score"]["score"]
    assert a["loss_over_levels"] == b["loss_over_levels"]
