"""Unchanged numbers (T0): four small jobs on the CIFAR-10 pictures give the numbers recorded from the library before
its sixth pass (the read over the ten levels, by level, the score, and the content hash of every unit of every build
folder the job wrote).

The recorded file is found through MD_BEFORE_JSON and the jobs are written under MD_T0_ROOT (the tests skip when either,
or MD_CIFAR_NPZ, is unset). The jobs take minutes each on a CPU, so each is its own test, and a later job reads the
folder of an earlier one: run them in the order a, b, c_source, c, d (one command each keeps every command short).
``capture`` runs all four in one call (the code that wrote the recorded file, run on the earlier library).
"""
import json
import os

import pytest

SMALL = dict(seed=0, device="cpu", scale=0.002, judge="fake", score=1, heldout=64, score_n=64, sheet=0, threads=4)
FOLDERS = {"a": "a_time_split_plain", "b": "b_content_plain_avg", "c_source": "c_source_plain_4k",
           "c": "c_time_split_copies_4k", "d": "d_content_experts_start"}


def _hashes(job: str) -> dict:
    """{build folder: {unit: content hash}} for every build folder under a job folder."""
    out = {}
    for dirpath, _, files in os.walk(job):
        if "build.json" in files:
            with open(os.path.join(dirpath, "build.json"), encoding="utf-8") as fh:
                b = json.load(fh)
            out[os.path.relpath(dirpath, job).replace("\\", "/")] = {u["name"]: u["hash"] for u in b["units"]}
    return out


def _numbers(r: dict, job: str) -> dict:
    sc = r.get("score") or {}
    d = {"loss_over_levels": r["loss_over_levels"], "by_level": r["by_level"], "score": sc.get("score"),
         "hashes": _hashes(job)}
    if r.get("averaged"):
        d["averaged"] = {k: {"loss_over_levels": v["loss_over_levels"], "by_level": v["by_level"],
                             "by_stage": v["by_stage"], "score": (v.get("score") or {}).get("score")}
                         for k, v in r["averaged"].items()}
    return d


def run_job(which: str, npz: str, root: str) -> dict:
    """One of the jobs (a, b, c_source, c, d) into root/<its folder> -> its numbers."""
    from geolip.modular_diffusion.experiments import content, time_split
    kw = dict(SMALL, data=npz, ref=os.path.join(root, "reference.npz"))
    j = os.path.join(root, FOLDERS[which])
    if which == "a":
        r = time_split.main(**dict(kw, variant="plain", out=j))
    elif which == "b":
        r = content.main(**dict(kw, variant="plain", out=j, avg="0.999"))
    elif which == "c_source":
        r = time_split.main(**dict(kw, variant="plain_4k", out=j))
    elif which == "c":
        r = time_split.main(**dict(kw, variant="copies_4k", out=j, source=os.path.join(root, FOLDERS["c_source"])))
    else:
        r = content.main(**dict(kw, variant="experts_start", out=j, source=os.path.join(root, FOLDERS["b"])))
    return _numbers(r, j)


def capture(npz: str, root: str) -> dict:
    """Run the four jobs (and the source of c) into ``root`` and return the numbers of a, b, c and d."""
    out = {}
    for which in ("a", "b", "c_source", "c", "d"):
        nums = run_job(which, npz, root)
        if which != "c_source":
            out[which] = nums
    return out


@pytest.mark.parametrize("which", ["a", "b", "c_source", "c", "d"])
def test_T0_the_recorded_numbers_are_unchanged(which):
    path = os.environ.get("MD_BEFORE_JSON", "")
    root = os.environ.get("MD_T0_ROOT", "")
    npz = os.environ.get("MD_CIFAR_NPZ", "")
    if not path or not root or not npz:
        pytest.skip("MD_BEFORE_JSON, MD_T0_ROOT or MD_CIFAR_NPZ is not set")
    with open(path, encoding="utf-8") as fh:
        before = json.load(fh)
    os.makedirs(root, exist_ok=True)
    for need in {"c": ["c_source"], "d": ["b"]}.get(which, []):
        if not os.path.exists(os.path.join(root, FOLDERS[need], "result.json")):
            pytest.skip(f"run the job {need} first (it is the source of {which})")
    now = json.loads(json.dumps(run_job(which, npz, root)))
    if which != "c_source":
        assert now == before[which], which
