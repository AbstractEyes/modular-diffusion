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


# The seventh pass: thirteen more small jobs recorded from the library before that pass (MD_BEFORE7_JSON, run under
# MD_T07_ROOT). A job's sources come first (the order of P7_ORDER); every job's numbers are recorded, sources included.
P7_ORDER = ["a", "b", "c_source", "c", "d_rand", "d_shared", "e_source", "e", "f_source", "f1", "f3", "g", "h"]
P7_FOLDERS = {k: "p7_" + k for k in P7_ORDER}
P7_NEEDS = {"c": ["c_source"], "d_rand": ["b"], "d_shared": ["b"], "e": ["e_source"], "f1": ["f_source"],
            "f3": ["f_source"], "g": ["c_source"]}


def run_job7(which: str, npz: str, root: str) -> dict:
    """One job of the seventh pass's record into root/p7_<which> -> its numbers."""
    from geolip.modular_diffusion.experiments import closed_forms, content, stages, time_split
    kw = dict(SMALL, data=npz, ref=os.path.join(root, "reference.npz"))
    j = os.path.join(root, P7_FOLDERS[which])
    src = lambda k: os.path.join(root, P7_FOLDERS[k])
    if which == "a":
        r = time_split.main(**dict(kw, variant="plain", out=j))
    elif which == "b":
        r = content.main(**dict(kw, variant="plain", out=j, avg="g8"))
    elif which == "c_source":
        r = content.main(**dict(kw, variant="blind", out=j, avg="g8"))
    elif which == "c":
        r = content.main(**dict(kw, variant="blind_experts_all", out=j, source=src("c_source"),
                                source_model="model_avg_g8"))
    elif which == "d_rand":
        r = content.main(**dict(kw, variant="rand_start", out=j, source=src("b")))
    elif which == "d_shared":
        r = content.main(**dict(kw, variant="shared_start", out=j, source=src("b"), source_model="model_avg_g8"))
    elif which == "e_source":
        r = stages.main(**dict(kw, variant="source", out=j))
    elif which == "e":
        r = stages.main(**dict(kw, variant="copies", cuts="0.25,0.65", out=j, source=src("e_source")))
    elif which == "f_source":
        r = time_split.main(**dict(kw, variant="plain_d4", out=j, avg="g8"))
    elif which == "f1":
        r = stages.main(**dict(kw, variant="tails", out=j, stem_from=src("f_source"), stem_model="model_avg_g8"))
    elif which == "f3":
        r = stages.main(**dict(kw, variant="tails", cuts="0.25,0.65", out=j, stem_from=src("f_source"),
                               stem_model="model_avg_g8"))
    elif which == "g":
        r = closed_forms.main(**dict(kw, variant="class_stage_heads", out=j, source=src("c_source"),
                                     source_model="model_avg_g8", batches=20))
    else:
        r = time_split.main(**dict(kw, variant="tails_mixed", out=j, avg="g8"))
    return _numbers(r, j)


@pytest.mark.parametrize("which", P7_ORDER)
def test_T0_the_numbers_recorded_before_the_seventh_pass_are_unchanged(which):
    path = os.environ.get("MD_BEFORE7_JSON", "")
    root = os.environ.get("MD_T07_ROOT", "")
    npz = os.environ.get("MD_CIFAR_NPZ", "")
    if not path or not root or not npz:
        pytest.skip("MD_BEFORE7_JSON, MD_T07_ROOT or MD_CIFAR_NPZ is not set")
    with open(path, encoding="utf-8") as fh:
        before = json.load(fh)
    os.makedirs(root, exist_ok=True)
    for need in P7_NEEDS.get(which, []):
        if not os.path.exists(os.path.join(root, P7_FOLDERS[need], "result.json")):
            pytest.skip(f"run the job {need} first (it is the source of {which})")
    now = json.loads(json.dumps(run_job7(which, npz, root)))
    assert now == before[which], which


# The eighth pass: fourteen more small jobs recorded from the library before that pass (MD_BEFORE8_JSON, run under
# MD_T08_ROOT), sources first (the order of P8_ORDER); every job's numbers are recorded, sources included.
P8_ORDER = ["a", "b", "c", "d", "e", "f", "g_tails", "g", "h_src0", "h_src1", "h_c0", "h_c1", "h_twin", "h_steps"]
P8_FOLDERS = {k: "p8_" + k for k in P8_ORDER}
P8_NEEDS = {"b": ["a"], "c": ["a"], "f": ["e"], "g_tails": ["a"], "g": ["g_tails"], "h_c0": ["h_src0"],
            "h_c1": ["h_src1"], "h_twin": ["h_c0", "h_c1"], "h_steps": ["h_c0"]}


def run_job8(which: str, npz: str, root: str) -> dict:
    """One job of the eighth pass's record into root/p8_<which> -> its numbers."""
    from geolip.modular_diffusion.experiments import chain_reads, content, stages, time_split
    # five sampling steps for the score: each job stays within a few minutes on a CPU
    kw = dict(SMALL, data=npz, ref=os.path.join(root, "reference.npz"), score_steps=5)
    j = os.path.join(root, P8_FOLDERS[which])
    src = lambda k: os.path.join(root, P8_FOLDERS[k])
    reads = dict(data=npz, ref=kw["ref"], device="cpu", judge="fake", score=1, heldout=64, score_n=64, score_steps=5,
                 sheet=0, threads=4)
    if which == "a":
        r = stages.main(**dict(kw, variant="source", out=j))
    elif which == "b":
        r = stages.main(**dict(kw, variant="copies", cuts="0.25,0.65", out=j, source=src("a")))
    elif which == "c":
        r = stages.main(**dict(kw, variant="copies", cuts="0.25,0.65", fresh=1, out=j, source=src("a")))
    elif which == "d":
        r = content.main(**dict(kw, variant="blind", out=j, avg="g8"))
    elif which == "e":
        r = content.main(**dict(kw, variant="blind9", out=j, avg="g8"))
    elif which == "f":
        r = content.main(**dict(kw, variant="snap9", out=j, source=src("e"), source_model="model_avg_g8"))
    elif which == "g_tails":
        r = stages.main(**dict(kw, variant="tails", cuts="0.25,0.65", out=j, stem_from=src("a"),
                               stem_model="model_avg_g8"))
    elif which == "g":
        r = stages.main(**dict(kw, variant="thaw", thaw="all", out=j, base=src("g_tails")))
    elif which in ("h_src0", "h_src1"):
        r = time_split.main(**dict(kw, variant="plain_4k", seed=int(which[-1]), out=j))
    elif which in ("h_c0", "h_c1"):
        r = time_split.main(**dict(kw, variant="copies_4k", seed=int(which[-1]), out=j,
                                   source=src("h_src" + which[-1])))
    elif which == "h_twin":
        r = chain_reads.main(**dict(reads, variant="twin", base=src("h_c0"), donor=src("h_c1"), stage=1, out=j))
    else:
        r = chain_reads.main(**dict(reads, variant="steps", base=src("h_c0"), plan="start", n=10, out=j))
    return _numbers(r, j)


@pytest.mark.parametrize("which", P8_ORDER)
def test_T0_the_numbers_recorded_before_the_eighth_pass_are_unchanged(which):
    path = os.environ.get("MD_BEFORE8_JSON", "")
    root = os.environ.get("MD_T08_ROOT", "")
    npz = os.environ.get("MD_CIFAR_NPZ", "")
    if not path or not root or not npz:
        pytest.skip("MD_BEFORE8_JSON, MD_T08_ROOT or MD_CIFAR_NPZ is not set")
    with open(path, encoding="utf-8") as fh:
        before = json.load(fh)
    os.makedirs(root, exist_ok=True)
    for need in P8_NEEDS.get(which, []):
        if not os.path.exists(os.path.join(root, P8_FOLDERS[need], "result.json")):
            pytest.skip(f"run the job {need} first (it is the source of {which})")
    now = json.loads(json.dumps(run_job8(which, npz, root)))
    assert now == before[which], which
