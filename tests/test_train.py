"""Whole runs: smoke runs against the original script (P5), the recorded CPU run (P6), repeatability (test 12) and the
two ways in (test 14)."""
import json
import math
import os
import subprocess
import sys

import pytest
import torch

from geolip.modular_diffusion import config, train
from geolip.modular_diffusion.train import RUN_PREFIX

LEGACY_PREFIX = "mr_"                    # the original script's output-name prefix
SECONDS = {"train_seconds", "seconds"}


def _same(a, b, path="$", skip=SECONDS):
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), (path, list(a), list(b) if isinstance(b, dict) else b)
        for k in a:
            if k in skip:
                continue
            _same(a[k], b[k], f"{path}.{k}", skip)
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]", skip)
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), path
    else:
        assert a == b, (path, a, b)


def _compare_results(old, new, out_old, out_new):
    """The result dicts equal except run seconds, the output folder and the file-name prefix; the flat weights equal."""
    old, new = dict(old), dict(new)
    assert new["tag"] == RUN_PREFIX + old["tag"][len(LEGACY_PREFIX):]
    tag_old, tag_new = old.pop("tag"), new.pop("tag")
    args_old = {k: v for k, v in old.pop("args").items() if k not in ("src", "out")}
    args_new = {k: v for k, v in new.pop("args").items() if k != "out"}
    assert args_old == args_new
    _same(old, new)
    w_old = torch.load(os.path.join(out_old, tag_old + ".pt"))
    w_new = torch.load(os.path.join(out_new, tag_new + ".pt"))
    assert list(w_old) == list(w_new)
    for k in w_old:
        assert torch.equal(w_old[k], w_new[k]), k


P5_CASES = {
    "pictures_base": ["bed=pictures", "anchor=1", "bands=3", "keep=1", "anchor_in=1"],
    "pictures_stage_soft_mdrop": ["bed=pictures", "anchor=1", "stages=3", "soft=0.05", "mdrop=0.25", "stage_body=3",
                                  "stage_soft=0.06", "content=0.25-0.65:2", "read_at=2"],
    "fields_base": ["bed=fields", "anchor=1", "bands=3", "keep=1"],
}


@pytest.mark.parametrize("case", sorted(P5_CASES))
def test_p5_smoke_run_matches_original(legacy, geofractal_src, synthetic_npz, tmp_path, case):
    common = ["smoke=1", "device=cpu", "seeds=0", f"data={synthetic_npz}"] + P5_CASES[case]
    out_old, out_new = str(tmp_path / "old"), str(tmp_path / "new")
    old = legacy.main(common + [f"out={out_old}", f"src={geofractal_src[0]}"])
    new = train.main(common + [f"out={out_new}"])
    assert len(old) == len(new) == 1
    _compare_results(old[0], new[0], out_old, out_new)
    with open(os.path.join(out_new, "log.txt"), encoding="utf-8") as fh:
        assert fh.read().splitlines()[-1] == "done"


def test_p6_recorded_cpu_run(cifar_npz, stored_runs, tmp_path):
    """The recorded 4-step CPU run on the real data: the same numbers to the last digit, the same weights."""
    out = str(tmp_path / "run")
    res = train.main(["smoke=1", "bed=pictures", "anchor=1", "bands=3", "keep=1", "anchor_in=1", "name=ain1",
                      "device=cpu", f"data={cifar_npz}", f"out={out}"])[0]
    assert res["loss_over_levels"] == 0.6963074098164579
    assert res["heldout_loss"] == 0.7686708960682154
    assert res["curve"] == [[0, 0.7040558457374573], [2, 0.6315953135490417], [3, 0.6559357643127441]]
    from conftest import find_stored, STORED_SMOKE_PT
    stored_pt = find_stored(stored_runs, "_pictures_ain1_s0_steps4_d128.pt", STORED_SMOKE_PT)
    with open(stored_pt[:-3] + ".json", encoding="utf-8") as fh:
        stored = json.load(fh)
    stored_args = {k: v for k, v in stored["args"].items() if k not in ("src", "out", "data")}
    got_args = {k: v for k, v in res["args"].items() if k not in ("out", "data")}
    # keys the recorded version did not have yet are at their defaults
    extra = {k: got_args.pop(k) for k in list(got_args) if k not in stored_args}
    assert extra == {k: train.DEFAULTS[k] for k in extra}
    assert stored_args == got_args
    mine = json.loads(json.dumps({k: v for k, v in res.items() if k not in ("tag", "args")}))
    _same({k: v for k, v in stored.items() if k not in ("tag", "args")}, mine)
    w_old = torch.load(stored_pt)
    w_new = torch.load(os.path.join(out, res["tag"] + ".pt"))
    assert list(w_old) == list(w_new)
    for k in w_old:
        assert torch.equal(w_old[k], w_new[k]), k


def test_12_same_seed_twice(tmp_path):
    kw = dict(bed="fields", steps=6, batch=16, heldout=64, log_every=2, anchor=1, stages="3", device="cpu")
    r1 = train.main(out=str(tmp_path / "a"), **kw)[0]
    r2 = train.main(out=str(tmp_path / "b"), **kw)[0]
    r1["args"].pop("out")
    r2["args"].pop("out")
    _same(r1, r2)
    w1 = torch.load(os.path.join(tmp_path / "a", r1["tag"] + ".pt"))
    w2 = torch.load(os.path.join(tmp_path / "b", r2["tag"] + ".pt"))
    assert list(w1) == list(w2) and all(torch.equal(w1[k], w2[k]) for k in w1)
    assert any(float(v.abs().max()) > 0 for k, v in w1.items() if k.endswith("head.proj.weight"))   # it trained


def test_14_parse_both_ways():
    a = config.parse(train.DEFAULTS, ["steps=7", "soft=0.05", "stages=3", "junk"])
    b = config.parse(train.DEFAULTS, steps=7, soft=0.05, stages=3)
    assert a == b
    assert a["steps"] == 7 and isinstance(a["soft"], float) and a["stages"] == "3"
    c = config.parse(train.DEFAULTS, ["mdrop=0"], lr=1)
    assert c["mdrop"] == 0.0 and isinstance(c["mdrop"], float) and c["lr"] == 1.0
    with pytest.raises(ValueError):
        config.parse(train.DEFAULTS, ["nope=1"])
    with pytest.raises(ValueError):
        config.parse(train.DEFAULTS, nope=1)
    assert config.in_notebook() is False


def test_14_main_ignores_sys_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest", "-k=bogus", "steps=999999", "unknown=1"])
    res = train.main(bed="fields", steps=2, batch=8, heldout=32, device="cpu", out=str(tmp_path))
    assert res[0]["args"]["steps"] == 2


def test_14_module_entry_point(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", HF_TOKEN="")
    proc = subprocess.run([sys.executable, "-m", "geolip.modular_diffusion.train", "bed=fields", "steps=4", "device=cpu",
                           "heldout=64", f"out={tmp_path}"], capture_output=True, text=True, env=env, timeout=200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "done"
