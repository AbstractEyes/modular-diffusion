"""Determinism options and the CPU load (T8), and the entry point's new experiments (T9)."""
import json
import os
import time

import pytest
import torch

from geolip.modular_diffusion import determinism, run
from geolip.modular_diffusion.experiments import closed_forms, content, load, stages, time_split

SMALL = dict(seed=0, device="cpu", scale=0.0005, dim=32, heldout=32, batch=8, log_every=1, warmup=1, score=0,
             sheet=0, threads=4)


def test_T8_det_warn_writes_the_list_and_restores_the_flag(synthetic_npz, tmp_path):
    assert not torch.are_deterministic_algorithms_enabled()
    out = str(tmp_path / "warn")
    r = time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=out, det="warn"))
    assert isinstance(r["nondeterministic"], list) and r["args"]["det"] == "warn"
    with open(os.path.join(out, "result.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored["nondeterministic"] == r["nondeterministic"]
    assert not torch.are_deterministic_algorithms_enabled()          # put back when the job ends
    plain = time_split.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=str(tmp_path / "plain")))
    assert "nondeterministic" not in plain and "det" not in plain["args"]
    assert plain["loss_over_levels"] == r["loss_over_levels"]


def test_T8_det_collects_warnings_raised_in_the_job():
    with determinism.mode("warn") as det:
        assert torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
        import warnings
        warnings.warn("a test message")
        warnings.warn("a test message")
        res = {}
        det.record(res)
    assert res["nondeterministic"] == ["a test message"]
    with determinism.mode("on"):                 # det=on takes the deterministic algorithms, it does not only warn
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.are_deterministic_algorithms_enabled()
    with pytest.raises(ValueError):
        determinism.check("sometimes")


def test_T8_det_on_sets_the_workspace_before_the_job(synthetic_npz, tmp_path, monkeypatch):
    monkeypatch.delenv(determinism.WORKSPACE_KEY, raising=False)
    seen = {}
    real = time_split.PicturesBed

    def bed_spy(*a, **k):                     # the job's first tensor on the device is the bed's
        seen["at_bed"] = os.environ.get(determinism.WORKSPACE_KEY)
        return real(*a, **k)

    monkeypatch.setattr(time_split, "PicturesBed", bed_spy)
    kw = dict(SMALL, variant="plain", data=synthetic_npz, out=str(tmp_path / "on"), det="on")
    r = run.main("time_split", [f"{k}={v}" for k, v in kw.items()])
    assert seen["at_bed"] == ":4096:8" and os.environ[determinism.WORKSPACE_KEY] == ":4096:8"
    assert r["workspace"]["found"] == ":4096:8"                     # run.main set it before the module ran
    assert r["workspace"]["card_touched_before"] is False and isinstance(r["nondeterministic"], list)
    monkeypatch.delenv(determinism.WORKSPACE_KEY)
    r2 = content.main(**dict(SMALL, variant="plain", data=synthetic_npz, out=str(tmp_path / "on2"), det="on"))
    assert r2["workspace"]["found"] is None and r2["workspace"]["value"] == ":4096:8"
    assert r2["args"]["det"] == "on"


def test_T8_the_cpu_load(tmp_path):
    out = str(tmp_path / "load")
    t0 = time.time()
    r = run.main("load", ["threads=2", "seconds=2", f"out={out}"])
    spent = time.time() - t0
    assert 2.0 <= r["spent_seconds"] < 3.5 and 2.0 <= spent < 6.0
    assert r["threads"] == 2 and r["seconds"] == 2.0 and r["products"] > 1
    with open(os.path.join(out, "log.txt"), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines[0].startswith("LOAD running") and lines[-1] == "done"
    with open(os.path.join(out, "result.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    assert {"threads", "seconds", "products"} <= set(stored)
    assert "| 2 | 2 |" in load.summarize(str(tmp_path))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4") or 4))


def test_T9_the_entry_point_knows_the_new_experiments():
    assert "stages" in run.EXPERIMENTS and "load" in run.EXPERIMENTS
    for name in run.EXPERIMENTS:
        mod = __import__(f"geolip.modular_diffusion.experiments.{name}", fromlist=["main"])
        assert callable(mod.main) and callable(mod.summarize)
    from geolip.modular_diffusion.experiments import chain_reads
    for mod in (time_split, content, stages):
        assert "score_last" in mod.DEFAULTS and "score_avg" in mod.DEFAULTS
    for mod in (time_split, content, closed_forms, stages, chain_reads, load):
        assert "det" in mod.DEFAULTS
