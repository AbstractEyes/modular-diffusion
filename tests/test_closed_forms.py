"""Heads solved by least squares (T6), the stored-state cache's passes (T7) and the closed-forms experiment end to end
(T8)."""
import json
import math
import os

import pytest
import torch

from geolip.modular_diffusion import closed, run
from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import read_build, read_weights
from geolip.modular_diffusion.cache import FeatureCache
from geolip.modular_diffusion.experiments import closed_forms, time_split
from geolip.modular_diffusion.forms import cluster_build
from geolip.modular_diffusion.linker import export_build, link
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.read import read
from geolip.modular_diffusion.train import fit_units, step_generator

FIT = dict(lr=1e-3, warmup=1, batch=16, cond_drop=0.1, heldout=32, device="cpu", log=lambda s: None, log_every=1)
BATCHES = 6
BATCH = 16


@pytest.fixture(scope="module")
def trained(synthetic_npz, tmp_path_factory):
    """A small trained one-head model, exported as a build folder."""
    bed = PicturesBed(synthetic_npz, "cpu")
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)
    res = fit_units(model, bed, 0, steps=5, expect_anchor_at_birth=True, **FIT)
    folder = str(tmp_path_factory.mktemp("trained") / "model")
    export_build(model, folder, seed=0, trained=["stem", "head"], recipe=res["recipe"],
                 marks={"loss_over_levels": res["loss_over_levels"], "frozen_unchanged": True})
    return bed, folder


@torch.no_grad()
def _direct_mse(model, bed, seed=0, batches=BATCHES, batch=BATCH, t_range=(0.0, 1.0)):
    """The training loss over the equations' batches, computed from the model's velocity (not from the equations)."""
    model.eval()
    total = 0.0
    for k in range(batches):
        gen = step_generator(seed, k, "cpu")
        x, cls = bed.train_batch(batch, gen)
        zt, t, c, target = model.draw(x, cls, gen, 0.1, "uniform", t_range)
        v = model.velocity(zt, t, c)
        total += float(((v.double() - target.double()) ** 2).mean())
    return total / batches


def test_T6_solved_head(trained):
    bed, folder = trained
    model = link(folder)
    direct_before = _direct_mse(model, bed)
    eq = closed.head_equations(model, bed, 0, BATCHES, batch=BATCH)
    assert eq["rows"] == BATCHES * BATCH * 256 and eq["G"].dtype == torch.float64
    w, b = closed.solve_head(eq, ridge=1e-6)
    prov = closed.install_head(model, w, b, eq)
    direct_after = _direct_mse(model, bed)
    assert direct_after <= direct_before
    assert abs(prov["mse_before"] - direct_before) / direct_before < 1e-6
    assert abs(prov["mse_after"] - direct_after) / direct_after < 1e-6
    assert prov["batches"] == BATCHES and prov["rows"] == eq["rows"] and prov["ridge"] == 1e-6
    assert prov["weight_norm_after"] > 0 and prov["t_range"] == [0.0, 1.0]
    assert model.head.proj.weight.dtype == torch.float32
    assert torch.equal(model.head.proj.weight, w.float()) and torch.equal(model.head.proj.bias, b.float())
    # solving twice changes nothing: the equations do not depend on the head
    eq2 = closed.head_equations(model, bed, 0, BATCHES, batch=BATCH)
    assert torch.equal(eq2["G"], eq["G"]) and torch.equal(eq2["R"], eq["R"])
    w2, b2 = closed.solve_head(eq2, ridge=1e-6)
    assert torch.equal(w2, w) and torch.equal(b2, b)
    before = model.head.proj.weight.detach().clone()
    closed.install_head(model, w2, b2, eq2)
    assert torch.equal(model.head.proj.weight, before)


def test_T6_stage_heads(trained, tmp_path):
    bed, folder = trained
    one = link(folder)
    closed.solve_model_head(one, bed, 0, BATCHES, batch=BATCH)
    model = link(folder)
    out = str(tmp_path / "staged")
    staged, provs = closed.solve_stage_heads(model, bed, 0, BATCHES, time_split.CUTS, time_split.SOFT,
                                             time_split.TRAIN_RANGES, out, base=folder, batch=BATCH)
    assert [p["t_range"] for p in provs] == [list(r) for r in time_split.TRAIN_RANGES]
    assert torch.equal(model.head.proj.weight, link(folder).head.proj.weight)      # the caller's model is untouched
    loaded, kind = time_split.load_job(out)
    assert kind == "clusters" and len(loaded.clusters) == 3
    assert len({id(c.stem) for c in loaded.clusters}) == 1 and len({id(c.head) for c in loaded.clusters}) == 3
    r_staged = read(loaded, bed, 32, "cpu")["loss_over_levels"]
    r_one = read(one, bed, 32, "cpu")["loss_over_levels"]
    assert math.isfinite(r_staged) and r_staged <= r_one + 1e-3, (r_staged, r_one)
    # each stage head is the solve on its own range (computed again here)
    for s, rng in enumerate(time_split.TRAIN_RANGES):
        c = link(folder)
        closed.solve_model_head(c, bed, 0, BATCHES, batch=BATCH, t_range=rng)
        assert torch.equal(c.head.proj.weight, loaded.clusters[s].head.proj.weight)


# ------------------------------------------------------------------ T7: the cache's passes
def test_T7_cache_passes(trained, tmp_path, monkeypatch):
    bed, folder = trained
    m1, names = closed_forms.members_model(folder, 0, bed)
    n = 3
    cfolder = str(tmp_path / "cache")
    cache = FeatureCache.write(m1, bed, 0, n, 8, cfolder, train=names)
    files = [f for f in os.listdir(cfolder) if f.startswith("step_")]
    assert len(files) == n
    assert cache.size_bytes() == sum(os.path.getsize(os.path.join(cfolder, f)) for f in files) > 0
    # passes=1 is the single pass
    m2, _ = closed_forms.members_model(folder, 0, bed)
    c1 = cache.fit(m1, lr=1e-3, warmup=2, log_every=1)
    c2 = FeatureCache.open(cfolder, m2).fit(m2, lr=1e-3, warmup=2, log_every=1, passes=1)
    assert c1 == c2 and len(c1) == n
    for nm in names:
        for k, v in m1.unit(nm).state_dict().items():
            assert torch.equal(v, m2.unit(nm).state_dict()[k])
    # two passes: step k reads stored step k mod n; the warm-up runs over k once
    m3, _ = closed_forms.members_model(folder, 0, bed)
    c3 = FeatureCache.open(cfolder, m3)
    seen, lrs = [], []
    real_batch = FeatureCache.batch

    def spy(self, step, device="cpu"):
        seen.append(step)
        return real_batch(self, step, device)

    real_step = torch.optim.Adam.step

    def step_spy(self, *a, **k):
        lrs.append(self.param_groups[0]["lr"])
        return real_step(self, *a, **k)

    monkeypatch.setattr(FeatureCache, "batch", spy)
    monkeypatch.setattr(torch.optim.Adam, "step", step_spy)
    curve = c3.fit(m3, lr=1e-3, warmup=2 * n, log_every=1, passes=2)
    assert seen == [0, 1, 2, 0, 1, 2]
    assert len(curve) == 2 * n
    assert lrs == pytest.approx([1e-3 * (k + 1) / (2 * n) for k in range(2 * n)])
    with pytest.raises(ValueError):
        c3.fit(m3, steps=2 * n + 1, passes=2)
    with pytest.raises(ValueError):
        c3.fit(m3, steps=n + 1)


# ------------------------------------------------------------------ T8: the experiment end to end
SMALL = dict(seed=0, device="cpu", heldout=32, batch=8, log_every=1, warmup=1, score=1, judge="fake", score_n=32,
             score_steps=2, sheet=0, threads=4)
SOURCE = dict(SMALL, scale=0.0005, dim=32)
EXTRA = {"solve", "as_trained", "curve", "seconds_per_step", "cache"}


@pytest.fixture(scope="module")
def runs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("closed_forms")
    for v in ("plain", "plain_d4"):
        time_split.main(**dict(SOURCE, variant=v, data=synthetic_npz, out=str(root / v)))
    cache = str(root / "cache")
    jobs = [("head_solved", "plain", dict(batches=3)), ("heads_staged_solved", "plain", dict(batches=3)),
            ("members_live", "plain_d4", dict(steps=4)), ("cache_write", "plain_d4", dict(steps=4, cache=cache)),
            ("members_cached", "plain_d4", dict(steps=4, cache=cache)),
            ("members_cached_2", "plain_d4", dict(steps=8, cache=cache, passes=2, read=0))]
    results = {}
    for name, src, kw in jobs:
        v = name if name != "members_cached_2" else "members_cached"
        args = dict(SMALL, variant=v, data=synthetic_npz, out=str(root / name), source=str(root / src), **kw)
        if name == "members_live":
            results[name] = run.main("closed_forms", [f"{k}={val}" for k, val in args.items()])
        else:
            results[name] = closed_forms.main(**args)
    return root, results


def test_T8_every_variant_writes_its_keys(runs):
    root, results = runs
    assert {r["variant"] for r in results.values()} == set(closed_forms.VARIANTS)
    for name, r in results.items():
        with open(root / name / "result.json", encoding="utf-8") as fh:
            stored = json.load(fh)
        assert set(time_split_keys()) | EXTRA == set(stored), name
        assert stored["experiment"] == "closed_forms"
        with open(root / name / "log.txt", encoding="utf-8") as fh:
            assert fh.read().splitlines()[-1] == "done"
    for name in ("head_solved", "heads_staged_solved", "members_live", "members_cached"):
        r = results[name]
        assert isinstance(r["loss_over_levels"], float) and len(r["by_stage"]) == 3, name
        assert r["score"]["n"] == 32, name
    hs = results["head_solved"]
    assert hs["as_trained"]["equals_stored"] is True
    assert set(hs["solve"]) == {"head"} and hs["solve"]["head"]["batches"] == 3
    st = results["heads_staged_solved"]
    assert st["kind"] == "clusters" and set(st["solve"]) == {"stage_0", "stage_1", "stage_2"}
    assert st["as_trained"]["equals_stored"] is True and len(st["alone"]) == 3
    assert os.path.exists(root / "heads_staged_solved" / "staged.json")
    cw = results["cache_write"]
    assert cw["cache"]["steps"] == 4 and cw["cache"]["bytes"] > 0 and cw["loss_over_levels"] is None
    assert results["members_cached_2"]["read"] is None and len(results["members_cached_2"]["curve"]) == 8


def time_split_keys():
    return {"experiment", "variant", "seed", "kind", "cuts", "soft", "parts", "numbers", "train_seconds", "read",
            "loss_over_levels", "by_level", "by_stage", "wrong_assembly", "alone", "score", "args", "torch", "device",
            "version"}


def test_T8_cached_members_equal_live_members(runs):
    root, results = runs
    live, cached = results["members_live"], results["members_cached"]
    assert live["curve"] == cached["curve"] and len(live["curve"]) == 4
    assert live["loss_over_levels"] == cached["loss_over_levels"]
    assert live["by_level"] == cached["by_level"]
    a, b = read_build(str(root / "members_live" / "model")), read_build(str(root / "members_cached" / "model"))
    assert a.names() == b.names() == ["anchor", "stem", "head", "member_b0", "member_b1", "member_b2"]
    for e in a.units:
        assert e.hash == b.entry(e.name).hash, e.name
    src = read_build(str(root / "plain_d4" / "model"))
    for n in ("anchor", "stem", "head"):                                  # the adopted units are the source's
        assert a.entry(n).hash == src.entry(n).hash
    assert live["parts"][0]["frozen_unchanged"] and cached["parts"][0]["frozen_unchanged"]


def test_T8_summarize_and_refusals(runs, synthetic_npz, tmp_path):
    root, results = runs
    text = closed_forms.summarize(str(root))
    lines = [x for x in text.splitlines() if x.startswith("| ") and not x.startswith("| variant")]
    assert len(lines) == len(results)
    with pytest.raises(ValueError):
        closed_forms.main(variant="head_solved", data=synthetic_npz, out=str(tmp_path / "a"), device="cpu")
    with pytest.raises(ValueError):
        closed_forms.main(variant="cache_write", data=synthetic_npz, out=str(tmp_path / "b"), device="cpu",
                          source=str(root / "plain_d4"))
    with pytest.raises(ValueError):
        run.main("no_such_experiment")
