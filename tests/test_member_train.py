"""Several member engines trained as one stack (CPU, tiny sizes): the written-out stacked forward against each engine's
own forward, the gradients of the summed loss, a stacked run against solo runs, resume and snapshots, the written
folders, and the rows of the told-own variant against the stored trainer's."""
import os

import pytest
import torch
from safetensors.torch import load_file

from geolip.modular_diffusion import collective as C
from geolip.modular_diffusion import member_train as T
from geolip.modular_diffusion import members as MB
from geolip.modular_diffusion import scenes
from geolip.modular_diffusion.train import step_generator

BED = scenes.ScenesBed("F")
KW = dict(batch=4, heldout=6, every=2, save_every=0, dim=128, heads=2, depth=1, warmup=2, log=lambda s: None)


def _wake(engine, seed):
    """Random non-zero numbers in the head and the modulation."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (engine.head.proj.weight, engine.head.proj.bias):
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        for blk in engine.blocks:
            w = blk.modulation[1].weight
            w.copy_(torch.randn(w.shape, generator=g) * 0.02)
    return engine


# ------------------------------------------------------------------ 1. the stacked forward
@pytest.mark.parametrize("variant", [MB.T1, MB.T2, MB.T2K])
def test_1_the_stacked_forward_equals_each_engines_own_forward(variant):
    classes = (1, 4, None)
    engines = [_wake(MB.born(variant, c, 0, depth=3), 30 + i) for i, c in enumerate(classes)]
    rows = [T.engine_rows(variant, c, BED, 3, torch.Generator().manual_seed(100 + i), 0.5, None, ())
            for i, c in enumerate(classes)]
    sb = T.stack_rows(rows, "cpu")
    with torch.no_grad():
        got = T.stacked_forward(T.stack_state(engines), 2, sb["z"], sb["t"], **sb["inputs"])
        want = [e.forward_tokens(r["z"], r["t"], **r["prep"]) for e, r in zip(engines, rows)]
    diff = max(float((got[g] - want[g]).abs().max()) for g in range(3))
    print(variant.name, "stacked against own forward, largest |difference|:", diff)
    assert diff < 1e-5


# ------------------------------------------------------------------ 2. the gradients
def test_2_the_summed_losss_gradient_on_each_slice_is_that_engines_own_gradient():
    variant, classes = MB.T2, (2, 6)
    engines = [_wake(MB.born(variant, c, 0, depth=2), 40 + i) for i, c in enumerate(classes)]
    rows = [T.engine_rows(variant, c, BED, 4, torch.Generator().manual_seed(200 + i), 0.5, None, ())
            for i, c in enumerate(classes)]
    state = {k: v.requires_grad_(True) for k, v in T.stack_state(engines).items()}
    sb = T.stack_rows(rows, "cpu")
    loss = sum(T.stack_losses(state, 2, sb))
    loss.backward()
    # cross.q: the context is one token, so the cross attention's softmax is 1 whatever the query is and the gradient of
    # cross.q is zero in exact arithmetic; both sides hold rounding noise there (about 1e-11), compared absolutely.
    worst, noise = 0.0, 0.0
    for g, (e, r) in enumerate(zip(engines, rows)):
        e.zero_grad()
        C.masked_loss(e.forward_tokens(r["z"], r["t"], **r["prep"]), r["target"], r["loss_mask"]).backward()
        for name, p in e.named_parameters():
            a, b = state[name].grad[g], p.grad
            if ".cross.q." in name:
                noise = max(noise, float(a.abs().max()), float(b.abs().max()))
                continue
            worst = max(worst, float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)))
    print("largest relative gradient difference:", worst, "; largest |gradient| of cross.q:", noise)
    assert worst < 1e-5 and noise < 1e-9


# ------------------------------------------------------------------ 3. a stack against solo runs
def test_3_three_stacked_steps_equal_two_solo_runs(tmp_path):
    both = T.train_members(MB.T2, (3, None), BED, 0, str(tmp_path / "both"), steps=3, **KW)
    worst = 0.0
    for g, cls in enumerate((3, None)):
        solo = T.train_members(MB.T2, (cls,), BED, 0, str(tmp_path / f"solo{g}"), steps=3, **KW)
        for k in both["state"]:
            worst = max(worst, float((both["state"][k][g] - solo["state"][k][0]).abs().max()),
                        float((both["average"][k][g] - solo["average"][k][0]).abs().max()))
    print("stack against solo runs, largest |difference| of weights and averages:", worst)
    assert worst < 1e-5


# ------------------------------------------------------------------ 4. resume
def test_4_a_run_stopped_and_started_again_reaches_the_unbroken_runs_numbers(tmp_path):
    whole = T.train_members(MB.T2, (5, None), BED, 1, str(tmp_path / "whole"), steps=4, **KW)
    part = T.train_members(MB.T2, (5, None), BED, 1, str(tmp_path / "part"), steps=4, until=2, **KW)
    assert not part["finished"]
    rest = T.train_members(MB.T2, (5, None), BED, 1, str(tmp_path / "part"), steps=4, **KW)
    assert rest["finished"]
    assert all(torch.equal(whole["state"][k], rest["state"][k]) and torch.equal(whole["average"][k], rest["average"][k])
               for k in whole["state"])
    again = T.train_members(MB.T2, (5, None), BED, 1, str(tmp_path / "whole"), steps=4, **KW)
    assert again.get("restored") and [r["last_read"] for r in again["engines"]] == \
        [r["last_read"] for r in whole["engines"]]


def test_4b_a_snapshot_at_step_2_of_4_is_the_run_of_2_steps(tmp_path):
    four = T.train_members(MB.T2, (7,), BED, 2, str(tmp_path / "four"), steps=4, snapshots=(2,), **KW)
    two = T.train_members(MB.T2, (7,), BED, 2, str(tmp_path / "two"), steps=2, **KW)
    tag = two["engines"][0]["tag"]
    assert four["finished"] and os.path.isdir(tmp_path / "four" / tag)
    for f in ("engine.safetensors", "engine_avg.safetensors"):
        a, b = load_file(str(tmp_path / "four" / tag / f)), load_file(str(tmp_path / "two" / tag / f))
        assert all(torch.equal(a[k], b[k]) for k in b), f
    import json
    ja = json.load(open(tmp_path / "four" / tag / "train.json", encoding="utf-8"))
    jb = json.load(open(tmp_path / "two" / tag / "train.json", encoding="utf-8"))
    assert ja["curve"] == jb["curve"] and ja["last_read"] == jb["last_read"]


# ------------------------------------------------------------------ 5. the folders
def test_5_the_written_folders_load_and_answer_as_the_stacks_slice(tmp_path):
    res = T.train_members(MB.T2K, (0, None), BED, 3, str(tmp_path / "k"), steps=2, **dict(KW, depth=2))
    rows = [T.engine_rows(MB.T2K, c, BED, 3, torch.Generator().manual_seed(300 + i), 0.5, None, ())
            for i, c in enumerate((0, None))]
    sb = T.stack_rows(rows, "cpu")
    with torch.no_grad():
        stacked = T.stacked_forward(res["average"], 2, sb["z"], sb["t"], **sb["inputs"])
        worst = 0.0
        for g, r in enumerate(res["engines"]):
            eng = MB.load_member(str(tmp_path / "k" / r["tag"]))
            assert eng.cls == r["cls"] and eng.card["held_pairs"] == []
            v = eng.forward_tokens(rows[g]["z"], rows[g]["t"], **rows[g]["prep"])
            worst = max(worst, float((v - stacked[g]).abs().max()))
    print("loaded engines against the stack's averaged slice:", worst)
    assert worst < 1e-5


# ------------------------------------------------------------------ 5b. inputs prepared from the CPU layout
@pytest.mark.parametrize("variant", [MB.T1, MB.T2, MB.T2K, MB.T3])
def test_5b_prepare_on_the_cpu_layout_equals_prepare_on_the_device_layout(variant):
    for cls in (2, None):
        b = MB.member_batch(variant, cls, BED, 6, torch.Generator().manual_seed(400))
        cpu = MB.prepare(variant, b["cells_cpu"], b["colors_cpu"], b["points_cpu"], b["told_cpu"], b["visible_cpu"])
        dev = MB.prepare(variant, b["cells"], b["colors"], b["points"], b["told"], b["visible"])
        for k in cpu:
            if cpu[k] is None:
                assert dev[k] is None, k
            elif isinstance(cpu[k], tuple):
                assert all(torch.equal(x, y.cpu()) for x, y in zip(cpu[k], dev[k])), k
            else:
                assert torch.equal(cpu[k], dev[k].cpu()), (variant.name, cls, k)


# ------------------------------------------------------------------ 6. the told-own rows are the stored trainer's rows
def test_6_t1_rows_of_step_0_are_the_stored_trainers_rows():
    for cls, kind in ((4, "addressed_thing"), (None, "addressed_base")):
        ours = T.step_rows(MB.T1, (cls,), BED, 9, 0, 5)[0]
        gen = step_generator(9, 0, "cpu")                                   # the draws of collective.train_engine
        b = C.kind_batch(kind, cls, BED, 5, gen, None)
        z, t, target = C.noisy(b["x"], gen)
        assert (torch.equal(ours["batch"]["x"], b["x"]) and torch.equal(ours["z"], z) and torch.equal(ours["t"], t)
                and torch.equal(ours["target"], target) and torch.equal(ours["loss_mask"], b["loss_mask"])
                and torch.equal(ours["batch"]["visible"], b["key_mask"])), kind


# ------------------------------------------------------------------ 7. one render and one prepare a step for the stack
def _reference_engine_rows(variant, cls, bed, n, gen, p_point, exclude, held_pairs):
    """Copies of ``engine_rows``, ``step_rows`` and ``stack_rows`` as they were before the stack's rows were built at
    once on the training device."""
    ex, held = (None, ()) if variant.told == "own" else (exclude, tuple(held_pairs))
    b = MB.member_batch(variant, cls, bed, n, gen, p_point, ex, held)
    z, t, target = C.noisy(b["x"], gen)
    prep = MB.prepare(variant, b["cells_cpu"], b["colors_cpu"], b["points_cpu"], b["told_cpu"], b["visible_cpu"])
    return {"batch": b, "z": z, "t": t, "target": target, "prep": prep, "loss_mask": b["loss_mask"],
            "loss_ok": T._loss_ok(variant, cls, b)}


def _reference_stack(variant, classes, bed, seed, step, n, p_point=0.5, exclude=None, held_pairs=(), device="cpu"):
    rows = [_reference_engine_rows(variant, cls, bed, n, step_generator(int(seed), int(step), "cpu"), p_point,
                                   exclude, held_pairs) for cls in classes]
    for r in rows:
        if not r["loss_ok"]:
            raise ValueError("a row of the batch has no token in its loss mask")

    def st(key):
        return torch.stack([r["prep"][key] for r in rows]).to(device)

    told = None
    if rows[0]["prep"]["told_angles"] is not None:
        told = tuple(torch.stack([r["prep"]["told_angles"][j] for r in rows]).to(device) for j in range(2))
    return {"z": torch.stack([r["z"] for r in rows]).to(device), "t": torch.stack([r["t"] for r in rows]).to(device),
            "target": torch.stack([r["target"] for r in rows]).to(device),
            "loss_mask": torch.stack([r["loss_mask"] for r in rows]).to(device),
            "inputs": {"content": st("content"), "query_angles": st("query_angles"), "key_angles": st("key_angles"),
                       "key_mask": st("key_mask"), "told_angles": told}}


def _same_stack(got, want, what):
    for k in ("z", "t", "target", "loss_mask"):
        assert torch.equal(got[k], want[k]), (what, k)
    for k in ("content", "query_angles", "key_angles", "key_mask"):
        assert torch.equal(got["inputs"][k], want["inputs"][k]), (what, k)
    if want["inputs"]["told_angles"] is None:
        assert got["inputs"]["told_angles"] is None, what
    else:
        assert all(torch.equal(a, b) for a, b in zip(got["inputs"]["told_angles"], want["inputs"]["told_angles"])), what


@pytest.mark.parametrize("name", list(MB.VARIANTS))
@pytest.mark.parametrize("bed_name", scenes.BEDS)
def test_7_stream_rows_of_a_step_are_bit_for_bit_the_rows_built_engine_by_engine(name, bed_name):
    variant, bed = MB.VARIANTS[name], scenes.ScenesBed(bed_name)
    kws = [dict()] if variant.told == "own" else [dict(), dict(p_point=1.0, exclude=8, held_pairs=((1, 4), (4, 1)))]
    for classes in ((2, 5, None), (6,), (None,)):
        for step in (0, 3):
            for i, kw in enumerate(kws):
                got = T.stack_step_rows(variant, classes, bed, 4, step, 7, noise="stream", **kw)
                want = _reference_stack(variant, classes, bed, 4, step, 7, **kw)
                _same_stack(got, want, (name, bed_name, classes, step, i))
                inp = got["inputs"]
                # prepare gives one tensor for both when the member digits are the null on both sides; so does the stack
                one = not variant.turn or variant.identity == "name"
                assert (inp["query_angles"] is inp["key_angles"]) == one, name


def test_7b_one_step_of_the_stack_leaves_the_weights_of_the_step_built_engine_by_engine():
    variant, classes, bed = MB.T2, (1, 8, None), scenes.ScenesBed("C")
    engines = [_wake(MB.born(variant, c, 0, depth=2), 60 + i) for i, c in enumerate(classes)]
    results = []
    for build in ("now", "before"):
        state = {k: v.clone().requires_grad_(True) for k, v in T.stack_state(engines).items()}
        opt = torch.optim.Adam(list(state.values()), lr=1e-3, betas=T.BETAS, eps=T.EPS, weight_decay=0.0)
        average = {k: v.detach().clone() for k, v in state.items()}
        for s in range(2):
            if build == "now":
                T._step(variant, classes, bed, 5, s, 6, 0.5, None, [], state, 2, opt, average, "cpu", 1e-3, 2)
            else:
                for grp in opt.param_groups:
                    grp["lr"] = 1e-3 * min(1.0, (s + 1) / 2)
                losses = T.stack_losses(state, 2, _reference_stack(variant, classes, bed, 5, s, 6))
                opt.zero_grad(set_to_none=True)
                sum(losses[1:], losses[0]).backward()
                opt.step()
                dk = (1 + s) / (s + 10)
                with torch.no_grad():
                    for k in state:
                        average[k].copy_(dk * average[k] + (1.0 - dk) * state[k].detach())
        results.append((state, average))
    (sa, aa), (sb, ab) = results
    assert all(torch.equal(sa[k], sb[k]) and torch.equal(aa[k], ab[k]) for k in sa)


def test_7c_device_noise_draws_other_numbers_the_same_for_every_engine_and_leaves_the_rest():
    variant, bed = MB.T2, scenes.ScenesBed("F")
    a = T.stack_step_rows(variant, (2, 5, None), bed, 4, 3, 7, noise="stream")
    b = T.stack_step_rows(variant, (2, 5, None), bed, 4, 3, 7, noise="device")
    assert not torch.equal(a["z"], b["z"]) and not torch.equal(a["t"], b["t"])
    assert torch.equal(b["t"][0], b["t"][1]) and torch.equal(b["t"][0], b["t"][2])        # one stream an engine
    assert torch.equal(a["loss_mask"], b["loss_mask"])
    assert all(torch.equal(a["inputs"][k], b["inputs"][k]) for k in ("content", "key_angles", "key_mask"))
    # on a CPU the device generator is a CPU one: its noise then t, the same pictures, the expressions of noisy
    gen = T.noise_generator(4, 3, "cpu")
    z0 = torch.randn((7, 3, 48, 48), generator=gen)
    t = torch.rand(7, generator=gen)
    tt = t.view(7, 1, 1, 1)
    for g, cls in enumerate((2, 5, None)):
        x = MB.member_batch(variant, cls, bed, 7, step_generator(4, 3, "cpu"))["x"]
        assert torch.equal(b["t"][g], t) and torch.equal(b["target"][g], x - z0)
        assert torch.equal(b["z"][g], (1 - tt) * z0 + tt * x)
    with pytest.raises(ValueError):
        T.stack_step_rows(variant, (2,), bed, 4, 3, 7, noise="card")


def test_7d_device_noise_a_stack_equals_its_solo_runs_and_a_snapshot_its_shorter_run(tmp_path):
    kw = dict(KW, noise="device")
    both = T.train_members(MB.T2, (3, None), BED, 0, str(tmp_path / "both"), steps=3, **kw)
    stream = T.train_members(MB.T2, (3, None), BED, 0, str(tmp_path / "stream"), steps=3, **KW)
    assert not all(torch.equal(both["state"][k], stream["state"][k]) for k in both["state"])
    worst = 0.0
    for g, cls in enumerate((3, None)):
        solo = T.train_members(MB.T2, (cls,), BED, 0, str(tmp_path / f"solo{g}"), steps=3, **kw)
        for k in both["state"]:
            worst = max(worst, float((both["state"][k][g] - solo["state"][k][0]).abs().max()),
                        float((both["average"][k][g] - solo["average"][k][0]).abs().max()))
    print("device noise: stack against solo runs, largest |difference| of weights and averages:", worst)
    assert worst < 1e-5
    tags_d = [r["tag"] for r in both["engines"]]
    tags_s = [r["tag"] for r in stream["engines"]]
    assert all(a != b and a == b + "_noisedevice" for a, b in zip(tags_d, tags_s))
    assert both["settings"]["noise"] == "device" and stream["settings"]["noise"] == "stream"
    import json
    tj = json.load(open(tmp_path / "both" / tags_d[0] / "train.json", encoding="utf-8"))
    assert tj["settings"]["noise"] == "device"
    four = T.train_members(MB.T2, (7,), BED, 2, str(tmp_path / "four"), steps=4, snapshots=(2,), **kw)
    two = T.train_members(MB.T2, (7,), BED, 2, str(tmp_path / "two"), steps=2, **kw)
    tag = two["engines"][0]["tag"]
    assert four["finished"] and tag.endswith("_noisedevice")
    for f in ("engine.safetensors", "engine_avg.safetensors"):
        a, b = load_file(str(tmp_path / "four" / tag / f)), load_file(str(tmp_path / "two" / tag / f))
        assert all(torch.equal(a[k], b[k]) for k in b), f
    # a stack's checkpoint is not continued under the other noise
    T.train_members(MB.T2, (5,), BED, 1, str(tmp_path / "part"), steps=4, until=2, **KW)
    with pytest.raises(ValueError):
        T.train_members(MB.T2, (5,), BED, 1, str(tmp_path / "part"), steps=4, **kw)


def test_7e_the_phase_clock_names_every_phase():
    r = T.time_stack(MB.T2, (0, None), BED, 0, batch=2, steps=3, from_step=1, dim=64, heads=1, depth=1,
                     noise="device", phases=True)
    assert list(r["phases"]) == list(T.PHASES) and all(v >= 0 for v in r["phases"].values())
    assert r["noise"] == "device" and "phases" not in T.time_stack(MB.T2, (0,), BED, 0, batch=2, steps=2, from_step=1,
                                                                    dim=64, heads=1, depth=1)
