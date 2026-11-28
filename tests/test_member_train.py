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
