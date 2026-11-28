"""The addressed collective's experiment module on tiny settings: every variant runs end to end with two classes and
the base, writes its files with the keys the module text names, and a compose job repeats exactly."""
import json
import math
import os

import pytest

from geolip.modular_diffusion import members as MB
from geolip.modular_diffusion import run
from geolip.modular_diffusion.experiments import addressed_collective as A
from geolip.modular_diffusion.experiments import toy_collective as TC

TINY = dict(dim=64, heads=1, depth=1, steps=2, part_steps=2, batch=4, heldout=8, every=1, save_every=0, count=6,
            ns="1,4", pointer_ns="4", sample_steps=2, chunk=0, k=6, exact_lists=4, ts="0.3", classes="0,1,base",
            device="cpu", threads=4, bed="F")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("addressed_runs")
    TC.main(variant="judge", bed="F", out=str(root / "judge_F"), judge_steps=2, judge_batch=8, device="cpu", threads=4)
    for kind, held, seed, depth in (("T2", "plus4", 0, 1), ("T2", "plus4", 1, 1), ("T1", "", 0, 1),
                                    ("C2", "", 0, 1), ("T2K", "", 0, 2)):
        A.main(variant="parts", kind=kind, held=held, seed=seed, out=str(root / f"parts_{kind}_{seed}"),
               **dict(TINY, depth=depth))
    return root


def _common(runs, kind="T2", held="plus4", depth=1):
    return dict(TINY, kind=kind, held=held, depth=depth, parts=str(runs), judge=str(runs / "judge_F"))


def _files(folder):
    assert os.path.isfile(os.path.join(folder, "result.json")) and os.path.isfile(os.path.join(folder, "result.md"))
    assert open(os.path.join(folder, "log.txt"), encoding="utf-8").read().rstrip().endswith("done")
    return json.load(open(os.path.join(folder, "result.json"), encoding="utf-8"))


def test_the_entry_point_reaches_the_experiment_and_the_arguments_parse(tmp_path):
    assert A.parse_classes("0-8,base") == list(range(9)) + [None] and A.parse_classes("2,base") == [2, None]
    assert len(A.held_pairs("plus4")) == 9 and (0, 4) in A.held_pairs("plus4") and A.held_pairs("") == ()
    r = run.main("addressed_collective", ["variant=timing", "kind=T2", "stacks=1,2", "steps=3", "from_step=1",
                                          "batch=2", "dim=64", "heads=1", "depth=1", f"out={tmp_path / 'timing'}"])
    assert [x["engines"] for x in r["timing"]] == [1, 2] and all(x["seconds_a_step_stack"] > 0 for x in r["timing"])
    _files(tmp_path / "timing")


def test_the_pointing_reads_by_k_add_up_to_the_pooled_reads_on_a_hand_made_judge_output():
    import torch
    pl = MB.pointer_lists(4, 200, 6004, distinct=True)
    g = torch.Generator().manual_seed(3)
    # a hand-made judge: right with probability 0.7 on each cell, else a wrong thing / color
    keep_t = torch.rand(pl.cells.shape, generator=g) < 0.7
    keep_c = torch.rand(pl.cells.shape, generator=g) < 0.7
    pt = torch.where(keep_t | (pl.cells < 0), pl.cells.clamp_min(0), (pl.cells.clamp_min(0) + 1) % 9)
    pt = torch.where(pl.cells < 0, torch.full_like(pt, 9), pt)
    pc = torch.where(keep_c, pl.colors.clamp_min(0), (pl.colors.clamp_min(0) + 1) % 9)
    r = A.pointing_reads_of(pt, pc, pl.cells, pl.colors, pl.points, pl.eligible)
    by_k = r["pointing_by_k"]
    ks = [int(k) for k, p in zip(pl.eligible.tolist(), pl.pointed.tolist()) if p]
    assert set(by_k) == {str(k) for k in set(ks)} and all(by_k[str(k)]["lists"] == ks.count(k) for k in set(ks))
    for absent in range(1, 9):
        if absent not in ks:
            assert str(absent) not in by_k
    assert sum(x["lists"] for x in by_k.values()) == r["pointed"] == r["pointing_color_right"]["n"]
    for key in ("pointing_thing_right", "pointing_color_right", "pointing_both_right"):
        pooled = sum(x[key]["mean"] * x["lists"] for x in by_k.values()) / r["pointed"]
        assert abs(pooled - r[key]["mean"]) < 1e-12, key
    assert all(abs(x["one_over_k"] - 1 / int(k)) < 1e-15 for k, x in by_k.items())
    assert list(by_k) == sorted(by_k, key=int)
    assert "pointing_by_k" not in A.pointing_reads_of(pt, pc, pl.cells, pl.colors, pl.points)


def test_the_eligible_reference_is_one_over_k_and_never_below_the_all_stated_one():
    for n in (3, 4, 6, 9):
        for only, held in ((False, ()), (False, A.held_pairs("plus4")), (True, A.held_pairs("plus4"))):
            pl = MB.pointer_lists(n, 300, 6000 + n, held, only_held=only, distinct=True)
            el = A.eligible_reference(pl)
            al = A.blind_reference(pl.cells, pl.colors, pl.points)
            ks = [int(k) for k, p in zip(pl.eligible.tolist(), pl.pointed.tolist()) if p]
            assert el["lists"] == al["lists"] == len(ks) == int(pl.pointed.sum())
            assert {int(k): c for k, c in el["k_histogram"].items()} == {k: ks.count(k) for k in set(ks)}
            if ks:
                assert abs(el["mean"] - sum(1 / k for k in ks) / len(ks)) < 1e-12
                assert el["mean"] >= al["mean"] - 1e-12 and max(ks) <= n - 1
            print("n", n, "only_held", only, "held", bool(held), ": eligible", el["mean"], "all stated", al["mean"],
                  "k", el["k_histogram"])


def test_parts_write_every_engine_and_their_results(runs):
    r = _files(runs / "parts_T2_0")
    assert [e["cls"] for e in r["engines"]] == [0, 1, None] and all("descended" in e for e in r["engines"])
    for e in r["engines"]:
        assert os.path.isfile(runs / "parts_T2_0" / e["tag"] / "train.json")


@pytest.mark.parametrize("kind,held,depth", [("T2", "plus4", 1), ("T1", "", 1)])
def test_compose_writes_the_stored_and_the_pointing_reads_and_repeats_exactly(runs, tmp_path, kind, held, depth):
    a = _common(runs, kind, held, depth)
    r = A.main(variant="compose", out=str(tmp_path / "c1"), **a)
    again = A.main(variant="compose", out=str(tmp_path / "c2"), **a)
    _files(tmp_path / "c1")
    assert os.path.isfile(tmp_path / "c1" / "sheet.jpg")
    assert set(r["stored"]) == {"1", "4"}
    for x in r["stored"].values():
        for key in ("cell_rate", "scene_right", "color_rate", "per_list", "by_judge"):
            assert key in x, key
    if kind == "T1":
        assert r["pointing"] == {}
    else:
        assert set(r["pointing"]) == {"unheld_pairs", "held_pairs"}
        for label in r["pointing"].values():
            x = label["4"]
            for key in ("stated_thing_right", "stated_color_right", "pointing_thing_right", "pointing_color_right",
                        "reference_all_stated", "reference_eligible", "distinct_colors"):
                assert key in x, key
            assert x["reference_all_stated"]["mean"] is None or abs(x["reference_all_stated"]["mean"] - 1 / 3) < 1e-12
            el = x["reference_eligible"]
            assert el["lists"] == x["reference_all_stated"]["lists"] == sum(el["k_histogram"].values())
            assert el["mean"] is None or el["mean"] >= x["reference_all_stated"]["mean"]
            assert {k: v["lists"] for k, v in x["pointing_by_k"].items()} == el["k_histogram"]
            for judge_reads_ in x["by_judge"].values():
                assert "pointing_by_k" in judge_reads_
        if any(lb["4"]["pointing_by_k"] for lb in r["pointing"].values()):
            assert "BY THE NUMBER OF ELIGIBLE REQUESTS k" in open(tmp_path / "c1" / "result.md", encoding="utf-8").read()
    for res in (r, again):                                                     # the clock is the one thing that differs
        for block in [res["stored"]] + list(res["pointing"].values()):
            for x in block.values():
                x.pop("seconds", None)
    assert json.dumps(r["stored"], default=str) == json.dumps(again["stored"], default=str)
    assert json.dumps(r["pointing"], default=str) == json.dumps(again["pointing"], default=str)


@pytest.mark.parametrize("kind,held,depth", [("T2", "plus4", 1), ("T1", "", 1), ("C2", "", 1), ("T2K", "", 2)])
def test_address_runs_every_read(runs, tmp_path, kind, held, depth):
    r = A.main(variant="address", out=str(tmp_path / "addr"), **dict(_common(runs, kind, held, depth), k=16))
    _files(tmp_path / "addr")
    if kind == "T1":
        assert "note" in r["find"]
    else:
        assert r["find"]["vary"] == ("content" if kind == "C2" else "query")
        assert set(r["find"]["by_t"]) == {"0.0", "0.3"}
        assert r["find"]["k_rows_drawn_per_engine"] == 16
        for x in r["find"]["by_t"].values():
            for key in ("rows", "skipped_same_color", "skipped_no_candidate", "share_nearer_asked_a", "blocks",
                        "rows_drawn", "both_nearer", "only_a", "only_b", "neither", "max_abs_answer_delta",
                        "per_engine"):
                assert key in x, key
            assert x["both_nearer"] + x["only_a"] + x["only_b"] + x["neither"] == x["rows"]
            assert x["both_nearer"] + x["only_a"] == x["nearer_asked_a"]
            assert x["rows_drawn"] == 16 * len(x["per_engine"]) == sum(e["drawn"] for e in x["per_engine"].values())
            assert x["rows"] == sum(e["used"] for e in x["per_engine"].values())
            for e in x["per_engine"].values():
                assert e["drawn"] == e["used"] + e["skipped_no_pointer"] + e["skipped_same_color"] + \
                    e["skipped_no_candidate"]
            for b in x["blocks"].values():
                for key in ("max_abs_weight_delta", "max_abs_score_delta", "rows_exactly_equal", "rows"):
                    assert key in b, key
                assert 0 <= b["rows_exactly_equal"] <= b["rows"] == x["rows"]
    x = r["by_t"]["0.3"]
    for key in ("serve", "together", "unturned_gate_identical", "neutral", "missing", "unheld", "tolerance"):
        assert key in x, key
    if x["scenes"]:
        assert x["unturned_gate_identical"] is True and x["missing"]["smallest_base"] == 1.0
        assert x["unheld"]["largest_member"] == 0.0 and x["serve"]["reverse_restores"] is True
    for t_key, xt in r["by_t"].items():                  # the neutral control never divides by its clamp
        nt = xt["neutral"]
        if xt["scenes"]:
            assert nt["old"]["n"] == nt["new"]["n"] == xt["scenes"], t_key
            for side in ("old", "new"):
                for q in ("median", "p95"):
                    assert math.isfinite(nt[side][q]) and nt[side][q] < 1e6, (kind, t_key, side, q, nt[side][q])
            assert nt["smallest_denominator"] > 1e-20, (kind, t_key, nt["smallest_denominator"])
    print(kind, "address scenes", x["scenes"], "skipped", x["skipped"], "; FIND rows",
          {t: v["rows"] for t, v in r["find"].get("by_t", {}).items()})


def _reference_find(a, col):
    """A copy of the experiment's FIND as it was before its batches were drawn once and its blocks read together."""
    import torch
    from geolip.modular_diffusion import scenes as S
    from geolip.modular_diffusion.spectral import patchify
    variant = col.variant
    vary = "content" if variant.identity == "content" else "query"
    bed = S.ScenesBed(a["bed"], "cpu")
    dev = a["device"]
    depth = len(next(iter(col.things.values())).blocks)
    out = {"vary": vary, "k_rows_drawn_per_engine": int(a["k"]), "by_t": {}}
    for t in A._levels(a):
        acc = {"rows": 0, "rows_drawn": 0, "skipped_same_color": 0, "skipped_no_candidate": 0,
               "skipped_no_pointer": 0, "nearer_asked_a": 0, "nearer_asked_b": 0, "both_nearer": 0, "only_a": 0,
               "only_b": 0, "neither": 0, "max_abs_answer_delta": 0.0, "per_engine": {}, "blocks": {}}
        sums = {}
        for cls, eng in col.things.items():
            gen = torch.Generator().manual_seed(A.FIND_SEED + 100 * cls)
            mb = MB.member_batch(variant, cls, bed, int(a["k"]), gen, 1.0, A._exclude(a), A.held_pairs(a["held"]))
            z_all = A._noisy_at(mb["x"], t, A.FIND_SEED + 100 * cls + 1 + int(round(1000 * t)))
            mine = {"drawn": int(mb["x"].shape[0]), "used": 0, "skipped_same_color": 0, "skipped_no_candidate": 0,
                    "skipped_no_pointer": 0}
            acc["per_engine"][str(cls)] = mine
            acc["rows_drawn"] += mine["drawn"]
            for i in range(mb["x"].shape[0]):
                own = [c for c in range(9) if int(mb["cells"][i, c]) == cls and int(mb["points"][i, c]) >= 0]
                if not own:
                    acc["skipped_no_pointer"] += 1
                    mine["skipped_no_pointer"] += 1
                    continue
                cell = own[0]
                ask_a = int(mb["points"][i, cell])
                ask_b, why = A._second_candidate(mb["cells"][i], mb["colors"][i], mb["points"][i], cls, ask_a)
                if ask_b is None:
                    acc["skipped_" + why] += 1
                    mine["skipped_" + why] += 1
                    continue
                mine["used"] += 1
                visible = mb["visible"][i:i + 1] if mb["visible"] is not None else None
                z = z_all[i:i + 1].to(dev)
                tt = torch.full((1,), float(t), device=dev)
                row = mb["cells"][i].tolist()
                ca = torch.tensor(S.COLOR_VALUES[int(mb["colors"][i, row.index(ask_a)])])
                cb = torch.tensor(S.COLOR_VALUES[int(mb["colors"][i, row.index(ask_b)])])
                acc["rows"] += 1
                for blk in range(depth):
                    r = MB.find_read(eng, variant, z, tt, mb["cells"][i:i + 1], mb["colors"][i:i + 1],
                                     mb["points"][i:i + 1], 0, cell, ask_a, ask_b, blk, visible=visible, vary=vary)
                    s = sums.setdefault(blk, {"w_asked_a": 0.0, "w_asked_b": 0.0, "change": 0.0, "groups": 0.0,
                                              "max_abs_weight_delta": 0.0, "max_abs_score_delta": 0.0,
                                              "rows_exactly_equal": 0})
                    wa, wb = r["weights_a"].double(), r["weights_b"].double()
                    s["w_asked_a"] = s["w_asked_a"] + wa[:, 0]
                    s["w_asked_b"] = s["w_asked_b"] + wb[:, 1]
                    s["change"] = s["change"] + 0.5 * ((wa[:, 0] - wb[:, 0]) + (wb[:, 1] - wa[:, 1]))
                    ga = r["scores_a"].double()[:, 0] - r["scores_b"].double()[:, 0]
                    gb = r["scores_b"].double()[:, 1] - r["scores_a"].double()[:, 1]
                    s["groups"] = s["groups"] + 0.5 * (ga + gb)
                    s["max_abs_weight_delta"] = max(s["max_abs_weight_delta"], float((wa - wb).abs().max()))
                    s["max_abs_score_delta"] = max(s["max_abs_score_delta"],
                                                   float((r["scores_a"].double() - r["scores_b"].double()).abs().max()))
                    s["rows_exactly_equal"] += int(torch.equal(r["weights_a"], r["weights_b"])
                                                   and torch.equal(r["answer_a"], r["answer_b"]))
                    if blk == 0:
                        acc["max_abs_answer_delta"] = max(acc["max_abs_answer_delta"], float(
                            (r["answer_a"].double() - r["answer_b"].double()).abs().max()))
                        zt = patchify(z, S.PATCH)[0, MB._tokens_of(cell, z.device)].cpu()
                        nearer = {}
                        for tag, ans, want, other in (("a", r["answer_a"], ca, cb), ("b", r["answer_b"], cb, ca)):
                            m = A._mean_color(zt + (1 - t) * ans.cpu())
                            nearer[tag] = float((m - want).norm()) < float((m - other).norm())
                            if nearer[tag]:
                                acc["nearer_asked_" + tag] += 1
                        acc[{(True, True): "both_nearer", (True, False): "only_a", (False, True): "only_b",
                             (False, False): "neither"}[(nearer["a"], nearer["b"])]] += 1
        rows = max(1, acc["rows"])
        for blk, s in sums.items():
            acc["blocks"][str(blk)] = {
                "weight_on_asked_under_ask_a": (s["w_asked_a"] / rows).tolist(),
                "weight_on_asked_under_ask_b": (s["w_asked_b"] / rows).tolist(),
                "mean_change_of_the_asked_weight": (s["change"] / rows).tolist(),
                "mean_change_by_group": {g: (s["groups"][:, j] / rows).tolist() for j, g in enumerate(MB.GROUPS)},
                "max_abs_weight_delta": s["max_abs_weight_delta"], "max_abs_score_delta": s["max_abs_score_delta"],
                "rows_exactly_equal": s["rows_exactly_equal"], "rows": acc["rows"]}
        acc["share_nearer_asked_a"] = acc["nearer_asked_a"] / acc["rows"] if acc["rows"] else None
        acc["share_nearer_asked_b"] = acc["nearer_asked_b"] / acc["rows"] if acc["rows"] else None
        out["by_t"][str(t)] = acc
    return out


@pytest.mark.parametrize("kind,held,depth", [("T2", "plus4", 1), ("T2K", "", 2), ("C2", "", 1)])
def test_find_with_one_batch_an_engine_and_one_pair_of_passes_writes_the_numbers_of_the_block_by_block_read(
        runs, kind, held, depth):
    from geolip.modular_diffusion.config import parse
    a = parse(A.DEFAULTS, (), **dict(_common(runs, kind, held, depth), k=16))
    col, _ = A.load_collective(a)
    got = A._find(a, col, lambda s: None)
    want = _reference_find(a, col)
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
    assert any(x["rows"] for x in got["by_t"].values()), "no row was read: the comparison would be empty"


def test_exchange_is_exact_at_one_step_and_over_the_chain(runs, tmp_path):
    r = A.main(variant="exchange", out=str(tmp_path / "ex"), swap_class=0, **_common(runs))
    _files(tmp_path / "ex")
    assert r["one_step"]["0.3"]["exchanged"] == [0] and all(v == 0.0 for v in r["one_step"]["0.3"]["others"].values())
    assert r["chain"]["largest_outside"] == 0.0 and r["chain"]["largest_outside_unclamped"] == 0.0
    print("chain: exchanged cells", r["chain"]["exchanged_cells"], "differing", r["chain"]["exchanged_cells_differing"])


def test_parts_with_held_pairs_restart_finished_and_resume_stopped_to_the_same_engines(tmp_path):
    kw = dict(TINY, kind="T2", held="plus4", seed=0, steps=3)
    whole = A.main(variant="parts", out=str(tmp_path / "whole"), **kw)
    again = A.main(variant="parts", out=str(tmp_path / "whole"), **kw)                 # finished: nothing runs
    assert whole["finished"] and again["finished"] and again.get("restored")
    assert [e["last_read"] for e in again["engines"]] == [e["last_read"] for e in whole["engines"]]
    part = A.main(variant="parts", out=str(tmp_path / "split"), until=2, **kw)
    assert part["finished"] is False and part["step"] == 2 and not part["engines"]
    rest = A.main(variant="parts", out=str(tmp_path / "split"), **kw)
    assert rest["finished"] and not rest.get("restored")
    same = A.main(variant="same", parts=f"{tmp_path / 'whole'},{tmp_path / 'split'}", out=str(tmp_path / "same"))
    assert same["same"] is True and same["engines_compared"] == 3 and not same["differing"]
    assert "the same engines" in open(tmp_path / "same" / "result.md", encoding="utf-8").read()
    other = A.main(variant="parts", out=str(tmp_path / "other"), **dict(kw, seed=1))
    assert other["finished"]
    diff = A.main(variant="same", parts=f"{tmp_path / 'whole'},{tmp_path / 'other'}", out=str(tmp_path / "diff"))
    assert diff["same"] is False and diff["engines_compared"] == 0 and len(diff["only_in_a"]) == 3
    assert A.same_engines(str(tmp_path / "whole"), str(tmp_path / "whole"))["same"] is True
    import shutil
    from safetensors.torch import load_file, save_file
    shutil.copytree(tmp_path / "whole", tmp_path / "tampered")
    tag = whole["engines"][1]["tag"]
    path = str(tmp_path / "tampered" / tag / "engine_avg.safetensors")
    w = load_file(path)
    name = sorted(w)[0]
    w[name] = w[name].clone()
    w[name].view(-1)[0] += 1e-6
    save_file(w, path)
    t = A.same_engines(str(tmp_path / "whole"), str(tmp_path / "tampered"))
    assert t["same"] is False and t["differing"] == [{"tag": tag, "file": "engine_avg.safetensors", "tensors": [name]}]


def test_engines_trained_with_device_noise_are_found_by_that_noise_only(tmp_path):
    from geolip.modular_diffusion.config import parse
    folder = tmp_path / "parts_device"
    r = A.main(variant="parts", kind="T2", seed=0, noise="device", out=str(folder), **TINY)
    assert r["noise"] == "device" and _files(folder)["noise"] == "device"
    assert all(e["tag"].endswith("_noisedevice") for e in r["engines"])
    for e in r["engines"]:
        tj = json.load(open(folder / e["tag"] / "train.json", encoding="utf-8"))
        assert tj["settings"]["noise"] == "device"
    a = parse(A.DEFAULTS, (), **dict(TINY, kind="T2", seed=0, parts=str(folder), noise="device"))
    col, used = A.load_collective(a)
    assert [u["cls"] for u in used] == [0, 1, None] and col.base is not None and col.classes == [0, 1]
    a_stream = parse(A.DEFAULTS, (), **dict(TINY, kind="T2", seed=0, parts=str(folder), noise="stream"))
    with pytest.raises(FileNotFoundError) as err:
        A.load_collective(a_stream)
    assert "T2_c0_bedF_seed0_steps2_d64h1x1_exnone_heldnone" in str(err.value)
    assert "_noisedevice" not in str(err.value).split(" under ")[0]
    t = A.main(variant="timing", kind="T2", stacks="1", steps=3, from_step=1, batch=2, dim=64, heads=1, depth=1,
               noise="device", out=str(tmp_path / "timing"))
    assert t["noise"] == "device" and t["timing"][0]["noise"] == "device" and t["timing"][0]["phases"]
    assert "PHASES" in open(tmp_path / "timing" / "result.md", encoding="utf-8").read()
