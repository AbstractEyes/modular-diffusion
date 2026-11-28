"""The addressed collective's experiment module on tiny settings: every variant runs end to end with two classes and
the base, writes its files with the keys the module text names, and a compose job repeats exactly."""
import json
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
                        "blind_reference", "distinct_colors"):
                assert key in x, key
            assert x["blind_reference"]["mean"] is None or abs(x["blind_reference"]["mean"] - 1 / 3) < 1e-12
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
        for x in r["find"]["by_t"].values():
            for key in ("rows", "skipped_same_color", "skipped_no_candidate", "share_nearer_asked_a", "blocks"):
                assert key in x, key
    x = r["by_t"]["0.3"]
    for key in ("serve", "together", "unturned_gate_identical", "neutral", "missing", "unheld", "tolerance"):
        assert key in x, key
    if x["scenes"]:
        assert x["unturned_gate_identical"] is True and x["missing"]["smallest_base"] == 1.0
        assert x["unheld"]["largest_member"] == 0.0 and x["serve"]["reverse_restores"] is True
    print(kind, "address scenes", x["scenes"], "skipped", x["skipped"], "; FIND rows",
          {t: v["rows"] for t, v in r["find"].get("by_t", {}).items()})


def test_exchange_is_exact_at_one_step_and_over_the_chain(runs, tmp_path):
    r = A.main(variant="exchange", out=str(tmp_path / "ex"), swap_class=0, **_common(runs))
    _files(tmp_path / "ex")
    assert r["one_step"]["0.3"]["exchanged"] == [0] and all(v == 0.0 for v in r["one_step"]["0.3"]["others"].values())
    assert r["chain"]["largest_outside"] == 0.0 and r["chain"]["largest_outside_unclamped"] == 0.0
    print("chain: exchanged cells", r["chain"]["exchanged_cells"], "differing", r["chain"]["exchanged_cells_differing"])
