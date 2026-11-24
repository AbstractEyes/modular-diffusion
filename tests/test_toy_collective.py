"""The toy collective's experiment module on tiny settings: every variant runs and writes the keys the table tool
reads; the rows find their engines; the exactness read is exact on the addressed collective."""
import json
import os

import pytest

from geolip.modular_diffusion import run
from geolip.modular_diffusion.experiments import toy_collective as T

TINY = dict(dim=64, heads=1, depth=1, steps=2, part_steps=2, model_steps=2, model_dim=64, model_heads=1, model_depth=1,
            batch=4, heldout=8, every=1, count=12, ns="1,9", sample_steps=2, chunk=0, exact_lists=12, leak_scenes=4,
            judge_steps=2, judge_batch=8, device="cpu", threads=4, save_every=0)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy_runs")
    T.main(variant="judge", bed="F", out=str(root / "judge_F"), **TINY)
    jobs = [("addressed_thing", "0-8", 0, "embedding"), ("addressed_base", "", 0, "embedding"),
            ("whole_thing", "0-8", 0, "embedding"), ("whole_empty", "", 0, "embedding"),
            ("whole_unconditional", "", 0, "embedding"), ("model_mix", "", 0, "embedding"),
            ("model_one", "", 0, "embedding"), ("addressed_thing", "0-8", 0, "rotary"),
            ("addressed_base", "", 0, "rotary"), ("addressed_thing", "0-8", 1, "embedding"),
            ("whole_thing", "0-8", 1, "embedding")]
    for kind, classes, seed, read in jobs:
        T.main(variant="parts", kind=kind, classes=classes, seed=seed, read=read, bed="F",
               out=str(root / f"parts_{kind}_{seed}_{read}"), **TINY)
    return root


def _common(runs):
    return dict(TINY, parts=str(runs), judge=str(runs / "judge_F"), bed="F")


def test_the_entry_point_knows_the_experiment():
    assert "toy_collective" in run.EXPERIMENTS
    assert T.parse_cells("01235678") == [0, 1, 2, 3, 5, 6, 7, 8] and T.parse_cells("0,4") == [0, 4]
    assert T.parse_cells("") is None and T.parse_classes("0-8") == list(range(9)) and T.parse_classes("") == [None]
    assert T.engine_tag("addressed_thing", 3, "F", 0, 3000, 128, 2, 3) == "addressed_thing_c3_F_s0_n3000_w128h2d3_embedding"


def test_parts_and_judge_write_their_results(runs):
    r = json.load(open(runs / "parts_addressed_thing_0_embedding" / "result.json", encoding="utf-8"))
    assert len(r["engines"]) == 9 and all("descended" in e and e["curve"] for e in r["engines"])
    j = json.load(open(runs / "judge_F" / "result.json", encoding="utf-8"))
    assert j["judge"]["ceiling"]["crops"] == 4500 and "holds_rates" in j["judge"]
    assert open(runs / "judge_F" / "log.txt", encoding="utf-8").read().rstrip().endswith("done")


@pytest.mark.parametrize("row", ["R1", "R1r", "R2", "R3", "R4", "R5a", "R6", "REAL"])
def test_compose_writes_every_key_the_table_reads(runs, tmp_path, row):
    r = T.main(variant="compose", row=row, out=str(tmp_path / row), **_common(runs))
    assert set(r["by_n"]) == {"1", "9"}
    for n, x in r["by_n"].items():
        for key in ("cell_rate", "color_rate", "scene_right", "scene_right_colors", "ghosts", "histogram",
                    "by_neighbor", "per_list", "ghost_ink"):
            assert key in x, (row, n, key)
        assert len(x["per_list"]["scene_right"]) == 12 and len(x["per_list"]["cell_rate"]) == 12
    assert "inconclusive" in r and isinstance(r["engines"], list)
    assert os.path.exists(tmp_path / row / "sheet.jpg") and os.path.exists(tmp_path / row / "result.md")
    if row == "R1":
        assert len(r["engines"]) == 10


def test_compose_on_bed_C_reads_tint_and_seams(runs, tmp_path):
    root = tmp_path / "C"
    T.main(variant="judge", bed="C", out=str(root / "judge_C"), **TINY)
    r = T.main(variant="compose", row="REAL", bed="C", judge=str(root / "judge_C"), parts="", out=str(root / "real"),
               **{k: v for k, v in TINY.items()})
    x = r["by_n"]["9"]
    assert x["tint"]["coherence_index"] is not None and abs(x["tint"]["coherence_index"]) < 1e-9
    assert set(x["seam"]) == {"base_base", "thing_base", "thing_thing"}


def test_the_address_never_trained_lists(runs, tmp_path):
    a = dict(_common(runs), count=12)
    c = T.lists_for(dict(T.DEFAULTS, **a, lists="center"))[1]
    o = T.lists_for(dict(T.DEFAULTS, **a, lists="others"))[1]
    assert bool((c["cells"][:, 4] >= 0).all()) and bool((o["cells"][:, 4] < 0).all())
    assert bool(((o["cells"] >= 0).sum(1) == 1).all())


def test_controls_exact_leak_wrong_run(runs, tmp_path):
    c = T.main(variant="controls", row="R1", out=str(tmp_path / "controls"), **_common(runs))
    assert set(c["controls"]) == {"b", "c", "d", "e", "f", "g"}
    assert c["controls"]["d"]["holds"] is True and c["controls"]["e"]["holds"] is True
    assert c["controls"]["g"]["holds"] is True
    cr = T.main(variant="controls", row="R1r", out=str(tmp_path / "controls_r"), **_common(runs))
    assert set(cr["controls"]) == {"c", "d", "f"} and cr["controls"]["d"]["holds"] is True
    e = T.main(variant="exact", rows="R1,R2,R3", out=str(tmp_path / "exact"), **_common(runs))
    assert e["exact"]["R1"]["exchange_exact"] is True and e["exact"]["R1"]["snap_in_exact"] is True
    assert e["exact"]["R2"]["snap_in_exact"] is True
    lk = T.main(variant="leak", leak_ns="2,9", leak_ts="0.45", out=str(tmp_path / "leak"), **_common(runs))
    assert len(lk["leak"]) == 2 and lk["leak"][0]["other_cells"] is not None and lk["leak"][1]["free_cells"] is None
    w = T.main(variant="wrong", wrong_ns="2", e="0.5", out=str(tmp_path / "wrong"), **_common(runs))
    assert [x["e"] for x in w["wrong"]] == [0.0, 0.5] and w["wrong"][0]["corruptions"]["kept"] == 24
    t = T.main(variant="timing", kind="addressed_thing", steps=4, from_step=1, out=str(tmp_path / "timing"),
               **{k: v for k, v in TINY.items() if k != "steps"})
    assert t["timing"]["seconds_a_step"] > 0
    assert "| R1 |" in T.summarize(str(tmp_path)) or "| job |" in T.summarize(str(tmp_path))
