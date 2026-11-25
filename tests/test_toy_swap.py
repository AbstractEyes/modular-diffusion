"""Exchanging named parts of the toy collective (``swap``) on tiny engines: an exchange changes nothing outside the
exchanged class's cells and something inside; an empty swap leaves a result as it was; each refusal says why; an
untrained part can be swapped in; the unexchanged reads stored beside equal a separate job without the exchange; the
result names every engine that served."""
import hashlib
import json

import pytest
import torch

from geolip.modular_diffusion import collective as C
from geolip.modular_diffusion import layout as L
from geolip.modular_diffusion.experiments import layout_write as W
from geolip.modular_diffusion.experiments import toy_collective as T

TINY = dict(dim=64, heads=1, depth=1, steps=2, part_steps=2, batch=4, heldout=8, every=1, count=12, ns="4,9",
            sample_steps=2, chunk=0, judge_steps=2, judge_batch=8, device="cpu", threads=4, save_every=0)
LONG = "addressed_thing_c8_F_s0_n4_w64h1d1_embedding"
WIDE = "addressed_thing_c8_F_s1_n2_w128h2d1_embedding"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("swap_runs")
    T.main(variant="judge", bed="F", out=str(root / "judge_F"), **TINY)
    T.main(variant="parts", kind="addressed_thing", classes="0-8", seed=0, bed="F", out=str(root / "at"), **TINY)
    T.main(variant="parts", kind="addressed_base", classes="", seed=0, bed="F", out=str(root / "ab"), **TINY)
    T.main(variant="parts", kind="addressed_thing", classes="3,8", seed=0, bed="F", out=str(root / "long"),
           **dict(TINY, steps=4))
    T.main(variant="parts", kind="addressed_thing", classes="8", seed=1, bed="F", out=str(root / "wide"),
           **dict(TINY, dim=128, heads=2))
    T.main(variant="parts", kind="addressed_thing", classes="8", seed=0, bed="C", out=str(root / "bedC"), **TINY)
    T.main(variant="parts", kind="addressed_thing", classes="8", seed=0, read="rotary", bed="F", out=str(root / "rot"),
           **TINY)
    T.main(variant="parts", kind="whole_thing", classes="8", seed=0, bed="F", out=str(root / "whole"), **TINY)
    W.main(variant="write", writer="fake:truth", form="cell", examples=1, captions="bed", first_n=3, repeat_n=2,
           out=str(root / "true_cell_ex1"))
    W.main(variant="write", writer="fake:broken", form="cell", examples=1, captions="bed", first_n=3, repeat_n=2,
           out=str(root / "broken_cell_ex1"))
    return root


def _args(runs):
    return dict(TINY, parts=str(runs), judge=str(runs / "judge_F"), bed="F", seed=0)


def _compose(runs, out, **kw):
    return T.main(variant="compose", out=str(out), **{**_args(runs), "row": "R1", **kw})


def _draw(runs, out, **kw):
    return T.main(variant="draw", answers=str(runs / "true_cell_ex1"), role="true", out=str(out),
                  **{**_args(runs), "row": "R1", **kw})


def _no_seconds(by_n):
    return {n: {k: v for k, v in x.items() if k != "seconds"} for n, x in by_n.items()}


def _no_rights(by_n):
    out = _no_seconds(by_n)
    for x in out.values():
        x["per_list"] = {k: v for k, v in x["per_list"].items() if k not in ("thing_right_cells", "cell_reads")}
    return out


@pytest.fixture(scope="module")
def plain(runs, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("swap_plain")
    return {"compose": _compose(runs, tmp / "compose"), "draw": _draw(runs, tmp / "draw")}


@pytest.fixture(scope="module")
def swapped(runs, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("swap_swapped")
    return {"compose": _compose(runs, tmp / "compose", swap=f"8:{LONG}"),
            "draw": _draw(runs, tmp / "draw", swap=f"8:{LONG}")}


def test_an_exchange_changes_nothing_outside_and_something_inside(swapped):
    for variant in ("compose", "draw"):
        x = swapped[variant]["exact"]
        assert x["holds"] is True and x["largest_outside"] == 0.0, variant
        assert x["cells_compared"] > 0 and x["exchanged_cells"] > 0, variant
        assert x["inside_share"] > 0, variant
    c = swapped["compose"]
    assert all(v["largest_outside"] == 0.0 for v in c["exact"]["by_n"].values())
    assert c["swap"] == {"8": LONG}
    # the star's reads may differ; the stored reads of the two collectives come from one job, list by list
    assert set(c["unswapped"]) == set(c["by_n"]) == {"4", "9"}
    for n in ("4", "9"):
        cells = c["by_n"][n]["per_list"]["thing_right_cells"]
        assert len(cells) == 12 and all(len(r) == 9 for r in cells)
        assert sum(v >= 0 for r in cells for v in r) == 12 * int(n)


def test_several_classes_and_a_wider_part_of_another_seed(runs, tmp_path):
    r = _compose(runs, tmp_path / "two", swap=f"8:{WIDE},3:addressed_thing_c3_F_s0_n4_w64h1d1_embedding", ns="9")
    assert r["exact"]["holds"] is True and r["exact"]["inside_share"] > 0
    ex = {e["serves"]: e for e in r["engines"] if e["exchanged"]}
    assert set(ex) == {3, 8} and ex[8]["shape"] == [128, 2, 1] and ex[8]["seed"] == 1 and ex[3]["steps"] == 4


def test_an_empty_swap_leaves_the_results_as_they_were(runs, plain, tmp_path):
    c = _compose(runs, tmp_path / "compose", swap="")
    assert _no_seconds(c["by_n"]) == _no_seconds(plain["compose"]["by_n"])
    assert c["engines"] == plain["compose"]["engines"]
    assert not {"swap", "exact", "unswapped"} & set(c) and not {"swap", "exact", "unswapped"} & set(plain["compose"])
    assert all("thing_right_cells" not in x["per_list"] for x in c["by_n"].values())
    d = _draw(runs, tmp_path / "draw", swap="")
    assert d["pictures_sha256"] == plain["draw"]["pictures_sha256"]
    assert d["records"] == plain["draw"]["records"] and d["tables"] == plain["draw"]["tables"]
    assert not {"swap", "exact", "tables_unswapped"} & set(d)
    assert all("judged_unswapped" not in r for r in d["records"])


def test_the_unexchanged_reads_equal_a_separate_job(plain, swapped):
    c, p = swapped["compose"], plain["compose"]
    assert _no_rights(c["unswapped"]) == _no_seconds(p["by_n"])
    d, q = swapped["draw"], plain["draw"]
    assert d["pictures_sha256_unswapped"] == q["pictures_sha256"]
    by = {r["id"]: r for r in q["records"]}
    for r in d["records"]:
        assert r["judged_unswapped"] == by[r["id"]]["judged"]
        assert (r["C_unswapped"], r["P_unswapped"]) == (by[r["id"]]["C"], by[r["id"]]["P"])
    assert d["tables_unswapped"] == q["tables"]


def test_the_result_names_every_engine_that_served(swapped):
    for variant in ("compose", "draw"):
        e = swapped[variant]["engines"]
        assert len(e) == 11, variant
        for x in e:
            assert {"tag", "serves", "steps", "shape", "trained", "exchanged", "descended", "seed"} <= set(x)
        plain = [x for x in e if not x["exchanged"]]
        assert [x["serves"] for x in plain] == list(range(9)) + ["base"]
        assert all(x["steps"] == 2 and x["shape"] == [64, 1, 1] for x in plain)
        ex = [x for x in e if x["exchanged"]]
        assert len(ex) == 1 and ex[0]["tag"] == LONG and ex[0]["serves"] == 8 and ex[0]["steps"] == 4
        assert ex[0]["trained"] == plain[8]["trained"] and ex[0]["descended"] is not None
        assert swapped[variant]["inconclusive"] == bool(swapped[variant]["not_descended"])


def test_a_newborn_part(runs, tmp_path):
    r = _compose(runs, tmp_path / "newborn", swap="8:newborn")
    assert r["exact"]["holds"] is True and r["exact"]["inside_share"] > 0
    nb = [x for x in r["engines"] if x["exchanged"]]
    assert len(nb) == 1 and nb[0]["newborn"] is True and nb[0]["tag"] == "newborn" and nb[0]["steps"] == 0
    assert nb[0]["shape"] == [64, 1, 1] and nb[0]["serves"] == 8
    assert len(r["engines"]) == 11 and "newborn" not in r["not_descended"]


@pytest.mark.parametrize("swap, words", [
    ("8:nothing_here", "no finished engine named nothing_here"),
    ("3:" + LONG, "its class is 8, not 3"),
    ("8:addressed_thing_c8_C_s0_n2_w64h1d1_embedding", "its bed is 'C', not the job's 'F'"),
    ("8:addressed_thing_c8_F_s0_n2_w64h1d1_rotary", "its read form is 'rotary', not the row's 'embedding'"),
    ("8:whole_thing_c8_F_s0_n2_w64h1d1_embedding", "its kind is 'whole_thing', not 'addressed_thing'"),
    (f"8:{LONG},8:newborn", "the class 8 is named twice"),
    ("9:newborn", "is not one of 0 to 8"),
    ("8", "is not a class:engine pair"),
])
def test_each_refusal_says_why(runs, tmp_path, swap, words):
    with pytest.raises(ValueError, match=words.replace("(", r"\(").replace(")", r"\)")):
        _compose(runs, tmp_path / "refused", swap=swap, ns="4")


def test_swap_refuses_another_row_or_variant(runs, tmp_path):
    with pytest.raises(ValueError, match="the addressed collective"):
        _compose(runs, tmp_path / "r2", swap=f"8:{LONG}", row="R2")
    with pytest.raises(ValueError, match="the variants compose, draw, not exact"):
        T.main(variant="exact", swap=f"8:{LONG}", out=str(tmp_path / "exact"), **_args(runs))
    with pytest.raises(ValueError, match="the variant draw draws the row R1"):
        _draw(runs, tmp_path / "drawr", swap=f"8:{LONG}", row="R1r")


@pytest.fixture(scope="module")
def second_judge(runs):
    T.main(variant="judge", bed="F", seed=1, out=str(runs / "judge_F_j1"), **TINY)
    return str(runs / "judge_F_j1")


def test_a_second_judge_reads_the_same_pictures(runs, plain, second_judge, tmp_path):
    two = f"{runs / 'judge_F'},{second_judge}"
    c = _compose(runs, tmp_path / "c2", judge=two)
    alone = _compose(runs, tmp_path / "c_alone", judge=second_judge)
    assert c["judges"] == ["judge_F", "judge_F_j1"]
    assert _no_seconds(c["by_n"]) == _no_seconds(plain["compose"]["by_n"])           # the first judge as before
    assert _no_seconds(c["by_judge"]["judge_F_j1"]) == _no_seconds(alone["by_n"])
    d = _draw(runs, tmp_path / "d2", judge=two)
    da = _draw(runs, tmp_path / "d_alone", judge=second_judge)
    assert d["pictures_sha256"] == plain["draw"]["pictures_sha256"] == da["pictures_sha256"]
    assert [{k: v for k, v in r.items() if k != "judged_by"} for r in d["records"]] == plain["draw"]["records"]
    assert [r["judged_by"]["judge_F_j1"] for r in d["records"]] == [r["judged"] for r in da["records"]]
    assert d["tables_by"]["judge_F_j1"] == da["tables"] and d["tables"] == plain["draw"]["tables"]


def test_a_second_judge_with_an_exchange(runs, second_judge, tmp_path):
    two = f"{runs / 'judge_F'},{second_judge}"
    c = _compose(runs, tmp_path / "c", judge=two, swap=f"8:{LONG}", ns="4")
    alone = _compose(runs, tmp_path / "ca", judge=second_judge, swap=f"8:{LONG}", ns="4")
    assert _no_seconds(c["unswapped_by_judge"]["judge_F_j1"]) == _no_seconds(alone["unswapped"])
    assert _no_seconds(c["by_judge"]["judge_F_j1"]) == _no_seconds(alone["by_n"])
    d = _draw(runs, tmp_path / "d", judge=two, swap=f"8:{LONG}")
    da = _draw(runs, tmp_path / "da", judge=second_judge, swap=f"8:{LONG}")
    assert [r["judged_unswapped_by"]["judge_F_j1"] for r in d["records"]] == [r["judged_unswapped"] for r in da["records"]]
    assert d["tables_unswapped_by"]["judge_F_j1"] == da["tables_unswapped"]


def test_two_judges_of_one_name_or_another_variant_are_refused(runs, second_judge, tmp_path):
    with pytest.raises(ValueError, match="the same base name 'judge_F'"):
        _compose(runs, tmp_path / "twice", judge=f"{runs / 'judge_F'},{runs / 'judge_F'}")
    with pytest.raises(ValueError, match="several judges apply to the variants compose, draw"):
        T.main(variant="controls", out=str(tmp_path / "ctl"), **{**_args(runs), "judge": f"{runs / 'judge_F'},{second_judge}"})


PLAIN = lambda kind, k: T.engine_tag(kind, k, "F", 0, 2, 64, 1, 1)


def _direct_pictures(runs, records, part8):
    """The pictures of the draw job's captions by a collective built here from the engines' own files (the plain parts,
    and ``part8`` for the star), not through ``swap``."""
    index = T.index_engines(str(runs))
    things = {k: C.load_engine(index[PLAIN("addressed_thing", k)]) for k in range(9)}
    things[8] = part8
    col = C.Collective(things, C.load_engine(index[PLAIN("addressed_base", None)]), addressed=True)
    drawn = [r for r in records if r["parsed"]]
    cells = torch.tensor([[-1 if c[0] is None else c[0] for c in L.request_cells(r["requests"])] for r in drawn])
    colors = torch.tensor([[-1 if c[1] is None else c[1] for c in L.request_cells(r["requests"])] for r in drawn])
    z = C.sample(lambda z_, t, c, k: col.answer(z_, t, c, k, "gate"), cells, colors, 0, 2, "cpu", 0,
                 noise=T.caption_noise(drawn))
    return T.pictures_sha256(T.made_pixels(z))


def test_the_exchanged_pictures_are_those_of_the_named_engine(runs, swapped):
    index = T.index_engines(str(runs))
    d = swapped["draw"]
    assert _direct_pictures(runs, d["records"], C.load_engine(index[LONG])) == d["pictures_sha256"]
    assert _direct_pictures(runs, d["records"], C.load_engine(index[PLAIN("addressed_thing", 8)])) == \
        d["pictures_sha256_unswapped"]


def test_a_newborn_part_is_the_born_engine(runs, tmp_path):
    d = _draw(runs, tmp_path / "nb", swap="8:newborn")
    born = C.born("addressed_thing", 8, 0, 64, 1, 1, "embedding").eval()
    assert _direct_pictures(runs, d["records"], born) == d["pictures_sha256"]


def test_every_engine_names_the_weights_it_read(runs, swapped):
    index = T.index_engines(str(runs))
    for variant in ("compose", "draw"):
        e = swapped[variant]["engines"]
        hashes = [x["weights_sha256"] for x in e]
        assert all(hashes) and len(set(hashes)) == len(hashes) == 11, variant
        ex = [x for x in e if x["exchanged"]][0]
        own = hashlib.sha256(open(f"{index[LONG]}/engine_avg.safetensors", "rb").read()).hexdigest()
        assert ex["weights_sha256"] == own == T.weights_sha256(index[LONG])


def test_the_exchange_is_real_and_exact_before_the_clamp(runs, swapped, tmp_path):
    for variant in ("compose", "draw"):
        x = swapped[variant]["exact"]
        assert x["real"] is True and x["largest_outside_unclamped"] == 0.0 and x["holds"] is True
    same = _compose(runs, tmp_path / "same", swap=f"8:{PLAIN('addressed_thing', 8)}", ns="4")
    assert same["exact"]["holds"] is True and same["exact"]["real"] is False
    assert same["exact"]["exchanged_cells"] > 0 and same["exact"]["inside_share"] == 0.0


def test_nothing_drawn_reads_nothing_and_the_pairs_are_still_checked(runs, tmp_path):
    common = dict(answers=str(runs / "broken_cell_ex1"), role="small")
    d = T.main(variant="draw", out=str(tmp_path / "none"), **{**_args(runs), **common, "swap": f"8:{LONG}"})
    assert d["drawn"] == 0 and d["exact"]["holds"] is None and d["exact"]["real"] is None
    assert len(d["engines"]) == 11
    with pytest.raises(ValueError, match="its class is 8, not 3"):
        T.main(variant="draw", out=str(tmp_path / "refused"), **{**_args(runs), **common, "swap": f"3:{LONG}"})


def test_the_stored_result_holds_the_exchange(swapped, runs, tmp_path):
    r = _compose(runs, tmp_path / "stored", swap=f"8:{LONG}", ns="4")
    stored = json.load(open(tmp_path / "stored" / "result.json", encoding="utf-8"))
    assert stored["exact"]["holds"] is True and stored["swap"] == {"8": LONG} and "4" in stored["unswapped"]
    assert stored["args"]["swap"] == f"8:{LONG}" and r["exact"] == stored["exact"]
