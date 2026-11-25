"""From a caption to a picture on the toy bed: the bed's captions (counts, wordings, one instance a cell, a true layout
scores 1), the rules that turn a written layout into requests, the four reads per caption (written right, handed right,
drawn as written, picture right) on hand-made cases, and the writer cell and the drawing variant end to end on tiny
engines with the true writer and a garbage writer."""
import hashlib
import json
import random

import pytest

from geolip.modular_diffusion import layout as L
from geolip.modular_diffusion import scenes as S
from geolip.modular_diffusion.experiments import layout_write as W
from geolip.modular_diffusion.experiments import toy_collective as T

BED = L.bed_prompts(0)
BY_ID = {p["id"]: p for p in BED}
# the sha256 of the default generator's prompts (json, sorted keys), measured before the bed's captions existed
DEFAULT_SHA = {0: "16cc5ad0f5ca0c79c571b746f215482967ab06baf43a10517684cfa4a21f05cd",
               1: "999895cb23988a377bccc96cd566712b88c99e06a236f260c899befc36d15e70"}


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_the_default_generator_is_unchanged():
    for seed, digest in DEFAULT_SHA.items():
        assert _sha(L.make_prompts(seed)) == digest
    assert all("vocabulary" not in p for p in L.make_prompts(0))


def test_the_bed_captions_counts_and_words():
    assert len(BED) == 600 and [sum(p["family"] == f for p in BED) for f in L.FAMILIES] == [100] * 6
    assert L.BED_THINGS == S.THINGS and L.BED_COLOURS == S.COLORS
    assert BED == L.bed_prompts(0) and BED != L.bed_prompts(1)
    assert BED[0]["text"] == "A realistic photo of a scene with one disc"
    words = set(L.BED_THINGS) | set(L.BED_PLURALS.values())
    for p in BED:
        assert p["vocabulary"] == "bed"
        assert all(a["thing"] in L.BED_THINGS for a in p["asks"]) and all(t in L.BED_THINGS for t in p["absent"])
        assert all(a["colour"] is None or a["colour"] in L.BED_COLOURS for a in p["asks"])
        nouns = set(L._words(p["text"])) & (set(L.THINGS) | set(L.PLURALS.values()))
        assert not nouns, (p["id"], nouns)        # no thing of the default lists in a bed caption
        assert set(L._words(p["text"])) & words, p["id"]
    # the wordings of the six families
    fam = {f: [p["text"] for p in BED if p["family"] == f] for f in L.FAMILIES}
    assert all(t.startswith("A realistic photo of a scene with ") for t in fam["count"] + fam["colour"] + fam["place"])
    assert all(t.startswith("A realistic photo of a scene without ") for t in fam["absent"])
    assert any(" pluses" in t or " crosses" in t for t in fam["count"])
    assert all(t.endswith(".") and "; the " in t for t in fam["combined"])
    sizes = [sum(a["count"] for a in p["asks"]) for p in BED if p["family"] == "combined"]
    assert [sizes.count(k) for k in (2, 3, 4, 5)] == [25] * 4
    counts = sorted({sum(a["count"] for a in p["asks"]) for p in BED if p["family"] == "count"})
    assert counts == [1, 2, 3, 4, 5]


def test_every_bed_true_layout_has_one_instance_a_cell_and_scores_one():
    for k, p in enumerate(BED):
        cells = [(i["col"], i["row"]) for i in p["truth"]]
        assert len(cells) == len(set(cells))
        r = L.caption_reads(p, k, L.render("cell", p["truth"]))
        assert r["W"] == 1 and r["H"] == 1 and not r["dropped"], p["id"]
    # the default generator may put several instances of one thing in one cell: the bed's may not
    many = [p for p in L.make_prompts(0) if len({(i["col"], i["row"]) for i in p["truth"]}) < len(p["truth"])]
    assert many


def test_a_bed_answer_is_read_with_the_bed_things():
    p = BY_ID["count_000"]                                    # one disc
    assert L.score(p, L.read("cell", "disc | none | left | top"), "cell")["address_right"] == 1
    assert L.score(p, L.read("cell", "circle | none | left | top"), "cell")["address_right"] == 0
    # the same label read without the bed's vocabulary names no thing
    assert L.thing_of("red disc") is None and L.thing_of("red disc", L.VOCABULARIES["bed"]) == "disc"
    assert L.thing_of("two pluses", L.VOCABULARIES["bed"]) == "plus"


# ---------------------------------------------------------------- answer -> requests, each rule
def _hand(answer, index=7):
    return L.bed_requests(L.read_table(answer), index)


def test_rule_a_thing_not_of_the_bed_is_dropped():
    h = _hand("kite | red | left | top\ndisc | red | centre | top")
    assert [q["thing"] for q in h["requests"]] == ["disc"]
    assert h["dropped"] == [{"line": 0, "label": "kite", "why": "not a thing of the bed"}]


def test_rule_no_address_is_dropped():
    h = _hand("disc | red | far | top\nstar | blue | right | bottom")
    assert [q["thing"] for q in h["requests"]] == ["star"] and h["dropped"][0]["why"] == "no address"


def test_rule_the_first_written_keeps_a_cell():
    h = _hand("disc | red | left | top\nstar | blue | left | top\ndisc | red | left | top")
    assert [(q["thing"], q["line"]) for q in h["requests"]] == [("disc", 0)]
    assert [d["why"] for d in h["dropped"]] == ["cell taken by another thing", "cell taken by the same thing"]


def test_rule_a_missing_or_foreign_colour_is_drawn_and_a_stated_one_kept():
    h = _hand("disc | none | left | top\nstar | white | right | top\nring | blue | centre | middle", index=42)
    rng = random.Random(9000 + 42)
    first, second = L.BED_COLOURS[rng.randrange(9)], L.BED_COLOURS[rng.randrange(9)]
    q = h["requests"]
    assert (q[0]["colour"], q[0]["colour_drawn"], q[0]["colour_why"]) == (first, True, "no colour")
    assert (q[1]["colour"], q[1]["colour_drawn"], q[1]["colour_why"]) == (second, True, "a colour not of the bed")
    assert (q[2]["colour"], q[2]["colour_drawn"]) == ("blue", False)
    assert _hand("disc | none | left | top", index=43)["requests"][0]["colour"] == \
        L.BED_COLOURS[random.Random(9043).randrange(9)]


def test_rule_an_unparsed_answer_hands_nothing():
    h = _hand("disc | red | left")
    assert h == {"parsed": False, "requests": [], "dropped": []}
    r = L.caption_reads(BY_ID["count_000"], 0, "disc | red | left")
    assert (r["parsed"], r["W"], r["H"]) == (0, 0, 0) and L.picture_reads(BY_ID["count_000"], r, None) == {"C": 0, "P": 0}


def test_a_table_dressed_answer_is_read_by_the_tolerant_read():
    p = BY_ID["count_000"]
    r = L.caption_reads(p, 0, L.dress_table("disc | none | left | top"))
    assert r["parsed"] == 1 and r["W"] == 1 and r["H"] == 1


# ---------------------------------------------------------------- W, C, P, H on hand-made cases
def _caption(text, asks, relations=(), places=(), family="combined"):
    return L._prompt("combined_000", family, text, asks, places, relations, [], []) | {"vocabulary": "bed"}


TWO_DISCS = _caption("A realistic photo of two red discs and a blue star; the discs are to the left of the star.",
                     [{"thing": "disc", "count": 2, "colour": "red"}, {"thing": "star", "count": 1, "colour": "blue"}],
                     relations=[{"a": "disc", "rel": "left of", "b": "star"}])


def _as_drawn(reads):
    return L.request_cells(reads["requests"])


def test_two_discs_in_two_cells_read_right_everywhere():
    r = L.caption_reads(TWO_DISCS, 0, "disc | red | left | top\ndisc | red | left | bottom\nstar | blue | right | middle")
    assert (r["W"], r["H"]) == (1, 1)
    assert L.picture_reads(TWO_DISCS, r, _as_drawn(r)) == {"C": 1, "P": 1}


def test_two_discs_in_one_cell_written_right_handed_wrong():
    r = L.caption_reads(TWO_DISCS, 0, "disc | red | left | top\ndisc | red | left | top\nstar | blue | right | middle")
    assert (r["W"], r["H"]) == (1, 0)                       # the scorer counts a cell conflict only between two things
    assert [d["why"] for d in r["dropped"]] == ["cell taken by the same thing"]
    assert L.picture_reads(TWO_DISCS, r, _as_drawn(r)) == {"C": 1, "P": 0}      # drawn exactly as handed, one disc short


def test_a_picture_that_differs_from_the_requests():
    r = L.caption_reads(TWO_DISCS, 0, "disc | red | left | top\ndisc | red | left | bottom\nstar | blue | right | middle")
    judged = _as_drawn(r)
    judged[0] = [None, None]                                   # the top-left disc not drawn
    assert L.picture_reads(TWO_DISCS, r, judged) == {"C": 0, "P": 0}
    judged = _as_drawn(r)
    judged[4] = [L.BED_THINGS.index("ring"), 0]                # a ghost in the centre cell
    assert L.picture_reads(TWO_DISCS, r, judged) == {"C": 0, "P": 0}
    judged = _as_drawn(r)
    judged[5] = [judged[5][0], L.BED_COLOURS.index("red")]     # the star drawn red: its stated colour broken
    assert L.picture_reads(TWO_DISCS, r, judged) == {"C": 0, "P": 0}


def test_a_colour_the_caption_does_not_state():
    p = _caption("A realistic photo of a disc and a blue star; the disc is to the left of the star.",
                 [{"thing": "disc", "count": 1, "colour": None}, {"thing": "star", "count": 1, "colour": "blue"}],
                 relations=[{"a": "disc", "rel": "left of", "b": "star"}])
    r = L.caption_reads(p, 5, "disc | none | left | top\nstar | blue | right | top")
    assert (r["W"], r["H"]) == (1, 1) and r["requests"][0]["colour_drawn"]
    judged = _as_drawn(r)
    assert L.picture_reads(p, r, judged) == {"C": 1, "P": 1}
    judged[0] = [judged[0][0], (judged[0][1] + 1) % 9]          # another colour than the drawn one: P may exceed C
    assert L.picture_reads(p, r, judged) == {"C": 0, "P": 1}


def test_a_drawn_colour_is_no_stated_colour_even_when_it_equals_the_caption_s():
    # a colour caption, one of its things written with the colour none (the other as the truth says), whose colour drawn
    # by the caption's seeded generator happens to be the very colour the caption states for it
    found = None
    for k, p in enumerate(BED):
        if p["family"] != "colour":
            continue
        for j in (0, 1):
            lines = []
            for i, t in enumerate(p["truth"]):
                colour = "none" if i == j else t["colour"]
                lines.append(f"{t['thing']} | {colour} | {L.COLUMNS[t['col']]} | {L.ROWS[t['row']]}")
            rng = random.Random(9000 + k)
            drawn = [L.BED_COLOURS[rng.randrange(9)] for i in range(len(p["truth"])) if i == j]
            if drawn[0] == p["truth"][j]["colour"]:
                found = (k, p, j, "\n".join(lines))
                break
        if found:
            break
    assert found, "no colour caption whose drawn colour equals its stated one"
    k, p, j, answer = found
    r = L.caption_reads(p, k, answer)
    q = r["requests"][j]
    assert q["colour"] == p["truth"][j]["colour"] and q["colour_drawn"] is True
    assert r["W"] == 0 and r["H"] == 0            # the drawn colour counts as no stated colour in H
    assert L.picture_reads(p, r, L.request_cells(r["requests"])) == {"C": 1, "P": 1}


def test_drawn_as_written_implies_picture_right_for_every_true_layout():
    for k, p in enumerate(BED):
        r = L.caption_reads(p, k, L.render("cell", p["truth"]))
        assert L.picture_reads(p, r, _as_drawn(r)) == {"C": 1, "P": 1}, p["id"]


# ---------------------------------------------------------------- the writer cell and the drawing variant, tiny
TINY = dict(dim=64, heads=1, depth=1, steps=2, part_steps=2, batch=4, heldout=8, every=1, count=12, ns="4",
            sample_steps=2, chunk=0, judge_steps=2, judge_batch=8, device="cpu", threads=4, save_every=0)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("prompt_runs")
    T.main(variant="judge", bed="F", out=str(root / "judge_F"), **TINY)
    for kind, classes, name in (("addressed_thing", "0-8", "at_F_s0"), ("addressed_base", "", "ab_F_s0")):
        T.main(variant="parts", kind=kind, classes=classes, seed=0, bed="F", out=str(root / name), **TINY)
    T.main(variant="compose", row="R1", bed="F", parts=f"{root / 'at_F_s0'},{root / 'ab_F_s0'}",
           judge=str(root / "judge_F"), out=str(root / "R1_F_s0"), **TINY)
    for mode in ("truth", "shuffled", "broken"):
        W.main(variant="write", writer=f"fake:{mode}", form="cell", examples=1, captions="bed", first_n=3,
               repeat_n=2, out=str(root / f"{mode}_cell_ex1"))
    return root


def _draw(root, mode, tmp):
    return T.main(variant="draw", answers=str(root / f"{mode}_cell_ex1"), parts=f"{root / 'at_F_s0'},{root / 'ab_F_s0'}",
                  judge=str(root / "judge_F"), bed="F", seed=0, role=mode, out=str(tmp / mode), **TINY)


def test_the_writer_cell_on_the_bed_captions(tiny):
    r = json.load(open(tiny / "truth_cell_ex1" / "result.json", encoding="utf-8"))
    assert r["captions"] == "bed" and r["prompts"] == 18 and r["real_prompts"] == 0
    assert all(b["scores"]["address_right"]["mean"] == 1.0 for b in r["summary"]["families"].values())
    assert r["controls"]["a"]["holds"] is True and r["controls"]["a"]["prompts"] == 606
    assert r["controls"]["e"]["holds"] is True
    with pytest.raises(ValueError):
        W.main(variant="write", writer="fake:truth", form="box", captions="bed", out=str(tiny / "refused"))


def test_the_drawing_variant_end_to_end(tiny, tmp_path):
    t = _draw(tiny, "truth", tmp_path)
    assert t["captions"] == 18 and t["drawn"] == 18 and len(t["engines"]) == 10
    for r in t["records"]:
        assert set(r) >= {"id", "family", "requests", "judged", "W", "C", "P", "H", "dropped"}
        assert r["W"] == 1 and r["H"] == 1 and len(r["judged"]) == 9
        assert r["C"] == 0 or r["P"] == 1
    assert (tmp_path / "truth" / "sheet.jpg").exists() and (tmp_path / "truth" / "result.md").exists()
    assert t["tables"]["all"]["n"] == 18 and set(t["tables"]["families"]) == set(L.FAMILIES)
    again = _draw(tiny, "truth", tmp_path / "again")
    assert again["pictures_sha256"] == t["pictures_sha256"]          # the same pictures on a second run
    g = _draw(tiny, "broken", tmp_path)
    assert g["drawn"] == 0 and all(r["P"] == 0 and r["C"] == 0 and r["W"] == 0 for r in g["records"])
    n = _draw(tiny, "shuffled", tmp_path)
    assert n["drawn"] == 18 and any(r["W"] == 0 for r in n["records"])
    assert "| draw_" not in T.summarize_drawn(str(tmp_path)) and "truth_cell_ex1" in T.summarize_drawn(str(tmp_path))


def test_the_same_noise_for_every_writer(tiny, tmp_path):
    rec = [{"id": "combined_002", "family": "combined"}, {"id": "count_000", "family": "count"}]
    z = T.caption_noise(rec)
    import geolip.modular_diffusion.collective as C
    assert bool((z[0] == C.start_noise(100, 7105)[2]).all()) and bool((z[1] == C.start_noise(100, 7100)[0]).all())


def test_the_parity_variant(tiny, tmp_path):
    p = T.main(variant="parity", parts=f"{tiny / 'at_F_s0'},{tiny / 'ab_F_s0'}", judge=str(tiny / "judge_F"),
               stored=str(tiny / "R1_F_s0"), real_row="", wrong_rows="", bed="F", seed=0, out=str(tmp_path / "parity"),
               **TINY)
    q = p["parity"]
    assert q["difference"] == 0.0 and q["holds"] is True and q["pictures_equal"] is None
    assert p["judge_control"]["holds_rates"] is not None and p["judge_control"]["holds"] is None   # no real row given
