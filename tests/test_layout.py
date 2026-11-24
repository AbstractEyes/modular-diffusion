"""Layouts: the prompts with known answers, the readers of the three forms, the address of an instance, every score on
hand-made answers, the published rules on hand-made boxes, the controls, the summaries and the paired comparison."""
import json
import math

import pytest

from geolip.modular_diffusion import layout as L

PROMPTS = L.make_prompts(0)
EXAMPLES = L.example_prompts()


def _prompt(family, asks, places=(), relations=(), absent=(), pid="hand_000"):
    return {"id": pid, "family": family, "text": "", "asks": list(asks), "places": list(places),
            "relations": list(relations), "absent": list(absent), "truth": []}


def ask(thing, count=1, colour=None):
    return {"thing": thing, "count": count, "colour": colour}


def cells(*lines):
    return L.read("cell", "\n".join(lines))


# ------------------------------------------------------------------ the prompts
def test_the_prompts_are_the_same_list_twice_600_and_100_a_family():
    assert PROMPTS == L.make_prompts(0)
    assert len(PROMPTS) == 600 and len({p["id"] for p in PROMPTS}) == 600
    for fam in L.FAMILIES:
        mine = [p for p in PROMPTS if p["family"] == fam]
        assert len(mine) == 100 and all(p["id"].startswith(fam + "_") for p in mine)
    assert L.make_prompts(1) != PROMPTS


def test_the_families_follow_their_rules():
    count = [p for p in PROMPTS if p["family"] == "count"]
    pairs = sorted((p["asks"][0]["thing"], p["asks"][0]["count"]) for p in count)
    assert pairs == sorted((t, n) for t in L.THINGS_SIMPLE for n in range(1, 6) for _ in range(2))
    for k in range(0, 100, 2):                                 # the two copies of a text have two different truths
        assert count[k]["text"] == count[k + 1]["text"] and count[k]["truth"] != count[k + 1]["truth"]
    absent = [p for p in PROMPTS if p["family"] == "absent"]
    assert sorted(p["absent"][0] for p in absent) == sorted(t for t in L.THINGS_SIMPLE for _ in range(10))
    assert all(p["asks"] == [] and p["truth"] == [] for p in absent)
    for fam in ("colour", "place", "relation"):
        mine = [p for p in PROMPTS if p["family"] == fam]
        assert len({p["text"] for p in mine}) == 100, fam
        for p in mine:
            assert len(p["asks"]) == 2 and p["asks"][0]["thing"] != p["asks"][1]["thing"]
    for p in PROMPTS:
        if p["family"] == "colour":
            assert p["asks"][0]["colour"] != p["asks"][1]["colour"] and all(a["colour"] in L.COLOURS for a in p["asks"])
    simple = {a["thing"] for p in PROMPTS if p["family"] in ("count", "colour", "place") for a in p["asks"]}
    assert simple <= set(L.THINGS_SIMPLE)
    every = {a["thing"] for p in PROMPTS for a in p["asks"]} | {t for p in PROMPTS for t in p["absent"]}
    assert not every & set(L.THINGS_EXAMPLES)                 # the examples share no thing with a test prompt


def test_every_combined_truth_is_conflict_free_and_satisfies_its_facts():
    comb = [p for p in PROMPTS if p["family"] == "combined"]
    sizes = [sum(a["count"] for a in p["asks"]) for p in comb]
    assert [sizes.count(n) for n in (2, 3, 4, 5)] == [25, 25, 25, 25]
    for p in comb:
        assert 2 <= len(p["asks"]) <= 3 and 1 <= len(p["places"]) + len(p["relations"]) <= 2
        owner = {}
        for i in p["truth"]:
            assert owner.setdefault((i["col"], i["row"]), i["thing"]) == i["thing"], p["id"]
        s = L.score(p, L.read("cell", L.render("cell", p["truth"])), "cell")
        assert s["spatial"] == 1 and s["no_conflict"] == 1, p["id"]
        assert p["text"].startswith("A realistic photo of ") and p["text"].endswith(".")


def test_articles_and_the_worked_examples():
    assert L.article("orange") == "an" and L.article("red") == "a" and L.article("umbrella") == "an"
    assert any("an orange" in p["text"] or "an apple" in p["text"] or "an elephant" in p["text"] for p in PROMPTS)
    assert [p["family"] for p in EXAMPLES] == list(L.FAMILIES)
    assert EXAMPLES[5]["text"] == ("A realistic photo of two green kites, a white sheep and a clock; the kites are to the "
                                   "left of the sheep and the clock is on the top.")
    assert {a["thing"] for p in EXAMPLES for a in p["asks"]} | {"umbrella"} == set(L.THINGS_EXAMPLES)


# ------------------------------------------------------------------ control (a)
@pytest.mark.parametrize("form", ["box", "cell"])
def test_control_a_the_generators_layouts_score_1_everywhere(form):
    for p in PROMPTS + EXAMPLES:
        s = L.score(p, L.read(form, L.render(form, p["truth"])), form)
        applies = L.APPLIES[p["family"]]
        assert s["address_right"] == 1 and all(s[k] == 1 for k in applies), (p["id"], s)
        assert all(s[k] is None for k in L.SCORES if k not in applies)
        assert s["instance_errors"] == 0 and s["kinds"] == []


def test_the_box_rule_and_the_asks():
    two = [L._inst("cat", None, 0, 0)] * 3
    assert L.boxes_of(two) == [[110, 110, 270, 270], [150, 150, 310, 310], [190, 190, 350, 350]]
    with pytest.raises(ValueError):
        L.boxes_of([L._inst("cat", None, 2, 2)] * 4)
    assert L.render("cell", []) == "none" and L.render("box", []) == "[]"
    a = L.ask("box", "A realistic photo of a scene with one cup")
    assert a.startswith(L.ASK_BOX + "\n\nDescription: A realistic photo of a scene with two kites\nAnswer:\n[")
    assert a.endswith("\n\nDescription: A realistic photo of a scene with one cup\nAnswer:") and a.count("Description:") == 7
    assert L.ask("cell", "x", examples=False) == L.ASK_CELL + "\n\nDescription: x\nAnswer:"
    assert "centre" in L.ASK_CELL and "\nAnswer with these lines" in L.ASK_CELL
    assert L.ask("chain_text", "P").startswith("There is no picture to see. Here is a description of one:\n\nP\n\nTreat")


# ------------------------------------------------------------------ the thing of a label, the address
def test_thing_of():
    assert L.thing_of("a red cup") == "cup"
    assert L.thing_of("three cats") == "cat"
    assert L.thing_of("cups") == "cup"
    assert L.thing_of("the knives") == "knife"
    assert L.thing_of("cup and saucer") == "cup"
    assert L.thing_of("dog bed") is None                  # two listed things: an extra thing
    assert L.thing_of("teapot") is None and L.thing_of("Sheep") == "sheep" and L.thing_of("buses") == "bus"
    assert L.colours_of("a Grey cat") == ["gray"]


def test_address_of_a_box():
    assert L.address_of_box([0, 0, 100, 100]) == (0, 0)
    assert L.address_of_box([400, 400, 600, 600]) == (1, 1)
    assert L.address_of_box([700, 0, 1000, 300]) == (2, 0)
    assert L.address_of_box([300, 0, 460, 100]) == (1, 0)        # center x 0.38 is not under 0.38: centre
    assert L.address_of_box([300, 0, 459, 100]) == (0, 0)
    assert L.address_of_box([300, 0, 461, 100]) == (1, 0)
    assert L.address_of_box([0, 0, 1200, 100]) is None           # outside the canvas: no address
    assert L.address_of_box([500, 0, 400, 100]) is None          # no area


# ------------------------------------------------------------------ the readers
def test_the_box_reader_fence_prose_and_lenient():
    body = '[{"bbox_2d": [10, 10, 200, 200], "label": "red cup"}]'
    r = L.read("box", "```json\n" + body + "\n```")
    assert r["parsed"] and r["instances"][0]["thing"] == "cup" and r["instances"][0]["colours"] == ["red"]
    assert r["instances"][0]["col"] == 0 and r["instances"][0]["row"] == 0
    prose = L.read("box", "Here is the layout: " + body + " I hope this helps.")
    assert not prose["parsed"] and prose["failure"] == "not json"
    assert prose["lenient"] and prose["lenient"][0]["thing"] == "cup"
    p = _prompt("count", [ask("cup")])
    s = L.score(p, prose)
    assert s["parsed"] == 0 and s["address_right"] == 0 and s["lenient_address_right"] == 1
    assert s["kinds"] == ["broken format"] and s["instance_errors"] == 1
    lenient = L.read("box", 'x [{"bbox": [1, 1, 50, 50], "name": "dog"}, {"box": [1, 1], "label": "cat"}] y')
    assert [i["thing"] for i in lenient["lenient"]] == ["dog"]
    assert L.read("box", "[]")["parsed"] and L.read("box", "[]")["instances"] == []


def test_the_cell_reader():
    r = cells("cup | red | left | top", "chair | none | center | middle", "dog | grey | right | bottom")
    assert r["parsed"] and [(i["thing"], i["col"], i["row"]) for i in r["instances"]] == [("cup", 0, 0), ("chair", 1, 1),
                                                                                           ("dog", 2, 2)]
    assert r["instances"][2]["colours"] == ["gray"] and r["instances"][1]["colours"] == []
    assert L.read("cell", "none")["instances"] == [] and L.read("cell", " None \n")["parsed"]
    bad = L.read("cell", "Here you go:\ncup | red | left | top")
    assert not bad["parsed"] and bad["failure"] == "wrong field count" and len(bad["lenient"]) == 1
    far = cells("cup | none | far | top")
    assert far["parsed"] and far["instances"][0]["col"] is None and far["instances"][0]["kinds"] == ["no place word"]


def test_control_c_every_broken_answer_gives_its_kind():
    for form in ("box", "cell", "chain"):
        res = L.broken_check(form)
        assert res and all(x["ok"] for x in res), res
    kinds = {k for _, k in L.BROKEN["box"]} | {k for _, k in L.BROKEN["cell"]}
    assert {"not json", "wrong keys", "outside canvas", "no place word", "wrong field count"} <= kinds


# ------------------------------------------------------------------ the scores, by hand
def test_an_extra_thing_fails_things():
    p = _prompt("count", [ask("cat", 2)])
    s = L.score(p, cells("cat | none | left | top", "cat | none | left | middle", "bowl | none | right | top"))
    assert s["things"] == 0 and s["counts"] == 1 and s["address_right"] == 0 and "extra thing" in s["kinds"]
    assert s["instance_errors"] == 0                       # both asked cats are written right
    s2 = L.score(p, cells("cat | none | left | top", "cat | none | left | middle", "teapot | none | right | top"))
    assert s2["things"] == 0                               # a label that names no listed thing is an extra thing


def test_a_wrong_count():
    p = _prompt("count", [ask("cat", 3)])
    s = L.score(p, cells("cat | none | left | top", "cat | none | left | middle"))
    assert s["counts"] == 0 and s["things"] == 1 and s["kinds"] == ["wrong count"] and s["instance_errors"] == 1


def test_a_colour_on_the_wrong_thing():
    p = _prompt("colour", [ask("cup", 1, "red"), ask("chair", 1, "blue")])
    s = L.score(p, cells("cup | blue | left | top", "chair | red | right | top"))
    assert s["colours"] == 0 and s["address_right"] == 0 and s["instance_errors"] == 2 and "wrong colour" in s["kinds"]
    ok = L.score(p, L.read("box", '[{"bbox_2d": [0, 0, 100, 100], "label": "red cup"}, '
                                  '{"bbox_2d": [900, 900, 1000, 1000], "label": "blue chair"}]'))
    assert ok["address_right"] == 1
    two = L.score(p, cells("cup | red | left | top", "chair | blue | right | top", ))
    assert two["colours"] == 1
    both = L.score(p, cells("red blue cup | none | left | top", "chair | blue | right | top"))
    assert both["colours"] == 0                            # a color that is not the stated one


def test_a_box_in_the_middle_column_fails_on_the_left():
    p = _prompt("place", [ask("cup"), ask("chair")], places=[{"thing": "cup", "axis": "col", "value": 0},
                                                              {"thing": "chair", "axis": "col", "value": 2}])
    r = L.read("box", '[{"bbox_2d": [300, 400, 500, 600], "label": "cup"}, {"bbox_2d": [800, 400, 900, 600], '
                      '"label": "chair"}]')
    s = L.score(p, r)
    assert r["instances"][0]["col"] == 1 and s["spatial"] == 0 and s["address_right"] == 0 and s["instance_errors"] == 1
    assert L.published(p, r)["lmd"] == 1                    # the centers' strict rule passes it: no grid there


def test_two_different_things_in_one_cell():
    p = _prompt("colour", [ask("cup", 1, "red"), ask("chair", 1, "blue")])
    s = L.score(p, cells("cup | red | left | top", "chair | blue | left | top"))
    assert s["no_conflict"] == 0 and s["kinds"] == ["shared cell"] and s["instance_errors"] == 2
    same = L.score(_prompt("count", [ask("cat", 2)]), cells("cat | none | left | top", "cat | none | left | top"))
    assert same["address_right"] == 1                       # one thing twice in a cell is no conflict


def test_the_absent_thing_written():
    p = _prompt("absent", [], absent=["cat"])
    assert L.score(p, L.read("cell", "none"))["address_right"] == 1
    s = L.score(p, cells("cat | none | left | top"))
    assert s["absent"] == 0 and s["kinds"] == ["absent written"] and s["things"] is None
    assert L.score(p, cells("dog | none | left | top"))["address_right"] == 1     # another thing is allowed here


def test_relations_and_unplaced_instances():
    p = _prompt("relation", [ask("horse"), ask("bus")], relations=[{"a": "horse", "rel": "left of", "b": "bus"}])
    assert L.score(p, cells("horse | none | left | top", "bus | none | centre | bottom"))["spatial"] == 1
    assert L.score(p, cells("horse | none | centre | top", "bus | none | centre | bottom"))["spatial"] == 0
    s = L.score(p, cells("horse | none | far | top", "bus | none | right | top"))
    assert s["placed"] == 0 and s["spatial"] == 0 and "no place word" in s["kinds"]
    up = _prompt("relation", [ask("apple"), ask("couch")], relations=[{"a": "apple", "rel": "above", "b": "couch"}])
    assert L.score(up, cells("apple | none | left | top", "couch | none | right | middle"))["address_right"] == 1
    assert L.score(up, cells("apple | none | left | middle", "couch | none | right | middle"))["spatial"] == 0


def test_an_unparsed_answer_scores_0_on_everything_that_applies():
    p = PROMPTS[400]
    s = L.score(p, L.read("cell", "I cannot do that."))
    assert all(s[k] == 0 for k in L.APPLIES["relation"]) and s["absent"] is None and s["address_right"] == 0


# ------------------------------------------------------------------ the chain reader
def test_chain_read_three_chains():
    one = L.chain_read("see: a table with three red cups and a dog\ncount: three cups\ncaption: three cups and a dog.")
    assert one["parsed"]
    things = [(i["thing"], i["colours"], i["col"]) for i in one["instances"]]
    assert things == [("cup", ["red"], None)] * 3 + [("dog", [], None)]
    assert all(i["kinds"] == ["not placed"] for i in one["instances"])     # no place line: not placed
    two = L.chain_read("name: a laptop and a cup\nplace: the laptop sits on the left, at the top\n"
                       "place: the cup is in the centre\ncaption: a desk.")
    by = {i["thing"]: i for i in two["instances"]}
    assert (by["laptop"]["col"], by["laptop"]["row"]) == (0, 0)
    assert by["cup"]["col"] is None and by["cup"]["kinds"] == ["not placed"]      # a column word and no row word
    three = L.chain_read("see: a horse and a bus\nrelate: the horse stands to the left of the bus\n"
                         "place: the bus is at the bottom right\ncaption: a street.")
    assert three["relations"] == [{"a": "horse", "rel": "left of", "b": "bus"}]
    by = {i["thing"]: i for i in three["instances"]}
    assert (by["bus"]["col"], by["bus"]["row"]) == (2, 2) and by["horse"]["col"] is None
    assert not L.chain_read("A horse and a bus on a street.")["parsed"]
    neg = L.chain_read("see: an empty street with no cats\ncaption: an empty street.")
    assert neg["parsed"] and neg["instances"] == []
    mid = L.chain_read("see: a dog\nplace: the dog is in the middle on the left")
    assert (mid["instances"][0]["col"], mid["instances"][0]["row"]) == (0, 1)


def test_records_reading():
    r = L.read("spec", json.dumps({"thing": {"name": "cat", "count": "3", "colour": "red", "position": "left"},
                                   "relation": [{"a": "cat", "rel": "left of", "b": "dog"}],
                                   "form": {"medium": "photograph"}}))
    assert r["parsed"] and [i["thing"] for i in r["instances"]] == ["cat", "cat", "cat", "dog"]
    assert r["instances"][0]["colours"] == ["red"] and r["instances"][0]["col"] == 0 and r["instances"][0]["row"] is None
    p = _prompt("count", [ask("cat", 3)])
    s = L.score(p, r)
    assert s["counts"] == 1 and s["things"] == 0 and s["placed"] == 0     # the related dog is an extra thing
    assert L.score(_prompt("absent", [], absent=["cat"]), L.read("spec", "{}"))["address_right"] == 1


# ------------------------------------------------------------------ the published rules
def _boxes(*pairs):
    return L.read("box", json.dumps([{"bbox_2d": b, "label": lab} for lab, b in pairs]))


def test_the_published_rules_on_hand_made_boxes():
    p = _prompt("place", [ask("cup"), ask("chair")], places=[{"thing": "cup", "axis": "col", "value": 0},
                                                              {"thing": "chair", "axis": "col", "value": 2}])
    near = _boxes(("cup", [400, 400, 500, 500]), ("chair", [420, 600, 520, 700]))       # centers 450 and 470
    out = L.published(p, near)
    assert out["lmd"] == 1                                 # strict: 450 < 470
    assert out["margin"] == 0                              # 470 > 450 + 0.1 (100 + 100) = 470 is not strict
    assert out["sectors"] == 0                             # the chair is mostly below the cup, not to its right
    far = _boxes(("cup", [0, 400, 100, 500]), ("chair", [800, 400, 900, 500]))
    assert L.published(p, far) == {"lmd": 1, "margin": 1, "sectors": 1}
    rel = _prompt("relation", [ask("apple"), ask("couch")], relations=[{"a": "apple", "rel": "above", "b": "couch"}])
    two = _boxes(("apple", [0, 0, 100, 100]), ("apple", [0, 900, 100, 1000]), ("couch", [0, 500, 100, 600]))
    out = L.published(rel, two)
    assert out["sectors"] == 1 and out["margin"] == 0 and out["lmd"] is None      # any pair passes; one box a name
    cnt = _prompt("count", [ask("cat", 2)])
    assert L.published(cnt, _boxes(("cat", [0, 0, 9, 9]), ("black cat", [20, 20, 40, 40])))["lmd"] == 1
    assert L.published(cnt, _boxes(("cat", [0, 0, 9, 9]), ("bobcat", [20, 20, 40, 40])))["lmd"] == 1    # by substring
    col = _prompt("colour", [ask("cup", 1, "red"), ask("chair", 1, "blue")])
    assert L.published(col, _boxes(("red cup", [0, 0, 9, 9]), ("chair", [20, 20, 40, 40])))["lmd"] == 0
    assert L.published(cnt, L.read("cell", "cat | none | left | top")) == {"lmd": None, "margin": None, "sectors": None}
    assert L.published(cnt, L.read("box", "no"))["lmd"] == 0
    assert L.published(_prompt("absent", [], absent=["cat"]), _boxes(("dog", [0, 0, 9, 9])))["lmd"] == 1


def test_score_real():
    r = cells("dog | none | left | top", "frisbee | none | right | top")
    s = L.score_real("A dog catches a frisbee in a park.", r)
    assert s == {"parsed": 1, "placed": 1, "no_conflict": 1, "written_things_in_caption": 1}
    s = L.score_real("Two dogs in a park.", cells("dog | none | left | top", "cat | none | left | top"))
    assert s["no_conflict"] == 0 and s["written_things_in_caption"] == 0
    assert L.score_real("x", L.read("cell", "??"))["parsed"] == 0


# ------------------------------------------------------------------ control (b)
def test_the_derangement():
    for n in (2, 3, 10, 100):
        p = L.derangement(n, 0)
        assert sorted(p) == list(range(n)) and all(p[i] != i for i in range(n)) and p == L.derangement(n, 0)
    fam = L.family_derangement(PROMPTS, 0)
    assert all(fam[k] != k and PROMPTS[fam[k]]["family"] == PROMPTS[k]["family"] for k in range(600))
    with pytest.raises(ValueError):
        L.derangement(1)


def test_control_b_a_truth_against_another_prompt_and_random_cells():
    fam = L.family_derangement(PROMPTS, 0)
    for f in ("count", "colour", "relation", "combined"):
        idx = [k for k, p in enumerate(PROMPTS) if p["family"] == f]
        v = [L.score(PROMPTS[fam[k]], L.truth_reading(PROMPTS[k], "box"))["address_right"] for k in idx]
        assert sum(v) / len(v) < 0.2, f
    rel = [p for p in PROMPTS if p["family"] == "relation"]
    v = [L.score(p, L.random_reading(p, 1000 + k))["spatial"] for k, p in enumerate(rel)]
    assert 0.2 <= sum(v) / len(v) <= 0.6
    # PLACE asks two placements (each 1/3 at random): the random reading reads about 1/9, under the registered band
    place = [p for p in PROMPTS if p["family"] == "place"]
    v = [L.score(p, L.random_reading(p, 1000 + k))["spatial"] for k, p in enumerate(place)]
    assert abs(sum(v) / len(v) - 1 / 9) < 0.08
    assert 0.03 <= sum(v) / len(v) <= 0.22                  # the amended band of PLACE
    r = L.random_reading(rel[0], 5)
    assert r == L.random_reading(rel[0], 5) and len(r["instances"]) == 2


# ------------------------------------------------------------------ distinct texts, the whole-row form, copied boxes
def test_rates_count_each_distinct_text_once():
    rows = [L.score(p, L.truth_reading(p, "cell"), "cell") for p in PROMPTS]
    assert all(len(r["text_key"]) == 16 for r in rows)
    s = L.summarize(rows)
    fam = s["families"]
    assert fam["count"]["n"] == 50 and fam["count"]["prompts"] == 100 and fam["count"]["texts"] == 50
    assert fam["absent"]["n"] == 10 and fam["absent"]["scores"]["address_right"]["n"] == 10
    assert all(fam[f]["n"] == 100 for f in ("colour", "place", "relation", "combined"))
    assert s["all"]["n"] == 460 and s["all"]["duplicates_disagree"] == 0
    assert fam["count"]["per_instance_error"]["asked"] == 150          # 2 x (1 + .. + 5) x 10 / 2
    d = L.compare(rows, rows, "address_right", "count")
    assert d["n"] == 50 and d["tie"] is True


def test_copies_scored_differently_are_all_kept():
    p = PROMPTS[0]
    q = PROMPTS[1]
    assert p["text"] == q["text"]
    good = L.score(p, L.truth_reading(p, "cell"))
    bad = L.score(q, L.read("cell", "nonsense"))
    s = L.summarize([good, bad])
    assert s["families"]["count"]["n"] == 2 and s["families"]["count"]["duplicates_disagree"] == 1
    assert s["families"]["count"]["scores"]["address_right"]["mean"] == 0.5
    other = [dict(good), dict(L.score(q, L.truth_reading(q, "cell")))]
    d = L.compare([good, bad], other, "address_right", "count")
    assert d["n"] == 2 and d["mean"] == -0.5                         # paired by prompt id


def test_question_lines_and_the_whole_row_reader():
    text = ("see: two red cups on a table and a dog\nplace: the dog is at the bottom right\ncaption: cups and a dog.\n"
            "question: thing name -> cup [record]\nquestion: thing count -> 2 [record]\n"
            "question: thing colour -> red [record]\nquestion: thing position -> left [record]\n"
            "question: rel1 a -> cup [record]\nquestion: rel1 rel -> left of [record]\nquestion: rel1 b -> dog [record]\n"
            "question: what time is it -> unknown [unknown]")
    rec = L.question_records(text)
    assert rec == {"thing": {"name": "cup", "count": "2", "colour": "red", "position": "left"},
                   "relation": [{"a": "cup", "rel": "left of", "b": "dog"}]}
    r = L.read("chain_full", text)
    assert r["parsed"] and r["form"] == "chain_full"
    cups = [i for i in r["instances"] if i["thing"] == "cup"]
    assert len(cups) == 2 and cups[0]["colours"] == ["red"] and cups[0]["col"] is None    # a column, no row: not placed
    dog = [i for i in r["instances"] if i["thing"] == "dog"]
    assert len(dog) == 1 and (dog[0]["col"], dog[0]["row"]) == (2, 2)
    assert r["relations"] == [{"a": "cup", "rel": "left of", "b": "dog"}]
    placed = L.read("chain_full", "see: a cat\nplace: the cat is at the top\nquestion: thing name -> cat [record]\n"
                                  "question: thing position -> right [record]\nquestion: thing count -> unknown [chain]")
    assert [(i["thing"], i["col"], i["row"]) for i in placed["instances"]] == [("cat", 2, 0)]     # column + chain row
    only_q = L.read("chain_full", "question: thing name -> cat [record]")
    assert not only_q["parsed"] and only_q["failure"] == "no step line"
    extra = L.read("chain_full", "see: a scene\nquestion: thing name -> teapot [record]")
    assert extra["instances"][0]["thing"] is None and extra["instances"][0]["label"] == "teapot"


def test_the_whole_row_form_written_by_code():
    for p in PROMPTS[::37] + EXAMPLES:
        r = L.read("chain_full", L.render("chain_full", p["truth"]))
        s = L.score(p, r)
        assert r["parsed"] and s["things"] in (None, 1) and s["counts"] in (None, 1), p["id"]
    t = L.render("chain_full", EXAMPLES[4]["truth"])
    assert "question: rel1 a -> suitcase [record]" in t and "question: rel1 rel -> above [record]" in t


def test_copied_boxes():
    ex = L.read("box", L.render("box", EXAMPLES[0]["truth"]))
    assert L.copied(ex) == 1 and L.score(EXAMPLES[0], ex)["copied"] == 1
    own = L.read("box", '[{"bbox_2d": [10, 10, 60, 60], "label": "cup"}]')
    assert L.copied(own) == 0
    mixed = L.read("box", json.dumps([{"bbox_2d": [110, 420, 270, 580], "label": "backpack"},
                                      {"bbox_2d": [10, 10, 60, 60], "label": "cup"}]))
    assert L.copied(mixed) == 0
    assert L.copied(L.read("box", "[]")) == 0 and L.copied(L.read("box", "x")) == 0
    assert L.copied(L.read("cell", "cup | none | left | top")) is None
    rows = [L.score(PROMPTS[0], ex, "box"), L.score(PROMPTS[2], own, "box")]
    assert L.summarize(rows)["families"]["count"]["copied"]["mean"] == 0.5


# ------------------------------------------------------------------ summaries
def test_summarize_and_compare_on_hand_made_rows():
    rows = [{"id": "a", "family": "count", "parsed": 1, "things": 1, "counts": 1, "colours": None, "placed": 1,
             "spatial": None, "no_conflict": None, "absent": None, "address_right": 1, "kinds": [], "instance_errors": 0,
             "instances_asked": 2, "published": {"lmd": 1, "margin": None, "sectors": None}},
            {"id": "b", "family": "count", "parsed": 0, "things": 0, "counts": 0, "colours": None, "placed": 0,
             "spatial": None, "no_conflict": None, "absent": None, "address_right": 0, "kinds": ["broken format"],
             "instance_errors": 3, "instances_asked": 3, "lenient_address_right": 1,
             "published": {"lmd": 0, "margin": None, "sectors": None}},
            {"id": "c", "family": "combined", "parsed": 1, "things": 1, "counts": 1, "colours": 1, "placed": 1,
             "spatial": 0, "no_conflict": 1, "absent": 1, "address_right": 0, "kinds": ["spatial broken"],
             "instance_errors": 1, "instances_asked": 4},
            {"id": "r", "family": "real", "parsed": 1, "placed": 0, "no_conflict": 1, "written_things_in_caption": 1}]
    s = L.summarize(rows)
    c = s["families"]["count"]["scores"]["address_right"]
    assert c["mean"] == 0.5 and c["n"] == 2 and math.isclose(c["se"], math.sqrt(0.25 / 2))
    assert s["all"]["n"] == 3 and s["all"]["per_instance_error"] == {"errors": 4, "asked": 9, "rate": 4 / 9}
    assert s["all"]["kinds"] == {"broken format": 1, "spatial broken": 1} and s["all"]["lenient_recovered"] == 1
    assert s["load_curve"] == [{"mean": 0.0, "se": 0.0, "n": 1, "instances": 4}]
    assert s["published"]["lmd_four_families"]["mean"] == 0.5 and s["real"]["placed"]["mean"] == 0.0
    other = [dict(r, address_right=1) for r in rows[:3]]
    d = L.compare(rows, other, "address_right")
    assert d["n"] == 3 and math.isclose(d["mean"], -2 / 3) and math.isclose(d["se"], math.sqrt(1 / 3) / math.sqrt(3))
    assert d["tie"] is (abs(d["mean"]) < 2 * d["se"])
    many = [dict(rows[0], id=f"m{k}") for k in range(20)]
    assert L.compare(many, [dict(r, address_right=0) for r in many])["tie"] is False      # 1 - 0 every time: se 0
    assert L.compare(rows, rows)["tie"] is True and L.compare(rows, rows)["mean"] == 0
    assert L.compare(rows, other, "address_right", "combined")["n"] == 1
    assert L.compare(rows, [], "address_right")["n"] == 0
