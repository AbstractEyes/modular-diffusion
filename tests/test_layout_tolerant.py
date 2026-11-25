"""Layouts: the nine-example set (its three added examples letter for letter, their layouts scoring 1), the asks of the
sets 0 and 1 unchanged (their sha256 equal to the hashes stored with a finished run's answers), the table-tolerant read
of the cell form on dressed and plain answers, broken rows, an empty answer and real answers of a 2B model, its control,
and the experiment end to end with the nine examples."""
import hashlib
import os

import pytest

from geolip.modular_diffusion import layout as L
from geolip.modular_diffusion import run
from geolip.modular_diffusion.experiments import layout_write as W

PROMPTS = L.make_prompts(0)
BY_ID = {p["id"]: p for p in PROMPTS}

# the sha256 of the full cell-form ask of these prompts, as stored with the answers of a finished run of version 0.11.0
# (the cell form with the six worked examples, and with none)
ASK_SHA_SIX = {
    "count_000": "d54f98306317be4a2831bc3fe0f6c93beaa933a34fc1518c8a4bc40363005360",
    "count_017": "fc5d3a82e4e0f1d313e88f3c4f6a3bf55ef55cdb90927bfeb7cc1fc86d6daaa5",
    "place_003": "d0d188f71833d5921cd7d0a93c533ee96b88eba30085b4308daf65557f801979",
    "place_050": "7790bda439816400112b7231870efe307ed351d7986d2b8d0feb28556a42d808",
    "combined_000": "d8a943ec673bb22b12c00cdcfca587665ea6f2f1747d49d569150bb4f9ef6266",
    "combined_077": "60e7b0e694916b03f3922d3f8197c1051de5b4deb15cae5ab355b9490c0bd7de",
}
ASK_SHA_NONE = {
    "count_000": "e90b488603f376729b6158f85fdaa3240d34fad2167171256056bb08e04e1954",
    "count_017": "0aa67d66a4d3c65f12d821bdaf841b5ccac550138b50f6d1f3e7f428a5955595",
    "place_003": "9cab0a076421119772be03d792979124a8fce451be1fc97275cf574dbb7ecdf2",
    "place_050": "d156fadfcb4bc60c367777c3af6cb1bbfd95b77ab158a63255f737734bac83b9",
    "combined_000": "062a2e491963c6d2e333c7ee7b5eff9300de8477c88cc762f4969599251c409e",
    "combined_077": "f97d82e27647250f53a2778376c92c84a08a8098ec5ead11b9f0d810a34a95b3",
}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ the example sets
def test_the_asks_of_the_six_and_of_none_are_unchanged():
    for pid, sha in ASK_SHA_SIX.items():
        assert _sha(L.ask("cell", BY_ID[pid]["text"], True)) == sha, pid
        assert _sha(L.ask("cell", BY_ID[pid]["text"], 1)) == sha, pid
    for pid, sha in ASK_SHA_NONE.items():
        assert _sha(L.ask("cell", BY_ID[pid]["text"], False)) == sha, pid
        assert _sha(L.ask("cell", BY_ID[pid]["text"], 0)) == sha, pid
    assert L.example_prompts() == L.example_prompts(1) == L.example_prompts(True) and len(L.example_prompts()) == 6
    assert L.example_prompts(0) == []


def test_the_three_added_examples_letter_for_letter():
    nine = L.example_prompts(2)
    assert len(nine) == 9 and nine[:6] == L.example_prompts(1)
    assert [p["id"] for p in nine[6:]] == ["example_7", "example_8", "example_9"]
    assert all(p["family"] == "combined" for p in nine[6:])
    texts = [p["text"] for p in nine[6:]]
    assert texts == [
        "A realistic photo of a blue vase, two umbrellas and a bench; the vase is above the bench and the umbrellas are on "
        "the right.",
        "A realistic photo of two suitcases and a yellow banana; the banana is to the left of the suitcases.",
        "A realistic photo of two black sheep, two red kites and a clock; the kites are on the top and the sheep are below "
        "the clock."]
    answers = [L.render("cell", p["truth"]) for p in nine[6:]]
    assert answers == [
        "vase | blue | centre | top\numbrella | none | right | top\numbrella | none | right | middle\n"
        "bench | none | centre | bottom",
        "suitcase | none | right | top\nsuitcase | none | right | bottom\nbanana | yellow | left | middle",
        "sheep | black | left | bottom\nsheep | black | right | bottom\nkite | red | left | top\nkite | red | right | top\n"
        "clock | none | centre | top"]
    things = {a["thing"] for p in nine for a in p["asks"]} | {"umbrella"}
    assert things == set(L.THINGS_EXAMPLES)                   # no new thing; none occurs in a test prompt
    assert nine[6]["asks"] == [{"thing": "vase", "count": 1, "colour": "blue"},
                               {"thing": "umbrella", "count": 2, "colour": None},
                               {"thing": "bench", "count": 1, "colour": None}]


@pytest.mark.parametrize("form", ["box", "cell"])
def test_the_nine_examples_layouts_score_1(form):
    for p in L.example_prompts(2):
        s = L.score(p, L.read(form, L.render(form, p["truth"])), form)
        assert s["address_right"] == 1 and s["instance_errors"] == 0 and s["kinds"] == [], (p["id"], s)
        if p["family"] == "combined":
            assert s["spatial"] == 1 and s["no_conflict"] == 1 and s["colours"] == 1


def test_the_nine_example_ask():
    text = BY_ID["combined_000"]["text"]
    six, nine = L.ask("cell", text, 1), L.ask("cell", text, 2)
    assert nine.count("Description:") == 10 and six.count("Description:") == 7
    head = six[:six.index("\n\nDescription: " + text)]
    assert nine.startswith(head + "\n\nDescription: A realistic photo of a blue vase, two umbrellas and a bench;")
    assert nine.endswith("clock | none | centre | top\n\nDescription: " + text + "\nAnswer:")
    with pytest.raises(ValueError):
        L.ask("cell", text, 3)
    with pytest.raises(ValueError):
        L.example_prompts(-1)
    assert W.cell_name("base4b", "cell", 2) == "base4b_cell_ex2" and W.cell_name("base", "cell", True) == "base_cell_ex1"


# ------------------------------------------------------------------ the table-tolerant read
def test_untable_takes_off_the_dressing_and_nothing_else():
    text, n = L.untable("```markdown\n| name | Color | column | row |\n| :--- | --- | ---: | --- |\n| cup | red | left | top |\n```")
    assert text == " cup | red | left | top " and n == {"fence": 2, "bars": 3, "header": 1, "separator": 1}
    plain = "cup | red | left | top\nchair | none | right | bottom"
    assert L.untable(plain) == (plain, {"fence": 0, "bars": 0, "header": 0, "separator": 0})
    assert L.untable("thing | colour | col")[1]["header"] == 1          # three fields of the header words
    assert L.untable("name | color")[1]["header"] == 0                  # two fields: not a header
    assert L.untable("name | color | place | row")[1]["header"] == 0    # a word that is no header word
    assert L.untable("---")[1]["separator"] == 1 and L.untable("| : |")[1]["separator"] == 0      # a dash is needed
    assert L.untable("| cup | red | left | top")[1]["bars"] == 0       # one bar only: left as it is
    # a fence: three or more backticks, then an optional info string without a backtick; white space around ignored
    for fence in ("```", "```markdown", "``` text", "```c++", "```plain-text", "````", "  ```json  "):
        assert L.untable(fence) == ("", {"fence": 1, "bars": 0, "header": 0, "separator": 0}), fence
    for line in ("cup `red` | none | left | top", "``", "``x``", "```x`", "a ```", "| ``` |"):
        assert L.untable(line)[1]["fence"] == 0, line                    # not a line that is only a fence
    text, n = L.untable("```plain-text\ncup | red | left | top\n```")
    assert text == "cup | red | left | top" and n["fence"] == 2


def test_the_tolerant_read_on_dressed_and_plain_answers():
    p = BY_ID["combined_000"]
    plain = L.render("cell", p["truth"])
    assert L.table_score(p, plain)["address_right"] == 1
    dressed = L.dress_table(plain)
    assert not L.read("cell", dressed)["parsed"]                        # the strict grammar refuses the table
    t = L.table_score(p, dressed)
    assert t["address_right"] == 1 and t["parsed"] == 1 and t["drops"]["header"] == 1 and t["drops"]["fence"] == 2
    none = L.table_score(BY_ID["absent_000"], L.dress_table("none"))
    assert none["address_right"] == 1                                    # '| none |' read as none
    assert L.score(BY_ID["absent_000"], L.read("cell", L.dress_table("none")), "cell")["address_right"] == 0
    with pytest.raises(ValueError):
        W.FakeWriter("table", "box", {})                                 # the dressed writer writes the cell form only


def test_the_tolerant_read_keeps_broken_rows_broken():
    p = BY_ID["count_000"]
    D = lambda fence=0, bars=0, header=0, separator=0: {"fence": fence, "bars": bars, "header": header,
                                                        "separator": separator}
    # (answer, the tolerant read's failure, the strict read's failure, what the dressing steps took): each case with a
    # step taking a line shows that the dressing came off and the broken row under it stayed broken
    cases = (
        ("```c++\n| name | color | column | row |\n|---|---|---|---|\n| backpack | none | left |\n```",
         "wrong field count", "wrong field count", D(2, 3, 1, 1)),
        ("| name | color | column | row |\n|  | none | left | top |", "empty label", "wrong field count", D(0, 2, 1, 0)),
        ("```\n| name | color | column | row |\n|---|---|---|---|\n```", "empty answer", "wrong field count",
         D(2, 2, 1, 1)),
        ("| row | col | name |", "empty answer", "wrong field count", D(0, 1, 1, 0)),
        # plain-read cases (no dressing to take off): the tolerant read must leave them as the strict read does
        ("Here you go:\nbackpack | none | left | top", "wrong field count", "wrong field count", D()),
        ("", "empty answer", "empty answer", D()),
        ("   \n\n", "empty answer", "empty answer", D()),
    )
    for answer, failure, strict_failure, drops in cases:
        r = L.read_table(answer)
        assert not r["parsed"] and r["failure"] == failure, (answer, r["failure"])
        assert r["table_drops"] == drops, (answer, r["table_drops"])
        assert L.read("cell", answer)["failure"] == strict_failure, answer
        s = L.table_score(p, answer)
        assert s["parsed"] == 0 and s["address_right"] == 0 and s["kinds"] == ["broken format"]
    far = L.table_score(p, "| backpack | none | far | top |")
    assert far["parsed"] == 1 and far["placed"] == 0 and "no place word" in far["kinds"]


def test_the_tolerant_read_on_real_answers_of_a_2b_model():
    # answers stored with a finished run (the cell form, a 2B model), with the prompts they answered
    one = BY_ID["count_000"]
    assert one["text"] == "A realistic photo of a scene with one backpack"
    ans = "name | color | column | row\n--- | --- | --- | ---\nbackpack | none | left | top"
    assert L.score(one, L.read("cell", ans), "cell")["address_right"] == 0        # the strict read: two extra things
    assert L.table_score(one, ans)["address_right"] == 1
    bottle = BY_ID["count_020"]
    assert bottle["text"] == "A realistic photo of a scene with one bottle"
    ans = "| bottle | clear | left | top |"
    assert L.score(bottle, L.read("cell", ans), "cell")["parsed"] == 0
    assert L.table_score(bottle, ans)["address_right"] == 1
    two = BY_ID["count_002"]
    ans = "name | color | column | row\nname | none | left | middle\nname | none | right | middle"
    t = L.table_score(two, ans)                                          # the header goes; rows named 'name' stay
    assert t["parsed"] == 1 and t["things"] == 0 and t["address_right"] == 0 and t["drops"]["header"] == 1
    cars = BY_ID["combined_081"]
    ans = ("| car | purple | left | top |\n|---|---|---|---|\n| car | purple | left | top |\n| truck | yellow | left | bottom |\n"
           "| truck | yellow | left | bottom |\n| truck | yellow | left | bottom |\n| truck | yellow | left | bottom |")
    t = L.table_score(cars, ans)
    assert t["parsed"] == 1 and t["counts"] == 0 and t["drops"] == {"fence": 0, "bars": 7, "header": 0, "separator": 1}


def test_rows_summaries_and_comparisons_under_the_tolerant_read():
    rows = []
    for p in PROMPTS[:12]:
        r = L.score(p, L.read("cell", L.dress_table(L.render("cell", p["truth"]))), "cell")
        r["table"] = L.table_score(p, L.dress_table(L.render("cell", p["truth"])))
        rows.append(r)
    s = L.summarize(rows)
    assert s["all"]["scores"]["address_right"]["mean"] == 0.0
    assert s["table"]["all"]["scores"]["address_right"]["mean"] == 1.0 and s["table"]["table"] is None
    assert s["table"]["answers_with"]["header"] == 12 and s["table"]["answers"] == 12
    lifted = L.table_rows(rows)
    assert len(lifted) == 12 and all("table" not in x and x["address_right"] == 1 for x in lifted)
    assert L.compare(lifted, rows, "address_right")["mean"] == 1.0
    plain = [L.score(p, L.truth_reading(p, "box"), "box") for p in PROMPTS[:12]]
    assert L.summarize(plain)["table"] is None and L.table_rows(plain) == []


# ------------------------------------------------------------------ control (e)
def test_control_e_holds_on_the_generators_layouts_and_can_fail(monkeypatch):
    every = PROMPTS + L.example_prompts(2)
    e = L.tolerant_check(every)
    assert e["holds"] is True
    assert e["strict_equal"] == {"prompts": 609, "differ": 0, "ids": [], "holds": True}
    assert e["dressed"]["strict_unparsed"] == 609 and e["dressed"]["tolerant_not_right"] == 0
    monkeypatch.setattr(L, "read_table", lambda answer: L.read("cell", answer))      # no dressing taken off
    bad = L.tolerant_check(every[:30])
    assert bad["holds"] is False and bad["dressed"]["tolerant_not_right"] == 30 and bad["strict_equal"]["holds"]


# ------------------------------------------------------------------ the experiment with the nine examples
def _cell(tmp_path, name, **kw):
    out = str(tmp_path / name)
    kw.setdefault("first_n", 4)
    kw.setdefault("repeat_n", 4)
    return run.main("layout_write", [f"out={out}"], variant="write", **kw), out


def test_a_cell_with_the_nine_examples_end_to_end(tmp_path):
    r, out = _cell(tmp_path, "truth_cell_ex2", writer="fake:truth", form="cell", examples=2)
    assert r["examples"] is True and r["example_set"] == 2 and len(r["asks"]["examples"]) == 9
    c = r["controls"]
    assert c["a"]["prompts"] == 609 and c["a"]["holds"] is True and c["a"]["writer_answers_failed"] == 0
    assert c["e"]["holds"] is True and c["e"]["strict_equal"]["prompts"] == 609 and c["e"]["answers_changed"] == 0
    rows = W.read_jsonl(os.path.join(out, "rows.jsonl"))
    assert all(isinstance(x["table"], dict) and x["table"]["address_right"] == x["address_right"] for x in rows)
    # the dressed writer: every answer its true layout inside a Markdown table. The strict read refuses every answer
    # (absent's '| none |' too: the fence and the header have the wrong field count), the tolerant read takes them all
    d, dout = _cell(tmp_path, "table_cell_ex2", writer="fake:table", form="cell", examples=2)
    for f in L.FAMILIES:
        assert d["summary"]["families"][f]["scores"]["address_right"]["mean"] == 0.0, f
        assert d["summary"]["families"][f]["scores"]["parsed"]["mean"] == 0.0, f
        assert d["summary"]["table"]["families"][f]["scores"]["address_right"]["mean"] == 1.0, f
    drows = W.read_jsonl(os.path.join(dout, "rows.jsonl"))
    assert all(x["address_right"] == 0 and x["table"]["address_right"] == 1 for x in drows)
    assert d["controls"]["e"]["holds"] is True and d["controls"]["e"]["answers_changed"] == 24
    assert d["summary"]["table"]["answers_with"] == {"fence": 24, "bars": 24, "header": 24, "separator": 24}
    outputs = W.read_jsonl(os.path.join(out, "outputs.jsonl"))
    first = outputs[0]
    assert first["ask_sha256"] == _sha(L.ask("cell", BY_ID[first["id"]]["text"], 2))
    md = open(os.path.join(out, "result.md"), encoding="utf-8").read()
    assert "table-tolerant read" in md and "| e |" in md
    six, _ = _cell(tmp_path, "truth_cell_ex1", writer="fake:truth", form="cell", examples=1)
    assert six["controls"]["a"]["prompts"] == 606 and six["controls"]["e"]["strict_equal"]["prompts"] == 606
    box, _ = _cell(tmp_path, "truth_box_ex1", writer="fake:truth", form="box", examples=1)
    assert box["summary"]["table"] is None and box["controls"]["e"]["holds"] is None
    with pytest.raises(ValueError):
        _cell(tmp_path, "bad", writer="fake:truth", form="cell", examples=3)


def test_the_comparison_carries_the_tolerant_read(tmp_path):
    _, a = _cell(tmp_path, "a", writer="fake:truth", form="cell", examples=2)
    _, b = _cell(tmp_path, "b", writer="fake:broken", form="cell", examples=2)
    r = run.main("layout_write", [f"out={tmp_path}/cmp"], variant="write_compare", a=a, b=b)
    assert r["table"]["address_right"]["all"]["mean"] == 1.0 and r["table"]["address_right"]["all"]["tie"] is False
    # the dressed writer against the plain true writer: 1 apart on every prompt by the strict read, 0 by the tolerant one
    _, t = _cell(tmp_path, "t", writer="fake:table", form="cell", examples=2)
    d = run.main("layout_write", [f"out={tmp_path}/cmp_table"], variant="write_compare", a=t, b=a)
    for f in list(L.FAMILIES) + ["all"]:
        one = f == "absent"            # one distinct text at 4 prompts a family: no standard error, so no tie is judged
        assert d["scores"]["address_right"][f]["mean"] == -1.0, f
        assert d["scores"]["address_right"][f]["tie"] is (None if one else False), f
        assert d["table"]["address_right"][f]["mean"] == 0.0, f
        assert d["table"]["address_right"][f]["tie"] is (None if one else True), f
    _, x = _cell(tmp_path, "x", writer="fake:truth", form="box", examples=1)
    r2 = run.main("layout_write", [f"out={tmp_path}/cmp2"], variant="write_compare", a=x, b=x)
    assert r2["table"] is None
    assert "table read" in W.summarize(str(tmp_path))
