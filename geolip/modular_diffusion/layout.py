"""Written layouts, their addresses and their scores: how often a writer (a language model, a parser, code) writes the
right set of addresses for a picture from a text prompt alone. Pure Python (the standard library only).

THE ADDRESS of a thing instance is one cell of a 3 x 3 grid. From a box (x1, y1, x2, y2) on the relative grid 0 to
1000 (the origin at the top left), the cell holding the box's center: the column is left when the center's x is under
0.38 of the width, right when it is above 0.62, else centre; the rows likewise on y (top, middle, bottom). From words,
the named cell. A box with a number outside 0 to 1000, or with x2 <= x1 or y2 <= y1, has no address.

THE PROMPTS (``make_prompts``): 600 prompts whose right answers are known by construction, 100 a family, drawn by
``random.Random(seed)`` only (never a set's order, never ``hash``), so the list is the same on every machine:

    count     "A realistic photo of a scene with three cats": ten things, every number 1 to 5, twice (two truths)
    absent    "A realistic photo of a scene without cats": ten a thing; the text is the same ten times and the copies
              are kept, so a deterministic writer answers the same ten times (as the published 400-prompt set does)
    colour    "... with a red cup and a blue chair": two different things, two different colors
    place     "... with a cup on the left and a chair on the right" (left / right or top / bottom)
    relation  "A realistic photo of a horse to the left of a bus" (to the left of, to the right of, above, below)
    combined  a layout drawn first (2 to 5 instances of two or three things, 25 prompts a size), then one or two facts
              that hold in it: "...; the kites are to the left of the sheep and the clock is on the top."

The first four families use ten things, the last two thirty. ``truth`` is one layout that satisfies the prompt; it
writes the worked examples and the scorer's own control, and answers are never scored by equality with it.

THE FORMS: "box" (a JSON list of {"bbox_2d": [x1, y1, x2, y2], "label": ...}, one entry a thing instance), "cell"
(one line a thing instance: name | color | column | row; the single line ``none`` when nothing is visible), "chain"
(a written chain of thought, ``step: text`` lines, read by a line reader) and "spec" (the records of a text parser).

THE READERS (``read``): strict grammars. box: after white space and ONE optional Markdown code fence are taken off,
the whole answer is a JSON list of objects with exactly the keys bbox_2d (four numbers) and label (a non-empty
string). cell: every non-empty line has exactly four fields separated by ``|``, or the answer is the single line
``none``; center and grey are read as centre and gray; a column or row that is no place word gives no address. The
lenient reads are kept beside and judge nothing: box, the first JSON list anywhere, entries with four numbers under
bbox_2d, bbox or box and a string under label or name; cell, every line with at least four fields, its first four.

THE THING OF A LABEL (``thing_of``): lower case, words of letters only; articles, color words and number words
dropped; a plural turned singular by a fixed table; the label names a thing when exactly one listed thing is among
its words; none, or two different ones, is an EXTRA thing.

THE SCORES (``score``), each 0 or 1 a prompt: parsed; things (the written things are exactly the asked ones); counts;
colours (every instance of a thing with a stated color carries it and no other); placed (every instance has an
address); spatial (every statement holds on the addresses: "on the left" = every instance in the left column; "A to
the left of B" = every instance of A in a column left of every instance of B; rows for above and below); no_conflict
(no cell holds instances of two different things); absent (no instance of the absent thing); address_right (the
product of the scores of the family). Families: count = parsed, things, counts, placed; absent = parsed, absent,
placed, no_conflict; colour = count's with colours and no_conflict; place and relation = colour's with spatial;
combined = all. A rate is a mean with the standard error sqrt(p (1 - p) / n) over the DISTINCT prompt texts of a family
(``distinct``: copies of one text scored alike count once, so count has 50 and absent 10; copies scored differently are
all kept and counted as disagreeing); two writers are compared text by text (``compare``): the mean difference, its
standard error sd / sqrt(n), and a tie when |mean| < 2 se.

THE WHOLE-ROW CHAIN FORM ("chain_full", ``chain_full_read``): the chain reader joined with the records stated by the
answer's ``question: <key> -> <value> [record|chain|unknown]`` lines. COPIED (``copied``, box readings only; it judges
nothing): every written box equals a box the box form writes for one of the six worked examples.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from typing import Dict, List, Optional, Sequence

# ------------------------------------------------------------------ the fixed lists
COLUMNS = ("left", "centre", "right")
ROWS = ("top", "middle", "bottom")
CUTS = (0.38, 0.62)
COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray")
COLOUR_SPELLINGS = {"grey": "gray"}
THINGS_SIMPLE = ("backpack", "book", "bottle", "bowl", "car", "cat", "chair", "cup", "dog", "laptop")
THINGS_HARD = THINGS_SIMPLE + ("apple", "bear", "bed", "bicycle", "bird", "boat", "bus", "cake", "carrot", "couch", "cow",
                               "elephant", "fork", "horse", "knife", "motorcycle", "pizza", "sandwich", "spoon", "truck")
THINGS_EXAMPLES = ("kite", "umbrella", "bench", "clock", "vase", "sheep", "suitcase", "banana")
THINGS = THINGS_HARD + THINGS_EXAMPLES
PLURALS = {"backpack": "backpacks", "book": "books", "bottle": "bottles", "bowl": "bowls", "car": "cars", "cat": "cats",
           "chair": "chairs", "cup": "cups", "dog": "dogs", "laptop": "laptops", "apple": "apples", "bear": "bears",
           "bed": "beds", "bicycle": "bicycles", "bird": "birds", "boat": "boats", "bus": "buses", "cake": "cakes",
           "carrot": "carrots", "couch": "couches", "cow": "cows", "elephant": "elephants", "fork": "forks",
           "horse": "horses", "knife": "knives", "motorcycle": "motorcycles", "pizza": "pizzas",
           "sandwich": "sandwiches", "spoon": "spoons", "truck": "trucks", "kite": "kites", "umbrella": "umbrellas",
           "bench": "benches", "clock": "clocks", "vase": "vases", "sheep": "sheep", "suitcase": "suitcases",
           "banana": "bananas"}
SINGULAR = {p: s for s, p in PLURALS.items()}
NUMBER_WORDS = ("one", "two", "three", "four", "five")
ARTICLES = ("a", "an", "the")
RELATIONS = ("left of", "right of", "above", "below")
RELATION_WORDS = {"left of": "to the left of", "right of": "to the right of", "above": "above", "below": "below"}
SIDES = {"left": ("col", 0), "right": ("col", 2), "top": ("row", 0), "bottom": ("row", 2)}
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
FAMILIES = ("count", "absent", "colour", "place", "relation", "combined")
PER_FAMILY = 100
COMBINED_SIZES = (2, 3, 4, 5)
CANVAS = 1000
BOX_SIDE = 160
BOX_SHIFT = 40
CELL_CENTERS = (190, 500, 810)
MAX_IN_CELL = 3                 # the box rule shifts a second and a third instance of one thing in one cell

SCORES = ("parsed", "things", "counts", "colours", "placed", "spatial", "no_conflict", "absent")
APPLIES = {"count": ("parsed", "things", "counts", "placed"),
           "absent": ("parsed", "absent", "placed", "no_conflict"),
           "colour": ("parsed", "things", "counts", "placed", "colours", "no_conflict"),
           "place": ("parsed", "things", "counts", "placed", "colours", "no_conflict", "spatial"),
           "relation": ("parsed", "things", "counts", "placed", "colours", "no_conflict", "spatial"),
           "combined": SCORES}
KINDS = ("broken format", "outside canvas", "no area", "no place word", "not placed", "extra thing", "missing thing",
         "wrong count", "wrong colour", "spatial broken", "shared cell", "absent written")
KIND_OF_SCORE = {"things": ("extra thing", "missing thing"), "counts": ("wrong count",), "colours": ("wrong colour",),
                 "placed": ("outside canvas", "no area", "no place word", "not placed"), "spatial": ("spatial broken",),
                 "no_conflict": ("shared cell",), "absent": ("absent written",), "parsed": ("broken format",)}

# the chain reader's words (the trained form's sixteen step names; its position and relation words)
CHAIN_STEPS = ("see", "name", "place", "count", "relate", "form", "light", "colour", "mood", "mean", "intend", "story",
               "compare", "change", "check", "say")
CHAIN_COUNT_STEPS = ("count", "see", "name")
CHAIN_COLUMN_WORDS = {"left": 0, "centre": 1, "center": 1, "right": 2}
CHAIN_ROW_WORDS = {"top": 0, "upper": 0, "bottom": 2, "lower": 2}
CHAIN_RELATIONS = ("left of", "right of", "above", "below", "behind", "in front of", "beside", "holding", "facing",
                   "looking at", "larger than", "smaller than", "same colour as", "touching", "inside", "under",
                   "leaning on", "reflected in")
CHAIN_COUNTS = {"no": 0, "zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NEGATIONS = ("no", "zero", "without")

# ------------------------------------------------------------------ the asks (data: the words given to a model)
ASK_BOX = ("You plan the layout of a picture. Read the description and list every thing that must be visible, one entry "
           "for each single thing (three cats are three entries). Give each a box.\nAnswer with a JSON list and nothing "
           "else. Each entry is {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"<the thing>\"}: x1, y1 is the box's top-left "
           "corner and x2, y2 its bottom-right corner, as whole numbers from 0 to 1000 (0, 0 is the picture's top-left "
           "corner; x grows to the right, y grows downward). Put the thing's color in the label when the description "
           "gives one (\"red cup\"). Follow every placement the description states. Boxes of different things must not "
           "share the same part of the picture. Do not list a thing the description says is absent. If nothing must be "
           "visible, answer [].")
ASK_CELL = ("You plan the layout of a picture. Read the description and list every thing that must be visible, one line "
            "for each single thing (three cats are three lines). Give each a place on a 3 by 3 grid.\nAnswer with these "
            "lines and nothing else. Each line is: name | color | column | row. name is the thing; color is its color "
            "when the description gives one, else none; column is left, centre or right; row is top, middle or bottom. "
            "Follow every placement the description states. Two different things must not have the same column and row. "
            "Do not list a thing the description says is absent. If nothing must be visible, answer with the single "
            "line: none")
ASK_CHAIN_TEXT = ("There is no picture to see. Here is a description of one:\n\n{prompt}\n\nTreat the description as what "
                  "the picture shows and think it through in writing, a closer look, ending with a caption.")
ASKS = {"box": ASK_BOX, "cell": ASK_CELL, "chain_text": ASK_CHAIN_TEXT}

# the six worked examples: (family, text, layout as (thing, colour, column, row))
EXAMPLES = (
    ("count", "A realistic photo of a scene with two kites", [("kite", None, 0, 1), ("kite", None, 2, 1)]),
    ("absent", "A realistic photo of a scene without umbrellas", []),
    ("colour", "A realistic photo of a scene with a red bench and a blue clock",
     [("bench", "red", 0, 2), ("clock", "blue", 2, 0)]),
    ("place", "A realistic photo of a scene with a vase on the left and a sheep on the right",
     [("vase", None, 0, 1), ("sheep", None, 2, 1)]),
    ("relation", "A realistic photo of a suitcase above a banana", [("suitcase", None, 1, 0), ("banana", None, 1, 2)]),
    ("combined", "A realistic photo of two green kites, a white sheep and a clock; the kites are to the left of the sheep "
     "and the clock is on the top.",
     [("kite", "green", 0, 1), ("kite", "green", 0, 2), ("sheep", "white", 2, 1), ("clock", None, 1, 0)]),
)


# ------------------------------------------------------------------ words
def article(word: str) -> str:
    """'an' before a word whose first letter is a vowel, else 'a'."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def plural(thing: str) -> str:
    return PLURALS[thing]


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _colour_word(w: str) -> Optional[str]:
    w = COLOUR_SPELLINGS.get(w, w)
    return w if w in COLOURS else None


def _listed(w: str) -> Optional[str]:
    """The listed thing a word names (singular or plural), or None."""
    if w in PLURALS:
        return w
    return SINGULAR.get(w)


def thing_of(label: str) -> Optional[str]:
    """The listed thing a label names, or None (no listed thing, or two different ones: an extra thing)."""
    dropped = set(ARTICLES) | set(COLOURS) | set(COLOUR_SPELLINGS) | set(NUMBER_WORDS)
    found: List[str] = []
    for w in _words(label):
        if w in dropped:
            continue
        t = _listed(w)
        if t is not None and t not in found:
            found.append(t)
    return found[0] if len(found) == 1 else None


def colours_of(text: str) -> List[str]:
    """The color words of a text, as a sorted list without repeats (grey read as gray)."""
    out = sorted({c for c in (_colour_word(w) for w in _words(text)) if c})
    return out


# ------------------------------------------------------------------ the prompts
def _prompt(pid: str, family: str, text: str, asks, places=(), relations=(), absent=(), truth=()) -> dict:
    return {"id": pid, "family": family, "text": text, "asks": [dict(a) for a in asks], "places": [dict(p) for p in places],
            "relations": [dict(r) for r in relations], "absent": list(absent), "truth": [dict(t) for t in truth]}


def _inst(thing, colour, col, row) -> dict:
    return {"thing": thing, "colour": colour, "col": int(col), "row": int(row)}


def _draw_cells(rng: random.Random, n: int, allowed: Sequence[int], used: Optional[Dict[int, int]] = None) -> List[int]:
    """n cells (0..8, col = c % 3, row = c // 3) drawn from ``allowed``, at most MAX_IN_CELL a cell."""
    used = {} if used is None else used
    cells = []
    for _ in range(n):
        free = [c for c in allowed if used.get(c, 0) < MAX_IN_CELL]
        c = free[rng.randrange(len(free))]
        used[c] = used.get(c, 0) + 1
        cells.append(c)
    return cells


def _count_text(n: int, thing: str) -> str:
    return f"A realistic photo of a scene with {NUMBER_WORDS[n - 1]} {thing if n == 1 else plural(thing)}"


def _count_prompts(rng: random.Random) -> List[dict]:
    out = []
    for thing in THINGS_SIMPLE:
        for n in range(1, 6):
            seen: List[List[int]] = []
            for _ in range(2):
                while True:
                    cells = sorted(_draw_cells(rng, n, range(9)))
                    if cells not in seen:
                        break
                seen.append(cells)
                truth = [_inst(thing, None, c % 3, c // 3) for c in cells]
                out.append(("count", _count_text(n, thing), [{"thing": thing, "count": n, "colour": None}], [], [], [],
                            truth))
    return out


def _absent_prompts(rng: random.Random) -> List[dict]:
    out = []
    for thing in THINGS_SIMPLE:
        for _ in range(10):
            out.append(("absent", f"A realistic photo of a scene without {plural(thing)}", [], [], [], [thing], []))
    return out


def _pair(rng: random.Random, things: Sequence[str]):
    a = things[rng.randrange(len(things))]
    b = things[rng.randrange(len(things))]
    while b == a:
        b = things[rng.randrange(len(things))]
    return a, b


def _colour_prompts(rng: random.Random) -> List[tuple]:
    out, texts = [], []
    while len(out) < PER_FAMILY:
        a, b = _pair(rng, THINGS_SIMPLE)
        ca, cb = _pair(rng, COLOURS)
        text = f"A realistic photo of a scene with {article(ca)} {ca} {a} and {article(cb)} {cb} {b}"
        if text in texts:
            continue
        texts.append(text)
        cells = rng.sample(range(9), 2)
        truth = [_inst(a, ca, cells[0] % 3, cells[0] // 3), _inst(b, cb, cells[1] % 3, cells[1] // 3)]
        out.append(("colour", text, [{"thing": a, "count": 1, "colour": ca}, {"thing": b, "count": 1, "colour": cb}],
                    [], [], [], truth))
    return out


def _on_side(rng: random.Random, side: str):
    axis, value = SIDES[side]
    other = rng.randrange(3)
    return (value, other) if axis == "col" else (other, value)


def _place_prompts(rng: random.Random) -> List[tuple]:
    out, texts = [], []
    sides = ("left", "right", "top", "bottom")
    while len(out) < PER_FAMILY:
        a, b = _pair(rng, THINGS_SIMPLE)
        side = sides[rng.randrange(4)]
        opp = OPPOSITE[side]
        text = f"A realistic photo of a scene with {article(a)} {a} on the {side} and {article(b)} {b} on the {opp}"
        if text in texts:
            continue
        texts.append(text)
        (ca, ra), (cb, rb) = _on_side(rng, side), _on_side(rng, opp)
        places = [{"thing": a, "axis": SIDES[side][0], "value": SIDES[side][1]},
                  {"thing": b, "axis": SIDES[opp][0], "value": SIDES[opp][1]}]
        out.append(("place", text, [{"thing": a, "count": 1, "colour": None}, {"thing": b, "count": 1, "colour": None}],
                    places, [], [], [_inst(a, None, ca, ra), _inst(b, None, cb, rb)]))
    return out


def _relation_holds_cells(rel: str, ca: int, ra: int, cb: int, rb: int) -> bool:
    return {"left of": ca < cb, "right of": ca > cb, "above": ra < rb, "below": ra > rb}[rel]


def _relation_prompts(rng: random.Random) -> List[tuple]:
    out, texts = [], []
    while len(out) < PER_FAMILY:
        a, b = _pair(rng, THINGS_HARD)
        rel = RELATIONS[rng.randrange(4)]
        text = f"A realistic photo of {article(a)} {a} {RELATION_WORDS[rel]} {article(b)} {b}"
        if text in texts:
            continue
        texts.append(text)
        while True:
            ca, ra, cb, rb = (rng.randrange(3) for _ in range(4))
            if _relation_holds_cells(rel, ca, ra, cb, rb):
                break
        out.append(("relation", text, [{"thing": a, "count": 1, "colour": None}, {"thing": b, "count": 1, "colour": None}],
                    [], [{"a": a, "rel": rel, "b": b}], [], [_inst(a, None, ca, ra), _inst(b, None, cb, rb)]))
    return out


def _facts_of(things: Sequence[str], truth: Sequence[dict]) -> List[tuple]:
    """Every checkable fact that holds in a layout: ("side", thing, side) and ("rel", a, rel, b), in a fixed order."""
    by = {t: [i for i in truth if i["thing"] == t] for t in things}
    facts = []
    for t in things:
        for side in ("left", "right", "top", "bottom"):
            axis, value = SIDES[side]
            if all(i[axis] == value for i in by[t]):
                facts.append(("side", t, side))
    for a in things:
        for b in things:
            if a == b:
                continue
            for rel in RELATIONS:
                if all(_relation_holds_cells(rel, i["col"], i["row"], j["col"], j["row"]) for i in by[a] for j in by[b]):
                    facts.append(("rel", a, rel, b))
    return facts


def _is_are(n: int) -> str:
    return "is" if n == 1 else "are"


def _named(thing: str, n: int) -> str:
    return thing if n == 1 else plural(thing)


def _combined_prompt(rng: random.Random, size: int) -> tuple:
    while True:
        kinds = 2 if size == 2 else (2, 3)[rng.randrange(2)]
        things = rng.sample(THINGS_HARD, kinds)
        counts = [1] * kinds
        for _ in range(size - kinds):
            counts[rng.randrange(kinds)] += 1
        coloured = [rng.random() < 0.5 for _ in things]
        palette = rng.sample(COLOURS, kinds)
        colours = [palette[k] if coloured[k] else None for k in range(kinds)]
        owner: Dict[int, str] = {}
        used: Dict[int, int] = {}
        truth = []
        ok = True
        for k, t in enumerate(things):
            allowed = [c for c in range(9) if owner.get(c, t) == t]
            if sum(MAX_IN_CELL - used.get(c, 0) for c in allowed) < counts[k]:
                ok = False
                break
            for c in _draw_cells(rng, counts[k], allowed, used):
                owner[c] = t
                truth.append(_inst(t, colours[k], c % 3, c // 3))
        if not ok:
            continue
        facts = _facts_of(things, truth)
        if not facts:
            continue
        m = min(len(facts), (1, 2)[rng.randrange(2)])
        chosen = rng.sample(facts, m)
        n_of = dict(zip(things, counts))
        items = []
        for t, n, c in zip(things, counts, colours):
            head = f"{c} {_named(t, n)}" if c else _named(t, n)
            items.append(f"{article(head)} {head}" if n == 1 else f"{NUMBER_WORDS[n - 1]} {head}")
        listed = ", ".join(items[:-1]) + " and " + items[-1]
        said, places, relations = [], [], []
        for f in chosen:
            if f[0] == "side":
                _, t, side = f
                said.append(f"the {_named(t, n_of[t])} {_is_are(n_of[t])} on the {side}")
                places.append({"thing": t, "axis": SIDES[side][0], "value": SIDES[side][1]})
            else:
                _, a, rel, b = f
                said.append(f"the {_named(a, n_of[a])} {_is_are(n_of[a])} {RELATION_WORDS[rel]} the {_named(b, n_of[b])}")
                relations.append({"a": a, "rel": rel, "b": b})
        text = f"A realistic photo of {listed}; {' and '.join(said)}."
        asks = [{"thing": t, "count": n, "colour": c} for t, n, c in zip(things, counts, colours)]
        return ("combined", text, asks, places, relations, [], truth)


def _combined_prompts(rng: random.Random) -> List[tuple]:
    return [_combined_prompt(rng, size) for size in COMBINED_SIZES for _ in range(PER_FAMILY // len(COMBINED_SIZES))]


def make_prompts(seed: int = 0) -> List[dict]:
    """The 600 prompts with known answers, 100 a family, in the order of FAMILIES; ids '<family>_<3 digits>'."""
    rng = random.Random(int(seed))
    out = []
    for family, maker in (("count", _count_prompts), ("absent", _absent_prompts), ("colour", _colour_prompts),
                          ("place", _place_prompts), ("relation", _relation_prompts), ("combined", _combined_prompts)):
        made = maker(rng)
        assert len(made) == PER_FAMILY, (family, len(made))
        for i, (fam, text, asks, places, relations, absent, truth) in enumerate(made):
            out.append(_prompt(f"{family}_{i:03d}", fam, text, asks, places, relations, absent, truth))
    return out


def example_prompts() -> List[dict]:
    """The six worked examples (their things occur in no test prompt), with their layouts as ``truth``."""
    out = []
    for k, (family, text, layout) in enumerate(EXAMPLES, start=1):
        truth = [_inst(t, c, col, row) for t, c, col, row in layout]
        asks, places, relations, absent = [], [], [], []
        for t, c, _, _ in layout:
            for a in asks:
                if a["thing"] == t:
                    a["count"] += 1
                    break
            else:
                asks.append({"thing": t, "count": 1, "colour": c})
        if family == "absent":
            absent = ["umbrella"]
        elif family == "place":
            places = [{"thing": "vase", "axis": "col", "value": 0}, {"thing": "sheep", "axis": "col", "value": 2}]
        elif family == "relation":
            relations = [{"a": "suitcase", "rel": "above", "b": "banana"}]
        elif family == "combined":
            relations = [{"a": "kite", "rel": "left of", "b": "sheep"}]
            places = [{"thing": "clock", "axis": "row", "value": 0}]
        out.append(_prompt(f"example_{k}", family, text, asks, places, relations, absent, truth))
    return out


# ------------------------------------------------------------------ writing a layout by code
def label_of(inst: dict) -> str:
    return f"{inst['colour']} {inst['thing']}" if inst.get("colour") else inst["thing"]


def boxes_of(instances: Sequence[dict]) -> List[List[int]]:
    """The box rule: centered on the cell's center, BOX_SIDE wide and high; the second and third instance of one thing in
    one cell shifted by BOX_SHIFT on both axes, so no two boxes are equal."""
    seen: Dict[tuple, int] = {}
    out = []
    half = BOX_SIDE // 2
    for i in instances:
        key = (i["thing"], i["col"], i["row"])
        k = seen.get(key, 0)
        if k >= MAX_IN_CELL:
            raise ValueError(f"more than {MAX_IN_CELL} instances of {i['thing']} in one cell")
        seen[key] = k + 1
        cx, cy = CELL_CENTERS[i["col"]] + BOX_SHIFT * k, CELL_CENTERS[i["row"]] + BOX_SHIFT * k
        out.append([cx - half, cy - half, cx + half, cy + half])
    return out


def render(form: str, instances: Sequence[dict]) -> str:
    """A layout ({"thing", "colour", "col", "row"} a thing instance) written by code in the form "box" or "cell"."""
    if form == "box":
        return json.dumps([{"bbox_2d": b, "label": label_of(i)} for i, b in zip(instances, boxes_of(instances))])
    if form == "cell":
        if not instances:
            return "none"
        return "\n".join(f"{i['thing']} | {i.get('colour') or 'none'} | {COLUMNS[i['col']]} | {ROWS[i['row']]}"
                         for i in instances)
    if form == "chain":
        return render_chain(instances)
    if form == "chain_full":
        return render_chain_full(instances)
    raise ValueError(f"render writes the forms box, cell, chain and chain_full, not {form!r}")


def render_chain(instances: Sequence[dict]) -> str:
    """A short chain in ``step: text`` lines for a layout (for tests and fake writers): a count line a thing and a place
    line for every thing whose instances share one cell."""
    things: List[str] = []
    for i in instances:
        if i["thing"] not in things:
            things.append(i["thing"])
    lines = []
    for t in things:
        mine = [i for i in instances if i["thing"] == t]
        n, c = len(mine), mine[0].get("colour")
        head = f"{c} {_named(t, n)}" if c else _named(t, n)
        lines.append(f"count: {NUMBER_WORDS[n - 1] if n <= 5 else n} {head}")
    for t in things:
        mine = [i for i in instances if i["thing"] == t]
        if len({(i["col"], i["row"]) for i in mine}) == 1:
            lines.append(f"place: the {t} is at the {ROWS[mine[0]['row']]} {COLUMNS[mine[0]['col']]}")
    if not lines:
        lines.append("see: an empty scene")
    lines.append("caption: a scene.")
    return "\n".join(lines)


def ask(form: str, text: str, examples: bool = True) -> str:
    """The ask of a form for a prompt text: box and cell, the ask, a blank line, the six worked examples (when
    ``examples``), then 'Description: <text>' and 'Answer:'; chain_text, the text sentence (no examples)."""
    if form == "chain_text":
        return ASK_CHAIN_TEXT.replace("{prompt}", text)
    if form not in ("box", "cell"):
        raise ValueError(f"ask writes the forms box, cell and chain_text, not {form!r}")
    blocks = []
    if examples:
        blocks = [f"Description: {p['text']}\nAnswer:\n{render(form, p['truth'])}" for p in example_prompts()]
    blocks.append(f"Description: {text}\nAnswer:")
    return ASKS[form] + "\n\n" + "\n\n".join(blocks)


# ------------------------------------------------------------------ reading
def address_of_box(box) -> Optional[tuple]:
    """(col, row) of the cell holding the box's center, or None when the box has no address."""
    kinds = box_kinds(box)
    if kinds:
        return None
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / CANVAS, (y1 + y2) / 2 / CANVAS
    return _cut(cx), _cut(cy)


def _cut(v: float) -> int:
    return 0 if v < CUTS[0] else 2 if v > CUTS[1] else 1


def box_kinds(box) -> List[str]:
    kinds = []
    if any(v < 0 or v > CANVAS for v in box):
        kinds.append("outside canvas")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        kinds.append("no area")
    return kinds


def _instance(label: str, colours: List[str], box=None, col=None, row=None, kinds=()) -> dict:
    return {"label": label, "thing": thing_of(label), "colours": sorted(set(colours)),
            "box": list(box) if box is not None else None, "col": col, "row": row, "kinds": list(kinds)}


def _box_instance(box, label: str) -> dict:
    kinds = box_kinds(box)
    addr = None if kinds else address_of_box(box)
    return _instance(label, colours_of(label), box=box, col=addr[0] if addr else None, row=addr[1] if addr else None,
                     kinds=kinds)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```") and len(t) >= 6:
        body = t[3:-3]
        nl = body.find("\n")
        if nl >= 0 and re.fullmatch(r"[A-Za-z0-9_-]*", body[:nl].strip()):
            body = body[nl + 1:]
        t = body.strip()
    return t


def _reading(form: str, parsed: bool, failure, instances, lenient=None, **more) -> dict:
    r = {"form": form, "parsed": bool(parsed), "failure": failure, "instances": instances, "lenient": lenient}
    r.update(more)
    return r


def _read_box(answer: str) -> dict:
    lenient = _lenient_box(answer)
    t = _strip_fence(answer)
    try:
        data = json.loads(t)
    except ValueError:
        return _reading("box", False, "not json", None, lenient)
    if not isinstance(data, list):
        return _reading("box", False, "not a list", None, lenient)
    out = []
    for e in data:
        if not isinstance(e, dict):
            return _reading("box", False, "entry not an object", None, lenient)
        if set(e) != {"bbox_2d", "label"}:
            return _reading("box", False, "wrong keys", None, lenient)
        b = e["bbox_2d"]
        if not (isinstance(b, list) and len(b) == 4 and all(_number(v) for v in b)):
            return _reading("box", False, "bad box list", None, lenient)
        if not isinstance(e["label"], str):
            return _reading("box", False, "label not a string", None, lenient)
        if not e["label"].strip():
            return _reading("box", False, "empty label", None, lenient)
        out.append(_box_instance(b, e["label"]))
    return _reading("box", True, None, out, lenient)


def _lenient_box(answer: str) -> Optional[List[dict]]:
    dec = json.JSONDecoder()
    first = None
    for m in re.finditer(r"\[", answer):
        try:
            data, _ = dec.raw_decode(answer[m.start():])
        except ValueError:
            continue
        if not isinstance(data, list):
            continue
        if first is None:
            first = data
        if any(isinstance(e, dict) for e in data):
            first = data
            break
    if first is None:
        return None
    out = []
    for e in first:
        if not isinstance(e, dict):
            continue
        b = next((e[k] for k in ("bbox_2d", "bbox", "box") if k in e), None)
        lab = next((e[k] for k in ("label", "name") if k in e and isinstance(e[k], str)), None)
        if isinstance(b, list) and len(b) == 4 and all(_number(v) for v in b) and lab is not None:
            out.append(_box_instance(b, lab))
    return out


def _cell_instance(fields: List[str]) -> dict:
    name, colour, col, row = (f.strip() for f in fields[:4])
    cw, rw = col.lower(), row.lower()
    cw = "centre" if cw == "center" else cw
    c = COLUMNS.index(cw) if cw in COLUMNS else None
    r = ROWS.index(rw) if rw in ROWS else None
    kinds = [] if (c is not None and r is not None) else ["no place word"]
    cols = colours_of(colour) + colours_of(name)
    return _instance(name, cols, col=c if not kinds else None, row=r if not kinds else None, kinds=kinds)


def _read_cell(answer: str) -> dict:
    lines = [ln for ln in answer.strip().splitlines() if ln.strip()]
    lenient = [_cell_instance(ln.split("|")) for ln in lines if len(ln.split("|")) >= 4]
    if len(lines) == 1 and lines[0].strip().lower() == "none":
        return _reading("cell", True, None, [], [])
    if not lines:
        return _reading("cell", False, "empty answer", None, lenient)
    out = []
    for ln in lines:
        fields = ln.split("|")
        if len(fields) != 4:
            return _reading("cell", False, "wrong field count", None, lenient)
        if not fields[0].strip():
            return _reading("cell", False, "empty label", None, lenient)
        out.append(_cell_instance(fields))
    return _reading("cell", True, None, out, lenient)


def records_reading(spec: dict) -> dict:
    """A parser's records ({"thing": {"name", "count", "colour", "position"}, "relation": [{"a", "rel", "b"}], ...}) as
    a reading: the main thing's instances (its count, color and column word; no row, so no address), then one instance
    of every other thing of its relations (no place)."""
    out = []
    thing = spec.get("thing") or {}
    named = []
    if thing.get("name"):
        try:
            n = int(thing.get("count") or 1)
        except ValueError:
            n = 1
        col = CHAIN_COLUMN_WORDS.get(str(thing.get("position") or "").lower())
        colours = colours_of(str(thing.get("colour") or ""))
        for _ in range(max(n, 0)):
            out.append(_instance(str(thing["name"]), colours, col=col, row=None, kinds=["not placed"]))
        named.append(str(thing["name"]))
    for r in spec.get("relation") or []:
        for side in ("a", "b"):
            name = str(r.get(side) or "")
            if name and name not in named:
                named.append(name)
                out.append(_instance(name, [], kinds=["not placed"]))
    rels = [{"a": thing_of(str(r.get("a"))), "rel": r.get("rel"), "b": thing_of(str(r.get("b")))}
            for r in spec.get("relation") or []]
    return _reading("spec", True, None, out, None, relations=rels)


def _read_spec(answer: str) -> dict:
    try:
        spec = json.loads(answer)
    except ValueError:
        return _reading("spec", False, "not json", None)
    if not isinstance(spec, dict):
        return _reading("spec", False, "not an object", None)
    return records_reading(spec)


LINE = re.compile(r"^\s*(?:[-*•]\s*)?\**([A-Za-z]+)\**\s*:\**\s*(.*)$")


def _mentions(words: List[str]):
    """(position, thing, count or None, colour or None, negated) of every listed noun of a line's words."""
    out = []
    for k, w in enumerate(words):
        t = _listed(w)
        if t is None:
            continue
        j = k - 1
        colour = _colour_word(words[j]) if j >= 0 else None
        while j >= 0 and _colour_word(words[j]):
            j -= 1
        before = words[j] if j >= 0 else ""
        count = int(before) if before.isdigit() else CHAIN_COUNTS.get(before)
        out.append((k, t, count, colour, before in NEGATIONS))
    return out


def chain_read(text: str) -> dict:
    """The line reader of the chain form: from ``step: text`` lines (the sixteen step names, and the caption line),
    per listed thing: its count (a count word or number before the noun in a count, see or name line; a thing named with
    no count is one instance; 'no' or 'zero' is a count of 0, and a negated mention names nothing), its colors (a color
    word directly before the noun), its column and row (a place line naming exactly one thing, with a position word of
    each axis: left / centre / right for the column, top / upper / bottom / lower for the row, middle for the row, or
    for the column when the line has a row word and no column word) and the relations (a relate or place line with two
    things around one of the trained relation words). A thing with no placing line is not placed. ``parsed`` is true
    when at least one step line exists."""
    parts = _chain_parts(text)
    if parts is None:
        return _reading("chain", False, "no step line", None, None, relations=[])
    out = []
    for t in parts["order"]:
        n = parts["count"].get(t, 1)
        c, r = parts["place"].get(t, (None, None))
        placed = c is not None and r is not None
        for _ in range(n):
            out.append(_instance(t, sorted(parts["colours"].get(t, ())), col=c if placed else None,
                                 row=r if placed else None, kinds=[] if placed else ["not placed"]))
    return _reading("chain", True, None, out, None, relations=parts["relations"])


def _chain_parts(text: str) -> Optional[dict]:
    """What the chain reader takes from the step and caption lines, per thing (None when no step line exists):
    order, count, colours, place (column, row; either may be None) and relations."""
    lines = []
    for ln in str(text).splitlines():
        m = LINE.match(ln)
        if m and m.group(1).lower() in CHAIN_STEPS + ("caption",):
            lines.append((m.group(1).lower(), m.group(2)))
    steps = [s for s, _ in lines if s != "caption"]
    if not steps:
        return None
    order: List[str] = []
    count: Dict[str, int] = {}
    colours: Dict[str, set] = {}
    place: Dict[str, tuple] = {}
    relations = []
    for step, body in lines:
        words = _words(body)
        ms = _mentions(words)
        for k, t, c, col, neg in ms:
            if step in CHAIN_COUNT_STEPS and t not in count and c is not None:
                count[t] = c
            if neg:
                continue
            if t not in order:
                order.append(t)
            if col:
                colours.setdefault(t, set()).add(col)
        things = []
        for _, t, _, _, neg in ms:
            if not neg and t not in things:
                things.append(t)
        if step == "place" and len(things) == 1 and things[0] not in place:
            cols = {CHAIN_COLUMN_WORDS[w] for w in words if w in CHAIN_COLUMN_WORDS}
            rows = {CHAIN_ROW_WORDS[w] for w in words if w in CHAIN_ROW_WORDS}
            middle = "middle" in words
            if middle and not rows:
                rows = {1}
            elif middle and not cols:
                cols = {1}
            c = cols.pop() if len(cols) == 1 else None
            r = rows.pop() if len(rows) == 1 else None
            place[things[0]] = (c, r)
        if step in ("relate", "place") and len(things) == 2:
            pos = {t: next(k for k, tt, _, _, neg in ms if tt == t and not neg) for t in things}
            a, b = sorted(things, key=lambda t: pos[t])
            between = " ".join(words[pos[a] + 1:pos[b]])
            rel = next((r for r in CHAIN_RELATIONS if re.search(r"\b" + re.escape(r) + r"\b", between)), None)
            if rel:
                relations.append({"a": a, "rel": rel, "b": b})
    return {"order": order, "count": count, "colours": colours, "place": place, "relations": relations}


# a question line of the trained whole-row form: `question: <key> -> <value> [record|chain|unknown]` (the pattern of the
# trained-ask package's chain reader; keys in its "record" form: thing name / count / colour / position, rel<i> a / rel / b)
QUESTION_LINE = re.compile(r"^\s*(?:[-*•]\s*)?\**question\**\s*:\**\s*(.*\S)\s*$", re.I)
QUESTION = re.compile(r"^(.*?)\s*->\s*(.*?)\s*\[(record|chain|unknown)\]\s*$", re.I)
QUESTION_THING_KEYS = ("name", "count", "colour", "position")
QUESTION_RELATION_KEY = re.compile(r"^rel(\d+) (a|rel|b)$")


def question_records(text: str) -> dict:
    """The records the question lines state: {"thing": {name, count, colour, position}, "relation": [{a, rel, b}]}.
    A value 'unknown' gives nothing; the first value of a key wins; a relation needs its three parts."""
    thing: Dict[str, str] = {}
    rels: Dict[int, Dict[str, str]] = {}
    for ln in str(text).splitlines():
        m = QUESTION_LINE.match(ln)
        q = QUESTION.match(m.group(1)) if m else None
        if not q:
            continue
        key, value = " ".join(q.group(1).lower().split()), q.group(2).strip()
        if not value or value.lower() == "unknown":
            continue
        if key.startswith("thing ") and key[6:] in QUESTION_THING_KEYS:
            thing.setdefault(key[6:], value)
            continue
        r = QUESTION_RELATION_KEY.match(key)
        if r:
            rels.setdefault(int(r.group(1)), {}).setdefault(r.group(2), value)
    out: dict = {}
    if thing:
        out["thing"] = thing
    full = [rels[i] for i in sorted(rels) if all(k in rels[i] for k in ("a", "rel", "b"))]
    if full:
        out["relation"] = [{"a": r["a"], "rel": r["rel"], "b": r["b"]} for r in full]
    return out


def _count_value(v) -> Optional[int]:
    w = str(v).strip().lower()
    if w.isdigit():
        return int(w)
    return CHAIN_COUNTS.get(w)


def chain_full_read(text: str) -> dict:
    """The reader of the trained whole-row form: ``chain_read`` of the step and caption lines JOINED with the records of
    the question lines (``question_records``). A thing named by either source is written once; its count, colors and
    column come from the question lines when they give them, else from the chain lines; its row only from a chain place
    line; the relations from both, without repeats. ``parsed`` is true when at least one step line exists."""
    parts = _chain_parts(text)
    if parts is None:
        return _reading("chain_full", False, "no step line", None, None, relations=[])
    rec = question_records(text)
    q = rec.get("thing") or {}
    order = list(parts["order"])
    count, colours = dict(parts["count"]), {t: set(v) for t, v in parts["colours"].items()}
    cols = {t: p[0] for t, p in parts["place"].items()}
    rows = {t: p[1] for t, p in parts["place"].items()}
    labels: Dict[str, str] = {t: t for t in order}
    if q.get("name"):
        key = thing_of(q["name"]) or "?" + " ".join(_words(q["name"]))
        if key not in labels:
            order.append(key)
            labels[key] = key if not key.startswith("?") else q["name"]
        n = _count_value(q.get("count")) if q.get("count") else None
        if n is not None:
            count[key] = n
        if q.get("colour") and colours_of(q["colour"]):
            colours[key] = set(colours_of(q["colour"]))
        c = CHAIN_COLUMN_WORDS.get(str(q.get("position") or "").strip().lower())
        if c is not None:
            cols[key] = c
    relations = list(parts["relations"])
    for r in rec.get("relation") or []:
        a, b = thing_of(r["a"]), thing_of(r["b"])
        for name, t in ((r["a"], a), (r["b"], b)):
            key = t or "?" + " ".join(_words(name))
            if key not in labels:
                order.append(key)
                labels[key] = key if t else name
        item = {"a": a, "rel": " ".join(str(r["rel"]).lower().split()), "b": b}
        if item not in relations:
            relations.append(item)
    out = []
    for t in order:
        c, r = cols.get(t), rows.get(t)
        placed = c is not None and r is not None
        for _ in range(count.get(t, 1)):
            out.append(_instance(labels[t], sorted(colours.get(t, ())), col=c if placed else None,
                                 row=r if placed else None, kinds=[] if placed else ["not placed"]))
    return _reading("chain_full", True, None, out, None, relations=relations, questions=rec)


def render_chain_full(instances: Sequence[dict]) -> str:
    """A chain and question lines for a layout (for tests and fake writers): the chain of ``render_chain``, then the first
    thing's name, count and color, its column word when all its instances share a column, and one rel1 triple when two
    things stand in one of the four relations."""
    lines = [render_chain(instances)]
    if instances:
        first = instances[0]["thing"]
        mine = [i for i in instances if i["thing"] == first]
        lines.append(f"question: thing name -> {first} [record]")
        lines.append(f"question: thing count -> {len(mine)} [record]")
        if mine[0].get("colour"):
            lines.append(f"question: thing colour -> {mine[0]['colour']} [record]")
        if len({i["col"] for i in mine}) == 1:
            lines.append(f"question: thing position -> {COLUMNS[mine[0]['col']]} [record]")
        things: List[str] = []
        for i in instances:
            if i["thing"] not in things:
                things.append(i["thing"])
        facts = [f for f in _facts_of(things, instances) if f[0] == "rel"]
        if facts:
            _, a, rel, b = facts[0]
            lines += [f"question: rel1 a -> {a} [record]", f"question: rel1 rel -> {rel} [record]",
                      f"question: rel1 b -> {b} [record]"]
    return "\n".join(lines)


def read(form: str, answer: str) -> dict:
    """The strict reading of an answer in a form ("box", "cell", "chain" or "spec"): {"form", "parsed", "failure",
    "instances", "lenient"}; an instance is {"label", "thing", "colours", "box", "col", "row", "kinds"}."""
    if form == "box":
        return _read_box(answer)
    if form == "cell":
        return _read_cell(answer)
    if form == "chain":
        return chain_read(answer)
    if form == "chain_full":
        return chain_full_read(answer)
    if form == "spec":
        return _read_spec(answer)
    raise ValueError(f"read knows the forms box, cell, chain, chain_full and spec, not {form!r}")


# ------------------------------------------------------------------ scoring
def _key(inst: dict) -> str:
    return inst["thing"] if inst["thing"] is not None else "?" + " ".join(_words(inst["label"]))


def _addressed(inst: dict) -> bool:
    return inst["col"] is not None and inst["row"] is not None


def _rel_holds(rel: str, i: dict, j: dict) -> bool:
    return _relation_holds_cells(rel, i["col"], i["row"], j["col"], j["row"])


def _statement_holds(st: dict, by: Dict[str, List[dict]]) -> bool:
    if "axis" in st:
        mine = by.get(st["thing"], [])
        return bool(mine) and all(_addressed(i) and i[st["axis"]] == st["value"] for i in mine)
    a, b = by.get(st["a"], []), by.get(st["b"], [])
    return (bool(a) and bool(b) and all(_addressed(i) for i in a + b)
            and all(_rel_holds(st["rel"], i, j) for i in a for j in b))


def _instance_breaks(inst: dict, thing: str, statements: List[dict], by: Dict[str, List[dict]]) -> bool:
    for st in statements:
        if "axis" in st:
            if st["thing"] == thing and not (_addressed(inst) and inst[st["axis"]] == st["value"]):
                return True
        elif thing in (st["a"], st["b"]):
            other = by.get(st["b"] if thing == st["a"] else st["a"], [])
            if not _addressed(inst) or not other or not all(_addressed(j) for j in other):
                return True
            if thing == st["a"] and not all(_rel_holds(st["rel"], inst, j) for j in other):
                return True
            if thing == st["b"] and not all(_rel_holds(st["rel"], j, inst) for j in other):
                return True
    return False


def text_key(text: str) -> str:
    """The first 16 hex digits of the sha256 of a prompt's text: rows of one text share it."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


_EXAMPLE_BOXES: List[tuple] = []


def example_boxes() -> List[tuple]:
    """Every box the box form writes for the six worked examples."""
    if not _EXAMPLE_BOXES:
        for p in example_prompts():
            _EXAMPLE_BOXES.extend(tuple(b) for b in boxes_of(p["truth"]))
    return list(_EXAMPLE_BOXES)


def copied(reading: dict) -> Optional[int]:
    """For a box reading: 1 when it is parsed, has at least one box, and every written box equals a box of the six worked
    examples; else 0. None for the other forms. It judges nothing."""
    if reading.get("form") != "box":
        return None
    if not reading.get("parsed") or not reading.get("instances"):
        return 0
    known = set(example_boxes())
    return int(all(tuple(i["box"]) in known for i in reading["instances"]))


def score(prompt: dict, reading: dict, form: Optional[str] = None) -> dict:
    """Every score of the prompt's family (0 or 1; None where a score does not apply), address_right, the failure kinds
    present, and per asked instance whether it is in error (instance_errors of instances_asked). An unparsed answer
    scores 0 on everything that applies; its lenient reading, when there is one, is scored as lenient_address_right."""
    family = prompt["family"]
    applies = APPLIES[family]
    asked = sum(int(a["count"]) for a in prompt["asks"])
    out = {"id": prompt["id"], "family": family, "form": form or reading.get("form"), "text_key": text_key(prompt["text"]),
           "copied": copied(reading)}
    if not reading.get("parsed"):
        for s in SCORES:
            out[s] = 0 if s in applies else None
        out.update({"address_right": 0, "kinds": ["broken format"], "instance_errors": asked, "instances_asked": asked,
                    "failure": reading.get("failure")})
        len_inst = reading.get("lenient")
        out["lenient_address_right"] = (None if len_inst is None else
                                        _score_instances(prompt, len_inst)["address_right"])
        return out
    s = _score_instances(prompt, reading["instances"])
    out.update(s)
    out["lenient_address_right"] = None
    return out


def _score_instances(prompt: dict, instances: List[dict]) -> dict:
    family = prompt["family"]
    applies = APPLIES[family]
    by: Dict[str, List[dict]] = {}
    for i in instances:
        by.setdefault(_key(i), []).append(i)
    asked_things = [a["thing"] for a in prompt["asks"]]
    written = list(by)
    kinds = set()
    # things
    extra = [k for k in written if k not in asked_things]
    missing = [t for t in asked_things if t not in by]
    things = int(not extra and not missing)
    if extra:
        kinds.add("extra thing")
    if missing:
        kinds.add("missing thing")
    counts = int(all(len(by.get(a["thing"], [])) == int(a["count"]) for a in prompt["asks"]))
    if not counts:
        kinds.add("wrong count")
    colour_ok = True
    for a in prompt["asks"]:
        if a.get("colour"):
            for i in by.get(a["thing"], []):
                if i["colours"] != [a["colour"]]:
                    colour_ok = False
    if not colour_ok:
        kinds.add("wrong colour")
    placed = int(all(_addressed(i) for i in instances))
    for i in instances:
        kinds.update(i.get("kinds") or ())
    statements = [dict(p) for p in prompt["places"]] + [dict(r) for r in prompt["relations"]]
    spatial = int(all(_statement_holds(st, by) for st in statements))
    if not spatial:
        kinds.add("spatial broken")
    cells: Dict[tuple, set] = {}
    for i in instances:
        if _addressed(i):
            cells.setdefault((i["col"], i["row"]), set()).add(_key(i))
    shared = {c for c, ks in cells.items() if len(ks) > 1}
    no_conflict = int(not shared)
    if shared:
        kinds.add("shared cell")
    absent = int(not any(t in by for t in prompt["absent"]))
    if not absent:
        kinds.add("absent written")
    values = {"parsed": 1, "things": things, "counts": counts, "colours": int(colour_ok), "placed": placed,
              "spatial": spatial, "no_conflict": no_conflict, "absent": absent}
    out = {s: (values[s] if s in applies else None) for s in SCORES}
    out["address_right"] = int(all(values[s] for s in applies))
    allowed = set()
    for s in applies:
        allowed.update(KIND_OF_SCORE[s])
    out["kinds"] = sorted(k for k in kinds if k in allowed)
    # per asked instance
    errors = 0
    for a in prompt["asks"]:
        t, c = a["thing"], int(a["count"])
        flags = []
        for i in by.get(t, []):
            bad = (bool(a.get("colour")) and i["colours"] != [a["colour"]]) or not _addressed(i) \
                or _instance_breaks(i, t, statements, by) \
                or (_addressed(i) and (i["col"], i["row"]) in shared)
            flags.append(bad)
        flags.sort()
        matched = flags[:c]
        errors += (c - len(matched)) + sum(matched)
    out["instance_errors"] = errors
    out["instances_asked"] = sum(int(a["count"]) for a in prompt["asks"])
    return out


def _substring_count(labels: List[str], name: str) -> int:
    return sum(name in lab.lower() for lab in labels)


def _center(box) -> tuple:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def _lmd_relation(rel: str, a, b) -> bool:
    (xa, ya, _, _), (xb, yb, _, _) = _center(a), _center(b)
    return {"left of": xa < xb, "right of": xa > xb, "above": ya < yb, "below": ya > yb}[rel]


def _margin_relation(rel: str, a, b, c: float = 0.1) -> bool:
    (xa, ya, wa, ha), (xb, yb, wb, hb) = _center(a), _center(b)
    return {"left of": xb > xa + c * (wa + wb), "right of": xa > xb + c * (wa + wb),
            "above": yb > ya + c * (ha + hb), "below": ya > yb + c * (ha + hb)}[rel]


def _sector_relation(rel: str, a, b) -> bool:
    """LayoutGPT's four equal sectors of the circle around B: 'A rel B' holds when A's center lies in that sector. The
    source's y axis points up, so in picture coordinates (y down) 'above' is the smaller y."""
    (xa, ya, _, _), (xb, yb, _, _) = _center(a), _center(b)
    dx, dy_up = xa - xb, -(ya - yb)
    d = math.hypot(dx, dy_up)
    if d == 0:
        return False
    if rel == "above":
        return dy_up / d >= math.sin(math.pi / 4)
    if rel == "below":
        return dy_up / d <= math.sin(-math.pi / 4)
    if rel == "left of":
        return dx / d < math.cos(3 * math.pi / 4)
    return dx / d > math.cos(math.pi / 4)


PLACE_AS_RELATION = {"left": "left of", "right": "right of", "top": "above", "bottom": "below"}


def _pair_statements(prompt: dict) -> List[tuple]:
    """(a, rel, b) of a place or relation prompt: 'A on the left and B on the right' is A left of B."""
    if prompt["family"] == "relation":
        return [(r["a"], r["rel"], r["b"]) for r in prompt["relations"]]
    first = prompt["places"][0]
    side = [s for s, v in SIDES.items() if v == (first["axis"], first["value"])][0]
    return [(first["thing"], PLACE_AS_RELATION[side], prompt["places"][1]["thing"])]


def published(prompt: dict, reading: dict) -> dict:
    """The published rules on the raw boxes, beside the address scores: ``lmd`` (count, absent, colour, place: a count
    per name by substring equal to the asked number, 0 for an absent thing; the color word in each such label; for
    place, one box a name and the two centers compared with a strict inequality, the other axis ignored), ``margin``
    (place and relation: the centers apart by more than 0.1 of the two boxes' summed widths, or heights) and
    ``sectors`` (place and relation: the four equal sectors, passing when ANY pair of boxes of the two names has the
    relation, as the source scores it). None for other families and for readings without boxes; 0 when unparsed."""
    family = prompt["family"]
    out = {"lmd": None, "margin": None, "sectors": None}
    if reading.get("form") != "box":
        return out
    lmd_fam = family in ("count", "absent", "colour", "place")
    pair_fam = family in ("place", "relation")
    if not reading.get("parsed"):
        return {"lmd": 0 if lmd_fam else None, "margin": 0 if pair_fam else None, "sectors": 0 if pair_fam else None}
    insts = reading["instances"]
    labels = [i["label"] for i in insts]

    def boxes(name):
        return [i["box"] for i in insts if name in i["label"].lower()]

    if lmd_fam:
        ok = all(_substring_count(labels, t) == 0 for t in prompt["absent"])
        for a in prompt["asks"]:
            mine = [lab for lab in labels if a["thing"] in lab.lower()]
            ok = ok and len(mine) == int(a["count"])
            if a.get("colour"):
                ok = ok and all(a["colour"] in lab.lower() for lab in mine)
        if family == "place" and ok:
            (ta, rel, tb), = _pair_statements(prompt)
            ok = _lmd_relation(rel, boxes(ta)[0], boxes(tb)[0])
        out["lmd"] = int(ok)
    if pair_fam:
        (ta, rel, tb), = _pair_statements(prompt)
        ba, bb = boxes(ta), boxes(tb)
        one_each = len(ba) == 1 and len(bb) == 1
        out["margin"] = int(one_each and _margin_relation(rel, ba[0], bb[0]))
        out["sectors"] = int(any(_sector_relation(rel, x, y) for x in ba for y in bb))
    return out


def score_real(text: str, reading: dict) -> dict:
    """A real caption (no key): parsed, placed, no_conflict, and written_things_in_caption (every written listed thing
    occurs in the caption as a word, singular or plural)."""
    if not reading.get("parsed"):
        return {"parsed": 0, "placed": 0, "no_conflict": 0, "written_things_in_caption": 0}
    insts = reading["instances"]
    words = set(_words(text))
    cells: Dict[tuple, set] = {}
    for i in insts:
        if _addressed(i):
            cells.setdefault((i["col"], i["row"]), set()).add(_key(i))
    things = [i["thing"] for i in insts if i["thing"] is not None]
    in_caption = all(t in words or PLURALS[t] in words for t in things)
    return {"parsed": 1, "placed": int(all(_addressed(i) for i in insts)),
            "no_conflict": int(all(len(k) == 1 for k in cells.values())), "written_things_in_caption": int(in_caption)}


# ------------------------------------------------------------------ the controls
def derangement(n: int, seed: int = 0) -> List[int]:
    """A fixed permutation of 0 .. n - 1 with no fixed point (Sattolo's shuffle of random.Random(seed): one cycle)."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    p = list(range(n))
    rng = random.Random(int(seed))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        p[i], p[j] = p[j], p[i]
    return p


def family_derangement(prompts: Sequence[dict], seed: int = 0) -> List[int]:
    """For each position of ``prompts``, the position of another prompt of the same family (a derangement inside each
    family)."""
    out = list(range(len(prompts)))
    for fam in FAMILIES:
        idx = [k for k, p in enumerate(prompts) if p["family"] == fam]
        if len(idx) < 2:
            continue
        for k, j in zip(idx, derangement(len(idx), seed)):
            out[k] = idx[j]
    return out


def truth_reading(prompt: dict, form: str = "cell") -> dict:
    """The prompt's own layout as a reading (as the form's reader would give it)."""
    return read(form, render(form, prompt["truth"]))


def random_reading(prompt: dict, seed: int) -> dict:
    """The right things, counts and colors at cells drawn at random, as a reading."""
    rng = random.Random(int(seed))
    insts = []
    for a in prompt["asks"]:
        for _ in range(int(a["count"])):
            c, r = rng.randrange(3), rng.randrange(3)
            insts.append(_instance(label_of({"thing": a["thing"], "colour": a.get("colour")}),
                                   [a["colour"]] if a.get("colour") else [], col=c, row=r))
    return _reading("random", True, None, insts, None)


BROKEN = {
    "box": [('[{"bbox_2d": [100, 100, 300, 300], "label": "cup"}', "not json"),
            ('{"bbox_2d": [100, 100, 300, 300], "label": "cup"}', "not a list"),
            ('["cup"]', "entry not an object"),
            ('[{"bbox_2d": [100, 100, 300, 300]}]', "wrong keys"),
            ('[{"bbox_2d": [100, 100, 300], "label": "cup"}]', "bad box list"),
            ('[{"bbox_2d": [100, 100, 300, 300], "label": " "}]', "empty label"),
            ('[{"bbox_2d": [100, 100, 1300, 300], "label": "cup"}]', "outside canvas"),
            ('[{"bbox_2d": [300, 100, 100, 300], "label": "cup"}]', "no area")],
    "cell": [("cup | none | left", "wrong field count"),
             ("cup | none | left | top | extra", "wrong field count"),
             (" | none | left | top", "empty label"),
             ("cup | none | far | top", "no place word")],
    "chain": [("A cup on a table, seen from above.", "no step line")],
    "chain_full": [("question: thing name -> cup [record]", "no step line")],
}
# the failure kinds that leave an answer PARSED but give its instance no address (the registered grammars read such
# an answer and fail PLACED: a number out of range in a box, a word that is no place in a cell)
UNPLACED_KINDS = ("outside canvas", "no area", "no place word")


def broken_check(form: str) -> List[dict]:
    """Control (c) for a form: each broken answer with the kind it must give and the kind it gave."""
    out = []
    for answer, kind in BROKEN[form]:
        r = read(form, answer)
        if kind in UNPLACED_KINDS:
            got = sorted({k for i in (r["instances"] or []) for k in i["kinds"]}) if r["parsed"] else [r["failure"]]
            ok = r["parsed"] and kind in got
        else:
            got = [r["failure"]]
            ok = (not r["parsed"]) and r["failure"] == kind
        out.append({"answer": answer, "kind": kind, "got": got, "parsed": r["parsed"], "ok": bool(ok)})
    return out


# ------------------------------------------------------------------ summaries
def rate(values: Sequence[int]) -> dict:
    vals = [v for v in values if v is not None]
    n = len(vals)
    if not n:
        return {"mean": None, "se": None, "n": 0}
    p = sum(vals) / n
    return {"mean": p, "se": math.sqrt(p * (1 - p) / n), "n": n}


SCORED_FIELDS = SCORES + ("address_right", "kinds", "instance_errors", "instances_asked", "lenient_address_right",
                           "published", "copied", "written_things_in_caption")


def _group_key(r: dict) -> tuple:
    return (r.get("family"), r["text_key"]) if r.get("text_key") else ("id", r.get("id"))


def _groups(rows: Sequence[dict]) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = {}
    for r in rows:
        out.setdefault(_group_key(r), []).append(r)
    return out


def _kept_of(group: List[dict]) -> List[dict]:
    first = [group[0].get(k) for k in SCORED_FIELDS]
    return group[:1] if all([r.get(k) for k in SCORED_FIELDS] == first for r in group) else list(group)


def distinct(rows: Sequence[dict]) -> tuple:
    """(kept rows, {"prompts", "texts", "duplicates_disagree"}): the rows of one family with the same text_key and the same
    scored fields collapse to one; when the copies of a text differ in any score, every copy is kept and the text is
    counted in duplicates_disagree. Rows without a text_key are distinct."""
    kept, disagree = [], 0
    groups = _groups(rows)
    for g in groups.values():
        k = _kept_of(g)
        disagree += int(len(k) > 1)
        kept += k
    return kept, {"prompts": len(rows), "texts": len(groups), "duplicates_disagree": disagree}


def _block(rows: Sequence[dict]) -> dict:
    """The rates of a set of rows, each distinct text counted once (``distinct``)."""
    rows, info = distinct(rows)
    out = {"n": len(rows), "scores": {s: rate([r.get(s) for r in rows]) for s in SCORES + ("address_right",)}}
    out.update(info)
    if any(r.get("copied") is not None for r in rows):
        out["copied"] = rate([r.get("copied") for r in rows])
    errors = sum(int(r.get("instance_errors") or 0) for r in rows)
    asked = sum(int(r.get("instances_asked") or 0) for r in rows)
    out["per_instance_error"] = {"errors": errors, "asked": asked, "rate": errors / asked if asked else None}
    kinds: Dict[str, int] = {}
    for r in rows:
        for k in r.get("kinds") or ():
            kinds[k] = kinds.get(k, 0) + 1
    out["kinds"] = {k: kinds[k] for k in KINDS if k in kinds}
    unparsed = [r for r in rows if r.get("parsed") == 0]
    out["unparsed"] = len(unparsed)
    out["lenient_recovered"] = sum(1 for r in unparsed if r.get("lenient_address_right") == 1)
    pub: Dict[str, dict] = {}
    for key in ("lmd", "margin", "sectors"):
        vals = [(r.get("published") or {}).get(key) for r in rows]
        if any(v is not None for v in vals):
            pub[key] = rate(vals)
    if pub:
        out["published"] = pub
    return out


def summarize(rows: Sequence[dict]) -> dict:
    """Per family and over the six synthetic families: every score's mean, standard error and n; the per-instance error;
    the failure kinds; the load curve on combined by the number of asked instances; the published rules (lmd over
    count, absent, colour and place; margin and sectors over place and relation); the real captions' scores."""
    syn = [r for r in rows if r.get("family") in FAMILIES]
    out = {"families": {f: _block([r for r in syn if r["family"] == f]) for f in FAMILIES
                        if any(r["family"] == f for r in syn)}}
    out["all"] = _block(syn)
    syn, _ = distinct(syn)                     # each distinct text once in the published rules and the load curve too
    lmd_rows = [r for r in syn if r["family"] in ("count", "absent", "colour", "place")]
    pair_rows = [r for r in syn if r["family"] in ("place", "relation")]
    out["published"] = {"lmd_four_families": rate([(r.get("published") or {}).get("lmd") for r in lmd_rows]),
                        "margin_place_relation": rate([(r.get("published") or {}).get("margin") for r in pair_rows]),
                        "sectors_place_relation": rate([(r.get("published") or {}).get("sectors") for r in pair_rows])}
    curve = []
    for k in sorted({int(r["instances_asked"]) for r in syn if r["family"] == "combined"}):
        sel = [r["address_right"] for r in syn if r["family"] == "combined" and int(r["instances_asked"]) == k]
        curve.append(dict(rate(sel), instances=k))
    out["load_curve"] = curve
    real, _ = distinct([r for r in rows if r.get("family") == "real"])
    if real:
        out["real"] = {k: rate([r.get(k) for r in real])
                       for k in ("parsed", "placed", "no_conflict", "written_things_in_caption")}
    return out


def compare(rows_a: Sequence[dict], rows_b: Sequence[dict], key: str = "address_right",
            family: Optional[str] = None) -> dict:
    """The text-by-text difference a - b of one score (``family``: one family; None: the six synthetic families): a pair
    is one distinct text's row in a and in b (``distinct``); when either side kept several copies of a text, the copies
    are paired by prompt id; rows without a text_key are paired by prompt id. -> mean, standard error sd / sqrt(n) (ddof
    1), n, and ``tie`` (|mean| < 2 se; an exactly zero mean with a zero standard error is a tie too)."""
    fams = FAMILIES if family is None else (family,)
    ga = _groups([r for r in rows_a if r.get("family") in fams])
    gb = _groups([r for r in rows_b if r.get("family") in fams])
    pairs = []
    for g, ra in ga.items():
        if g not in gb:
            continue
        ka, kb = _kept_of(ra), _kept_of(gb[g])
        if len(ka) == 1 and len(kb) == 1:
            pairs.append((ka[0], kb[0]))
        else:
            b_by = {r["id"]: r for r in gb[g]}
            pairs += [(r, b_by[r["id"]]) for r in ra if r["id"] in b_by]
    diffs = []
    for x, y in pairs:
        va, vb = x.get(key), y.get(key)
        if va is None or vb is None:
            continue
        diffs.append(float(va) - float(vb))
    n = len(diffs)
    if not n:
        return {"mean": None, "se": None, "n": 0, "tie": None}
    m = sum(diffs) / n
    se = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) / math.sqrt(n) if n > 1 else None
    tie = None if se is None else (abs(m) < 2 * se or (m == 0 and se == 0))
    return {"mean": m, "se": se, "n": n, "tie": tie}
