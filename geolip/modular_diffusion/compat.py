"""Compatibility with the flat argument set and the single-file weight layout of the original research script.

build_from_args(a, field_shape) -> Build      the units an argument set describes, in composition order
legacy_parameter_map(model)                   {flat key: parameter} (embed., time., cls., body.i., body_head.,
                                               members.j., stage_blocks.s.)
to_legacy_state(model) / load_legacy_state     the flat state dict, both ways (members matched by name)
load_legacy_files(model, paths)               several flat weight files, each with the JSON result beside it naming
                                               its members (shared parts from each file in turn)
describe(a)                                   the run name an argument set gives
parse_extra(spec)                             "b@lo-hi[#tag]" items

A flat weight file holds parameters only: the anchor's spectrum is not in it (it is measured from the data).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from .build import Build, UnitEntry
from .rotary import BANDS3, WHOLE
from .units import NULL_CLASS
from .windows import stage_edges

GROUPS = {"2": [(2, 3, 4, 5, 6, 7), (0, 1, 8, 9)], "10": [(c,) for c in range(10)]}
TIME_FREQ = 64


def num(x: float) -> str:
    """A window edge as the decimal string that reads back to the same float."""
    return repr(float(x))


def parse_extra(spec: str) -> List[Tuple[str, Any, str, str]]:
    """``b@lo-hi[#tag]`` items -> (name, band, lo text, hi text); band is 0, 1, 2 or "a" (every band).
    The name is built from the typed text: x<tag>_b<b>_<lo>-<hi>."""
    out = []
    for item in (x for x in spec.split(",") if x):
        body, _, tag = item.partition("#")
        b, win = body.split("@")
        lo, hi = win.split("-")
        out.append((f"x{tag}_b{b}_{lo}-{hi}", b if b == "a" else int(b), lo, hi))
    names = [n for n, _, _, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"two extra members share a name (give them tags): {names}")
    return out


def _code(kind: str) -> list:
    if kind in ("whole", "rung"):
        return list(WHOLE)
    raise NotImplementedError(f"position code {kind!r} is not supported")


def build_from_args(a: Dict[str, Any], field_shape: Sequence[int]) -> Build:
    """The build of an argument set. Unit order: anchor, stem, stage units, head, members (base, extra, content)."""
    if int(a.get("xdeep", 0)):
        raise NotImplementedError("xdeep (a member reading another member's state) is not supported")
    plain = a["arch"] == "plain"
    if a["arch"] not in ("plain", "collective"):
        raise ValueError(f"unknown arch {a['arch']!r}")
    c, hh, ww = (int(x) for x in field_shape)
    grid = int(a["grid"])
    patch = hh // grid
    d, heads = int(a["dim"]), int(a["heads"])
    nb = 1 if plain else int(a["bands"])
    geometry = {"field": [c, hh, ww], "grid": grid, "patch": patch, "token_dim": c * patch * patch, "dim": d,
                "heads": heads, "time_freq": TIME_FREQ, "classes": NULL_CLASS + 1, "bands": nb}
    edges = [] if plain else stage_edges(a["stages"])
    units = [UnitEntry("anchor", "Anchor", {"site": "anchor"}, {"output": bool(a["anchor"])}),
             UnitEntry("stem", "Stem", {"site": "trunk", "slot": 0},
                       {"depth": int(a["plain_depth"] if plain else a["body_depth"]), "anchor_in": int(a["anchor_in"]),
                        "code": _code(a["body_code"])})]
    sb = a["stage_body"]
    if sb and not plain:
        cuts = [0.0] + stage_edges(sb) + [1.0]
        kind = "GrowthBlock" if int(a["stage_zero"]) else "StageBody"
        for s in range(len(cuts) - 1):
            units.append(UnitEntry(f"stage_{s}", kind, {"site": "trunk", "slot": 1, "window": [num(cuts[s]), num(cuts[s + 1])],
                                                        "soft": num(a["stage_soft"])},
                                   {"index": s}, stream="stage_blocks", index=s))
    if plain or int(a["body_out"]):
        units.append(UnitEntry("head", "Head", {"site": "out", "band": "all"}, {}))
    if not plain:
        keep = bool(int(a["keep"])) and nb > 1
        cuts = [0.0] + list(edges) + [1.0]
        specs = []                                       # (name, band, lo text, hi text, classes)
        if int(a["base"]):
            specs += [(f"b{b}_s{s}", b, num(cuts[s]), num(cuts[s + 1]), None) for s in range(len(cuts) - 1) for b in range(nb)]
        for name, b, lo, hi in parse_extra(a["extra"]):
            specs.append((name, "all" if b == "a" else b, lo, hi, None))
        for item in (x for x in a["content"].split(";") if x):
            win, gname = item.split(":")
            lo, hi = win.split("-")
            for gi, classes in enumerate(GROUPS[gname]):
                for b in range(nb):
                    specs.append((f"c{gi}_b{b}_{lo}-{hi}", b, lo, hi, list(classes)))
        for name, b, lo, hi, classes in specs:
            if a["codes"] == "rung" and nb == 3 and b != "all":
                code = [BANDS3[b][1], BANDS3[b][2]]
            else:
                code = _code("whole" if a["codes"] == "rung" else a["codes"])
            address = {"site": "out", "band": b, "window": [lo, hi], "soft": num(a["soft"]), "classes": classes}
            units.append(UnitEntry(name, "BandMember" if classes is None else "ContentExpert", address,
                                   {"depth": int(a["member_depth"]), "code": code, "keep": keep}))
    build = Build(name=describe(a), geometry=geometry, units=units, info={"stage_edges": list(edges)})
    build.validate()
    return build


# ------------------------------------------------------------------ the flat weight layout
def _legacy_prefixes(model) -> List[Tuple[str, Any, str]]:
    """(flat prefix, unit, unit-relative prefix) in the flat file's order."""
    from .units import BandMember, Head, StageBody, Stem
    rows = []
    stem = model.stem
    rows.append(("embed.", stem, "embed."))
    rows.append(("time.", stem, "time."))
    rows.append(("cls.", stem, "cls."))
    rows.append(("body.", stem, "blocks."))
    if model.head is not None:
        rows.append(("body_head.", model.head, ""))
    for j, m in enumerate(model.members):
        rows.append((f"members.{j}.", m, ""))
    for u in model.stage_units:
        rows.append((f"stage_blocks.{int(u.spec['index'])}.", u, "block."))
    return rows


def legacy_parameter_map(model) -> Dict[str, Tensor]:
    """{flat key: the model's parameter} for every parameter of the model."""
    out = {}
    for flat, unit, rel in _legacy_prefixes(model):
        for k, p in unit.named_parameters():
            if k.startswith(rel):
                out[flat + k[len(rel):]] = p
    n_all = sum(1 for _ in model.parameters())
    if len(out) != n_all:
        raise RuntimeError(f"the flat map covers {len(out)} of {n_all} parameters")
    return out


def to_legacy_state(model) -> Dict[str, Tensor]:
    """The flat state dict (CPU copies), keys in the flat file's order."""
    return {k: p.detach().cpu().clone() for k, p in legacy_parameter_map(model).items()}


def load_legacy_state(model, state: Dict[str, Tensor], member_names: Sequence[str]) -> Dict[str, Any]:
    """Load a flat state dict. Shared parts (stem, head, stage blocks) by key, present ones only; members by name
    (``member_names[j]`` is the name of ``members.j``), each one strictly. Returns what was loaded."""
    pmap = legacy_parameter_map(model)
    own = {m.name: j for j, m in enumerate(model.members)}
    got, unused, shared = [], [], set()
    per: Dict[str, Dict[str, Tensor]] = {}
    with torch.no_grad():
        for k, v in state.items():
            if k.startswith("members."):
                i, rest = k[len("members."):].split(".", 1)
                per.setdefault(member_names[int(i)], {})[rest] = v
            elif k in pmap:
                pmap[k].copy_(v.to(pmap[k].device))
                shared.add(k)
    for name, sd in per.items():
        if name in own:
            model.members[own[name]].load_state_dict(sd, strict=True)
            got.append(name)
        else:
            unused.append(name)
    return {"members_loaded": got, "members_unused": unused, "shared": shared}


def load_legacy_files(model, paths: Sequence[str], device=None) -> Dict[str, Any]:
    """Load parts from one or more flat weight files (each with ``<file>.json`` beside it, whose "members" list names
    its members). Shared parts come from each file in turn; members are matched by name."""
    got, unused, shared_seen = [], [], set()
    for path in paths:
        state = torch.load(path, map_location=device or "cpu")
        with open(path[:-3] + ".json", encoding="utf-8") as fh:
            names = json.load(fh)["members"]
        r = load_legacy_state(model, state, names)
        got += r["members_loaded"]
        unused += r["members_unused"]
        shared_seen.update(r["shared"])
    stage_new = bool(model.stage_units) and not any(k.startswith("stage_blocks.") for k in shared_seen)
    return {"files": list(paths), "members_loaded": got, "members_new": [m.name for m in model.members if m.name not in got],
            "members_unused": unused, "stage_new": bool(stage_new)}


# ------------------------------------------------------------------ names
def describe(a: Dict[str, Any]) -> str:
    """The run name an argument set gives (used in output file names)."""
    if a["arch"] == "plain":
        return (f"plain{a['plain_depth']}" + ("_anchor" if a["anchor"] else "") + (f"_ain{a['anchor_in']}" if a["anchor_in"] else "")
                + ("" if a["body_code"] == "whole" else f"_body{a['body_code']}"))
    s = f"b{a['bands']}{'k' if a['keep'] and a['bands'] > 1 else 'u'}_{a['codes']}_st{a['stages']}"
    s += (f"soft{a['soft']:g}" if a["soft"] > 0 else "") + ("_anchor" if a["anchor"] else "") + ("" if a["body_out"] else "_nobodyhead")
    s += (f"_bd{a['body_depth']}" if a["body_depth"] != 1 else "") + (f"_md{a['member_depth']}" if a["member_depth"] != 1 else "")
    s += ("_x" + a["extra"].replace(",", "+").replace("@", "at").replace("#", "t") if a["extra"] else "")
    s += (f"_train-{a['train'].replace(',', '+')}" if a["train"] != "all" else "")
    s += ("_c" + a["content"].replace(":", "g").replace(";", "+") if a["content"] else "") + ("" if a["equalize"] else "_noeq")
    s += ("" if a["body_code"] == "whole" else f"_body{a['body_code']}") + ("_from" if a["init"] else "")
    s += ("" if a["base"] else "_nobase") + (f"_mdrop{a['mdrop']:g}" if a["mdrop"] > 0 else "") + (f"_sb{a['stage_body']}" if a["stage_body"] else "")
    s += ("_xdeep" if a["xdeep"] else "") + (f"_ain{a['anchor_in']}" if a["anchor_in"] else "") + ("_sz" if a["stage_zero"] else "")
    s += f"_ssoft{a['stage_soft']:g}" if a["stage_soft"] > 0 else ""
    s += "_tln" if a["t_sampling"] == "logit_normal" else ""
    return s
