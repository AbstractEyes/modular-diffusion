"""Builds of the stage forms: one cluster (anchor, stem, an optional whole-range body, head), and one model whose
windowed bodies (tails) are mixed by the stage weights before one head.

    cluster_build(field_shape, stem_depth=7)                 anchor, stem, head (the plain build of the flat arguments)
    cluster_build(field_shape, stem_depth=4, body_depth=3)   anchor, stem, body, head
    mixed_tails_build(field_shape, stem_depth=4, tail_depth=3, cuts=(0, 0.25, 0.65, 1), soft=0.06)
                                                             anchor, stem, body_0 .. body_{S-1}, head
    adopt(model, folder, ["stem"])                           the named units' weights from a build folder
    add_class_experts(build, [[0], [1]], window=(0, 0.25))   one ContentExpert per class set (expert_c0, expert_c1)
    add_band_members(build, (0, 1, 2))                       one band-kept BandMember per band (member_b0 ..)
    with_class_experts(folder, seed, [[9]])                  a finished build folder + newborn class experts
    add_class_experts(build, [[0], [1]], gate="symbol")      parts gated by a random symbol (expert_s0, expert_s1)
    add_class_experts(build, [[0]], depth=0)                 a part with no block: a class head on the trunk's states
    with_parts(folder, seed, [dict(classes=..., ...), ...])  several groups of newborn parts at once

Every unit keeps a fixed name (anchor, stem, body, head), so the units of different clusters are interchangeable.
``cluster_build(stem_depth=k)`` gives exactly the unit entries and geometry of the flat arguments ``arch=plain
plain_depth=k anchor=1 anchor_in=1`` (``compat.build_from_args``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .build import Build, UnitEntry, canonical_json, read_build, read_card, read_weights, state_hash, unit_files
from .compat import TIME_FREQ, num
from .rotary import WHOLE
from .units import NULL_CLASS


def geometry(field_shape: Sequence[int], grid: int = 16, dim: int = 128, heads: int = 2, bands: int = 1) -> Dict[str, Any]:
    """The build geometry: the same keys and values ``compat.build_from_args`` writes."""
    c, hh, ww = (int(x) for x in field_shape)
    patch = hh // int(grid)
    return {"field": [c, hh, ww], "grid": int(grid), "patch": patch, "token_dim": c * patch * patch, "dim": int(dim),
            "heads": int(heads), "time_freq": TIME_FREQ, "classes": NULL_CLASS + 1, "bands": int(bands)}


def _edge(x) -> str:
    """A window edge as stored: a string is kept as written, a number becomes its exact decimal text."""
    return x if isinstance(x, str) else num(x)


def _anchor_stem(stem_depth: int, anchor_out: bool, anchor_in: int, anchor_symbols: int = 0,
                 class_window: Optional[Sequence] = None):
    aspec = {"output": bool(anchor_out)}
    if int(anchor_symbols) > 0:
        aspec["symbols"] = int(anchor_symbols)
    sspec = {"depth": int(stem_depth), "anchor_in": int(anchor_in), "code": list(WHOLE)}
    if class_window is not None:
        lo, hi = class_window
        sspec["class_window"] = [_edge(lo), _edge(hi)]
    return [UnitEntry("anchor", "Anchor", {"site": "anchor"}, aspec),
            UnitEntry("stem", "Stem", {"site": "trunk", "slot": 0}, sspec)]


def _head():
    return UnitEntry("head", "Head", {"site": "out", "band": "all"}, {})


def cluster_build(field_shape: Sequence[int], *, stem_depth: int, body_depth: int = 0, dim: int = 128, heads: int = 2,
                  grid: int = 16, anchor_out: bool = True, anchor_in: int = 1, name: str = "cluster",
                  body_stream: Optional[str] = None, anchor_symbols: int = 0, class_window: Optional[Sequence] = None,
                  bands: int = 1) -> Build:
    """anchor, stem (``stem_depth`` blocks), a whole-range body of ``body_depth`` blocks (none when 0), head.

    anchor_symbols K > 0: the per-class anchor (spec "symbols"; born from ``Model.born(..., class_stats=)``).
    class_window (lo, hi): the stem reads the class only where lo <= t < hi (strings are stored as written).
    bands: the geometry's band count (3 for band-kept members). The defaults give the build of earlier versions."""
    units = _anchor_stem(stem_depth, anchor_out, anchor_in, anchor_symbols, class_window)
    if int(body_depth) > 0:
        units.append(UnitEntry("body", "Body", {"site": "trunk", "slot": 1}, {"depth": int(body_depth)}, stream=body_stream))
    units.append(_head())
    build = Build(name=name, geometry=geometry(field_shape, grid, dim, heads, bands), units=units,
                  info={"stage_edges": []})
    build.validate()
    return build


def expert_name(classes: Sequence[int], prefix: str = "expert") -> str:
    """expert_c3 for [3], expert_c0-1 for [0, 1]."""
    return f"{prefix}_c" + "-".join(str(int(c)) for c in classes)


def symbol_name(symbols: Sequence[int], prefix: str = "expert") -> str:
    """expert_s3 for the symbol [3] (a symbol-gated part), expert_s0-1 for [0, 1]."""
    return f"{prefix}_s" + "-".join(str(int(c)) for c in symbols)


def add_class_experts(build: Build, classes: Sequence[Sequence[int]], window=(0.0, 1.0), soft: float = 0.0,
                      depth: int = 1, prefix: str = "expert", gate: str = "class") -> List[str]:
    """Append one ContentExpert per class set to ``build`` (in place) and return their names. Each: site "out", band
    "all", the window and the crossfade half-width, the class list; spec depth, the whole position code, keep False;
    its random stream is its own name.

    depth 0: a part with no block, the head's own reading of the trunk's last states (a norm and one zero-born linear
    layer to the token values). gate "symbol": each list is a list of symbols, not classes (spec "gate": "symbol";
    named <prefix>_s<symbols>); gate "class" writes the entry of earlier versions (no "gate" key)."""
    if gate not in ("class", "symbol"):
        raise ValueError(f"gate is class or symbol, not {gate!r}")
    names = []
    lo, hi = window
    for cs in classes:
        cs = [int(c) for c in cs]
        if not cs:
            raise ValueError("an expert needs at least one class")
        n = expert_name(cs, prefix) if gate == "class" else symbol_name(cs, prefix)
        spec = {"depth": int(depth), "code": list(WHOLE), "keep": False}
        if gate == "symbol":
            spec["gate"] = "symbol"
        build.units.append(UnitEntry(n, "ContentExpert",
                                     {"site": "out", "band": "all", "window": [_edge(lo), _edge(hi)], "soft": _edge(soft),
                                      "classes": cs}, spec))
        names.append(n)
    build.validate()
    return names


def add_band_members(build: Build, bands: Sequence[int] = (0, 1, 2), depth: int = 1, window=(0.0, 1.0),
                     soft: float = 0.0, keep: bool = True, prefix: str = "member") -> List[str]:
    """Append one zero-born BandMember per band (named member_b0, ...) to ``build`` (in place); returns their names.
    Each: site "out", the band, the window and half-width, no class set; spec depth, the whole position code, keep. A
    band-kept member needs a geometry of 3 bands."""
    names = []
    lo, hi = window
    for b in bands:
        n = f"{prefix}_b{int(b)}"
        build.units.append(UnitEntry(n, "BandMember",
                                     {"site": "out", "band": int(b), "window": [_edge(lo), _edge(hi)], "soft": _edge(soft),
                                      "classes": None},
                                     {"depth": int(depth), "code": list(WHOLE), "keep": bool(keep)}))
        names.append(n)
    build.validate()
    return names


def with_class_experts(folder: str, seed: int, classes: Sequence[Sequence[int]], window=(0.0, 1.0), soft: float = 0.0,
                       depth: int = 1, prefix: str = "expert", device="cpu", gate: str = "class"):
    """A finished build folder as a model with newborn class experts added -> (model, expert names).

    The folder is linked strictly (every unit file checked against its card's content hash); the folder itself is
    not changed. The experts are born from (seed, their name), zero-born, so the new model's velocity equals the
    source's exactly (checked on the linker's probe batch). Export the trained model with ``base=folder`` and
    ``trained=<expert names>``; removing the experts (``without_units``) gives back the source model."""
    return with_parts(folder, seed, [dict(classes=classes, window=window, soft=soft, depth=depth, prefix=prefix,
                                          gate=gate)], device=device)


def with_parts(folder: str, seed: int, groups: Sequence[Dict[str, Any]], device="cpu"):
    """A finished build folder with several groups of newborn parts -> (model, part names): each group is the keyword
    arguments of ``add_class_experts`` (classes, window, soft, depth, prefix, gate), appended in order. As
    ``with_class_experts`` otherwise (the newborn parts change no digit of the source's output)."""
    from .linker import born_gap, link
    from .model import Model
    from .units import born
    src = link(folder, device="cpu", strict=True)
    b = src.build
    new = Build(name=b.name, geometry=dict(b.geometry), units=[UnitEntry.from_dict(e.to_dict()) for e in b.units],
                info=dict(b.info))
    names = []
    for g in groups:
        names += add_class_experts(new, **g)
    units = list(src.units)
    for n in names:
        e = new.entry(n)
        units.append(born(e.type, seed, e.name, e.address, e.spec, new.geometry, stream=e.stream, index=e.index))
    model = Model(new, units)
    gap = born_gap(model, src)
    if gap != 0.0:
        raise AssertionError(f"the newborn experts change the source model's output: {gap}")
    return model.to(device), names


def without_units(model, names: Iterable[str]):
    """The model with the named units left out (the same unit objects otherwise; absence, never a zero weight)."""
    from .model import Model
    drop = set(names)
    b = model.build
    nb = Build(name=b.name, geometry=dict(b.geometry), units=[e for e in b.units if e.name not in drop], info=dict(b.info))
    return Model(nb, [u for u in model.units if u.name not in drop])


def mixed_tails_build(field_shape: Sequence[int], *, stem_depth: int, tail_depth: int, cuts: Sequence[float], soft: float,
                      dim: int = 128, heads: int = 2, grid: int = 16, anchor_out: bool = True, anchor_in: int = 1,
                      name: str = "mixed_tails", stream: str = "body") -> Build:
    """anchor, stem, one windowed body per stage (body_s acts on [cuts[s], cuts[s + 1]) with the crossfade half-width
    ``soft``; every body is drawn from the same stream, so they are born equal), head."""
    cuts = [float(c) for c in cuts]
    if cuts[0] != 0.0 or cuts[-1] != 1.0 or any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"cuts must rise from 0 to 1: {cuts}")
    units = _anchor_stem(stem_depth, anchor_out, anchor_in)
    for s in range(len(cuts) - 1):
        units.append(UnitEntry(f"body_{s}", "Body",
                               {"site": "trunk", "slot": 1, "window": [num(cuts[s]), num(cuts[s + 1])], "soft": num(soft)},
                               {"depth": int(tail_depth)}, stream=stream, index=0))
    units.append(_head())
    build = Build(name=name, geometry=geometry(field_shape, grid, dim, heads), units=units,
                  info={"stage_edges": cuts[1:-1]})
    build.validate()
    return build


def adopt(model, folder: str, names: Iterable[str]) -> None:
    """Load the named units' weights from a build folder into ``model`` (in place). Each weight file's content hash is
    checked against its card (and its build entry); the unit's type, address and spec must equal the model's."""
    b = read_build(folder)
    for name in names:
        entry = b.entry(name)
        target = model.unit(name)
        if (entry.type, canonical_json([entry.address, entry.spec])) != (target.unit_type, canonical_json([target.address, target.spec])):
            raise ValueError(f"unit {name!r} of {folder} is not the model's unit (type, address or spec differ)")
        card = read_card(unit_files(b.folder, name)[1])
        wpath = os.path.join(b.folder, entry.file) if entry.file else unit_files(b.folder, name)[0]
        state = read_weights(wpath)
        h = state_hash(entry.type, entry.address, entry.spec, state)
        if h != card.hash or (entry.hash is not None and h != entry.hash):
            raise ValueError(f"unit {name!r} of {folder}: the weight file's content hash differs from its card")
        dev = next((x.device for x in list(target.parameters()) + list(target.buffers())), torch.device("cpu"))
        with torch.no_grad():
            target.load_owned({k: v.to(dev) for k, v in state.items()})
