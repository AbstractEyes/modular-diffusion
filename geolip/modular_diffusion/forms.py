"""Builds of the stage forms: one cluster (anchor, stem, an optional whole-range body, head), and one model whose
windowed bodies (tails) are mixed by the stage weights before one head.

    cluster_build(field_shape, stem_depth=7)                 anchor, stem, head (the plain build of the flat arguments)
    cluster_build(field_shape, stem_depth=4, body_depth=3)   anchor, stem, body, head
    mixed_tails_build(field_shape, stem_depth=4, tail_depth=3, cuts=(0, 0.25, 0.65, 1), soft=0.06)
                                                             anchor, stem, body_0 .. body_{S-1}, head
    adopt(model, folder, ["stem"])                           the named units' weights from a build folder

Every unit keeps a fixed name (anchor, stem, body, head), so the units of different clusters are interchangeable.
``cluster_build(stem_depth=k)`` gives exactly the unit entries and geometry of the flat arguments ``arch=plain
plain_depth=k anchor=1 anchor_in=1`` (``compat.build_from_args``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Sequence

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


def _anchor_stem(stem_depth: int, anchor_out: bool, anchor_in: int):
    return [UnitEntry("anchor", "Anchor", {"site": "anchor"}, {"output": bool(anchor_out)}),
            UnitEntry("stem", "Stem", {"site": "trunk", "slot": 0},
                      {"depth": int(stem_depth), "anchor_in": int(anchor_in), "code": list(WHOLE)})]


def _head():
    return UnitEntry("head", "Head", {"site": "out", "band": "all"}, {})


def cluster_build(field_shape: Sequence[int], *, stem_depth: int, body_depth: int = 0, dim: int = 128, heads: int = 2,
                  grid: int = 16, anchor_out: bool = True, anchor_in: int = 1, name: str = "cluster",
                  body_stream: Optional[str] = None) -> Build:
    """anchor, stem (``stem_depth`` blocks), a whole-range body of ``body_depth`` blocks (none when 0), head."""
    units = _anchor_stem(stem_depth, anchor_out, anchor_in)
    if int(body_depth) > 0:
        units.append(UnitEntry("body", "Body", {"site": "trunk", "slot": 1}, {"depth": int(body_depth)}, stream=body_stream))
    units.append(_head())
    build = Build(name=name, geometry=geometry(field_shape, grid, dim, heads), units=units, info={"stage_edges": []})
    build.validate()
    return build


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
