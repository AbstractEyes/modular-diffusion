"""The build: a model's geometry and its units in composition order, stored as JSON.

    {"format": "modular-diffusion/build/1", "name": ...,
     "geometry": {"field": [C, H, W], "grid", "patch", "token_dim", "dim", "heads", "time_freq", "classes", "bands"},
     "units": [{"name", "type", "address", "spec", "file", "hash"}, ...],
     "info": {...}}                        descriptive fields only (never read to build the model)

Windows are stored as decimal strings and read with float(), so a stored edge never drifts by re-rendering.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BUILD_FORMAT = "modular-diffusion/build/1"


@dataclass
class UnitEntry:
    name: str
    type: str
    address: Dict[str, Any] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    stream: Optional[str] = None          # the random stream of its birth (default: its name)
    index: int = 0                        # its position in a shared stream
    file: Optional[str] = None
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type, "address": self.address, "spec": self.spec,
                "stream": self.stream, "index": self.index, "file": self.file, "hash": self.hash}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnitEntry":
        return cls(name=d["name"], type=d["type"], address=dict(d.get("address", {})), spec=dict(d.get("spec", {})),
                   stream=d.get("stream"), index=int(d.get("index", 0)), file=d.get("file"), hash=d.get("hash"))


@dataclass
class Build:
    name: str
    geometry: Dict[str, Any]
    units: List[UnitEntry] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    format: str = BUILD_FORMAT

    def names(self) -> List[str]:
        return [u.name for u in self.units]

    def entry(self, name: str) -> UnitEntry:
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(name)

    def validate(self) -> None:
        names = self.names()
        if len(set(names)) != len(names):
            raise ValueError(f"two units share a name: {names}")

    def to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "name": self.name, "geometry": self.geometry,
                "units": [u.to_dict() for u in self.units], "info": self.info}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Build":
        if d.get("format") != BUILD_FORMAT:
            raise ValueError(f"not a build file of format {BUILD_FORMAT!r}: {d.get('format')!r}")
        return cls(name=d["name"], geometry=dict(d["geometry"]), units=[UnitEntry.from_dict(u) for u in d["units"]],
                   info=dict(d.get("info", {})), format=d["format"])


def write_build(build: Build, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(build.to_dict(), fh, indent=1, sort_keys=False)
        fh.write("\n")


def read_build(path: str) -> Build:
    with open(path, encoding="utf-8") as fh:
        return Build.from_dict(json.load(fh))
