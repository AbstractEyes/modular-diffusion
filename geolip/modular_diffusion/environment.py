"""The environment check: the Python and torch builds, whether a card is visible, and every exact pin of a requirements
file against what is installed.

    from geolip.modular_diffusion import environment; environment.check("requirements.txt")
    python -m geolip.modular_diffusion.environment requirements=requirements.txt     # exit code 1 when a pin differs

The numbers the tests compare to the last digit were made with the pinned set; a different torch build or card can move
the last digits of a result (the check says so, it does not refuse).
"""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import sys
from typing import Dict, Optional

from . import __version__
from .config import in_notebook, parse

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*==\s*([^\s;]+)")


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _applies(marker: str) -> bool:
    if not marker:
        return True
    try:
        from packaging.markers import Marker
        return Marker(marker).evaluate()
    except Exception:                                   # an unreadable marker: the pin is checked anyway
        return True


def read_pins(path: str) -> Dict[str, str]:
    """{name: version} of every ``name==version`` line of a requirements file (environment markers applied)."""
    pins = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh.read().splitlines():
            line = raw.split(" #", 1)[0].strip()
            if not line or line.startswith(("#", "-")):
                continue
            spec, _, marker = line.partition(";")
            if not _applies(marker.strip()):
                continue
            m = _PIN.match(spec.strip())
            if m:
                pins[canonical_name(m.group(1))] = m.group(3)
    return pins


def installed_version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check(requirements: Optional[str] = "requirements.txt", verbose: bool = True) -> dict:
    """-> {"package", "python", "platform", "torch", "torch_cuda", "card", "pins": {name: [pinned, installed]},
    "pin_differences": [...], "problems": [...]}. ``requirements``: the requirements file to compare with (skipped when
    it is None or absent)."""
    rep = {"package": __version__, "python": platform.python_version(), "platform": sys.platform, "problems": [],
           "pins": {}, "pin_differences": []}
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["torch_cuda"] = torch.version.cuda
        rep["card"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        rep["torch"] = None
        rep["problems"].append("torch is not installed")
    if requirements and os.path.exists(requirements):
        rep["requirements"] = requirements
        for name, want in read_pins(requirements).items():
            have = installed_version(name)
            rep["pins"][name] = [want, have]
            if have != want:
                rep["pin_differences"].append(f"{name} pinned {want}, installed {have or 'nothing'}")
        rep["problems"] += rep["pin_differences"]
    else:
        rep["requirements"] = None
    if verbose:
        _print(rep)
    return rep


def _print(rep: dict) -> None:
    print(f"modular-diffusion {rep['package']}: environment check")
    card = rep.get("card") or "no card visible"
    print(f"  Python {rep['python']} ({rep['platform']}) | torch {rep.get('torch') or 'missing'} (CUDA {rep.get('torch_cuda')}) | {card}")
    if rep["requirements"]:
        n = len(rep["pin_differences"])
        print(f"  pins of {rep['requirements']}: {len(rep['pins'])} checked, " + ("all match" if not n else f"{n} differ"))
        for name, (want, have) in rep["pins"].items():
            print(f"    {name:<18} pinned {want:<16} installed {have or 'nothing'}" + ("" if want == have else "   DIFFERS"))
    else:
        print("  no requirements file given or found; pins not checked")
    for p in rep["problems"]:
        print(f"  PROBLEM: {p}")
    print("  ready" if not rep["problems"] else f"  {len(rep['problems'])} problem(s)")


DEFAULTS = dict(requirements="requirements.txt")


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    return check(a["requirements"] or None)


if __name__ == "__main__":
    sys.exit(1 if main([] if in_notebook() else sys.argv[1:])["problems"] else 0)
