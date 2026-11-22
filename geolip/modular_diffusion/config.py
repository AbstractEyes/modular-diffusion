"""Key=value configuration, read the same way from a command line and from Python keywords.

    parse(DEFAULTS, ["steps=100", "device=cpu"])       command-line items
    parse(DEFAULTS, steps=100, device="cpu")           keywords (a notebook or a library call)

Every key must already exist in the defaults; its type is the default's type. A library call never reads sys.argv:
only a module's ``if __name__ == "__main__"`` line passes it in.
"""
from __future__ import annotations

import sys
from typing import Any, Iterable, Mapping


def in_notebook() -> bool:
    """True inside a running IPython kernel (a notebook cell), where sys.argv holds the kernel's own arguments."""
    ip = sys.modules.get("IPython")
    if ip is None:
        return False
    get = getattr(ip, "get_ipython", None)
    return bool(get is not None and get() is not None)


def coerce(default: Any, value: Any) -> Any:
    """``value`` converted to the type of ``default``. Strings from a command line convert as ``type(default)(value)``."""
    if isinstance(default, str):
        return str(value)
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        if isinstance(value, str):
            return int(value)
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"expected an integer, got {value!r}")
            return int(value)
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def parse(defaults: Mapping[str, Any], argv: Iterable[str] = (), **overrides: Any) -> dict:
    """defaults, then ``key=value`` items, then keywords. An unknown key raises ValueError; items without '=' are ignored."""
    a = dict(defaults)
    for item in argv:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        _set(a, k, v)
    for k, v in overrides.items():
        _set(a, k, v)
    return a


def _set(a: dict, key: str, value: Any) -> None:
    if key not in a:
        raise ValueError(f"unknown argument {key!r}; known: {sorted(a)}")
    a[key] = coerce(a[key], value)
