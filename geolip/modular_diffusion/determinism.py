"""The determinism switch of a job (the ``det=`` argument of every experiment).

    det=        nothing changes (the default)
    det=warn    torch.use_deterministic_algorithms(True, warn_only=True) for the job: nothing is computed differently,
                but every operation that is not deterministic on the device raises a warning; the warnings are
                collected, and the result gains "nondeterministic": their sorted unique messages
    det=on      torch.use_deterministic_algorithms(True) for the job: every operation takes its deterministic
                algorithm (one that has none raises an error), and CUBLAS_WORKSPACE_CONFIG=:4096:8 is put into the
                process environment first (the matrix library reads it when it first touches the card, so it must be
                set before that: ``run.main`` sets it before the experiment module is imported, and ``mode`` sets it
                again when it is still absent)

Why it matters on a card: the backward pass of the memory-efficient attention kernel is not deterministic by default
(det=warn names it), so two runs of one job can differ in the last digits of every weight; det=on selects its
deterministic algorithm. On the CPU both modes change no number.

    with determinism.mode(a["det"]) as det:
        ...
        det.record(result)            # adds "nondeterministic" when the switch is on

The previous setting of the deterministic flag and the warning filters are put back when the job ends.
"""
from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Dict, Iterator, List

import torch

MODES = ("", "warn", "on")
WORKSPACE = ":4096:8"
WORKSPACE_KEY = "CUBLAS_WORKSPACE_CONFIG"


def check(mode: str) -> str:
    mode = str(mode or "")
    if mode not in MODES:
        raise ValueError(f"det must be empty, warn or on, not {mode!r}")
    return mode


def set_workspace() -> Dict[str, Any]:
    """Put the matrix library's fixed workspace into the process environment (when it is not set already) -> what was
    found and whether the card had already been touched."""
    before = os.environ.get(WORKSPACE_KEY)
    if not before:
        os.environ[WORKSPACE_KEY] = WORKSPACE
    initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    return {"found": before, "value": os.environ[WORKSPACE_KEY], "card_touched_before": initialized}


class _State:
    def __init__(self, mode: str):
        self.mode = mode
        self.caught: List[warnings.WarningMessage] = []
        self.workspace: Dict[str, Any] = {}

    def messages(self) -> List[str]:
        return sorted({str(w.message) for w in self.caught})

    def record(self, result: dict) -> None:
        """Add "nondeterministic" (and, for det=on, "workspace") to a result when the switch is on."""
        if self.mode:
            result["nondeterministic"] = self.messages()
            if self.mode == "on":
                result["workspace"] = dict(self.workspace)


@contextlib.contextmanager
def mode(value: str) -> Iterator[_State]:
    """The switch for the body of the ``with`` (see the module text)."""
    value = check(value)
    state = _State(value)
    if not value:
        yield state
        return
    if value == "on":
        state.workspace = set_workspace()
    was = torch.are_deterministic_algorithms_enabled()
    was_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state.caught = caught                       # the live list: ``record`` sees every warning raised so far
        torch.use_deterministic_algorithms(True, warn_only=(value == "warn"))
        try:
            yield state
        finally:
            state.caught = list(caught)
            torch.use_deterministic_algorithms(was, warn_only=was_warn)
