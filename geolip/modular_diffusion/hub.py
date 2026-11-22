"""Upload a folder to the Hugging Face Hub, with retries. The token comes from the environment variable HF_TOKEN only and
is never printed: any message that would carry it has it replaced by <token>.

    from geolip.modular_diffusion import hub
    hub.ship("runs/base", "user/repo", "base", "the base runs")
    python -m geolip.modular_diffusion.hub folder=runs/base repo=user/repo path_in_repo=base message="the base runs"

A fault of the Hub is retried with a doubling wait (15 s up to 10 min); ``ship`` never raises for it, and returns the
commit id, or None when every attempt failed.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Optional, Sequence

from .config import in_notebook, parse

IGNORE = ("run.log", "DONE", "FAILED", "STOP", "*.tmp", "__pycache__/*")


def token() -> str:
    """The Hub token from HF_TOKEN ("" when it is not set)."""
    return os.environ.get("HF_TOKEN", "")


def _hide(text: str, tok: str) -> str:
    return text.replace(tok, "<token>") if tok else text


def ship(folder: str, repo: str, path_in_repo: str, message: str, attempts: int = 8, repo_type: str = "model",
         ignore_patterns: Sequence[str] = IGNORE, api=None, wait_s: float = 15.0, max_wait_s: float = 600.0,
         sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = print) -> Optional[str]:
    """Upload ``folder`` to ``repo`` under ``path_in_repo``. -> the commit id, or None. ``api`` replaces the Hub client
    (by default ``huggingface_hub.HfApi(token=token())``, imported here)."""
    tok = token()
    if api is None:
        if not tok:
            say("NOT SHIPPED: HF_TOKEN is not set")
            return None
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
    n = sum(len(fs) for _, _, fs in os.walk(folder))
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(folder) for f in fs)
    say(f"shipping {n} files, {size / 1e6:.1f} MB, from {folder} to {repo}/{path_in_repo}")
    wait = wait_s
    for attempt in range(1, attempts + 1):
        try:
            info = api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=folder, path_in_repo=path_in_repo,
                                     commit_message=message, ignore_patterns=list(ignore_patterns))
            oid = getattr(info, "oid", None) or str(info)
            say(f"SHIPPED {_hide(str(oid), tok)}")
            return str(oid)
        except Exception as e:                       # a failed upload is a line, never an exception
            text = _hide(repr(e), tok)
            say(f"  try {attempt} of {attempts} failed: {text[:400]}")
            if attempt == attempts:
                say(f"NOT SHIPPED {text[:200]}")
                return None
            sleep(wait)
            wait = min(wait * 2, max_wait_s)
    return None


DEFAULTS = dict(folder="", repo="", path_in_repo="", message="", attempts=8, repo_type="model")


def main(argv=(), **overrides) -> Optional[str]:
    a = parse(DEFAULTS, argv, **overrides)
    for k in ("folder", "repo", "message"):
        if not a[k]:
            raise ValueError(f"{k}= is required")
    return ship(a["folder"], a["repo"], a["path_in_repo"], a["message"], attempts=int(a["attempts"]), repo_type=a["repo_type"])


if __name__ == "__main__":
    main([] if in_notebook() else sys.argv[1:])
