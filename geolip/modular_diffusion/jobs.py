"""A roster runner: one job at a time from a text file, with a stop file, DONE / FAILED markers, a start guard and a wall
cap enforced by this (the parent) process.

Roster lines (re-read every round, so lines can be added while it runs):

    tag | wall cap in minutes | program | arguments [| needs=<tag>[,<tag>]]
          program: a Python module name (run as ``python -m <module>``) or a path to a .py file
          needs: the job waits until each named job holds DONE (and fails when one FAILED)
          the runner appends ``out=<out>/<tag>``; a line starting with # is a comment; a line END ends the roster

Every round: if the stop file exists, no new job starts and the runner ends. Otherwise the first line whose job folder
holds neither DONE nor FAILED runs when the guard allows (by default: the card's memory in use, read from nvidia-smi, at
most ``card_room_mib`` with no other Python process on the card, and at least ``ram_free_gb`` of free memory, read with
psutil). The job's output goes to <out>/<tag>/run.log; when it ends, DONE (exit code 0) or FAILED is written into its
folder with the exit code and the seconds. A job that outlives its wall cap keeps running and the
log notes it; with ``end_at_cap`` the runner ends it (a hard stop) and marks it FAILED.

    from geolip.modular_diffusion import jobs
    jobs.run_roster("roster.txt", out="runs", stop="runs/STOP", log="runs/roster.log")
    python -m geolip.modular_diffusion.jobs roster=roster.txt out=runs stop=runs/STOP log=runs/roster.log
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import in_notebook, parse

Job = Tuple[str, float, str, List[str]]
Guard = Callable[[], Tuple[bool, str]]


def parse_roster(path: str) -> Tuple[List[Job], bool, Dict[str, List[str]]]:
    """-> (jobs, ended, needs): the roster's jobs in order, whether an END line was reached, and {tag: [needed tags]}
    for the lines with a fifth field ``needs=<tag>[,<tag>]``."""
    jobs: List[Job] = []
    needs: Dict[str, List[str]] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return jobs, False, needs
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "END":
            return jobs, True, needs
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 5 and parts[4].startswith("needs="):
            wanted = [x.strip() for x in parts[4][len("needs="):].split(",") if x.strip()]
            parts = parts[:4]
        elif len(parts) != 4:
            continue
        else:
            wanted = []
        try:
            jobs.append((parts[0], float(parts[1]), parts[2], parts[3].split()))
        except ValueError:
            continue
        if wanted:
            needs[parts[0]] = wanted
    return jobs, False, needs


def read_roster(path: str) -> Tuple[List[Job], bool]:
    """-> (jobs, ended): the roster's jobs in order, and whether an END line was reached."""
    jobs, ended, _ = parse_roster(path)
    return jobs, ended


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def nvidia_smi(args: Sequence[str]) -> str:
    try:
        return subprocess.run(["nvidia-smi"] + list(args), capture_output=True, text=True, timeout=30,
                              creationflags=_no_window()).stdout
    except Exception as e:                              # the guard must never crash the runner
        return "ERR " + repr(e)


def card_room(limit_mib: int, smi: Callable[[Sequence[str]], str] = nvidia_smi) -> Tuple[bool, str]:
    """The card holds at most ``limit_mib`` of memory in use and no other Python process."""
    used = smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).strip().split("\n")[0]
    apps = smi(["--query-compute-apps=process_name", "--format=csv,noheader"]).lower()
    try:
        ok = int(used) <= limit_mib
    except ValueError:
        return False, "card unreadable: " + used[:60]
    if "python" in apps:
        return False, "another Python process holds the card"
    return ok, f"card memory in use {used} MiB (limit {limit_mib})"


def ram_room(min_free_gb: float) -> Tuple[bool, str]:
    """At least ``min_free_gb`` of free memory (psutil, imported here)."""
    try:
        import psutil
    except ImportError:
        return False, "psutil is not installed (pip install modular-diffusion[guard]); free memory unknown"
    free = psutil.virtual_memory().available / 1e9
    return free >= min_free_gb, f"free memory {free:.1f} GB (minimum {min_free_gb:g})"


def default_guard(card_room_mib: Optional[int], ram_free_gb: Optional[float]) -> Guard:
    """The start guard: card room (skipped when card_room_mib is None) and free memory (skipped when ram_free_gb is None
    or not above 0)."""
    def guard() -> Tuple[bool, str]:
        notes = []
        if card_room_mib is not None:
            ok, why = card_room(int(card_room_mib))
            notes.append(why)
            if not ok:
                return False, why
        if ram_free_gb is not None and ram_free_gb > 0:
            ok, why = ram_room(float(ram_free_gb))
            notes.append(why)
            if not ok:
                return False, why
        return True, "; ".join(notes) or "no guard"
    return guard


def _command(python: str, program: str, args: Sequence[str], out_dir: str) -> List[str]:
    head = [python, "-u", program] if program.endswith(".py") else [python, "-u", "-m", program]
    return head + list(args) + ["out=" + out_dir.replace("\\", "/")]


def run_job(job: Job, out: str, python: str = sys.executable, env: Optional[Dict[str, str]] = None,
            poll_s: float = 2.0, grace_s: float = 30.0, end_at_cap: bool = False) -> Tuple[int, float, bool]:
    """Run one job in <out>/<tag>, its output to run.log. -> (exit code, seconds, ended by the wall cap).

    A job past its wall cap is left running unless ``end_at_cap`` is set. Ending a job is a hard stop on some systems, so
    it is opt-in: set it only on a machine nobody else is using."""
    tag, cap_min, program, args = job
    d = os.path.join(out, tag)
    os.makedirs(d, exist_ok=True)
    cmd = _command(python, program, args, d)
    t0 = time.time()
    capped = False
    with open(os.path.join(d, "run.log"), "ab") as log:
        p = subprocess.Popen(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=_no_window())
        while p.poll() is None:
            if end_at_cap and time.time() - t0 > cap_min * 60:
                capped = True
                p.terminate()
                try:
                    p.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                break
            time.sleep(poll_s)
    return p.returncode, time.time() - t0, capped


def run_roster(roster: str, out: str, stop: str, log: str, python: str = sys.executable, card_room_mib: Optional[int] = 14000,
               ram_free_gb: Optional[float] = 8.0, max_hours: float = 11.0, env: Optional[Dict[str, str]] = None,
               guard: Optional[Guard] = None, idle_exit: bool = True, poll_s: float = 2.0, wait_s: float = 30.0,
               hide_token: bool = True, end_at_cap: bool = False, claim: bool = False,
               after_job: Optional[Callable[[str, str, bool], None]] = None) -> dict:
    """Run the roster's jobs one at a time until the stop file appears, an END line is reached with nothing pending,
    the roster runs dry (``idle_exit``), or ``max_hours`` pass. Returns {"done", "failed", "capped", "ended"}.

    env: variables added to the runner's own environment for every job (an empty HF_TOKEN is added when
    ``hide_token``, so a job cannot write to the Hub). guard: a callable -> (ok, reason) replacing the default guard.

    claim: a job is taken by creating <out>/<tag>/CLAIM exclusively, and a job whose CLAIM exists is not pending, so
    several runners (processes or threads) can share one roster; the CLAIM of a job whose launch raised is removed.
    after_job(tag, folder, ok): called after a job's marker is written; an exception inside it is logged and the
    runner goes on. needs: a job whose roster line names ``needs=<tag>[,<tag>]`` is pending only when every named job's
    folder holds DONE; when a named job FAILED (or is not in the roster), the job is marked FAILED with that reason."""
    guard = guard or default_guard(card_room_mib, ram_free_gb)
    job_env = dict(os.environ)
    job_env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
    if hide_token:
        job_env["HF_TOKEN"] = ""
    job_env.update(env or {})
    folder = os.path.dirname(log)
    if folder:
        os.makedirs(folder, exist_ok=True)

    def note(msg: str) -> None:
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")

    t0 = time.time()
    result = {"done": [], "failed": [], "capped": [], "ended": ""}
    note(f"roster start (pid {os.getpid()}): {roster}")
    last_note = 0.0
    while True:
        if time.time() - t0 >= max_hours * 3600:
            result["ended"] = "the runner's time is up"
            break
        if os.path.exists(stop):
            result["ended"] = "the stop file is present"
            note("stop file seen; no new job starts")
            break
        jobs, end, needs = parse_roster(roster)
        marked = lambda tag, m: os.path.exists(os.path.join(out, tag, m))
        tags = {j[0] for j in jobs}
        pending = [j for j in jobs if not marked(j[0], "DONE") and not marked(j[0], "FAILED")
                   and not (claim and marked(j[0], "CLAIM"))]
        ready, blocked, newly_failed = [], [], False
        for j in pending:
            wanted = needs.get(j[0], [])
            bad = [w for w in wanted if marked(w, "FAILED") or (w not in tags and not marked(w, "DONE"))]
            if bad:
                reason = ", ".join(f"{w} {'FAILED' if marked(w, 'FAILED') else 'is not in the roster'}" for w in bad)
                d = os.path.join(out, j[0])
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "FAILED"), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + f"not run: needs {reason}\n")
                result["failed"].append(j[0])
                note(f"END {j[0]} FAILED: not run, needs {reason}")
                _after(after_job, j[0], d, False, note)
                newly_failed = True
            elif all(marked(w, "DONE") for w in wanted):
                ready.append(j)
            else:
                blocked.append(j)
        if not ready:
            if blocked:
                time.sleep(wait_s)
                continue
            if newly_failed:
                continue                                   # a failure may reach further jobs: look again
            if end or idle_exit:
                result["ended"] = "END line reached" if end else "nothing pending"
                note(result["ended"])
                break
            time.sleep(wait_s)
            continue
        ok, why = guard()
        if not ok:
            if time.time() - last_note > 300:
                note(f"waiting: {why}; next {ready[0][0]}")
                last_note = time.time()
            time.sleep(wait_s)
            continue
        job = ready[0]
        tag, cap_min, program, args = job
        folder_j = os.path.join(out, tag)
        if claim:
            os.makedirs(folder_j, exist_ok=True)
            try:
                os.close(os.open(os.path.join(folder_j, "CLAIM"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue                                   # another runner took it first
        note(f"START {tag} (cap {cap_min:g} min; {why}; {len(pending) - 1} more pending):{program} {' '.join(args)}")
        try:
            rc, secs, capped = run_job(job, out, python, job_env, poll_s, end_at_cap=end_at_cap)
        except Exception as e:
            rc, secs, capped = -999, 0.0, False
            note(f"launch of {tag} raised {e!r}")
            if claim:
                try:
                    os.remove(os.path.join(folder_j, "CLAIM"))
                except OSError:
                    pass
        mark = "DONE" if rc == 0 and not capped else "FAILED"
        os.makedirs(folder_j, exist_ok=True)
        with open(os.path.join(folder_j, mark), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + f"rc={rc} seconds={secs:.0f}" + (" wall cap" if capped else "") + "\n")
        result["done" if mark == "DONE" else "failed"].append(tag)
        if capped:
            result["capped"].append(tag)
        past = " (ended by the wall cap)" if capped else (" (ran past its wall cap)" if secs > cap_min * 60 else "")
        note(f"END {tag} rc={rc} {mark} after {secs / 60:.1f} min" + past)
        _after(after_job, tag, folder_j, mark == "DONE", note)
    note("roster end: " + result["ended"])
    return result


def _after(after_job, tag: str, folder: str, ok: bool, note: Callable[[str], None]) -> None:
    """Call ``after_job(tag, folder, ok)``; an exception is logged, never raised."""
    if after_job is None:
        return
    try:
        after_job(tag, folder, ok)
    except Exception as e:
        note(f"after_job for {tag} raised {e!r}")


DEFAULTS = dict(roster="", out="", stop="", log="", python=sys.executable, card_room_mib=14000, ram_free_gb=8.0,
                max_hours=11.0, idle_exit=1)


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    for k in ("roster", "out", "stop", "log"):
        if not a[k]:
            raise ValueError(f"{k}= is required")
    return run_roster(a["roster"], a["out"], a["stop"], a["log"], python=a["python"], card_room_mib=int(a["card_room_mib"]),
                      ram_free_gb=float(a["ram_free_gb"]), max_hours=float(a["max_hours"]), idle_exit=bool(a["idle_exit"]))


if __name__ == "__main__":
    print(main([] if in_notebook() else sys.argv[1:]))
