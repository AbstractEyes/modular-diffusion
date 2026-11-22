"""The roster runner (test 15), the Hub upload (test 16) and the environment check."""
import os
import sys
import time
import types

import pytest

from geolip.modular_diffusion import environment, hub, jobs

ALWAYS = lambda: (True, "test guard")
NEVER = lambda: (False, "test guard says no")


def _script(folder, name, body):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    return path.replace("\\", "/")


def _roster(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ test 15
def test_15_stop_file_means_no_new_job(tmp_path):
    ok = _script(str(tmp_path), "ok.py", "print('hello')\n")
    roster = _roster(str(tmp_path / "roster.txt"), [f"j1 | 1 | {ok} | a=1"])
    stop = str(tmp_path / "STOP")
    open(stop, "w").close()
    out = str(tmp_path / "out")
    r = jobs.run_roster(roster, out, stop, str(tmp_path / "log.txt"), guard=ALWAYS)
    assert r["ended"] == "the stop file is present" and r["done"] == [] and r["failed"] == []
    assert not os.path.exists(os.path.join(out, "j1"))


def test_15_guard_false_means_no_start(tmp_path):
    ok = _script(str(tmp_path), "ok.py", "print('hello')\n")
    roster = _roster(str(tmp_path / "roster.txt"), [f"j1 | 1 | {ok} | a=1"])
    out = str(tmp_path / "out")
    t0 = time.time()
    r = jobs.run_roster(roster, out, str(tmp_path / "STOP"), str(tmp_path / "log.txt"), guard=NEVER,
                        max_hours=1.0 / 3600, wait_s=0.1)
    assert time.time() - t0 < 30
    assert r["ended"] == "the runner's time is up" and not os.path.exists(os.path.join(out, "j1"))
    with open(tmp_path / "log.txt", encoding="utf-8") as fh:
        assert "test guard says no" in fh.read()


def test_15_markers_and_the_wall_cap(tmp_path):
    ok = _script(str(tmp_path), "ok.py", "import sys\nprint('args', sys.argv[1:])\n")
    bad = _script(str(tmp_path), "bad.py", "import sys\nsys.exit(3)\n")
    slow = _script(str(tmp_path), "slow.py", "import time\ntime.sleep(120)\n")
    roster = _roster(str(tmp_path / "roster.txt"), [
        "# a comment", f"j_ok | 1 | {ok} | a=1 b=2", f"j_bad | 1 | {bad} | x=1", f"j_slow | 0.03 | {slow} | y=1",
        "END", f"j_after | 1 | {ok} | never=1"])
    out = str(tmp_path / "out")
    t0 = time.time()
    r = jobs.run_roster(roster, out, str(tmp_path / "STOP"), str(tmp_path / "log.txt"), guard=ALWAYS, poll_s=0.1,
                        end_at_cap=True)
    assert time.time() - t0 < 60
    assert r["done"] == ["j_ok"] and r["failed"] == ["j_bad", "j_slow"] and r["capped"] == ["j_slow"]
    assert r["ended"] == "END line reached"
    assert os.path.exists(os.path.join(out, "j_ok", "DONE"))
    with open(os.path.join(out, "j_ok", "run.log"), encoding="utf-8") as fh:
        assert "'a=1', 'b=2', 'out=" in fh.read()
    with open(os.path.join(out, "j_bad", "FAILED"), encoding="utf-8") as fh:
        assert "rc=3" in fh.read()
    with open(os.path.join(out, "j_slow", "FAILED"), encoding="utf-8") as fh:
        assert "wall cap" in fh.read()
    assert not os.path.exists(os.path.join(out, "j_after"))
    # a second pass skips the marked jobs
    r2 = jobs.run_roster(roster, out, str(tmp_path / "STOP"), str(tmp_path / "log.txt"), guard=ALWAYS, poll_s=0.1)
    assert r2["done"] == [] and r2["failed"] == []


def test_15_default_guard_parts(monkeypatch):
    def smi(used, apps):
        return lambda args: used if "--query-gpu=memory.used" in args else apps
    assert jobs.card_room(14000, smi("1200\n", ""))[0] is True
    assert jobs.card_room(14000, smi("15000\n", ""))[0] is False
    assert jobs.card_room(14000, smi("1200\n", "python.exe\n"))[0] is False
    assert jobs.card_room(14000, smi("ERR FileNotFoundError", ""))[0] is False
    fake = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=20e9))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert jobs.ram_room(8.0)[0] is True and jobs.ram_room(32.0)[0] is False
    monkeypatch.setitem(sys.modules, "psutil", None)                      # not installed
    ok, why = jobs.ram_room(8.0)
    assert ok is False and "psutil" in why
    assert jobs.default_guard(None, None)() == (True, "no guard")


# ------------------------------------------------------------------ test 16
class _FakeApi:
    def __init__(self, fail_times, secret):
        self.calls, self.fail_times, self.secret = 0, fail_times, secret

    def upload_folder(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"server said no to Bearer {self.secret}")
        return types.SimpleNamespace(oid="abc123")


def test_16_token_only_from_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert hub.token() == "hf_from_env"
    monkeypatch.delenv("HF_TOKEN")
    assert hub.token() == ""
    said = []
    assert hub.ship(".", "user/repo", "x", "m", say=said.append) is None       # no token and no client: nothing tried
    assert said == ["NOT SHIPPED: HF_TOKEN is not set"]


def test_16_retries_and_the_token_never_appears(monkeypatch, tmp_path, capsys):
    secret = "hf_SECRET_value_123"
    monkeypatch.setenv("HF_TOKEN", secret)
    (tmp_path / "a.txt").write_text("x")
    waits, said = [], []
    api = _FakeApi(2, secret)
    oid = hub.ship(str(tmp_path), "user/repo", "runs", "msg", api=api, sleep=waits.append, say=said.append)
    assert oid == "abc123" and api.calls == 3 and waits == [15.0, 30.0]
    assert said[-1] == "SHIPPED abc123"
    assert all(secret not in s for s in said) and any("<token>" in s for s in said)
    waits, said = [], []
    api = _FakeApi(99, secret)
    assert hub.ship(str(tmp_path), "user/repo", "runs", "msg", attempts=4, api=api, sleep=waits.append, say=said.append) is None
    assert api.calls == 4 and waits == [15.0, 30.0, 60.0] and said[-1].startswith("NOT SHIPPED")
    assert all(secret not in s for s in said)
    hub.ship(str(tmp_path), "user/repo", "runs", "msg", attempts=2, api=_FakeApi(99, secret), sleep=lambda s: None)
    assert secret not in capsys.readouterr().out


# ------------------------------------------------------------------ the environment check
def test_environment_check_against_pins(tmp_path, capsys):
    req = tmp_path / "requirements.txt"
    req.write_text("--extra-index-url https://example.invalid/whl\n# comment\nnumpy==0.0.1\nsafetensors==%s  # pinned\n"
                   "not-a-real-package-xyz==1.0\nplain-floor>=1\n" % environment.installed_version("safetensors"),
                   encoding="utf-8")
    rep = environment.check(str(req))
    assert set(rep["pins"]) == {"numpy", "safetensors", "not-a-real-package-xyz"}
    assert rep["pins"]["safetensors"][0] == rep["pins"]["safetensors"][1]
    assert len(rep["pin_differences"]) == 2 and rep["problems"] == rep["pin_differences"]
    out = capsys.readouterr().out
    assert "DIFFERS" in out and "2 problem(s)" in out
    assert environment.check(None, verbose=False)["requirements"] is None
