"""The roster runner's additions (test 12): two runners sharing a roster by claims never run a job twice; after_job is
called and its exception does not stop the runner; needs= orders jobs and carries a failure on."""
import os
import threading

from geolip.modular_diffusion import jobs

ALWAYS = lambda: (True, "test guard")


def _script(folder, name, body):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    return path.replace("\\", "/")


def _roster(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def test_12_two_runners_with_claims_never_run_a_job_twice(tmp_path):
    seen = str(tmp_path / "seen").replace("\\", "/")
    os.makedirs(seen)
    # every run of the job leaves its own file <tag>.<process id> (two processes appending to one file can lose a line)
    job = _script(str(tmp_path), "job.py", "import os, sys, time\nname = sys.argv[1]\n"
                  f"open(os.path.join({seen!r}, name + '.' + str(os.getpid())), 'w').close()\ntime.sleep(0.5)\n")
    tags = [f"j{i}" for i in range(6)]
    roster = _roster(str(tmp_path / "roster.txt"), [f"{t} | 1 | {job} | {t}" for t in tags])
    out = str(tmp_path / "out")
    results = []

    def runner(k):
        results.append(jobs.run_roster(roster, out, str(tmp_path / "STOP"), str(tmp_path / f"log{k}.txt"), guard=ALWAYS,
                                       poll_s=0.05, wait_s=0.05, claim=True))
    threads = [threading.Thread(target=runner, args=(k,)) for k in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    ran = [n.rsplit(".", 1)[0] for n in os.listdir(seen)]
    assert sorted(ran) == sorted(tags)                                  # each job exactly once
    assert sorted(results[0]["done"] + results[1]["done"]) == sorted(tags)
    assert all(os.path.exists(os.path.join(out, t, "DONE")) and os.path.exists(os.path.join(out, t, "CLAIM")) for t in tags)


def test_12_after_job_and_needs(tmp_path):
    order = str(tmp_path / "order.txt").replace("\\", "/")
    ok = _script(str(tmp_path), "ok.py", f"import sys\nopen({order!r}, 'a').write(sys.argv[1] + '\\n')\n")
    bad = _script(str(tmp_path), "bad.py", "import sys\nsys.exit(4)\n")
    roster = _roster(str(tmp_path / "roster.txt"), [
        f"late | 1 | {ok} | late | needs=early",
        f"early | 1 | {ok} | early",
        f"after_bad | 1 | {ok} | after_bad | needs=broken",
        f"chain | 1 | {ok} | chain | needs=after_bad,early",
        f"broken | 1 | {bad} | x=1",
        f"ghost | 1 | {ok} | ghost | needs=missing"])
    calls = []

    def after(tag, folder, good):
        calls.append((tag, good, os.path.basename(folder)))
        if tag == "early":
            raise RuntimeError("after_job fails here")

    out = str(tmp_path / "out")
    r = jobs.run_roster(roster, out, str(tmp_path / "STOP"), str(tmp_path / "log.txt"), guard=ALWAYS, poll_s=0.05,
                        wait_s=0.05, after_job=after)
    with open(order, encoding="utf-8") as fh:
        assert fh.read().split() == ["early", "late"]
    assert r["done"] == ["early", "late"]
    assert sorted(r["failed"]) == ["after_bad", "broken", "chain", "ghost"]
    with open(os.path.join(out, "after_bad", "FAILED"), encoding="utf-8") as fh:
        assert "needs broken FAILED" in fh.read()
    with open(os.path.join(out, "chain", "FAILED"), encoding="utf-8") as fh:
        assert "after_bad FAILED" in fh.read()
    with open(os.path.join(out, "ghost", "FAILED"), encoding="utf-8") as fh:
        assert "missing is not in the roster" in fh.read()
    assert ("early", True, "early") in calls and ("late", True, "late") in calls and ("broken", False, "broken") in calls
    assert {t for t, _, _ in calls} == {"early", "late", "broken", "after_bad", "chain", "ghost"}
    with open(tmp_path / "log.txt", encoding="utf-8") as fh:
        assert "after_job for early raised" in fh.read()
    assert jobs.read_roster(roster)[0][0] == ("late", 1.0, ok, ["late"])        # four fields, as before
    assert jobs.parse_roster(roster)[2]["chain"] == ["after_bad", "early"]
