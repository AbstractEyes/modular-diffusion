"""The report builders (P7): stored reports rebuilt byte for byte from stored result files, and the builders read result
files of both name prefixes.

    MD_STORED_REPORTS   the folder of the stored reports
    MD_STORED_RUNS      the folder of the stored results (one subfolder per job)
    MD_REPORT_CASES     a JSON file naming, per case, the job folders and the stored report to rebuild

The parity cases are skipped when a variable is unset; the last test needs nothing stored.
"""
import json
import os
import shutil

import pytest

from geolip.modular_diffusion import report, train


def _env(name):
    v = os.environ.get(name, "")
    if not v:
        pytest.skip(f"{name} is not set")
    return v


def _cases():
    with open(_env("MD_REPORT_CASES"), encoding="utf-8") as fh:
        return json.load(fh)


def _run(name):
    return os.path.join(_env("MD_STORED_RUNS"), name)


def _stored(name):
    with open(os.path.join(_env("MD_STORED_REPORTS"), name), "rb") as fh:
        return fh.read()


def test_p7_seed_spread_rebuilds_the_stored_report():
    case = _cases()["seed_spread"]
    text = report.seed_spread([_run(n) for n in case["runs"]], title=case["title"])
    assert text.encode("utf-8") == _stored(case["stored"])


def test_p7_check_merge_rebuilds_the_stored_report():
    case = _cases()["merge"]
    text = report.check_merge(_run(case["body"]), [_run(n) for n in case["solos"]], _run(case["joint"]),
                              _run(case["merged"]), _run(case["end_to_end"]))
    assert text.encode("utf-8") == _stored(case["stored"])


def test_p7_check_expansion_rebuilds_the_stored_report(tmp_path):
    case = _cases()["expansion"]
    out = str(tmp_path / "expansion.md")
    text = report.check_expansion(_run(case["base"]), _run(case["one"]), _run(case["map"]), out=out)
    assert text.encode("utf-8") == _stored(case["stored"])
    with open(out, "rb") as fh:
        assert fh.read() == _stored(case["stored"])


def test_p7_grade_builders_run_on_stored_results():
    case = _cases()["grade"]
    g = report.grade("base", [_run(case["base"])])
    assert case["row"] in g and case["numbers"] in g
    m = report.grade_map("map", _run(case["base"]), _run(case["map"]), _run(case["content"]))
    assert all(word in m for word in case["map_words"])


def test_report_reads_both_prefixes_and_skips_build_files(tmp_path):
    out = str(tmp_path / "runs")
    for seed in (0, 1):
        train.main(bed="fields", anchor=1, dim=32, steps=2, batch=8, heldout=32, device="cpu", out=out, seeds=str(seed), sheet=0)
    names = sorted(os.listdir(out))
    assert any(n.endswith(".build.json") for n in names)
    files = report.result_files(out)
    assert [os.path.basename(p)[:3] for p in files] == ["md_", "md_"]
    # the same results under the original prefix read the same
    old = str(tmp_path / "old")
    os.makedirs(old)
    for p in files:
        shutil.copyfile(p, os.path.join(old, "mr_" + os.path.basename(p)[3:]))
    new_text = report.seed_spread([out])
    old_text = report.seed_spread([old])
    assert new_text == old_text and "| seed | 0 | 1 |" in new_text
    assert report.grade("t", [out]) == report.grade("t", [old])            # rows are named by the tag inside the file
