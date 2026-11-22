"""Hygiene (test 17): the package reads clean to an outside engineer and runs anywhere (no module-path tricks, no
command-line parser at import, no registry, no machine paths, no in-house words); the notebook pins a released tag no
newer than the package, carries its Open-in-Colab link, and is stored without outputs."""
import glob
import json
import os
import re

import geolip.modular_diffusion as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "geolip", "modular_diffusion")
NOTEBOOKS = sorted(glob.glob(os.path.join(ROOT, "colab", "*.ipynb")))
COLAB = "https://colab.research.google.com/github/AbstractEyes/modular-diffusion/blob/"

FORBIDDEN = {
    "module path": r"__file__",
    "argparse": r"\bargparse\b",
    "registry": r"\bwinreg\b",
    "drive-letter path": r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]",
    "in-house words": r"(?i)mr_collective|\bdocket|\bnight\b|\bcanon\b|\bstatutes?\b|\bgauges?\b|\barm\b|\bPhil\b",
}


def _sources():
    files = sorted(glob.glob(os.path.join(PKG, "*.py")))
    assert len(files) >= 20, files
    return [(p, open(p, encoding="utf-8").read()) for p in files]


def _notebooks():
    assert NOTEBOOKS, "no notebook under colab/"
    return [(p, json.load(open(p, encoding="utf-8"))) for p in NOTEBOOKS]


def test_17_package_sources_are_clean():
    for path, text in _sources():
        for what, pattern in FORBIDDEN.items():
            m = re.search(pattern, text)
            assert m is None, (os.path.basename(path), what, text[max(0, m.start() - 40):m.end() + 40])


def test_17_notebook_text_is_clean():
    for path, nb in _notebooks():
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        for what in ("module path", "argparse", "registry", "drive-letter path", "in-house words"):
            assert re.search(FORBIDDEN[what], text) is None, (path, what)


def test_17_notebook_pins_a_tag_no_newer_than_the_package_and_links_itself():
    current = tuple(int(x) for x in md.__version__.split("."))
    for path, nb in _notebooks():
        head = "".join(nb["cells"][0]["source"])
        assert nb["cells"][0]["cell_type"] == "markdown"
        link = re.search(re.escape(COLAB) + r"(v\d+\.\d+\.\d+)/colab/" + re.escape(os.path.basename(path)), head)
        assert link, path
        c1 = "".join(nb["cells"][1]["source"])
        pin = re.search(r"modular-diffusion@v(\d+)\.(\d+)\.(\d+)", c1)
        assert pin, path
        pinned = tuple(int(x) for x in pin.groups())
        assert pinned <= current, (path, pinned, current)
        assert link.group(1) == "v%d.%d.%d" % pinned
        floor = re.search(r">= \((\d+), (\d+), (\d+)\)", c1)
        assert floor and tuple(int(x) for x in floor.groups()) <= pinned, path


def test_17_notebook_cells_have_no_outputs():
    for path, nb in _notebooks():
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code) == 3, path
        for c in code:
            assert c.get("outputs") == [] and c.get("execution_count") is None, path


NEW_FILES = ["geolip/modular_diffusion/staged.py", "geolip/modular_diffusion/forms.py", "geolip/modular_diffusion/score.py",
             "geolip/modular_diffusion/run.py", "geolip/modular_diffusion/experiments/__init__.py",
             "geolip/modular_diffusion/experiments/time_split.py", "tests/test_staged.py", "tests/test_body_range.py",
             "tests/test_fit_units.py", "tests/test_body_linker.py", "tests/test_score.py", "tests/test_data.py",
             "tests/test_roster_more.py", "tests/test_time_split.py",
             "geolip/modular_diffusion/experiments/chain_reads.py", "tests/test_sample_grid.py", "tests/test_chain_reads.py"]
MORE_WORDS = r"(?i)\bE\d\d\b|\bmd0|\bmr_|\bmind\b|prototype night"


def test_17_subpackage_sources_are_clean():
    files = sorted(glob.glob(os.path.join(PKG, "**", "*.py"), recursive=True))
    sub = [p for p in files if os.path.dirname(p) != PKG]
    assert sub, "no subpackage sources found"
    for path in files:
        text = open(path, encoding="utf-8").read()
        for what, pattern in FORBIDDEN.items():
            m = re.search(pattern, text)
            assert m is None, (os.path.relpath(path, ROOT), what, text[max(0, m.start() - 40):m.end() + 40])


def test_17_new_files_are_clean():
    for rel in NEW_FILES:
        path = os.path.join(ROOT, *rel.split("/"))
        text = open(path, encoding="utf-8").read()
        for what, pattern in dict(FORBIDDEN, more=MORE_WORDS).items():
            m = re.search(pattern, text)
            assert m is None, (rel, what, text[max(0, m.start() - 40):m.end() + 40])
    for path in NOTEBOOKS:
        text = "\n".join("".join(c["source"]) for c in json.load(open(path, encoding="utf-8"))["cells"])
        assert re.search(MORE_WORDS, text) is None, path


def test_17_readme_links_the_notebook():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    for path in NOTEBOOKS:
        assert re.search(re.escape(COLAB) + r"v\d+\.\d+\.\d+/colab/" + re.escape(os.path.basename(path)), text), path
