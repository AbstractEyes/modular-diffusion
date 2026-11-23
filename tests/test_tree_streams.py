"""The streams of copies (T5, T6, P4b): a sibling stream (stream=1 gives other weights than stream=0, the same weights
when repeated; stream=0 is the default job); copies that inherit a restricted training stream (no batch holds a class-9
picture) and copies that drop it (without_classes=none: some batch holds one)."""
import json
import os

import pytest

from geolip.modular_diffusion.beds import PicturesBed
from geolip.modular_diffusion.build import read_build, read_card, unit_files
from geolip.modular_diffusion.experiments import stages
from geolip.modular_diffusion.train import step_generator

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=16, batch=8, log_every=100, warmup=1, score=0,
             judge="fake", sheet=0, threads=4)
CUTS = "0.25,0.65"


def _hashes(folder):
    return {e.name: e.hash for e in read_build(folder).units}


def _tree_hashes(job):
    return {s: _hashes(os.path.join(job, f"stage_{s}")) for s in range(3)}


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("tree_streams")
    d = dict(SMALL, data=synthetic_npz)
    j = lambda n: str(root / n)
    g = dict(d, avg="g8")                       # one average (the carried one) keeps the file's time short
    r = {}
    r["src"] = stages.main(**dict(d, variant="source", out=j("src")))
    r["cop"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("cop"), source=j("src")))
    r["s1"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("s1"), source=j("src"), stream=1))
    r["s1b"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("s1b"), source=j("src"), stream=1))
    r["s0"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("s0"), source=j("src"), stream=0))
    r["src9"] = stages.main(**dict(d, variant="source", out=j("src9"), without_classes="9"))
    r["cop9"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("cop9"), source=j("src9")))
    r["cop9_all"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("cop9_all"), source=j("src9"),
                                       without_classes="none", batch=64))
    return root, r


def test_T5_a_sibling_stream(jobs):
    root, r = jobs
    assert _tree_hashes(str(root / "s0")) == _tree_hashes(str(root / "cop"))
    assert r["s0"]["loss_over_levels"] == r["cop"]["loss_over_levels"] and "stream" not in r["s0"]["args"]
    assert all("stream_seed" not in p for p in r["s0"]["parts"])
    one, again = _tree_hashes(str(root / "s1")), _tree_hashes(str(root / "s1b"))
    assert one == again and r["s1"]["loss_over_levels"] == r["s1b"]["loss_over_levels"]
    base = _tree_hashes(str(root / "cop"))
    assert all(one[s]["head"] != base[s]["head"] for s in range(3))
    seed1 = 0 + stages.GROW_STREAM * (stages.SIBLING_STREAM + 1)
    assert all(p["stream_seed"] == seed1 for p in r["s1"]["parts"]) and r["s1"]["args"]["stream"] == 1
    assert [p["start_step"] for p in r["s1"]["parts"]] == [p["start_step"] for p in r["cop"]["parts"]]
    assert [p["steps"] for p in r["s1"]["parts"]] == [p["steps"] for p in r["cop"]["parts"]]


def test_T6_copies_inherit_the_restricted_stream(jobs, synthetic_npz):
    root, r = jobs
    c9 = r["cop9"]
    assert c9["data"] == {"training_stream": {"without_classes": [9]}}
    card = read_card(unit_files(str(root / "cop9" / "stage_1"), "head")[1])
    assert card.data["training_stream"] == {"without_classes": [9]}
    # no batch of the copies' steps holds a class-9 picture: the draws of the recorded stream at those global steps
    bed = PicturesBed(synthetic_npz, "cpu").restricted(without_classes=(9,))
    for p in c9["parts"]:
        for s in range(p["start_step"], p["start_step"] + p["steps"]):
            _, y = bed.train_batch(SMALL["batch"], step_generator(0, s, "cpu"))
            assert not bool((y == 9).any())
    with open(root / "cop9" / "result.json", encoding="utf-8") as fh:
        assert json.load(fh)["data"]["training_stream"] == {"without_classes": [9]}
    assert "data" not in r["cop"]
    # the unrestricted copies differ from the restricted ones
    assert _tree_hashes(str(root / "cop9"))[0]["head"] != _tree_hashes(str(root / "cop"))[0]["head"]


def test_P4b_copies_can_drop_the_sources_restriction(jobs, synthetic_npz):
    root, r = jobs
    c = r["cop9_all"]
    assert c["data"]["training_stream"] is None and c["args"]["without_classes"] == "none"
    assert c["data"]["training_stream_choice"] == {"without_classes": "none", "source": {"without_classes": [9]}}
    assert "training_stream" not in read_card(unit_files(str(root / "cop9_all" / "stage_0"), "head")[1]).data
    bed = PicturesBed(synthetic_npz, "cpu")
    seen9 = False
    for p in c["parts"]:
        for s in range(p["start_step"], p["start_step"] + p["steps"]):
            _, y = bed.train_batch(64, step_generator(0, s, "cpu"))
            seen9 = seen9 or bool((y == 9).any())
    assert seen9                                                         # some batch holds a class-9 picture
    assert "without_classes" not in r["cop9"]["args"]                    # the default inherits (T6 above)
