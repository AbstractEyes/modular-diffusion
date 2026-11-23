"""A blind source (T3): stages source with class_window=blind is content's blind to the bit (the loss over the levels
and every unit's content hash), and with without_classes=9 as well content's blind9; the window is stored as written and
the restriction recorded as content records it. (The source with defaults against the record before the pass: T0.)"""
import pytest

from geolip.modular_diffusion.build import read_build, read_card, unit_files
from geolip.modular_diffusion.experiments import content, stages

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=16, batch=8, log_every=100, warmup=1, score=0,
             judge="fake", sheet=0, threads=4)


def _hashes(folder):
    return {e.name: e.hash for e in read_build(folder).units}


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("blind_source")
    d = dict(SMALL, data=synthetic_npz)
    j = lambda n: str(root / n)
    r = {}
    r["blind"] = content.main(**dict(d, variant="blind", out=j("blind")))
    r["blind9"] = content.main(**dict(d, variant="blind9", out=j("blind9")))
    r["src_blind"] = stages.main(**dict(d, variant="source", out=j("src_blind"), class_window="blind", total=6000))
    r["src_blind9"] = stages.main(**dict(d, variant="source", out=j("src_blind9"), class_window="blind",
                                         without_classes="9", total=6000))
    r["src_window"] = stages.main(**dict(d, variant="source", out=j("src_window"), class_window="0,0.25", avg="none"))
    return root, r


def test_T3_a_blind_source_is_contents_blind(jobs):
    root, r = jobs
    for mine, theirs in (("src_blind", "blind"), ("src_blind9", "blind9")):
        assert r[mine]["loss_over_levels"] == r[theirs]["loss_over_levels"], mine
        assert r[mine]["by_level"] == r[theirs]["by_level"], mine
        assert _hashes(str(root / mine / "model")) == _hashes(str(root / theirs / "model")), mine
    assert r["src_blind9"]["data"] == {"training_stream": {"without_classes": [9]}}
    assert r["src_blind9"]["args"]["class_window"] == "blind" and r["src_blind9"]["args"]["without_classes"] == "9"
    assert "data" not in r["src_blind"] and "parts_off" not in r["src_blind"]
    assert read_build(str(root / "src_blind" / "model")).entry("stem").spec["class_window"] == ["0", "0"]
    assert read_build(str(root / "src_window" / "model")).entry("stem").spec["class_window"] == ["0", "0.25"]
    card = read_card(unit_files(str(root / "src_blind9" / "model"), "head")[1])
    assert card.data["training_stream"] == {"without_classes": [9]}


def test_T3_refusals(synthetic_npz, tmp_path):
    d = dict(SMALL, data=synthetic_npz)
    for i, kw in enumerate((dict(variant="source", class_window="0.5"), dict(variant="source", without_classes="none"),
                            dict(variant="tails", class_window="blind", stem_from="x"),
                            dict(variant="source", parts="row:9"), dict(variant="source", stream=1))):
        with pytest.raises(ValueError):
            stages.main(**dict(d, out=str(tmp_path / f"r{i}"), **kw))
