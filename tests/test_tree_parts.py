"""Parts in sources and copies (T4, T5; T3 is test_blind_source.py, the streams test_tree_streams.py): a source born
with parts (birth gap 0.0, parts_off, exactness, the state covers the parts) and copies that continue it strictly;
parts born at the fork (a partial state over exactly the parts' parameters; refusals). (The defaults against the record
of the library before the pass: T0.)"""
import pytest

from geolip.modular_diffusion.build import read_build, read_card, unit_files
from geolip.modular_diffusion.experiments import stages
from geolip.modular_diffusion.linker import link
from geolip.modular_diffusion.train import read_state

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=16, batch=8, log_every=100, warmup=1, score=0,
             judge="fake", sheet=0, threads=4)
CUTS = "0.25,0.65"


def _hashes(folder):
    return {e.name: e.hash for e in read_build(folder).units}


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("tree_parts")
    d = dict(SMALL, data=synthetic_npz, ref=str(root / "reference.npz"))
    j = lambda n: str(root / n)
    r = {}
    r["src_parts"] = stages.main(**dict(d, variant="source", out=j("src_parts"), parts="experts10"))
    r["cop_parts"] = stages.main(**dict(d, variant="copies", cuts=CUTS, out=j("cop_parts"), source=j("src_parts")))
    r["src"] = stages.main(**dict(d, variant="source", out=j("src")))
    g = dict(d, avg="g8")                       # one average (the carried one) keeps the file's time short
    r["fork10"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("fork10"), source=j("src"),
                                     parts="experts10"))
    r["fork1"] = stages.main(**dict(g, variant="copies", cuts=CUTS, out=j("fork1"), source=j("src"), parts="shared1"))
    return root, r


def test_T4_defaults_are_not_recorded(jobs):
    _, r = jobs
    assert "class_window" not in r["src"]["args"] and "parts" not in r["src"]["args"]
    assert "parts_off" not in r["src"] and "data" not in r["src"] and "birth_gap" not in r["src"]


def test_T4_a_source_born_with_parts(jobs):
    root, r = jobs
    sp = r["src_parts"]
    names = [f"expert_c{c}" for c in range(10)]
    assert sp["birth_gap"] == 0.0 and sp["part_units"] == names and sp["parts_spec"] == "experts10"
    assert sp["parts_off"]["loss_over_levels"] != sp["loss_over_levels"]
    assert set(sp["parts_off"]) == {"loss_over_levels", "by_level", "by_stage", "class_worth"}
    assert sp["exact"]["others_max_abs"] == 0.0 and sp["exact"]["expert"] == "expert_c0"
    assert "parts_off" in sp["averaged"]["g8"] and sp["averaged"]["g8"]["exact"]["others_max_abs"] == 0.0
    state = read_state(str(root / "src_parts" / "state"))
    model = link(str(root / "src_parts" / "model"))
    part_params = {n for n, p in model.named_parameters()
                   if any(p is q for u in names for q in model.unit(u).parameters())}
    assert part_params and part_params <= set(state["m"])
    assert set(_hashes(str(root / "src_parts" / "model_avg_g8"))) >= set(names)
    assert sp["parts"][0]["seconds_per_step"] > 0 and "peak_memory_mib" in sp["parts"][0]
    assert sp["parts"][0]["trainable"] == "all"


def test_T4_copies_continue_every_unit_of_a_source_with_parts(jobs):
    root, r = jobs
    cp = r["cop_parts"]
    assert cp["part_units"] == [f"expert_c{c}" for c in range(10)]
    assert cp["exact"]["others_max_abs"] == 0.0 and "parts_off" in cp
    assert cp["parts_off"]["loss_over_levels"] != cp["loss_over_levels"]
    for p in cp["parts"]:
        assert "optimizer_new" not in p and p["start_step"] == r["src_parts"]["parts"][0]["steps"]
        assert p["averages_carried"] == ["g8"]
    for s in range(3):
        h = _hashes(str(root / "cop_parts" / f"stage_{s}"))
        assert set(h) == set(_hashes(str(root / "src_parts" / "model")))
    assert set(cp["averaged"]) == {"g8", "tail100"}
    assert all("parts_off" in cp["averaged"][k] for k in ("g8", "tail100"))


def test_T4_a_part_that_never_took_a_step_continues_without_state(synthetic_npz, tmp_path):
    """Two steps of two rows reach at most four classes: the other classes' parts never take a step, so the source's
    trainer state holds no entry for them. The copies continue them as they were (no moments) and say which."""
    d = dict(SMALL, data=synthetic_npz, batch=2, scale=0.0005)
    stages.main(**dict(d, variant="source", out=str(tmp_path / "s"), parts="experts10"))
    state = read_state(str(tmp_path / "s" / "state"))
    cp = stages.main(**dict(d, variant="copies", cuts=CUTS, out=str(tmp_path / "c"), source=str(tmp_path / "s")))
    for p in cp["parts"]:
        new = p["optimizer_new"]
        assert new and all(n not in state["m"] for n in new)
    assert cp["exact"]["others_max_abs"] == 0.0 and "parts_off" in cp
    card = read_card(unit_files(str(tmp_path / "c" / "stage_0"), "stem")[1])
    assert card.recipe["optimizer_state"] == "continued, partial"


def test_T5_parts_born_at_the_fork(jobs, synthetic_npz, tmp_path):
    root, r = jobs
    for key, spec in (("fork10", "experts10"), ("fork1", "shared1")):
        res = r[key]
        model = link(str(root / key / "stage_0"))
        names = res["part_units"]
        want = sorted(n for n, p in model.named_parameters()
                      if any(p is q for u in names for q in model.unit(u).parameters()))
        for p in res["parts"]:
            assert p["birth_gap"] == 0.0 and p["optimizer_new"] == want, key
            assert p["start_step"] == r["src"]["parts"][0]["steps"]
        assert res["parts_spec"] == spec and res["exact"]["others_max_abs"] == 0.0
        assert "parts_off" in res and "g8" in res["averaged"]
        # the parts' cards were trained here; every other unit continued the source's run
        card = read_card(unit_files(str(root / key / "stage_0"), names[0])[1])
        assert card.recipe["optimizer_state"] == "continued, partial"
    assert r["fork1"]["part_units"] == ["shared_c0-1-2-3-4-5-6-7-8-9"]
    d = dict(SMALL, data=synthetic_npz, variant="copies", cuts=CUTS)
    with pytest.raises(ValueError):                                       # a source that holds parts
        stages.main(**dict(d, out=str(tmp_path / "a"), source=str(root / "src_parts"), parts="experts10"))
    with pytest.raises(ValueError):                                       # with fresh copies
        stages.main(**dict(d, out=str(tmp_path / "b"), source=str(root / "src"), parts="experts10", fresh=1))
    with pytest.raises(ValueError):                                       # a row is no part of copies
        stages.main(**dict(d, out=str(tmp_path / "c"), source=str(root / "src"), parts="row:9"))
    with pytest.raises(ValueError):
        stages.main(**dict(d, out=str(tmp_path / "e"), source=str(root / "src"), parts="experts11"))


