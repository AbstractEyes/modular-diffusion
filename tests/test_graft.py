"""Grafts of trained parts onto another parent (T8): on the same base without training the parts job's g8 read
exactly; on a sibling tree the job runs, a parent differs, the read differs, the cards keep the parents the parts were
trained with and the tree links only with the graft switch; trained on further; a parts-spec mismatch refused. One
stage's parts for every stage (P5b). The new keys in json and in the summary (T10)."""
import json
import os

import pytest
import torch

from geolip.modular_diffusion.build import read_build, read_card, read_weights, unit_files
from geolip.modular_diffusion.experiments import stages
from geolip.modular_diffusion.linker import StaleUnit
from geolip.modular_diffusion.staged import link_staged, read_staged
from geolip.modular_diffusion.units import NULL_CLASS

SMALL = dict(seed=0, device="cpu", scale=0.001, dim=32, heldout=16, batch=8, log_every=100, warmup=1, score=0,
             judge="fake", score_n=32, score_steps=2, sheet=0, threads=4)
CUTS = "0.25,0.65"


def _tensors(folder):
    b = read_build(folder)
    return {e.name: read_weights(os.path.join(b.folder, e.file)) for e in b.units}


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("graft")
    d = dict(SMALL, data=synthetic_npz, ref=str(root / "reference.npz"))
    j = lambda n: str(root / n)
    t = dict(d, avg="tail100")                    # the bases: only the tree the parts stand on (keeps the file short)
    r = {}
    r["src"] = stages.main(**dict(d, variant="source", out=j("src")))
    r["cop"] = stages.main(**dict(t, variant="copies", cuts=CUTS, out=j("cop"), source=j("src")))
    r["sib"] = stages.main(**dict(t, variant="copies", cuts=CUTS, out=j("sib"), source=j("src"), stream=1))
    r["one"] = stages.main(**dict(t, variant="copies", out=j("one"), source=j("src")))
    r["p_row_expert_9"] = stages.main(**dict(d, variant="parts", out=j("p_row_expert_9"), base=j("cop"),
                                             parts="row+expert:9", only_classes="9"))
    graft = dict(d, variant="parts", parts="row+expert:9", only_classes="9", init_from=j("p_row_expert_9"))
    r["g_same"] = stages.main(**dict(graft, out=j("g_same"), base=j("cop"), total=0))
    r["g_sib"] = stages.main(**dict(graft, out=j("g_sib"), base=j("sib"), total=0))
    r["g_train"] = stages.main(**dict(graft, out=j("g_train"), base=j("sib"), total=300))
    r["row_one"] = stages.main(**dict(d, variant="parts", out=j("row_one"), base=j("one"), parts="row:9",
                                      only_classes="9"))
    r["g_stage0"] = stages.main(**dict(d, variant="parts", out=j("g_stage0"), base=j("cop"), parts="row:9",
                                       only_classes="9", init_from=j("row_one"), init_stage=0, total=0))
    return root, r


def test_T8_grafts(jobs, synthetic_npz, tmp_path):
    root, r = jobs
    same, sib, trained, src = r["g_same"], r["g_sib"], r["g_train"], r["p_row_expert_9"]
    assert same["graft"]["parents_equal"] == [True, True, True] and same["graft"]["init_stage"] == -1
    assert same["averaged"]["g8"]["loss_over_levels"] == src["averaged"]["g8"]["loss_over_levels"]
    assert same["averaged"]["g8"]["by_level"] == src["averaged"]["g8"]["by_level"]
    assert same["loss_over_levels"] == src["averaged"]["g8"]["loss_over_levels"]   # total=0: the same weights twice
    assert all(p["steps"] == 0 for p in same["parts"])
    assert False in sib["graft"]["parents_equal"]
    assert sib["averaged"]["g8"]["loss_over_levels"] != src["averaged"]["g8"]["loss_over_levels"]
    assert sib["exact"]["others_max_abs"] == 0.0 and sib["parts_off"]["equals_base"] is True
    # the grafted parts' cards say what they were trained with; the tree links only with the graft switch
    card = read_card(unit_files(str(root / "g_sib" / "stage_0"), "expert_c9")[1])
    first = read_card(unit_files(str(root / "p_row_expert_9" / "avg_g8" / "stage_0"), "expert_c9")[1])
    assert card.parents == first.parents and card.recipe["graft"]["onto"] != first.parents
    with pytest.raises(StaleUnit):
        link_staged(str(root / "g_sib"))
    assert len(link_staged(str(root / "g_sib"), grafts=True).clusters) == 3
    assert len(link_staged(str(root / "g_same")).clusters) == 3                       # same parents: strict links
    # trained on the sibling: the grafted parts move, their cards record where they came from
    assert all(p["steps"] > 0 for p in trained["parts"])
    moved = _tensors(str(root / "g_train" / "stage_0"))["expert_c9"]
    start = _tensors(str(root / "p_row_expert_9" / "avg_g8" / "stage_0"))["expert_c9"]
    assert any(not torch.equal(moved[k], start[k]) for k in start)
    card = read_card(unit_files(str(root / "g_train" / "stage_0"), "expert_c9")[1])
    assert card.recipe["graft"]["first_trained_with"] == first.parents
    assert len(link_staged(str(root / "g_train")).clusters) == 3
    with pytest.raises(ValueError):                                                   # a parts-spec mismatch
        stages.main(**dict(SMALL, data=synthetic_npz, variant="parts", out=str(tmp_path / "x"), base=str(root / "cop"),
                           parts="experts10", init_from=str(root / "p_row_expert_9"), total=0))


def test_P5b_one_stages_parts_for_every_stage(jobs, synthetic_npz, tmp_path):
    root, r = jobs
    g = r["g_stage0"]
    assert read_staged(str(root / "one" / "avg_tail100"))["clusters"] == ["stage_0"]
    assert g["graft"]["init_stage"] == 0 and len(g["graft"]["parents_equal"]) == 3
    assert g["exact"]["others_max_abs"] == 0.0 and g["parts_off"]["equals_base"] is True
    rows = [_tensors(str(root / "g_stage0" / f"stage_{s}"))["stem"]["cls.weight"][9] for s in range(3)]
    want = _tensors(str(root / "row_one" / "avg_g8" / "stage_0"))["stem"]["cls.weight"][9]
    assert all(torch.equal(x, want) for x in rows)
    assert not torch.equal(want, _tensors(str(root / "cop" / "avg_tail100" / "stage_0"))["stem"]["cls.weight"][NULL_CLASS])
    with pytest.raises(ValueError):                                       # stage by stage needs as many stages
        stages.main(**dict(SMALL, data=synthetic_npz, variant="parts", out=str(tmp_path / "y"), base=str(root / "cop"),
                           parts="row:9", init_from=str(root / "row_one"), total=0))
    with pytest.raises(ValueError):                                       # a stage the init job does not hold
        stages.main(**dict(SMALL, data=synthetic_npz, variant="parts", out=str(tmp_path / "z"), base=str(root / "cop"),
                           parts="row:9", init_from=str(root / "row_one"), init_stage=1, total=0))


def test_T10_the_graft_keys_are_json_and_the_summary_holds_the_jobs(jobs):
    root, r = jobs
    for name, res in r.items():
        json.dumps(res)
        with open(root / name / "result.json", encoding="utf-8") as fh:
            assert json.load(fh)["variant"] == res["variant"], name
    text = stages.summarize(str(root))
    rows = [x for x in text.splitlines() if x.startswith("| parts |")]
    assert len(rows) == 2 * 6                                             # the last weights and g8 of each parts job
