"""Band members on an averaged body (T7): three band-kept members, or one unfiltered member, on a finished job's averaged
folder; the base frozen (its hashes the source folder's); trained, averaged, read and scored; the three band members'
independence ratio within 1e-6 of 1. And the closed-forms summary on these jobs (T9)."""
import json
import os

import pytest

from geolip.modular_diffusion.build import read_build
from geolip.modular_diffusion.experiments import closed_forms, time_split
from geolip.modular_diffusion.linker import link

SMALL = dict(seed=0, device="cpu", heldout=32, batch=8, log_every=1, warmup=1, score=1, judge="fake", score_n=32,
             score_steps=2, sheet=0, threads=4)


@pytest.fixture(scope="module")
def jobs(synthetic_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("members_avg")
    d = dict(SMALL, data=synthetic_npz, ref=str(root / "reference.npz"))
    time_split.main(**dict(d, variant="plain_d4", out=str(root / "plain_d4"), avg="g8", scale=0.0005, dim=32))
    r = {}
    for v in ("members_avg", "member_all_avg"):
        r[v] = closed_forms.main(**dict(d, variant=v, out=str(root / v), source=str(root / "plain_d4"),
                                        source_model="model_avg_g8", scale=0.0005, reader=1))
    r["members_avg_steps"] = closed_forms.main(**dict(d, variant="members_avg", out=str(root / "members_avg_steps"),
                                                      source=str(root / "plain_d4"), source_model="model_avg_g8",
                                                      steps=4, avg="none"))
    return root, r


def test_T7_members_on_an_averaged_body(jobs):
    root, r = jobs
    src = read_build(str(root / "plain_d4" / "model_avg_g8"))
    for v, names in (("members_avg", ["member_b0", "member_b1", "member_b2"]), ("member_all_avg", ["member_all"])):
        res = r[v]
        b = read_build(str(root / v / "model"))
        assert [e.name for e in b.units] == ["anchor", "stem", "head"] + names
        for e in src.units:                                       # the averaged base, frozen
            assert b.entry(e.name).hash == e.hash, (v, e.name)
        assert res["parts"][0]["trainable"] == names and res["parts"][0]["frozen_unchanged"] is True
        assert res["parts"][0]["steps"] == time_split.steps_of(6000, 0.0005)
        assert res["parts"][0]["t_range"] == [0.0, 1.0]
        assert res["as_source"]["equals_stored"] is True and res["as_source"]["folder"] == "model_avg_g8"
        assert set(res["averaged"]) == {"g8"} and res["averaged"]["g8"]["score"]["n"] == 32
        assert "class_read" in res["score"] and "class_read" in res["averaged"]["g8"]["score"]
        assert isinstance(res["loss_over_levels"], float)
        m = link(str(root / v / "model"))
        assert [u.name for u in m.members] == names
        for e in read_build(str(root / v / "model_avg_g8")).units:
            if e.name in ("anchor", "stem", "head"):
                assert e.hash == src.entry(e.name).hash
    ratio = r["members_avg"]["read"]["independence_ratio_over_levels"]
    assert abs(ratio - 1.0) < 1e-6, ratio
    assert abs(r["members_avg"]["averaged"]["g8"]["independence_ratio_over_levels"] - 1.0) < 1e-6
    one = link(str(root / "member_all_avg" / "model")).unit("member_all")
    assert one.band == "all" and one.keep is False
    assert r["members_avg_steps"]["parts"][0]["steps"] == 4 and not r["members_avg_steps"].get("averaged")


def test_T9_the_closed_forms_summary_holds_the_member_jobs(jobs):
    root, _ = jobs
    text = closed_forms.summarize(str(root))
    assert "| members_avg | 0 |" in text and "| member_all_avg | 0 |" in text
