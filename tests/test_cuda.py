"""The experiment path on a card: the same end-to-end runs as the CPU suite's, with every tensor on the card (a unit
moved to a device while its probe, mask or reference stays behind is caught only here).

Off by default: set MD_TEST_CUDA=1 on a machine with a card (the suite itself runs with the card hidden). Small and short:
width 32, a few steps per part, under a gigabyte of card memory."""
import os

import pytest
import torch

from geolip.modular_diffusion.experiments import chain_reads, closed_forms, content, time_split
from geolip.modular_diffusion.staged import link_staged

pytestmark = pytest.mark.skipif(os.environ.get("MD_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                                reason="MD_TEST_CUDA=1 and a visible card are needed")

SMALL = dict(seed=0, device="cuda", scale=0.002, dim=32, heldout=64, batch=16, log_every=1, warmup=1, score=1,
             judge="fake", score_n=40, score_steps=3, sheet=1, threads=2, mem_fraction=0.0)
SOURCES = {"tails_alone": ("stem_from", "plain_d4"), "copies_soft": ("source", "plain")}


def test_experiment_variants_on_the_card(synthetic_npz, tmp_path):
    results = {}
    for v in ("plain_d4", "plain", "tails_mixed", "split_soft", "tails_joint", "tails_alone", "copies_soft"):
        kw = dict(SMALL, variant=v, data=synthetic_npz, out=str(tmp_path / v), ref=str(tmp_path / "reference.npz"))
        if v in SOURCES:
            key, src = SOURCES[v]
            kw[key] = str(tmp_path / src)
        results[v] = time_split.main(**kw)
    for v, r in results.items():
        assert r["device"] == "cuda" and isinstance(r["loss_over_levels"], float), v
        assert "score" in r["score"], (v, r["score"])
        assert all(p["peak_memory_mib"] is not None and p["peak_memory_mib"] > 0 for p in r["parts"]), v
        assert os.path.exists(tmp_path / v / "samples.png")
    staged = link_staged(str(tmp_path / "tails_joint"), device="cuda")
    assert len({id(c.stem) for c in staged.clusters}) == 1
    assert next(staged.parameters()).device.type == "cuda"
    # one read of each chain-reads variant on the card, from the jobs above, each with a score
    split, plain = str(tmp_path / "split_soft"), str(tmp_path / "plain")
    read_kw = dict(device="cuda", heldout=64, score=1, judge="fake", score_n=40, score_steps=3, sheet=1, threads=2,
                   mem_fraction=0.0, data=synthetic_npz, ref=str(tmp_path / "reference.npz"))
    chain = {"twin": dict(variant="twin", base=split, donor=split, stage=1),
             "swap": dict(variant="swap", base=plain, donor=split, stage=1),
             "wrong": dict(variant="wrong", base=split),
             "blend": dict(variant="blend", base=split, soft=0.0),
             "steps": dict(variant="steps", base=split, plan="width", n=6)}
    for name, kw in chain.items():
        r = chain_reads.main(**dict(read_kw, out=str(tmp_path / "chain" / name), **kw))
        assert r["device"] == "cuda" and isinstance(r["loss_over_levels"], float), name
        assert "score" in r["score"], (name, r["score"])
        assert os.path.exists(tmp_path / "chain" / name / "samples.png"), name
    assert chain_reads.summarize(str(tmp_path / "chain")).count("\n") == 2 + len(chain)


def test_content_and_closed_forms_on_the_card(synthetic_npz, tmp_path):
    """Every content variant and every closed-form variant end to end on the card: the per-class anchor, the class
    window, class experts on a finished model, the class reader, averaged weights, a head by a solve, stage heads by a
    solve, band members through the live body and from stored states."""
    ref = str(tmp_path / "reference.npz")
    small = dict(SMALL, data=synthetic_npz, ref=ref)
    c = {}
    sources = {"experts_start": "plain", "blind_experts_start": "blind", "blind_experts_all": "blind", "snap9": "blind9"}
    for v in ("plain", "blind", "blind9", "class_start", "blind_canchor", "canchor", "anchors", "experts_start",
              "blind_experts_start", "blind_experts_all", "snap9"):
        kw = dict(small, variant=v, out=str(tmp_path / "content" / v), reader=1)
        if v in sources:
            kw["source"] = str(tmp_path / "content" / sources[v])
        if v in ("plain", "blind_canchor", "blind_experts_start"):
            kw["avg"] = "0.5,0.9"
        c[v] = content.main(**kw)
    for v, r in c.items():
        assert r["device"] == "cuda" and isinstance(r["loss_over_levels"], float), v
        assert "score" in r["score"] and "class_read" in r["score"], (v, r["score"])
        assert len(r["by_class"]) == 10 and len(r["anchor_by_level"]) == 10, v
    assert "reader_check" in c["plain"]["score"]                        # the job that computed the reference file
    for v in ("experts_start", "blind_experts_start", "blind_experts_all", "snap9"):
        assert c[v]["exact"]["others_max_abs"] == 0.0, (v, c[v]["exact"])
    for v in ("plain", "blind_canchor", "blind_experts_start"):
        assert set(c[v]["averaged"]) == {"0.5", "0.9"}, v
        assert all(isinstance(a["loss_over_levels"], float) and "score" in a["score"] for a in c[v]["averaged"].values()), v
    assert "averaged" not in c["blind"]
    # the class anchor differs from the pooled one where the class is given
    assert c["anchors"]["anchor_by_level"] != c["anchors"]["anchor_pooled_by_level"]
    assert c["blind"]["anchor_by_level"] == c["blind"]["anchor_pooled_by_level"]

    plain = str(tmp_path / "content" / "plain")
    d4 = str(tmp_path / "d4")
    time_split.main(**dict(small, variant="plain_d4", out=d4, score=0, sheet=0))
    f = {}
    f["head_solved"] = closed_forms.main(**dict(small, variant="head_solved", source=plain, batches=4,
                                                out=str(tmp_path / "closed" / "head_solved")))
    f["heads_staged_solved"] = closed_forms.main(**dict(small, variant="heads_staged_solved", source=plain, batches=4,
                                                        out=str(tmp_path / "closed" / "heads_staged_solved")))
    member_kw = dict(small, source=d4, steps=1500)                     # scaled: three steps
    f["members_live"] = closed_forms.main(**dict(member_kw, variant="members_live", out=str(tmp_path / "closed" / "live")))
    cache = str(tmp_path / "cache")
    f["cache_write"] = closed_forms.main(**dict(member_kw, variant="cache_write", cache=cache,
                                                out=str(tmp_path / "closed" / "cache_write")))
    f["members_cached"] = closed_forms.main(**dict(member_kw, variant="members_cached", cache=cache,
                                                   out=str(tmp_path / "closed" / "cached")))
    f["members_cycle"] = closed_forms.main(**dict(member_kw, variant="members_cached", cache=cache, steps=3000, passes=2,
                                                  out=str(tmp_path / "closed" / "cycle")))
    f["members_cpu"] = closed_forms.main(**dict(member_kw, variant="members_cached", cache=cache, device="cpu", read=0,
                                                out=str(tmp_path / "closed" / "cpu")))
    assert f["head_solved"]["solve"]["head"]["mse_after"] <= f["head_solved"]["solve"]["head"]["mse_before"] + 1e-9
    assert f["head_solved"]["as_trained"]["equals_stored"] is True       # the same weights on the same card
    assert f["heads_staged_solved"]["kind"] == "clusters" and len(f["heads_staged_solved"]["alone"]) == 3
    assert f["cache_write"]["cache"]["bytes"] > 0
    # stored states on the card give what the live body gives, to the last digit
    assert f["members_cached"]["curve"] == f["members_live"]["curve"]
    assert f["members_cached"]["loss_over_levels"] == f["members_live"]["loss_over_levels"]
    # the CPU reads the same stored states: the same losses up to the two devices' rounding
    for (k1, l1), (k2, l2) in zip(f["members_cpu"]["curve"], f["members_cached"]["curve"]):
        assert k1 == k2 and abs(l1 - l2) < 1e-4, (k1, l1, l2)
    assert f["members_cpu"]["device"] == "cpu" and f["members_cpu"]["loss_over_levels"] is None
    for name in ("head_solved", "heads_staged_solved", "members_live", "members_cached", "members_cycle"):
        assert f[name]["device"] == "cuda" and isinstance(f[name]["loss_over_levels"], float), name
        assert "score" in f[name]["score"], (name, f[name]["score"])
