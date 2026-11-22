"""The experiment path on a card: the same end-to-end runs as the CPU suite's, with every tensor on the card (a unit
moved to a device while its probe, mask or reference stays behind is caught only here).

Off by default: set MD_TEST_CUDA=1 on a machine with a card (the suite itself runs with the card hidden). Small and short:
width 32, a few steps per part, under a gigabyte of card memory.

The third function runs its jobs with det=on (``determinism``): by default the attention kernel's backward pass is not
deterministic on a card, so two runs of one job can differ in the last digits; with the switch on they must be equal.
The matrix library's fixed workspace has to be in the environment before anything touches the card, so it is set here,
before torch is imported."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402

from geolip.modular_diffusion.experiments import chain_reads, closed_forms, content, time_split  # noqa: E402
from geolip.modular_diffusion.staged import link_staged  # noqa: E402

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


def test_stages_symbols_and_class_heads_on_the_card(synthetic_npz, tmp_path):
    """On the card, every job with det=on: a run continued from a saved trainer state equals the straight run (the last
    weights and the carried average), the straight run repeats, copies with 3 and 8 stages, copies with a new optimizer,
    tails, a grown model, symbol-gated parts, class heads by a solve; and the warn mode names what is not deterministic."""
    from geolip.modular_diffusion.experiments import stages
    small = dict(SMALL, data=synthetic_npz, ref=str(tmp_path / "reference.npz"), det="on")
    root = tmp_path / "stages"
    plain = time_split.main(**dict(small, variant="plain", out=str(root / "plain"), avg="g8"))
    src = stages.main(**dict(small, variant="source", out=str(root / "source"), total=4000, score=0, sheet=0))
    assert os.path.exists(root / "source" / "state" / "state.json")
    copy = dict(small, variant="copies", source=str(root / "source"), source_model="model", total=2000, reader=1)
    k1 = stages.main(**dict(copy, out=str(root / "k1"), cuts="", score_last=0, score_avg="tail100"))
    # the control of the stage experiment: continued over the whole range it is the straight run, on a card too, and
    # the straight run repeats to the last digit
    again = time_split.main(**dict(small, variant="plain", out=str(root / "plain_again"), avg="g8", score=0, sheet=0))
    assert again["loss_over_levels"] == plain["loss_over_levels"]
    assert k1["loss_over_levels"] == plain["loss_over_levels"]
    assert k1["averaged"]["g8"]["loss_over_levels"] == plain["averaged"]["g8"]["loss_over_levels"]
    assert k1["as_source"]["equals_stored"] is True
    assert k1["score"] is None and k1["averaged"]["g8"]["score"] is None
    assert "class_read" in k1["averaged"]["tail100"]["score"]
    s = {}
    for name, kw in (("k3", dict(cuts="0.25,0.65")), ("k8", dict(cuts="0.064,0.145,0.262,0.411,0.588,0.764,0.908")),
                     ("k3_fresh", dict(cuts="0.25,0.65", fresh=1))):
        s[name] = stages.main(**dict(copy, out=str(root / name), **kw))
        assert set(s[name]["averaged"]) == {"g8", "tail100"} and len(s[name]["by_stage"]) == 3, name
        assert "score" in s[name]["score"] and "class_read" in s[name]["averaged"]["tail100"]["score"], name
    assert len(s["k8"]["parts"]) == 8 and s["k8"]["soft"] == 0.016
    assert all(p["start_step"] == src["parts"][0]["steps"] for p in s["k3"]["parts"])
    assert all(p["start_step"] == 0 for p in s["k3_fresh"]["parts"])
    time_split.main(**dict(small, variant="plain_d4", out=str(root / "d4"), avg="g8", score=0, sheet=0))
    stem = dict(stem_from=str(root / "d4"), stem_model="model_avg_g8")
    for name, kw in (("tails_k1", dict(cuts="")), ("tails_k3", dict(cuts="0.25,0.65"))):
        r = stages.main(**dict(small, variant="tails", out=str(root / name), total=6000, score_last=0, **stem, **kw))
        assert r["as_source"]["equals_stored"] is True and "score" in r["averaged"]["g8"]["score"], name
        assert all(p["frozen_unchanged"] for p in r["parts"]), name
    g = stages.main(**dict(small, variant="grow", out=str(root / "grow"), blocks=2, steps_each=2000, **stem))
    assert [x["birth_gap"] for x in g["steps"]] == [0.0, 0.0] and all(x["earlier_unchanged"] for x in g["steps"])
    assert g["steps"][0]["stream_seed"] != g["steps"][1]["stream_seed"] and "score" in g["score"]
    assert stages.summarize(str(root)).count("| copies |") >= 4

    croot = tmp_path / "content"
    for v in ("plain", "blind"):
        content.main(**dict(small, variant=v, out=str(croot / v), reader=1, avg="g8"))
    mid = content.main(**dict(small, variant="class_to_mid", out=str(croot / "class_to_mid"), reader=1, avg="g8"))
    assert mid["class_worth"]["by_stage"][2] == 0.0 and mid["class_worth"]["by_stage"][0] != 0.0
    on_plain = dict(source=str(croot / "plain"), source_model="model_avg_g8")
    on_blind = dict(source=str(croot / "blind"), source_model="model_avg_g8")
    for v, base in (("experts_start", on_plain), ("rand_start", on_plain), ("shared_start", on_plain),
                    ("blind_experts_all", on_blind), ("blind_rand_all", on_blind), ("blind_shared_all", on_blind)):
        r = content.main(**dict(small, variant=v, out=str(croot / v), reader=1, avg="g8", score_last=0, **base))
        assert r["as_source"]["equals_stored"] is True, v
        assert r["exact"]["others_max_abs"] == 0.0, (v, r["exact"])
        assert "class_read" in r["averaged"]["g8"]["score"], v

    froot = tmp_path / "closed"
    for v in ("class_heads", "class_stage_heads"):
        r = closed_forms.main(**dict(small, variant=v, out=str(froot / v), batches=4, reader=1, **on_blind))
        assert r["as_source"]["equals_stored"] is True and r["exact"]["others_max_abs"] == 0.0, (v, r["exact"])
        assert r["solve"]["all"]["mse_after"] <= r["solve"]["all"]["mse_before"] + 1e-9, v
        assert "class_read" in r["score"], v
    hs = closed_forms.main(**dict(small, variant="head_solved", out=str(froot / "head_avg"), batches=4, **on_plain))
    assert hs["as_source"]["equals_stored"] is True
    assert hs["workspace"]["value"] == ":4096:8" and isinstance(hs["nondeterministic"], list)
    assert not any("attention" in m.lower() for m in hs["nondeterministic"]), hs["nondeterministic"]
    r = content.main(**dict(small, variant="plain", out=str(croot / "det_warn"), det="warn", score=0, sheet=0))
    assert any("attention" in m.lower() for m in r["nondeterministic"]), r["nondeterministic"]
