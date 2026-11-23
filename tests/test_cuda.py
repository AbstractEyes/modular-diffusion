"""The experiment path on a card: the same end-to-end runs as the CPU suite's, with every tensor on the card (a unit
moved to a device while its probe, mask or reference stays behind is caught only here).

Off by default: set MD_TEST_CUDA=1 on a machine with a card (the suite itself runs with the card hidden). Small and short:
width 32, a few steps per part, under a gigabyte of card memory.

The third function runs its jobs with det=on (``determinism``): by default the attention kernel's backward pass is not
deterministic on a card, so two runs of one job can differ in the last digits; with the switch on they must be equal.
The matrix library's fixed workspace has to be in the environment before anything touches the card, so it is set here,
before torch is imported.

The fourth function (the address frame, identity-born and solve-born parts, the thaw pass, members on an averaged body)
also runs on the CPU with MD_TEST_DEVICE=cpu, which checks its logic where no card is free; only a card checks a card.
So does the fifth (parts in a tree of stage copies, a row of the class table as a part, grafts, chain reads on an
averaged tree)."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402

from geolip.modular_diffusion.experiments import chain_reads, closed_forms, content, time_split  # noqa: E402
from geolip.modular_diffusion.staged import link_staged  # noqa: E402

DEV = os.environ.get("MD_TEST_DEVICE", "cuda")          # cpu: the fourth function's logic without a card
pytestmark = pytest.mark.skipif(os.environ.get("MD_TEST_CUDA") != "1" or (DEV == "cuda" and not torch.cuda.is_available()),
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


def test_frame_births_thaw_and_members_on_the_card(synthetic_npz, tmp_path):
    """Every job with det=on. The address frame gives the stored gates on the device, bit for bit, and a job through
    it repeats; identity-born and solve-born experts (ten, and thirty by class and stage): a solve-born model at birth
    IS the model of the solved class heads; a class snapped in by a solve; the thaw pass at every level on a tails
    build and on a grown one; band members and one unfiltered member on an averaged body; the reference run with a
    scored tail average."""
    from geolip.modular_diffusion import frame
    from geolip.modular_diffusion.experiments import stages
    from geolip.modular_diffusion.windows import window_weight
    small = dict(SMALL, device=DEV, data=synthetic_npz, ref=str(tmp_path / "reference.npz"), det="on")

    # the frame on the device: the gates as the units computed them before, and exact silence in the amplitude forms
    gen = torch.Generator().manual_seed(0)
    t = torch.cat([torch.rand(500, generator=gen), torch.tensor([0.0, 0.19, 0.25, 0.31, 0.59, 0.65, 0.71, 1.0])]).to(DEV)
    cls = torch.randint(0, 11, (t.shape[0],), generator=gen).to(DEV)
    owners = [frame.owner(lo, hi, 0.06, [c]) for lo, hi in ((0.0, 0.25), (0.25, 0.65), (0.65, 1.0)) for c in range(10)]
    w = frame.projective(owners, t, cls)
    for i, o in enumerate(owners):
        ref = window_weight(t, o.lo, o.hi, o.soft)
        ref = ref * o.class_mask(ref.device)[cls]
        assert torch.equal(w[:, i], ref), i
    for c in range(10):                                   # a class's three stage parts hand over: their gates sum to one
        rows = cls == c
        total = sum(w[rows, s * 10 + c] for s in range(3))
        assert float((total - 1).abs().max()) < 1e-6
    assert float(w[cls == 10].abs().max()) == 0.0         # the null class is on no part's address
    for form in ("plain", "discounted"):
        a = frame.aleph(owners, t, cls, U=5.0, form=form)
        assert a.device.type == t.device.type and bool(((w == 0) == (a == 0)).all()), form

    croot = tmp_path / "content"
    base = {}
    for v in ("plain", "blind", "blind9"):
        base[v] = content.main(**dict(small, variant=v, out=str(croot / v), reader=1, avg="g8"))
    on = lambda v: dict(source=str(croot / v), source_model="model_avg_g8")
    part = dict(small, reader=1, avg="g8", score_last=0)
    # a stored kind of job through the frame repeats to the last digit
    one = content.main(**dict(part, variant="blind_experts_all", out=str(croot / "all_1"), **on("blind")))
    two = content.main(**dict(part, variant="blind_experts_all", out=str(croot / "all_2"), **on("blind")))
    assert one["loss_over_levels"] == two["loss_over_levels"]
    assert one["averaged"]["g8"]["loss_over_levels"] == two["averaged"]["g8"]["loss_over_levels"]

    froot = tmp_path / "closed"
    heads = {v: closed_forms.main(**dict(small, variant=v, out=str(froot / v), batches=4, reader=1, **on("blind")))
             for v in ("class_heads", "class_stage_heads")}
    e = {}
    for v in ("blind_experts10_id", "blind_experts10_solved", "blind_experts10_id_1500", "blind_experts10_solved_1500",
              "blind_experts30_id", "blind_experts30_solved"):
        e[v] = content.main(**dict(part, variant=v, out=str(croot / v), batches=4, **on("blind")))
        assert e[v]["device"] == DEV and e[v]["as_source"]["equals_stored"] is True, v
        assert e[v]["exact"]["others_max_abs"] == 0.0, (v, e[v]["exact"])
        assert "class_read" in e[v]["averaged"]["g8"]["score"], v
        assert ("at_birth" in e[v]) == ("solved" in v), v
    assert len(e["blind_experts30_id"]["parts"][0]["trainable"]) == 30
    # born by a solve: before any training the model is the model of the solved heads
    assert e["blind_experts10_solved"]["at_birth"]["loss_over_levels"] == heads["class_heads"]["loss_over_levels"]
    assert e["blind_experts30_solved"]["at_birth"]["loss_over_levels"] == heads["class_stage_heads"]["loss_over_levels"]
    assert "at_birth" not in e["blind_experts10_id"] and "solve" in e["blind_experts10_solved"]

    # a class the body never saw: a head by a solve on that class's pictures, and a solve-born expert trained on them
    h9 = closed_forms.main(**dict(small, variant="class_heads", out=str(froot / "h9"), batches=4, reader=1, classes="9",
                                  only_classes="9", **on("blind9")))
    h9s = closed_forms.main(**dict(small, variant="class_stage_heads", out=str(froot / "h9s"), batches=4, reader=1,
                                   classes="9", only_classes="9", **on("blind9")))
    s9 = content.main(**dict(part, variant="snap9_solved", out=str(croot / "snap9_solved"), batches=4, **on("blind9")))
    old9 = content.main(**dict(part, variant="snap9", out=str(croot / "snap9_avg"), **on("blind9")))
    for name, r in (("h9", h9), ("h9s", h9s), ("snap9_solved", s9), ("snap9", old9)):
        assert r["as_source"]["equals_stored"] is True and r["exact"]["others_max_abs"] == 0.0, (name, r["exact"])
    assert s9["at_birth"]["loss_over_levels"] == h9["loss_over_levels"]

    # the thaw pass: from a tails build at every level, and from a grown build
    sroot = tmp_path / "stages"
    time_split.main(**dict(small, variant="plain_d4", out=str(sroot / "d4"), avg="g8", score=0, sheet=0))
    stem = dict(stem_from=str(sroot / "d4"), stem_model="model_avg_g8")
    stages.main(**dict(small, variant="tails", out=str(sroot / "tails"), total=6000, cuts="", score=0, sheet=0, **stem))
    stages.main(**dict(small, variant="grow", out=str(sroot / "grow"), blocks=2, steps_each=2000, score=0, sheet=0, **stem))
    th = {}
    for level in ("tail", "1", "2", "all"):
        th[level] = stages.main(**dict(small, variant="thaw", out=str(sroot / f"thaw_{level}"), base=str(sroot / "tails"),
                                       thaw=level, total=2000, reader=1, score_last=0))
    th["grown"] = stages.main(**dict(small, variant="thaw", out=str(sroot / "thaw_grown"), base=str(sroot / "grow"),
                                     thaw="all", total=2000, reader=1, score_last=0))
    for level, r in th.items():
        assert r["device"] == DEV and r["as_source"]["equals_stored"] is True, level
        assert all(p["frozen_unchanged"] for p in r["parts"]), level
        assert set(r["averaged"]) == {"g8", "tail100"} and r["score"] is None, level
        assert "class_read" in r["averaged"]["tail100"]["score"] and r["averaged"]["g8"]["score"] is None, level
        assert r["thaw"]["stream_seed"] != r["seed"] and r["parts"][0]["seconds_per_step"] > 0, level
        if DEV == "cuda":
            assert r["parts"][0]["peak_memory_mib"] > 0, level
    trained = [th[k]["numbers"]["trained"] for k in ("tail", "1", "2", "all")]
    assert trained == sorted(trained) and len(set(trained)) == 4, trained
    assert th["all"]["numbers"]["trained"] == th["all"]["numbers"]["all"]

    # band members and one unfiltered member on an averaged body; the reference run with a scored tail average
    m3 = closed_forms.main(**dict(small, variant="members_avg", out=str(froot / "members_avg"), reader=1, score_last=0,
                                  **on("plain")))
    m1 = closed_forms.main(**dict(small, variant="member_all_avg", out=str(froot / "member_all_avg"), reader=1,
                                  score_last=0, **on("plain")))
    for name, r in (("members_avg", m3), ("member_all_avg", m1)):
        assert r["as_source"]["equals_stored"] is True and "class_read" in r["averaged"]["g8"]["score"], name
    assert abs(m3["averaged"]["g8"]["independence_ratio_over_levels"] - 1) < 1e-5
    ref = stages.main(**dict(small, variant="source", out=str(sroot / "ref_long"), total=6500, avg="g8,tail100", reader=1,
                             score_last=0, score_avg="tail100"))
    assert "class_read" in ref["averaged"]["tail100"]["score"] and ref["averaged"]["g8"]["score"] is None


def test_tree_parts_grafts_and_chain_reads_on_the_card(synthetic_npz, tmp_path):
    """Every job with det=on. A blind source born with ten parts and copies of it; parts born at the fork of a plain
    source; the five parts specs on a finished tree; a graft without training and one with; one stage's parts for every
    stage; the chain read on an averaged tree; one job repeated gives the same numbers."""
    from geolip.modular_diffusion.experiments import stages
    small = dict(SMALL, device=DEV, data=synthetic_npz, ref=str(tmp_path / "reference.npz"), det="on", score=0,
                 sheet=0)
    j = lambda n: str(tmp_path / n)
    sp = stages.main(**dict(small, variant="source", out=j("src_blind_parts"), class_window="blind",
                            parts="experts10"))
    assert sp["device"] == DEV and sp["birth_gap"] == 0.0 and sp["exact"]["others_max_abs"] == 0.0
    cp = stages.main(**dict(small, variant="copies", cuts="0.25,0.65", out=j("cop_parts"), source=j("src_blind_parts")))
    assert cp["exact"]["others_max_abs"] == 0.0 and "parts_off" in cp
    stages.main(**dict(small, variant="source", out=j("src")))
    fork = stages.main(**dict(small, variant="copies", cuts="0.25,0.65", out=j("fork"), source=j("src"),
                              parts="shared1"))
    assert all(p["birth_gap"] == 0.0 and p["optimizer_new"] for p in fork["parts"])
    stages.main(**dict(small, variant="copies", cuts="0.25,0.65", out=j("cop"), source=j("src")))
    stages.main(**dict(small, variant="copies", cuts="0.25,0.65", out=j("sib"), source=j("src"), stream=1))
    stages.main(**dict(small, variant="copies", out=j("one"), source=j("src")))
    res = {}
    for spec in ("experts10", "shared1", "expert:9", "row:9", "row+expert:9"):
        kw = dict(only_classes="9") if ":" in spec else {}
        tag = spec.replace(":", "_").replace("+", "_")
        res[spec] = stages.main(**dict(small, variant="parts", out=j("p_" + tag), base=j("cop"), parts=spec, **kw))
        r = res[spec]
        assert r["device"] == DEV and r["as_source"]["equals_stored"] is True, spec
        assert r["parts_off"]["equals_base"] is True and r["exact"]["others_max_abs"] == 0.0, spec
        if spec.startswith("row"):
            assert r["row_birth_gap"] == 0.0 and all(p["rows_frozen_equal"] for p in r["parts"]), spec
        else:
            assert r["birth_gap"] == 0.0, spec
    again = stages.main(**dict(small, variant="parts", out=j("p_row_9_again"), base=j("cop"), parts="row:9",
                               only_classes="9"))
    assert again["loss_over_levels"] == res["row:9"]["loss_over_levels"]
    assert again["averaged"]["g8"]["loss_over_levels"] == res["row:9"]["averaged"]["g8"]["loss_over_levels"]
    graft = dict(small, variant="parts", parts="row+expert:9", only_classes="9", init_from=j("p_row_expert_9"))
    g0 = stages.main(**dict(graft, out=j("g0"), base=j("cop"), total=0))
    assert g0["graft"]["parents_equal"] == [True, True, True]
    assert g0["averaged"]["g8"]["loss_over_levels"] == res["row+expert:9"]["averaged"]["g8"]["loss_over_levels"]
    g1 = stages.main(**dict(graft, out=j("g1"), base=j("sib"), total=300))
    assert False in g1["graft"]["parents_equal"] and g1["exact"]["others_max_abs"] == 0.0
    stages.main(**dict(small, variant="parts", out=j("row_one"), base=j("one"), parts="row:9", only_classes="9"))
    gs = stages.main(**dict(small, variant="parts", out=j("g_stage0"), base=j("cop"), parts="row:9", only_classes="9",
                            init_from=j("row_one"), init_stage=0, total=0))
    assert gs["exact"]["others_max_abs"] == 0.0 and gs["graft"]["init_stage"] == 0
    stages.main(**dict(small, variant="source", seed=1, out=j("src1")))
    stages.main(**dict(small, variant="copies", seed=1, cuts="0.25,0.65", out=j("cop1"), source=j("src1")))
    tw = chain_reads.main(variant="twin", base=j("cop") + "/avg_tail100", donor=j("cop1") + "/avg_tail100", stage=1,
                          data=synthetic_npz, out=j("twin"), device=DEV, heldout=64, score=0, det="on")
    assert isinstance(tw["loss_over_levels"], float)
