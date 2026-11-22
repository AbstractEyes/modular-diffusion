"""The experiment path on a card: the same end-to-end runs as the CPU suite's, with every tensor on the card (a unit
moved to a device while its probe, mask or reference stays behind is caught only here).

Off by default: set MD_TEST_CUDA=1 on a machine with a card (the suite itself runs with the card hidden). Small and short:
width 32, a few steps per part, under a gigabyte of card memory."""
import os

import pytest
import torch

from geolip.modular_diffusion.experiments import time_split
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
