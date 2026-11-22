"""Build folders with a Body (test 9): the round trip through export and link, the parents by rule 2, adopt checks the
hash, a stem adopted and frozen keeps its hash and leaves its source folder byte-identical, the stale list."""
import os

import pytest
import torch
from safetensors.torch import save_file

from geolip.modular_diffusion import train
from geolip.modular_diffusion.beds import make_bed
from geolip.modular_diffusion.build import content_hash, read_build, read_card, read_weights, unit_files
from geolip.modular_diffusion.forms import adopt, cluster_build
from geolip.modular_diffusion.linker import export_build, link, recorded_parents, stale_list
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.train import fit_units


def _bed():
    return make_bed(dict(train.DEFAULTS, bed="fields"), "cpu")


def _snapshot(folder):
    return {n: open(os.path.join(folder, n), "rb").read() for n in sorted(os.listdir(folder))}


def _probe(bed, n=8):
    gen = torch.Generator().manual_seed(5)
    return (torch.randn((n,) + tuple(bed.field_shape), generator=gen), torch.linspace(0, 1, n),
            torch.full((n,), 10, dtype=torch.long))


def test_9_round_trip_parents_and_stale_list(tmp_path):
    bed = _bed()
    model = Model.born(cluster_build(bed.field_shape, stem_depth=1, body_depth=2, dim=32), 0, bed.spectrum)
    fit_units(model, bed, 0, steps=3, batch=8, heldout=32, warmup=1, log=lambda s: None)
    folder = str(tmp_path / "m")
    b = export_build(model, folder, seed=0)
    assert b.names() == ["anchor", "stem", "body", "head"]
    linked = link(folder)
    with torch.no_grad():
        assert torch.equal(linked.velocity(*_probe(bed)), model.velocity(*_probe(bed)))
    assert linked.body is not None and content_hash(linked.body) == content_hash(model.body)
    assert recorded_parents(b, "body") == ["anchor", "stem", "head"]
    assert recorded_parents(b, "head") == ["anchor", "stem", "body"]
    assert read_card(unit_files(folder, "body")[1]).birth["state"] == "random"
    assert stale_list(read_build(folder), "stem") == ["body", "head"]


def test_9_adopt_a_frozen_stem(tmp_path):
    bed = _bed()
    src = Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum)
    fit_units(src, bed, 0, steps=3, batch=8, heldout=32, warmup=1, log=lambda s: None)
    sf = str(tmp_path / "src")
    export_build(src, sf, seed=0)
    start = _snapshot(sf)
    stem_hash = read_build(sf).entry("stem").hash
    new = Model.born(cluster_build(bed.field_shape, stem_depth=1, body_depth=1, dim=32), 0, bed.spectrum)
    adopt(new, sf, ["stem"])
    assert content_hash(new.stem) == stem_hash
    res = fit_units(new, bed, 0, steps=3, trainable=["body", "head"], batch=8, heldout=32, warmup=1, log=lambda s: None)
    assert res["frozen_unchanged"] is True and content_hash(new.stem) == stem_hash
    nf = str(tmp_path / "new")
    export_build(new, nf, seed=0, base=sf, trained=["body", "head"], marks={"frozen_unchanged": True})
    assert _snapshot(sf) == start
    linked = link(nf)
    assert content_hash(linked.stem) == stem_hash
    assert read_card(unit_files(nf, "stem")[1]) == read_card(unit_files(sf, "stem")[1])      # the source's card kept
    assert stale_list(read_build(nf), "stem") == ["body", "head"]
    # a changed weight file is refused
    wpath = unit_files(sf, "stem")[0]
    sd = read_weights(wpath)
    k = sorted(sd)[0]
    sd[k] = sd[k].clone()
    sd[k].view(-1)[0] += 1e-6
    save_file(sd, wpath)
    with pytest.raises(ValueError, match="content hash"):
        adopt(Model.born(cluster_build(bed.field_shape, stem_depth=1, dim=32), 0, bed.spectrum), sf, ["stem"])
    with pytest.raises(ValueError):                                       # another stem depth: not the same unit
        adopt(Model.born(cluster_build(bed.field_shape, stem_depth=2, dim=32), 0, bed.spectrum), sf, ["stem"])
