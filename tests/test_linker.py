"""Unit files, the linker and conversion: snap_out restores the start (test 8), the linker refuses a changed file and
computes the stale closure (test 9), members trained alone merge like members trained together (test 11), and a flat
weight file converted to a build folder reads exactly like the flat model."""
import filecmp
import json
import os
import shutil
import zlib

import pytest
import torch
from safetensors.torch import save_file

from geolip.modular_diffusion import compat, train
from geolip.modular_diffusion.beds import make_bed
from geolip.modular_diffusion.build import (BUILD_FILE, CARD_SUFFIX, content_hash, load_unit, read_build, read_card, read_weights,
                                            save_unit, UnitCard, unit_files)
from geolip.modular_diffusion.linker import (StaleUnit, export_build, from_legacy, link, recorded_parents, snap_in,
                                             snap_out, stale_list)
from geolip.modular_diffusion.model import Model
from geolip.modular_diffusion.read import read

FIELD = (3, 32, 32)


def _args(**kw):
    return dict(train.DEFAULTS, bed="pictures", dim=64, anchor=1, anchor_in=1, **kw)


def _spectrum():
    return torch.rand(FIELD, generator=torch.Generator().manual_seed(1), dtype=torch.float64) + 0.1


def _perturb(model, names=None, salt=""):
    """Give the named units (default: all) small seeded values on top of their weights, as training would."""
    with torch.no_grad():
        for u in model.units:
            if names is not None and u.name not in names:
                continue
            for k, p in u.named_parameters():
                g = torch.Generator().manual_seed(zlib.crc32(f"{salt}{u.name}/{k}".encode()))
                p.add_(torch.randn(p.shape, generator=g) * 0.02)


def _probe(n=12, seed=3):
    gen = torch.Generator().manual_seed(seed)
    zt = torch.randn(n, *FIELD, generator=gen)
    t = torch.linspace(0.0, 1.0, n)
    cls = torch.arange(n) % 11
    return zt, t, cls


def _velocity(model):
    with torch.no_grad():
        return model.velocity(*_probe())


def _body(tmp_path):
    """A trained-looking body (anchor, stem, head) exported to a folder."""
    model = Model.born(compat.build_from_args(_args(base=0), FIELD), 0, _spectrum())
    _perturb(model)
    folder = str(tmp_path / "body")
    export_build(model, folder, seed=0)
    return model, folder


def _solo(tmp_path, body_model, body_folder, extra, salt="", name="solo"):
    """One member trained alone on the frozen body, exported against the body's folder."""
    model = Model.born(compat.build_from_args(_args(base=0, extra=extra), FIELD), 0, _spectrum())
    for u in body_model.units:
        model.unit(u.name).load_owned(u.owned_state())
    new = [m.name for m in model.members]
    _perturb(model, new, salt)
    folder = str(tmp_path / name)
    export_build(model, folder, seed=0, base=body_folder, trained=new, marks={"frozen_unchanged": True})
    return model, folder, new[0]


def _snapshot(folder):
    return {n: open(os.path.join(folder, n), "rb").read() for n in sorted(os.listdir(folder))}


# ------------------------------------------------------------------ test 8
def test_8_snap_in_then_out_restores_the_start(tmp_path):
    body, bf = _body(tmp_path)
    start = _snapshot(bf)
    v0 = _velocity(link(bf))
    assert torch.equal(v0, _velocity(body))
    solo, sf, member = _solo(tmp_path, body, bf, "1@0.25-0.65")
    b = snap_in(bf, os.path.join(sf, member + CARD_SUFFIX))
    assert b.names()[-1] == member
    merged = link(bf)
    assert torch.equal(_velocity(merged), _velocity(solo))
    assert not torch.equal(_velocity(merged), v0)
    with pytest.raises(StaleUnit) as err:                       # the head and the member were fit with the stem
        snap_out(bf, "stem")
    assert err.value.unit == "stem" and err.value.stale == ["head", member]
    assert _snapshot(bf) != start
    snap_out(bf, member)
    assert _snapshot(bf) == start                               # the build file, the hashes and every file
    assert torch.equal(_velocity(link(bf)), v0)


def test_8_newborn_snap_in_measures_born_gap(tmp_path):
    body, bf = _body(tmp_path)
    start = _snapshot(bf)
    nb = Model.born(compat.build_from_args(_args(base=0, extra="2@0-1"), FIELD), 0, _spectrum())
    for u in body.units:
        nb.unit(u.name).load_owned(u.owned_state())
    folder = str(tmp_path / "newborn")
    export_build(nb, folder, seed=0, base=bf, trained=["x_b2_0-1"], marks={"frozen_unchanged": True})
    card = read_card(os.path.join(folder, "x_b2_0-1" + CARD_SUFFIX))
    assert card.birth["hash"] == card.hash and card.marks["born_gap"] == 0.0
    snap_in(bf, os.path.join(folder, "x_b2_0-1" + CARD_SUFFIX))
    assert torch.equal(_velocity(link(bf)), _velocity(body))
    snap_out(bf, "x_b2_0-1")
    assert _snapshot(bf) == start


# ------------------------------------------------------------------ test 9
def test_9_linker_refuses_a_changed_file_and_names_it(tmp_path):
    model = Model.born(compat.build_from_args(_args(extra="1@0.25-0.65", content="0.25-0.65:2", stage_body="3"), FIELD),
                       0, _spectrum())
    _perturb(model)
    folder = str(tmp_path / "b")
    export_build(model, folder, seed=0)
    linked = link(folder)
    assert torch.equal(_velocity(linked), _velocity(model))
    for u in model.units:                                         # the hash is stable across save and load
        assert content_hash(linked.unit(u.name)) == content_hash(u) == read_build(folder).entry(u.name).hash
        unit, card = load_unit(folder, u.name)
        assert content_hash(unit) == card.hash
    wpath = unit_files(folder, "stem")[0]
    sd = read_weights(wpath)
    k = sorted(sd)[3]
    sd[k] = sd[k].clone()
    sd[k].view(-1)[0] += 1e-6
    save_file(sd, wpath)
    with pytest.raises(StaleUnit) as err:
        link(folder)
    assert err.value.unit == "stem" and "content hash" in err.value.reason


def test_9_stale_list_and_unvouched_pairs(tmp_path):
    body, bf = _body(tmp_path)
    _, sa, ma = _solo(tmp_path, body, bf, "1@0-0.65", salt="a", name="solo_a")
    _, sb, mb = _solo(tmp_path, body, bf, "0@0-1", salt="b", name="solo_b")
    snap_in(bf, os.path.join(sa, ma + CARD_SUFFIX))
    snap_in(bf, os.path.join(sb, mb + CARD_SUFFIX))
    b = read_build(bf)
    # the closure of "lists X as a parent": the body's units list each other, the members list the body
    assert stale_list(b, "stem") == ["head", ma, mb]
    assert stale_list(b, "head") == ["stem", ma, mb]
    assert stale_list(b, "anchor") == ["stem", "head", ma, mb]
    assert stale_list(b, ma) == [] and stale_list(b, mb) == []
    assert recorded_parents(b, ma) == ["anchor", "stem", "head"]   # bands 1 and 0 do not interact
    # a second member of band 1, trained apart from the first, overlaps it in band and window: neither card vouches
    _, sc, mc = _solo(tmp_path, body, bf, "1@0.25-1#c", salt="c", name="solo_c")
    before = _snapshot(bf)
    with pytest.raises(StaleUnit) as err:
        snap_in(bf, os.path.join(sc, mc + CARD_SUFFIX))
    assert {ma, mc} <= set(err.value.stale)
    assert _snapshot(bf) == before                                # the refused snap_in left the folder as it was


def test_9_card_round_trip(tmp_path):
    model = Model.born(compat.build_from_args(_args(), FIELD), 0, _spectrum())
    u = model.unit("b1_s0")
    card = save_unit(u, UnitCard(name=u.name, type=u.unit_type, parents={"stem": "0" * 64}), str(tmp_path))
    again = read_card(unit_files(str(tmp_path), u.name)[1])
    assert again == card and again.hash == content_hash(u)
    unit, _ = load_unit(str(tmp_path), u.name)
    for k, v in u.owned_state().items():
        assert torch.equal(v, unit.owned_state()[k]), k


# ------------------------------------------------------------------ test 11
def test_11_members_trained_alone_merge_like_members_trained_together(tmp_path):
    common = dict(bed="fields", base=0, keep=1, codes="rung", anchor=1, anchor_in=1, dim=64, batch=16, heldout=128,
                  log_every=10, device="cpu", seeds="0", sheet=0)
    body = train.main(out=str(tmp_path / "body"), steps=30, **common)[0]
    body_pt = os.path.join(tmp_path / "body", body["tag"] + ".pt")
    init = dict(init=body_pt, train="new", equalize=0, steps=20)
    solos = [train.main(out=str(tmp_path / f"solo{b}"), extra=f"{b}@0-1", **init, **common)[0] for b in range(3)]
    joint = train.main(out=str(tmp_path / "joint"), extra="0@0-1,1@0-1,2@0-1", **init, **common)[0]
    assert all(r["frozen_unchanged"] for r in solos + [joint])

    bed = make_bed(dict(train.DEFAULTS, **common), "cpu")
    bf = str(tmp_path / "merged")
    from_legacy(body_pt, bf, bed.spectrum)
    for b, r in enumerate(solos):
        sf = str(tmp_path / f"solo{b}_units")
        from_legacy(os.path.join(tmp_path / f"solo{b}", r["tag"] + ".pt"), sf, bed.spectrum, base=bf)
        for shared in ("anchor", "stem", "head"):                 # the frozen units kept the body's cards
            assert filecmp.cmp(unit_files(sf, shared)[1], unit_files(bf, shared)[1], shallow=False)
        snap_in(bf, os.path.join(sf, f"x_b{b}_0-1" + CARD_SUFFIX))
    merged = read(link(bf), bed, common["heldout"], "cpu")
    for lv in joint["levels"]:
        assert abs(merged["levels"][lv]["error"]["all"] - joint["levels"][lv]["error"]["all"]) < 1e-6, lv
    gain_sum = sum(body["loss_over_levels"] - r["loss_over_levels"] for r in solos)
    gain = body["loss_over_levels"] - merged["loss_over_levels"]
    assert abs(gain - gain_sum) < 1e-6, (gain, gain_sum)
    assert gain != 0.0                     # the members moved (on Gaussian fields the anchor is already optimal)


# ------------------------------------------------------------------ flat files
def test_legacy_round_trip_through_units(tmp_path):
    out = str(tmp_path / "run")
    r = train.main(bed="fields", anchor=1, stages="3", content="", dim=64, steps=3, batch=8, heldout=32, device="cpu",
                   out=out, sheet=0)[0]
    pt = os.path.join(out, r["tag"] + ".pt")
    bed = make_bed(dict(train.DEFAULTS, bed="fields"), "cpu")
    folder = str(tmp_path / "units")
    from_legacy(pt, folder, bed.spectrum)
    flat = torch.load(pt)
    again = compat.to_legacy_state(link(folder))
    assert list(flat) == list(again)
    for k in flat:
        assert torch.equal(flat[k], again[k]), k


def test_stored_base_file_as_a_build_folder_reads_the_same(stored_runs, tmp_path):
    """The stored 6,000-step base weights, converted to unit files and linked, read exactly like the flat model."""
    from conftest import STORED_BASE_PT, find_stored, NPZ_SHA256, sha256
    pt = find_stored(stored_runs, "_pictures_b3k_rung_st1_anchor_ain1_s0_steps6000_d128.pt", STORED_BASE_PT)
    npz = os.environ.get("MD_CIFAR_NPZ", "")
    if not npz:
        pytest.skip("MD_CIFAR_NPZ is not set")
    assert sha256(npz) == NPZ_SHA256
    bed = make_bed(dict(train.DEFAULTS, bed="pictures", data=npz), "cpu")
    folder = str(tmp_path / "base")
    b = from_legacy(pt, folder, bed.spectrum)
    assert b.names() == ["anchor", "stem", "head", "b0_s0", "b1_s0", "b2_s0"]
    flat, _ = compat.units_from_legacy(pt, None, bed.spectrum)
    linked = link(folder)
    r_flat = read(flat, bed, 256, "cpu")
    r_link = read(linked, bed, 256, "cpu")
    assert json.dumps(r_flat, sort_keys=True) == json.dumps(r_link, sort_keys=True)
    state = torch.load(pt)
    again = compat.to_legacy_state(linked)
    assert list(state) == list(again) and all(torch.equal(state[k], again[k]) for k in state)
