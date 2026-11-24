"""Capture runs: what one band engine trained three ways needs from the library. The trainer's random start, its reads
during training (they change no number of the training, and a resumed job keeps the same list), a finished job without
its checkpoint (read back, never trained again); the experiment's probe made of held-out rows in file order, its read
curve and its exchange variant; sampling a probe in groups (two models start from the same noise); the file-order row
reader; parquet rows after a skip; shipping a folder without its weights."""
import io
import json
import os
import types

import pytest
import torch

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion import hub  # noqa: E402
from geolip.modular_diffusion.capture import BANDS, reads  # noqa: E402
from geolip.modular_diffusion.capture import train as T  # noqa: E402
from geolip.modular_diffusion.capture.assemble import initial_noise, sample  # noqa: E402
from geolip.modular_diffusion.capture.data import (parquet_pictures, read_index, read_shard,  # noqa: E402
                                                   rows_in_order, synthetic_shards)
from geolip.modular_diffusion.capture.engine import build_engine, fresh_engine, param_count  # noqa: E402
from geolip.modular_diffusion.capture.teacher import tiny_teacher  # noqa: E402
from geolip.modular_diffusion.experiments import capture  # noqa: E402
from geolip.modular_diffusion.experiments.capture import random_source  # noqa: E402

PROMPTS = ["a red house", "two boats", "a cat"]


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


@pytest.fixture(scope="module")
def probe(teacher):
    states = teacher.text_states(teacher.tokenize(PROMPTS))
    return reads.make_probe(states, seed=11, per_band=2, latent_shape=(4, 16, 16), empty=teacher.empty_states(1))


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sorted(sa) == sorted(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


# ------------------------------------------------------------------ the trainer
def test_reads_during_training_change_no_number(teacher, probe, tmp_path):
    source = random_source(0, (4, 16, 16))
    kw = dict(steps=5, batch=2, seed=4, lr=1e-3, rho=0.2, log_every=0, log=lambda s: None)
    plain = T.train_band_engine(teacher, "tiny", "clean", source, out=str(tmp_path / "plain"), **kw)
    modes = []

    def on_every(n, engine, averaged):
        modes.append((engine.training, averaged.training))
        torch.randn(7)                                            # the global generator: the trainer never uses it
        live = reads.band_read(engine, teacher, "clean", probe)["mse"]
        mean = reads.band_read(averaged, teacher, "clean", probe)["mse"]
        return {"live": live, "averaged": mean}

    read = T.train_band_engine(teacher, "tiny", "clean", source, out=str(tmp_path / "read"), every=2,
                               on_every=on_every, **kw)
    assert _same(plain["engine"], read["engine"]) and _same(plain["average"], read["average"])
    assert plain["log"] == read["log"] and plain["reads"] == []
    assert [r["step"] for r in read["reads"]] == [2, 4, 5]                 # every 2 updates and after the last
    assert modes == [(False, False)] * 3 and read["engine"].training      # eval for the call, train again after
    last = read["reads"][-1]["value"]
    assert last["averaged"] == reads.band_read(read["average"], teacher, "clean", probe)["mse"]
    assert last["live"] == reads.band_read(read["engine"], teacher, "clean", probe)["mse"]
    with open(tmp_path / "read" / "train.json", encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored["reads"] == read["reads"] and stored["trained"] == param_count(read["engine"])
    assert stored["seconds_a_step"] > 0 and stored["seconds_a_step"] == read["seconds_a_step"]
    assert stored["settings"]["init"] == "inherit"


def test_a_resumed_job_keeps_the_same_reads(teacher, probe, tmp_path):
    source = random_source(2, (4, 16, 16))

    def on_every(n, engine, averaged):
        return {"averaged": reads.band_read(averaged, teacher, "middle", probe)["mse"], "n": n}

    kw = dict(steps=7, batch=2, seed=1, lr=1e-3, save_every=2, log_every=0, log=lambda s: None, every=3,
              on_every=on_every)
    whole = T.train_band_engine(teacher, "tiny", "middle", source, out=str(tmp_path / "whole"), **kw)
    first = T.train_band_engine(teacher, "tiny", "middle", source, out=str(tmp_path / "parts"), until=4, **kw)
    assert not first["finished"] and [r["step"] for r in first["reads"]] == [3]
    ck = torch.load(tmp_path / "parts" / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ck["reads"] == first["reads"] and len(ck["step_seconds"]) == 4
    rest = T.train_band_engine(teacher, "tiny", "middle", source, out=str(tmp_path / "parts"), **kw)
    assert rest["finished"] and [r["step"] for r in rest["reads"]] == [3, 6, 7]
    assert rest["reads"] == whole["reads"]                                  # the list of a job that never stopped
    assert _same(whole["engine"], rest["engine"]) and _same(whole["average"], rest["average"])


def test_a_finished_job_without_its_checkpoint_is_not_trained_again(teacher, tmp_path):
    source = random_source(3, (4, 16, 16))
    out = tmp_path / "job"
    kw = dict(steps=3, batch=2, seed=0, lr=1e-3, log_every=0, keep_checkpoint=False)
    one = T.train_band_engine(teacher, "tiny", "noisy", source, out=str(out), log=lambda s: None, **kw)
    assert one["finished"]
    assert sorted(os.listdir(out)) == ["engine.json", "engine.safetensors", "engine_avg.safetensors", "train.json"]
    with open(out / "train.json", "rb") as fh:
        before = fh.read()
    lines = []
    calls = []

    def boom(step, batch):                                       # any batch asked for would mean training again
        calls.append(step)
        return source(step, batch)

    again = T.train_band_engine(teacher, "tiny", "noisy", boom, out=str(out), log=lines.append, **kw)
    assert calls == [] and again["restored"] and again["finished"] and again["step"] == 3
    assert any(x.startswith("FINISHED already") for x in lines)
    assert _same(again["engine"], one["engine"]) and _same(again["average"], one["average"])
    assert again["log"] == one["log"] and again["trained"] == one["trained"]
    assert again["seconds_a_step"] == one["seconds_a_step"]
    assert not os.path.isfile(out / "checkpoint.pt")
    with open(out / "train.json", "rb") as fh:
        assert fh.read() == before
    with pytest.raises(ValueError):                              # other settings: never taken as this job
        T.train_band_engine(teacher, "tiny", "noisy", source, out=str(out), log=lambda s: None, **dict(kw, lr=2e-3))
    os.remove(out / "engine_avg.safetensors")
    with pytest.raises(ValueError, match="neither"):             # never started over
        T.train_band_engine(teacher, "tiny", "noisy", source, out=str(out), log=lambda s: None, **kw)
    # the default keeps the checkpoint, as before
    T.train_band_engine(teacher, "tiny", "noisy", source, out=str(tmp_path / "kept"), log=lambda s: None,
                        **dict(kw, keep_checkpoint=True))
    assert os.path.isfile(tmp_path / "kept" / "checkpoint.pt")


def test_a_random_start_is_fresh_engine_and_not_the_teachers(teacher, tmp_path):
    source = random_source(0, (4, 16, 16))
    kw = dict(steps=2, batch=2, seed=5, lr=1e-3, log_every=0, log=lambda s: None)
    start = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "r0"), init="random", until=0,
                                **kw)
    assert start["step"] == 0 and not start["finished"]
    assert _same(start["engine"], fresh_engine(teacher, "small", seed=5))
    assert not _same(start["engine"], build_engine(teacher, "small"))
    rand = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "r"), init="random", **kw)
    inh = T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "i"), **kw)
    assert not _same(rand["engine"], inh["engine"]) and rand["settings"]["init"] == "random"
    with pytest.raises(ValueError):                                       # init must match on resume
        T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "r0"), **kw)
    with pytest.raises(ValueError):
        T.train_band_engine(teacher, "small", "clean", source, out=str(tmp_path / "x"), init="zeros", **kw)
    # a checkpoint made before init was a setting counts as inherit
    old = tmp_path / "old"
    T.train_band_engine(teacher, "small", "clean", source, out=str(old), until=1, **kw)
    ck = torch.load(old / "checkpoint.pt", map_location="cpu", weights_only=True)
    del ck["settings"]["init"]
    torch.save(ck, old / "checkpoint.pt")
    with pytest.raises(ValueError):
        T.train_band_engine(teacher, "small", "clean", source, out=str(old), init="random", **kw)
    done = T.train_band_engine(teacher, "small", "clean", source, out=str(old), **kw)
    assert done["finished"] and _same(done["engine"], inh["engine"])


# ------------------------------------------------------------------ data
def test_rows_in_order_across_shard_edges(tmp_path):
    folder = str(tmp_path / "shards")
    index = synthetic_shards(folder, rows=11, shard_size=4, size=4, seed=2)
    assert index["counts"] == [4, 4, 3]
    parts = [read_shard(os.path.join(folder, n)) for n in read_index(folder)["shards"]]
    lat = torch.cat([p[0] for p in parts]).float()
    ids = torch.cat([p[1] for p in parts]).long()
    for first, n in ((0, 11), (3, 2), (4, 4), (2, 7), (7, 4), (10, 1), (0, 1)):
        x, t = rows_in_order(folder, first, n)
        assert x.dtype == torch.float32 and t.dtype == torch.long and x.shape[0] == n
        assert torch.equal(x, lat[first:first + n]) and torch.equal(t, ids[first:first + n]), (first, n)
    with pytest.raises(ValueError, match="11 rows"):
        rows_in_order(folder, 8, 4)
    with pytest.raises(ValueError):
        rows_in_order(folder, 0, 0)


def test_parquet_pictures_after_a_skip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from PIL import Image

    def png(k):
        buf = io.BytesIO()
        Image.new("RGB", (10 + k, 8), (k, 2 * k, 3 * k)).save(buf, format="PNG")
        return buf.getvalue()

    rows = 150                                                  # more than one batch of 64 rows
    table = pa.table({"image": [{"bytes": png(k), "path": None} for k in range(rows)],
                      "answer": [[f"caption {k}"] for k in range(rows)], "id": list(range(rows))})
    path = str(tmp_path / "rows.parquet")
    pq.write_table(table, path, row_group_size=50)
    pics, caps, meta = parquet_pictures(path, 5, skip=70)
    assert caps == [f"caption {k}" for k in range(70, 75)] and [m["id"] for m in meta] == list(range(70, 75))
    assert [p.size for p in pics] == [(10 + k, 8) for k in range(70, 75)]
    assert parquet_pictures(path, 3)[1] == parquet_pictures(path, 3, skip=0)[1] == ["caption 0", "caption 1",
                                                                                     "caption 2"]
    _, caps, meta = parquet_pictures(path, 0, skip=140)         # n 0: every row after the skip
    assert caps == [f"caption {k}" for k in range(140, 150)] and meta[-1]["id"] == 149
    assert len(parquet_pictures(path, 0)[1]) == rows
    _, caps, _ = parquet_pictures(path, 2, skip=63)             # across a batch edge
    assert caps == ["caption 63", "caption 64"]
    with pytest.raises(ValueError, match="fewer"):
        parquet_pictures(path, 11, skip=140)
    with pytest.raises(ValueError, match="none after"):
        parquet_pictures(path, 0, skip=150)
    bad = pa.table({"image": [{"bytes": png(k), "path": None} for k in range(4)], "answer": ["a", "b", " ", "d"]})
    pq.write_table(bad, str(tmp_path / "bad.parquet"))
    with pytest.raises(ValueError, match="row 2 of"):            # the row's number in the file
        parquet_pictures(str(tmp_path / "bad.parquet"), 2, skip=1)
    assert parquet_pictures(str(tmp_path / "bad.parquet"), 1, skip=3)[1] == ["d"]


# ------------------------------------------------------------------ sampling in groups
class _Recorder(torch.nn.Module):
    """A part that keeps the latents it is given at the sampler's first step (one record a sampling group)."""

    def __init__(self, inner, start):
        super().__init__()
        self.inner, self.start, self.first = inner, int(start), []

    def forward(self, z, t, states):
        if int(t.reshape(-1)[0]) == self.start:
            self.first.append(z.detach().clone())
        own = next(self.inner.parameters()).dtype
        return self.inner(z.to(own), t, encoder_hidden_states=states.to(own)).sample.to(z.dtype)


def test_sampling_in_groups_starts_two_models_from_the_same_noise(teacher):
    states = teacher.text_states(teacher.tokenize(PROMPTS + ["a road", "a bowl"]))
    probe = reads.make_probe(states, seed=21, per_band=2, latent_shape=(4, 16, 16), empty=teacher.empty_states(1))
    one = reads.sample_probe(teacher, probe, 4, 3.0, group=0)
    whole = sample(teacher, probe.states, probe.empty, 21, 4, 3.0, latent_shape=(4, 16, 16))
    assert torch.equal(one, whole)                                # one group: the one-batch sampling, to the bit
    assert reads.sample_groups(5, 2) == [(0, 2), (2, 2), (4, 1)] and reads.sample_groups(5, 9) == [(0, 5)]
    assert reads.group_seed(21, 0) == 21 and reads.group_seed(21, 2) == 21 + 2 * 100_003
    grouped = reads.sample_probe(teacher, probe, 4, 3.0, group=2)
    assert grouped.shape == one.shape
    for j, (first, n) in enumerate(reads.sample_groups(5, 2)):     # the rule: group j from group_seed(seed, j)
        z = initial_noise(reads.group_seed(21, j), n, (4, 16, 16))
        want = sample(teacher, probe.states[first:first + n], probe.empty, 0, 4, 3.0, z=z)
        assert torch.equal(grouped[first:first + n], want), j
    # two different models see identical start noise, row by row
    a, b = _Recorder(teacher.unet, 751), _Recorder(build_engine(teacher, "tiny").unet, 751)   # 751: the first of 4
    reads.sample_probe(a, probe, 4, 3.0, group=2)
    reads.sample_probe(b, probe, 4, 3.0, group=2)
    assert len(a.first) == len(b.first) == 3
    assert all(torch.equal(x, y) for x, y in zip(a.first, b.first))


# ------------------------------------------------------------------ the experiment
def _shards(tmp_path, rows=10):
    folder = str(tmp_path / "rows")
    synthetic_shards(folder, rows=rows, shard_size=4, size=16, seed=9)
    return folder


def test_the_probe_from_held_out_rows_in_file_order(teacher, tmp_path):
    from geolip.modular_diffusion.config import parse
    folder = _shards(tmp_path)
    a = parse(capture.DEFAULTS, (), probe_data=folder, probe_first=5, probe_rows=3, chunk=2, probe_per_band=2)
    p = capture._probe(a, teacher)
    x0, ids = rows_in_order(folder, 5, 3)
    assert torch.equal(p.x0, x0) and torch.equal(p.states, teacher.text_states(ids)) and p.batch == 2
    assert p.seed == a["probe_seed"] and p.empty is not None
    with pytest.raises(ValueError):
        capture._probe(dict(a, probe_rows=0), teacher)


def test_the_band_engine_variant_with_held_out_rows_reads_and_a_random_start(tmp_path):
    folder = _shards(tmp_path)
    out = tmp_path / "job"
    args = dict(variant="band_engine", tiny=1, shape="tiny", band="clean", steps=5, batch=2, seed=3, init="random",
                read_every=2, keep_checkpoint=0, probe_data=folder, probe_first=6, probe_rows=4, probe_per_band=2,
                chunk=3, log_every=0, threads=2, out=str(out))
    r = capture.main(**args)
    assert r["train"]["finished"] and r["train"]["init"] == "random" and not r["train"]["checkpoint_kept"]
    assert r["train"]["seconds_a_step"] > 0 and r["train"]["trained"] > 0
    assert r["probe"]["rows"] == 4 and r["probe"]["first"] == 6 and r["probe"]["order"] == "file"
    curve = r["curve"]
    assert [c["step"] for c in curve] == [0, 2, 4, 5]
    assert curve[0]["live"] == curve[0]["averaged"] == r["reads"]["born"]["mse"]
    # the born read is the random start's (fresh_engine from the job's seed), not the inherited engine's
    teacher = tiny_teacher(3)
    p = capture._probe(dict(capture.DEFAULTS, **{k: v for k, v in args.items() if k in capture.DEFAULTS}), teacher)
    want = reads.band_read(fresh_engine(teacher, "tiny", seed=3), teacher, "clean", p)["mse"]
    assert r["reads"]["born"]["mse"] == want and r["reads"]["born"]["init"] == "random"
    assert curve[-1]["averaged"] == r["reads"]["trained_clean"]["mse"]       # the last read is the end read
    assert r["reads"]["teacher_copy"]["mse"] == 0.0
    with open(out / "result.md", encoding="utf-8") as fh:
        assert "during training" in fh.read()
    again = capture.main(**args)                                             # finished: read back, not trained
    assert again["train"]["restored"] and again["curve"] == curve
    assert again["reads"]["trained_clean"] == r["reads"]["trained_clean"]
    inherit = capture.main(**dict(args, init="inherit", out=str(tmp_path / "inh"), read_every=0))
    assert "curve" not in inherit and inherit["reads"]["born"]["mse"] < r["reads"]["born"]["mse"]


def test_the_exchange_variant(tmp_path):
    common = dict(tiny=1, probe_n=3, probe_per_band=2, sample_steps=4, group=2, threads=2)
    job = tmp_path / "job"
    capture.main(variant="band_engine", shape="tiny", band="clean", steps=2, batch=2, out=str(job), log_every=0,
                 **common)
    parts = f"clean={job},middle=teacher,noisy=teacher"
    same = capture.main(variant="exchange", parts=parts, band="clean", other=str(job), out=str(tmp_path / "same"),
                        **common)
    ex = same["exchange"]
    for b in BANDS:                                   # the same part put back for itself: every number exactly 0
        x = ex["bands"][b]
        assert x["change"] == 0.0 and x["largest_change_where_silent"] == 0.0 and x["before"] == x["after"]
    assert ex["final_latent_change"] == 0.0 and ex["sample_group"] == 2
    tsame = capture.main(variant="exchange", parts=parts, band="middle", other="teacher",
                         out=str(tmp_path / "tsame"), **common)
    assert all(tsame["exchange"]["bands"][b]["change"] == 0.0 for b in BANDS)
    assert tsame["exchange"]["final_latent_change"] == 0.0
    copy = capture.main(variant="exchange", parts=parts, band="clean", other="teacher", out=str(tmp_path / "copy"),
                        **common)
    cx = copy["exchange"]["bands"]
    assert all(cx[b]["after"] == 0.0 for b in BANDS)  # the teacher's copy: the reads against the teacher become 0
    assert cx["clean"]["before"] > 0.0 and cx["clean"]["change"] < 0.0
    assert cx["noisy"]["largest_change_where_silent"] == 0.0
    assert copy["exchange"]["final_latent_change"] > 0.0
    with open(tmp_path / "copy" / "result.md", encoding="utf-8") as fh:
        assert "largest change where silent" in fh.read()
    with pytest.raises(ValueError):
        capture.main(variant="exchange", parts=parts, band="clean", out=str(tmp_path / "x"), **common)


def test_the_assembly_variant_samples_in_groups(tmp_path):
    common = dict(tiny=1, probe_n=3, probe_per_band=2, sample_steps=4, threads=2)
    parts = "clean=teacher,middle=teacher,noisy=teacher"
    one = capture.main(variant="assembly", parts=parts, group=0, out=str(tmp_path / "a"), **common)
    two = capture.main(variant="assembly", parts=parts, group=2, out=str(tmp_path / "b"), **common)
    assert one["final_latent_change"] == 0.0 and two["final_latent_change"] == 0.0 and two["sample_group"] == 2


# ------------------------------------------------------------------ shipping without weights
class _FilteringApi:
    """A stand-in for the Hub client that applies the ignore patterns as the Hub's own filter does."""

    def __init__(self):
        self.kw, self.sent = None, None

    def upload_folder(self, **kw):
        self.kw = kw
        root = kw["folder_path"]
        rel = sorted(os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/")
                     for r, _, fs in os.walk(root) for f in fs)
        try:
            from huggingface_hub.utils import filter_repo_objects
            self.sent = sorted(filter_repo_objects(rel, ignore_patterns=kw["ignore_patterns"]))
        except ImportError:
            import fnmatch
            self.sent = [p for p in rel if not any(fnmatch.fnmatch(p, q) for q in kw["ignore_patterns"])]
        return types.SimpleNamespace(oid="c0ffee")


def test_ship_leaves_the_weights_out(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    folder = tmp_path / "runs"
    for rel in ("job/engine.safetensors", "job/engine_avg.safetensors", "job/checkpoint.pt", "job/train.json",
                "job/result.json", "job/log.txt", "top.safetensors", "checkpoint.pt", "result.md", "run.log"):
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    said = []
    api = _FilteringApi()
    assert hub.ship(str(folder), "user/repo", "runs", "m", api=api, say=said.append,
                    ignore=("*.safetensors", "checkpoint.pt")) == "c0ffee"
    assert api.sent == ["job/log.txt", "job/result.json", "job/train.json", "result.md"]
    assert said[0].startswith("shipping 5 files") and "left out" in said[0]       # run.log is counted, as before
    assert all(p in api.kw["ignore_patterns"] for p in hub.IGNORE)                # today's patterns are kept
    plain = _FilteringApi()
    said = []
    hub.ship(str(folder), "user/repo", "runs", "m", api=plain, say=said.append)
    assert plain.kw["ignore_patterns"] == list(hub.IGNORE)                        # an existing caller: unchanged
    assert "job/engine.safetensors" in plain.sent and said[0].startswith("shipping 10 files")
    assert hub.extra_patterns(["a/*.bin", "*.pt", "*.pt"]) == ["a/*.bin", "*.pt", "*/*.pt"]
