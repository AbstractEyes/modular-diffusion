"""Capture, pictures: latents decoded to pictures; an assembly's latents sampled group by group with the probe's rule
(the same start noise as ``reads.sample_probe``); the kernel distance written out by hand, its blocks, and a control
that must fail; the text-to-picture judges; the contact sheet; the tiny image decoder; the caption picked from a list;
the parts born from the teacher; the pictures and picture_compare variants on the tiny teacher."""
import io
import json
import os
import shutil

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion.capture import BANDS, pictures, reads  # noqa: E402
from geolip.modular_diffusion.capture.assemble import Assembly  # noqa: E402
from geolip.modular_diffusion.capture.schedule import band_weights  # noqa: E402
from geolip.modular_diffusion.capture.teacher import tiny_teacher, tiny_vae  # noqa: E402
from geolip.modular_diffusion.experiments import capture  # noqa: E402

PROMPTS = ["a red house", "two boats", "a cat", "a road"]
SHAPE = (4, 16, 16)


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0, vae=True)


# ------------------------------------------------------------------ decoding and the decoder
def test_tiny_vae_and_the_teacher_with_it():
    vae = tiny_vae(0)
    assert sum(p.numel() for p in vae.parameters()) < 10_000
    assert not any(p.requires_grad for p in vae.parameters()) and not vae.training
    with torch.no_grad():
        x = vae.decode(torch.randn(2, 4, 16, 16)).sample
    assert x.shape == (2, 3, 32, 32)
    t = tiny_teacher(0, vae=True)
    assert t.vae is not None and tiny_teacher(0).vae is None
    a, b = tiny_vae(0).state_dict(), tiny_vae(1).state_dict()
    assert all(torch.equal(a[k], tiny_vae(0).state_dict()[k]) for k in a)            # drawn from the seed
    assert not all(torch.equal(a[k], b[k]) for k in a)
    plain, with_vae = tiny_teacher(0), tiny_teacher(0, vae=True)                         # the rest is unchanged
    sp, sv = plain.unet.state_dict(), with_vae.unet.state_dict()
    assert all(torch.equal(sp[k], sv[k]) for k in sp)


def test_decode(teacher):
    g = torch.Generator().manual_seed(3)
    z = torch.randn(5, 4, 16, 16, generator=g) * 0.2
    one = pictures.decode(teacher.vae, z, batch=2)
    assert one.shape == (5, 3, 32, 32) and one.dtype == torch.uint8 and one.device.type == "cpu"
    assert torch.equal(one, pictures.decode(teacher.vae, z, batch=8))
    other = pictures.decode(teacher.vae, z.flip(0), batch=2)
    assert not torch.equal(one[0], other[0])
    # the formula: x = vae.decode(z / scale), clamped, (x + 1) 127.5 rounded
    with torch.no_grad():
        x = teacher.vae.decode(z / 0.18215).sample
    assert torch.equal(one, ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8))


# ------------------------------------------------------------------ sampling group by group
def test_sample_latents_follows_the_probe_rule(teacher):
    states = teacher.text_states(teacher.tokenize(PROMPTS))
    empty = teacher.empty_states(1)
    kw = dict(steps=4, guidance=3.0, latent_shape=SHAPE)
    lines = []
    one = pictures.sample_latents(teacher, states, empty, 31, group=2, log=lines.append, **kw)
    assert one.dtype == torch.float32 and one.shape == (4,) + SHAPE and len(lines) == 2
    assert torch.equal(one, pictures.sample_latents(teacher, states, empty, 31, group=2, **kw))     # (a)
    first = pictures.sample_latents(teacher, states[:2], empty, 31, group=2, **kw)                 # (b)
    assert torch.equal(one[:2], first)
    probe = reads.make_probe(states, seed=31, per_band=2, latent_shape=SHAPE, empty=empty)          # (c)
    assert torch.equal(one, reads.sample_probe(teacher, probe, 4, 3.0, group=2))
    model = Assembly({"clean": capture._load_part(teacher, "born:tiny", {}), "middle": teacher, "noisy": teacher})
    assert torch.equal(pictures.sample_latents(model, states, empty, 31, group=0, **kw),
                       reads.sample_probe(model, probe, 4, 3.0, group=0))


# ------------------------------------------------------------------ the kernel distance
def test_kid_by_hand():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    # k(x, y) = (x . y / 2 + 1) ** 3
    # within a, i != j: a0.a1 = 0 -> 1; a0.a2 = 1 -> 1.5^3 = 3.375; a1.a2 = 1 -> 3.375; each pair twice: 15.5, / 6
    # within b, i != j: every dot is 0 -> 1; 6 terms: 6, / 6
    # across: dots 0 2 0 / 1 0 2 / 1 2 2 -> 1 8 1 / 1 1 8 / 1 8 8 = 37, mean 37 / 9
    want = 15.5 / 6 + 6.0 / 6 - 2.0 * 37.0 / 9
    assert pictures._mmd2(a, b) == pytest.approx(want, abs=1e-12)
    assert pictures._mmd2(b, a) == pytest.approx(want, abs=1e-12)                   # symmetric
    a4 = np.vstack([a, [[2.0, 1.0]]])
    b4 = np.vstack([b, [[1.0, 2.0]]])
    r = pictures.kid(a4, b4, blocks=2)
    assert r["full"] == pytest.approx(pictures.kid(b4, a4, blocks=2)["full"], abs=1e-12)
    assert r["full"] == pytest.approx(pictures._mmd2(a4, b4), abs=1e-12)


def test_kid_blocks_and_errors():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((9, 3)), rng.standard_normal((8, 3)) + 0.5
    r = pictures.kid(a, b, blocks=4)
    assert r["blocks"] == 4 and r["block_rows"] == 2 and r["na"] == 9 and r["nb"] == 8
    per = [pictures._mmd2(a[2 * k:2 * k + 2], b[2 * k:2 * k + 2]) for k in range(4)]
    assert r["block_mean"] == pytest.approx(np.mean(per), abs=1e-12)
    assert r["block_std"] == pytest.approx(np.std(per, ddof=1), abs=1e-12)
    assert len(r["block_values"]) == 4 and r["block_values"] == pytest.approx(per, abs=1e-12)
    assert all(type(v) is float for v in r["block_values"])
    assert float(np.mean(r["block_values"])) == pytest.approx(r["block_mean"], abs=1e-15)
    assert r["full"] == pytest.approx(pictures._mmd2(a, b), abs=1e-12)
    assert set(r) == {"full", "block_mean", "block_std", "block_values", "blocks", "block_rows", "na", "nb"}
    with pytest.raises(ValueError):
        pictures.kid(a, b, blocks=1)
    with pytest.raises(ValueError):
        pictures.kid(a, b, blocks=5)                              # 8 // 5 = 1 row a block


def test_kid_reads_near_zero_on_one_sample_and_larger_when_shifted():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 8))
    same = pictures.kid(x[:200], x[200:], blocks=4)
    shifted = pictures.kid(x[:200], rng.standard_normal((200, 8)) + 1.0, blocks=4)
    assert abs(same["full"]) < 0.05
    assert shifted["full"] > 0.5 and shifted["full"] > 20 * abs(same["full"])      # the control that must fail
    assert shifted["block_mean"] > 0.5


# ------------------------------------------------------------------ the judges and the sheet
def test_fake_clip_judge_and_contact_sheet():
    g = torch.Generator().manual_seed(0)
    pics = torch.randint(0, 256, (3, 3, 32, 32), generator=g, dtype=torch.uint8)
    ids = torch.randint(0, 49408, (3, 77), generator=g)
    judge = pictures.make_clip_judge("fake")
    assert judge.name == "fake-clip" and pictures.make_clip_judge("") is None
    s = judge.scores(pics, ids)
    assert s.dtype == np.float64 and s.shape == (3,) and np.all(np.abs(s) <= 100.0)
    assert np.array_equal(s, pictures.FakeClipJudge().scores(pics, ids))
    assert not np.array_equal(s, judge.scores(pics.flip(0), ids))                       # the picture matters
    assert not np.array_equal(s, judge.scores(pics, ids.flip(1)))                       # the ids matter
    sheet = pictures.contact_sheet(pics, columns=2, size=40)
    assert sheet.size == (80, 80) and sheet.mode == "RGB"
    assert pictures.contact_sheet(pics, columns=4, size=16).size == (48, 16)
    with pytest.raises(ValueError):
        pictures.contact_sheet(pics[:0])


def test_prepare_pictures():
    pics = torch.randint(0, 256, (2, 3, 40, 30), dtype=torch.uint8)
    x = pictures.prepare_pictures(pics, 24, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert x.shape == (2, 3, 24, 24) and x.dtype == torch.float32
    same = pictures.prepare_pictures(pics[..., :30, :], 30, (0.0,) * 3, (1.0,) * 3)        # no resize: only the scale
    assert torch.equal(same, pics[..., :30, :].float() / 255.0)


@pytest.mark.skipif(not os.environ.get("MD_CLIP_PATH"), reason="set MD_CLIP_PATH to a local CLIP folder")
def test_clip_judge_against_the_library_processor():
    from PIL import Image
    from transformers import CLIPProcessor, CLIPTokenizer
    path = os.environ["MD_CLIP_PATH"]
    judge = pictures.ClipJudge(path)
    proc = CLIPProcessor.from_pretrained(path, local_files_only=True)
    tok = CLIPTokenizer.from_pretrained(path, local_files_only=True)
    ids = tok(["a red square", "a blue circle", "a green field", "a gray wall"], padding="max_length", max_length=77,
              truncation=True, return_tensors="pt").input_ids
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(4, 3, 8, 8), dtype=np.uint8)
    pics = torch.from_numpy(np.repeat(np.repeat(base, 40, axis=2), 50, axis=3).copy())    # 320 x 400, smooth blocks
    ours = judge.scores(pics, ids)
    images = [Image.fromarray(p.permute(1, 2, 0).numpy()) for p in pics]
    with torch.no_grad():
        px = proc(images=images, return_tensors="pt")["pixel_values"]
        img = pictures._features(judge.model.get_image_features(pixel_values=px))
        txt = pictures._features(judge.model.get_text_features(input_ids=ids))
    theirs = pictures._cosine_score(img, txt)
    print("ours", ours.tolist(), "theirs", theirs.tolist())
    assert np.all(np.abs(ours - theirs) < 1.0), (ours, theirs)


@pytest.mark.skipif(not os.environ.get("MD_CLIP_PATH"), reason="set MD_CLIP_PATH to a local CLIP folder")
def test_clip_judge_against_the_library_whole_forward():
    """The projected features and the pooled position of an end-token-padded prompt: the judge against the model's
    whole forward (its own embeddings, with the attention mask and the library's processor)."""
    from PIL import Image
    from transformers import CLIPProcessor, CLIPTokenizer
    path = os.environ["MD_CLIP_PATH"]
    judge = pictures.ClipJudge(path)
    proc = CLIPProcessor.from_pretrained(path, local_files_only=True)
    tok = CLIPTokenizer.from_pretrained(path, local_files_only=True)
    enc = tok(["a red square", "a blue circle", "a green field", "a gray wall"], padding="max_length", max_length=77,
              truncation=True, return_tensors="pt")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(4, 3, 8, 8), dtype=np.uint8)
    pics = torch.from_numpy(np.repeat(np.repeat(base, 40, axis=2), 50, axis=3).copy())
    ours = judge.scores(pics, enc.input_ids)
    images = [Image.fromarray(p.permute(1, 2, 0).numpy()) for p in pics]
    with torch.no_grad():
        px = proc(images=images, return_tensors="pt")["pixel_values"]
        out = judge.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, pixel_values=px)
    theirs = pictures._cosine_score(out.image_embeds, out.text_embeds)
    print("ours", ours.tolist(), "whole forward", theirs.tolist())
    assert np.all(np.abs(ours - theirs) < 1.0), (ours, theirs)


# ------------------------------------------------------------------ the caption picked from a list
def test_parquet_pictures_pick(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from PIL import Image
    from geolip.modular_diffusion.capture.data import parquet_pictures

    def png(k):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (k, k, k)).save(buf, format="PNG")
        return buf.getvalue()

    table = pa.table({"image": [{"bytes": png(k), "path": None} for k in range(4)],
                      "answer": [[f"first {k}", f"second {k}"] for k in range(3)] + [["only 3"]]})
    path = str(tmp_path / "lists.parquet")
    pq.write_table(table, path)
    assert parquet_pictures(path, 3)[1] == ["first 0", "first 1", "first 2"]
    assert parquet_pictures(path, 3, pick=0)[1] == parquet_pictures(path, 3)[1]
    assert parquet_pictures(path, 3, pick=1)[1] == ["second 0", "second 1", "second 2"]
    assert parquet_pictures(path, 1, skip=3)[1] == ["only 3"]
    with pytest.raises(ValueError, match="row 3 of"):
        parquet_pictures(path, 2, skip=2, pick=1)
    plain = pa.table({"image": [{"bytes": png(k), "path": None} for k in range(2)], "answer": ["a", "b"]})
    pq.write_table(plain, str(tmp_path / "plain.parquet"))
    assert parquet_pictures(str(tmp_path / "plain.parquet"), 2)[1] == ["a", "b"]
    with pytest.raises(ValueError, match="row 0 of"):
        parquet_pictures(str(tmp_path / "plain.parquet"), 2, pick=1)


# ------------------------------------------------------------------ parts born from the teacher
def test_born_parts(teacher):
    loaded = {}
    full = capture._load_part(teacher, "born:full", loaded)
    assert full.text and not full.training and capture._load_part(teacher, "born:full", loaded) is full
    notext = capture._load_part(teacher, "born:2222/nomid/3:notext", loaded)
    assert notext.text is False and notext.shape.text == "2222/nomid/3"
    assert capture._load_part(teacher, "born:tiny:notext", loaded).text is False
    assert capture._load_part(teacher, "born:tiny", loaded).text is True
    for bad in ("born:", "born:huge", "born:tiny:other", "born:1x11/mid/4", "born::notext"):
        with pytest.raises(ValueError, match="born:<shape>:notext"):
            capture._load_part(teacher, bad, {})
    capture.cast_engines(loaded, "bf16")
    assert all(next(e.parameters()).dtype == torch.bfloat16 for e in loaded.values())
    assert next(teacher.unet.parameters()).dtype == torch.float32


def test_the_assembly_variant_with_born_parts(tmp_path):
    common = dict(tiny=1, probe_n=3, probe_per_band=4, sample_steps=4, group=2, threads=2)
    copy = capture.main(variant="assembly", parts="clean=born:full,middle=teacher,noisy=teacher",
                        out=str(tmp_path / "full"), **common)
    assert copy["final_latent_change"] == 0.0
    assert all(copy["reads"][b]["mse"] == 0.0 for b in BANDS)            # the full shape with the prompt: a copy
    cut = capture.main(variant="assembly", parts="clean=born:tiny,middle=teacher,noisy=teacher",
                       out=str(tmp_path / "tiny"), **common)
    assert cut["reads"]["clean"]["mse"] > 0.0 and cut["final_latent_change"] > 0.0
    clean = BANDS.index("clean")
    for b in ("middle", "noisy"):                                         # where the clean band is silent: unchanged
        silent = [t for t in cut["reads"][b]["steps"] if float(band_weights(int(t))[0, clean]) == 0.0]
        assert silent, b
        for t in silent:
            assert cut["reads"][b]["per_step"][str(t)] == copy["reads"][b]["per_step"][str(t)] == 0.0
    ex = capture.main(variant="exchange", parts="clean=teacher,middle=teacher,noisy=teacher", band="clean",
                      other="born:full", out=str(tmp_path / "ex"), **common)
    assert ex["final_latent_change"] == 0.0
    with pytest.raises(ValueError, match="born:<shape>"):
        capture.main(variant="assembly", parts="clean=born:huge,middle=teacher,noisy=teacher",
                     out=str(tmp_path / "bad"), **common)


# ------------------------------------------------------------------ the pictures variants
COMMON = dict(tiny=1, pic_n=6, group=3, sample_steps=4, guidance=3.0, judge="fake", clip="fake", sheet_n=4,
              decode_batch=4, threads=2)
TEACHER_PARTS = "clean=teacher,middle=teacher,noisy=teacher"


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pictures")
    engine = root / "engine"
    capture.main(variant="band_engine", tiny=1, shape="tiny", band="clean", steps=2, batch=2, init="random",
                 probe_n=3, probe_per_band=2, log_every=0, threads=2, out=str(engine))
    out = {"root": root, "engine": str(engine)}
    out["t1"] = capture.main(variant="pictures", parts=TEACHER_PARTS, out=str(root / "t1"), **COMMON)
    out["t2"] = capture.main(variant="pictures", parts=TEACHER_PARTS, out=str(root / "t2"), **COMMON)
    out["rows"] = capture.main(variant="pictures", source="rows", out=str(root / "rows"), **COMMON)
    out["rand"] = capture.main(variant="pictures", parts=f"clean={engine},middle=teacher,noisy=teacher",
                               out=str(root / "rand"), **COMMON)
    return out


def _npz(path):
    with np.load(os.path.join(path, "pictures.npz"), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def test_the_pictures_variant(jobs):
    root = jobs["root"]
    t1, t2, rows, rand = _npz(root / "t1"), _npz(root / "t2"), _npz(root / "rows"), _npz(root / "rand")
    assert t1["features"].dtype == np.float32 and t1["features"].shape == (6, 64)
    assert t1["rows"].dtype == np.int64 and t1["rows"].tolist() == list(range(6))
    assert t1["clip"].dtype == np.float32 and t1["clip"].shape == (6,)
    assert np.array_equal(t1["features"], t2["features"]) and np.array_equal(t1["clip"], t2["clip"])
    assert not np.array_equal(t1["features"], rows["features"])
    assert not np.array_equal(t1["features"], rand["features"])
    r = jobs["t1"]
    for k in ("experiment", "variant", "args", "teacher", "torch", "version", "n", "source", "parts", "part_dtype",
              "judge", "feature_dim", "clip_judge", "clip_mean", "clip_std", "seconds_sampling", "seconds_decoding",
              "seconds_judging", "seconds"):
        assert k in r, k
    assert r["n"] == 6 and r["source"] == "parts" and r["judge"].startswith("fake") and r["feature_dim"] == 64
    assert r["clip_judge"] == "fake-clip" and "probe" not in r
    assert r["clip_mean"] == pytest.approx(float(t1["clip"].astype(np.float64).mean()), abs=1e-4)
    for name in ("pictures.npz", "sheet.jpg", "result.json", "result.md", "log.txt"):
        assert os.path.isfile(root / "t1" / name), name
    from PIL import Image
    with Image.open(root / "t1" / "sheet.jpg") as im:
        assert im.size == (4 * 256, 256)
    with open(root / "t1" / "log.txt", encoding="utf-8") as fh:
        assert fh.read().rstrip().endswith("done")
    with open(root / "t1" / "result.json", encoding="utf-8") as fh:
        assert json.load(fh)["n"] == 6
    assert jobs["rows"]["source"] == "rows" and jobs["rows"]["seconds_sampling"] < jobs["t1"]["seconds_sampling"]


def test_the_pictures_variant_in_half_precision_and_its_errors(jobs, tmp_path):
    parts = f"clean={jobs['engine']},middle=teacher,noisy=teacher"
    half = capture.main(variant="pictures", parts=parts, part_dtype="bf16", sheet_n=0, out=str(tmp_path / "h"),
                        **{k: v for k, v in COMMON.items() if k != "sheet_n"})
    assert half["part_dtype"] == "bf16" and not os.path.isfile(tmp_path / "h" / "sheet.jpg")
    assert not np.array_equal(_npz(tmp_path / "h")["features"], _npz(jobs["root"] / "rand")["features"])
    loaded = {}
    teacher = tiny_teacher(0, vae=True)
    capture._parts(dict(capture.DEFAULTS, parts=parts), teacher, loaded)
    capture.cast_engines(loaded, "bf16")
    assert [next(e.parameters()).dtype for e in loaded.values()] == [torch.bfloat16]
    for bad in (dict(pic_n=0), dict(source="latents"), dict(part_dtype="fp8"), dict(parts="")):
        with pytest.raises(ValueError):
            capture.main(variant="pictures", out=str(tmp_path / "bad"), **dict(COMMON, **dict(dict(parts=parts), **bad)))


def test_the_decoder_stays_full_precision(tmp_path):
    a = dict(capture.DEFAULTS, tiny=1, dtype="bf16")
    teacher, _ = capture._teacher(a, vae=True)
    assert next(teacher.vae.parameters()).dtype == torch.float32
    assert next(teacher.unet.parameters()).dtype == torch.bfloat16


def test_the_picture_compare_variant(jobs, tmp_path):
    root = jobs["root"]
    same = capture.main(variant="picture_compare", a=str(root / "t1"), b=str(root / "t2"), blocks=3,
                        out=str(tmp_path / "same"))
    assert same["frechet"] < 1e-6 and same["clip_diff_mean"] == 0.0 and same["prefix"] is False
    assert same["n"] == same["n_a"] == same["n_b"] == 6 and same["kid"]["blocks"] == 3
    assert same["clip_a"] == same["clip_b"]
    self_ = capture.main(variant="picture_compare", a=str(root / "t1"), b=str(root / "t1"), blocks=3,
                         out=str(tmp_path / "self"))
    assert self_["frechet"] < 1e-6 and self_["clip_diff_mean"] == 0.0
    diff = capture.main(variant="picture_compare", a=str(root / "t1"), b=str(root / "rand"), blocks=3,
                        out=str(tmp_path / "diff"))
    assert diff["frechet"] > 0.0 and diff["clip_diff_se"] is not None
    with open(tmp_path / "diff" / "log.txt", encoding="utf-8") as fh:
        assert fh.read().rstrip().endswith("done")
    assert os.path.isfile(tmp_path / "diff" / "result.md")
    # unequal rows, paired: raises naming the first difference
    moved = tmp_path / "moved"
    shutil.copytree(root / "t2", moved)
    d = _npz(moved)
    d["rows"] = np.array([0, 1, 2, 9, 4, 5], dtype=np.int64)
    np.savez(os.path.join(moved, "pictures.npz"), **d)
    with pytest.raises(ValueError, match="position 3"):
        capture.main(variant="picture_compare", a=str(root / "t1"), b=str(moved), blocks=3, out=str(tmp_path / "x"))
    unpaired = capture.main(variant="picture_compare", a=str(root / "t1"), b=str(moved), blocks=3, paired=0,
                            out=str(tmp_path / "u"))
    assert "clip_diff_mean" not in unpaired and unpaired["frechet"] < 1e-6
    with pytest.raises(ValueError):
        capture.main(variant="picture_compare", a=str(root / "t1"), out=str(tmp_path / "y"))


def test_picture_compare_first_n_and_equal_features(jobs, tmp_path):
    root = jobs["root"]
    head = capture.main(variant="picture_compare", a=str(root / "t1"), b=str(root / "t1"), first_n=4, blocks=2,
                        out=str(tmp_path / "head"))
    assert head["n"] == 4 and head["first_n"] == 4 and head["features_equal"] is True
    assert head["max_abs_feature_difference"] == 0.0 and head["n_a"] == head["n_b"] == 6
    assert head["kid"]["block_rows"] == 2 and head["clip_diff_mean"] == 0.0
    with pytest.raises(ValueError, match="first_n"):
        capture.main(variant="picture_compare", a=str(root / "t1"), b=str(root / "t1"), first_n=7, blocks=2,
                     out=str(tmp_path / "long"))
    other = capture.main(variant="picture_compare", a=str(root / "t1"), b=str(root / "rand"), blocks=3,
                         out=str(tmp_path / "other"))
    assert other["features_equal"] is False and other["max_abs_feature_difference"] > 0.0
    assert other["first_n"] == 0 and other["n"] == 6
    loose = capture.main(variant="picture_compare", a=str(root / "t1"), b=str(root / "rows"), paired=0, first_n=4,
                         blocks=2, out=str(tmp_path / "loose"))
    assert loose["n"] == 4 and loose["rows_compared_a"] == loose["rows_compared_b"] == 4
    assert loose["features_equal"] is False


def test_picture_compare_on_a_common_prefix(jobs, tmp_path):
    root = jobs["root"]
    kw = dict(COMMON, group=2, sheet_n=0)
    six = capture.main(variant="pictures", parts=TEACHER_PARTS, out=str(tmp_path / "six"), **kw)
    four = capture.main(variant="pictures", parts=TEACHER_PARTS, out=str(tmp_path / "four"), **dict(kw, pic_n=4))
    assert six["n"] == 6 and four["n"] == 4
    r = capture.main(variant="picture_compare", a=str(tmp_path / "six"), b=str(tmp_path / "four"), blocks=2,
                     out=str(tmp_path / "c"))
    assert r["prefix"] is True and r["n"] == 4 and r["n_a"] == 6 and r["n_b"] == 4
    assert r["frechet"] < 1e-6 and r["clip_diff_mean"] == 0.0
    assert np.array_equal(_npz(tmp_path / "six")["features"][:4], _npz(tmp_path / "four")["features"])
    assert r["features_equal"] is True and r["max_abs_feature_difference"] == 0.0
    both = capture.main(variant="picture_compare", a=str(tmp_path / "six"), b=str(tmp_path / "four"), blocks=2,
                        first_n=4, out=str(tmp_path / "c4"))                      # the prefix, then first_n inside it
    assert both["prefix"] is True and both["n"] == 4 and both["features_equal"] is True
    with pytest.raises(ValueError, match="first_n"):
        capture.main(variant="picture_compare", a=str(tmp_path / "six"), b=str(tmp_path / "four"), blocks=2,
                     first_n=5, out=str(tmp_path / "c5"))
    other = capture.main(variant="pictures", parts=TEACHER_PARTS, noise_seed=2001, out=str(tmp_path / "other"),
                         **dict(kw, pic_n=4))
    assert other["n"] == 4
    with pytest.raises(ValueError, match="noise_seed"):
        capture.main(variant="picture_compare", a=str(tmp_path / "six"), b=str(tmp_path / "other"), blocks=2,
                     out=str(tmp_path / "d"))
    loose = capture.main(variant="picture_compare", a=str(tmp_path / "six"), b=str(tmp_path / "other"), blocks=2,
                         paired=0, out=str(tmp_path / "e"))
    assert loose["prefix"] is False and loose["n"] is None and loose["frechet"] > 0.0
    # a group of another size gives other noise: refused when paired
    with pytest.raises(ValueError, match="group"):
        capture.main(variant="picture_compare", a=str(root / "t1"), b=str(tmp_path / "four"), blocks=2,
                     out=str(tmp_path / "f"))
    # the rows' own pictures sample nothing: no sampling setting is compared
    rows = capture.main(variant="picture_compare", a=str(root / "rows"), b=str(tmp_path / "four"), blocks=2,
                        out=str(tmp_path / "g"))
    assert rows["prefix"] is True and rows["n"] == 4
