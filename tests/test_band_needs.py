"""The band-needs experiment on the tiny random teacher: the whole reads variant at small sizes (its keys, its exact
controls, the shares, the seeds' own rows), its resume from the part files, a partial job, the timing variant with its
exact row-forward counts, the count of the prompt read at the real sizes, and pictures read from a parquet file."""
import io
import json
import os
import shutil

import pytest

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion.capture import BANDS  # noqa: E402
from geolip.modular_diffusion.capture.assemble import ddim_timesteps  # noqa: E402
from geolip.modular_diffusion.capture.data import parquet_pictures, synthetic_shards  # noqa: E402
from geolip.modular_diffusion.capture.measure import single_changes  # noqa: E402
from geolip.modular_diffusion.capture.schedule import band_weights  # noqa: E402
from geolip.modular_diffusion.experiments import band_needs  # noqa: E402

SMALL = dict(rows=4, prompts=4, group=2, chunk=2, per_band=2, steps=6, seeds="0,1", cost_batch=2, cost_repeats=1)

# The row-forwards of SMALL, counted by hand.
# Born reads: 4 rows x 2 probe steps x 3 bands = 24 a forward set. Per seed: controls 24 (teacher) + 24 (full copy)
# + 48 (inherited and random base); r1 24 + 6 x 24; r2 24 + 3 x 24: 360 a seed, 720 for two.
# r4: ten shapes, each the teacher and the engine, one warm-up and one timed forward of 2 rows: 10 x 2 x 2 x 2 = 80.
# r3: six DDIM steps 831, 665, 499, 333, 167, 1; the step 333 lies in the crossfade of clean and middle, every other
# step in one band only. A sampling with one band's prompt changed runs 1 part a step and 2 at step 333 when that band
# is clean or middle: 7 + 7 + 6 = 20 part-steps over the bands, 40 for empty and shuffled. The reference and the empty
# prompt everywhere add 12, the repeat of the first group 6. Rows a forward: 2 prompts x 2 (guidance).
# Group 0: 4 x (40 + 18) = 232; group 1: 4 x (40 + 12) = 208; 440 a seed, 880 for two.
HAND = {"reads": 720, "r4": 80, "r3": 880}


def _numbers(r):
    return {k: r[k] for k in ("controls", "r1", "r2", "r3", "r4")}


def _no_seconds(x):
    """The result without measured seconds (they differ from run to run)."""
    if isinstance(x, dict):
        return {k: _no_seconds(v) for k, v in x.items() if k not in ("seconds", "teacher_seconds", "cost")}
    if isinstance(x, list):
        return [_no_seconds(v) for v in x]
    return x


@pytest.fixture(scope="module")
def first(tmp_path_factory):
    out = tmp_path_factory.mktemp("band_needs") / "job"
    return str(out), band_needs.main(tiny=1, out=str(out), **SMALL)


def test_the_reads_variant(first):
    out, r = first
    for k in ("experiment", "variant", "args", "torch", "version", "device", "card", "switches", "teacher", "data",
              "controls", "r1", "r2", "r3", "r4", "seconds"):
        assert k in r, k
    assert r["experiment"] == "band_needs" and r["variant"] == "reads"
    assert r["switches"] == {"cuda.matmul.allow_tf32": False, "cudnn.allow_tf32": False, "cudnn.benchmark": False}
    assert r["data"]["latent_shape"] == [4, 16, 16] and r["data"]["used"] == 8
    assert sorted(r["controls"]) == ["0", "1"]
    for s in ("0", "1"):
        c = r["controls"][s]
        assert all(c["full"][b] == 0.0 for b in BANDS)                     # the teacher's own copy reads exactly 0
        assert all(c["random_base"][b] > c["inherited_base"][b] > 0.0 for b in BANDS)
        assert all(c["ratio"][b] == c["random_base"][b] / c["inherited_base"][b] for b in BANDS)
        assert c["repeat"] == 0.0 and r["r3"][s]["repeat"] == 0.0         # the reference repeats itself exactly
        assert sorted(r["r1"][s]["table"]) == sorted(BANDS)
        assert all(sorted(r["r1"][s]["table"][b]) == sorted(n for n, _ in single_changes()) for b in BANDS)
        assert all(sorted(r["r2"][s][b]) == ["base", "small", "tiny"] for b in BANDS)
        assert len(r["r3"][s]["groups"]) == 2
    assert [x["name"] for x in r["r4"]] == ["full"] + [n for n, _ in single_changes()] + ["base", "small", "tiny"]
    assert all(x["seconds"] > 0 and x["teacher_seconds"] > 0 for x in r["r4"])
    assert r["row_forwards"]["reads"]["total"] == HAND["reads"]
    assert r["row_forwards"]["r4"]["total"] == HAND["r4"] and r["row_forwards"]["r3"]["total"] == HAND["r3"]
    names = sorted(os.listdir(os.path.join(out, "parts")))
    assert sorted(n[:-5] for n in names) == sorted(["controls_s0", "controls_s1", "r1_s0", "r1_s1", "r2_s0", "r2_s1",
                                                    "r4", "r3_s0_g000", "r3_s0_g001", "r3_s1_g000", "r3_s1_g001"])
    with open(os.path.join(out, "log.txt"), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines[-1] == "done"
    assert any(x.startswith("job size: 11 units") for x in lines)
    assert any("11/11 units done" in x for x in lines)
    with open(os.path.join(out, "result.md"), encoding="utf-8") as fh:
        md = fh.read()
    assert "## r1" in md and "## r2" in md and "## r3" in md and "## r4" in md and "seed gap" in md
    with open(os.path.join(out, "result.json"), encoding="utf-8") as fh:
        assert json.load(fh)["r1"] == r["r1"]


def test_shares_are_values_over_the_empty_everywhere_change(first):
    _, r = first
    for s, x in r["r3"].items():
        e = x["empty_everywhere"]
        assert e > 0.0
        for b in BANDS:
            t = x["table"][b]
            assert t["empty_share"] == t["empty"] / e and t["shuffled_share"] == t["shuffled"] / e
        # the seed's numbers: the groups' means weighted by rows (two groups of two rows here)
        g = x["groups"]
        assert e == (2 * g[0]["empty_everywhere"] + 2 * g[1]["empty_everywhere"]) / 4
        assert [y["first_row"] for y in g] == ([0, 2] if s == "0" else [4, 6])
        assert g[0]["seed"] == 100000 * (int(s) + 1) and g[1]["seed"] == 100000 * (int(s) + 1) + 1


def test_the_seeds_use_different_rows(first):
    _, r = first
    for b in BANDS:
        assert r["r1"]["0"]["table"][b] != r["r1"]["1"]["table"][b]
        assert r["r3"]["0"]["table"][b]["empty"] != r["r3"]["1"]["table"][b]["empty"]


def test_a_second_call_runs_nothing_again(first):
    out, r = first
    again = band_needs.main(tiny=1, out=out, **SMALL)
    assert again["units"]["ran"] == [] and len(again["units"]["read_back"]) == 11
    assert _numbers(again) == _numbers(r)


def test_a_stopped_job_recomputes_exactly_the_missing_units(first, tmp_path):
    out, r = first
    copy = str(tmp_path / "copy")
    shutil.copytree(out, copy)
    for name in ("r1_s1", "r3_s0_g001", "r4"):
        os.remove(os.path.join(copy, "parts", name + ".json"))
    again = band_needs.main(tiny=1, out=copy, **SMALL)
    assert again["units"]["ran"] == ["r1_s1", "r4", "r3_s0_g001"]
    assert again["r1"] == r["r1"] and again["r3"] == r["r3"] and again["controls"] == r["controls"]
    assert _no_seconds(again["r4"]) == _no_seconds(r["r4"])
    with open(os.path.join(copy, "log.txt"), encoding="utf-8") as fh:
        text = fh.read()
    assert text.count("unit r1_s1 ran in") == 2 and text.count("unit r1_s0 ran in") == 1


def test_only_r1_and_its_controls(tmp_path):
    r = band_needs.main(tiny=1, out=str(tmp_path / "r1"), do="r1", **SMALL)
    assert sorted(os.listdir(tmp_path / "r1" / "parts")) == ["controls_s0.json", "controls_s1.json", "r1_s0.json",
                                                            "r1_s1.json"]
    assert r["r2"] == {} and r["r3"] == {} and r["r4"] == []
    assert all(r["controls"][s]["full"][b] == 0.0 for s in ("0", "1") for b in BANDS)
    # controls 24 + 24 + 48, r1 24 + 6 x 24: 11 x 24 a seed
    assert r["row_forwards"]["r3"]["total"] == 0 and r["row_forwards"]["reads"]["total"] == 11 * 24 * 2


def test_too_few_rows_raise(tmp_path):
    data = str(tmp_path / "data")
    synthetic_shards(data, rows=7, shard_size=4, size=16)
    with pytest.raises(ValueError, match="7 rows"):
        band_needs.main(tiny=1, data=data, out=str(tmp_path / "job"), **SMALL)
    with pytest.raises(ValueError):
        band_needs.main(tiny=1, out=str(tmp_path / "job"), **dict(SMALL, prompts=5))      # a last group of one row
    with pytest.raises(ValueError):
        band_needs.main(tiny=1, out=str(tmp_path / "job"), **dict(SMALL, rows=5))         # more probe rows than prompts


def test_the_timing_variant(tmp_path):
    from geolip.modular_diffusion import run                     # through the entry point, key=value items
    argv = ["variant=timing", "tiny=1", f"out={tmp_path / 'timing'}"] + [f"{k}={v}" for k, v in SMALL.items()]
    r = run.main("band_needs", argv)
    assert all(v > 0 for v in r["minutes"].values()) and all(v > 0 for v in r["seconds_a_forward"].values())
    assert r["minutes"]["total"] == pytest.approx(r["minutes"]["reads"] + r["minutes"]["r4"] + r["minutes"]["r3"])
    assert {k: r["row_forwards"][k]["total"] for k in HAND} == HAND
    with open(os.path.join(tmp_path, "timing", "timing.json"), encoding="utf-8") as fh:
        assert json.load(fh)["minutes"] == r["minutes"]


def test_the_prompt_read_count_at_the_real_sizes():
    """prompts 128 in groups of 32, 50 steps, guidance 7.5, two seeds: derived step by step from the band weights."""
    ts = ddim_timesteps(50)
    w = band_weights(ts)
    changed = 0                                                   # part-steps of one sampling, summed over the bands
    for b in range(3):
        for i in range(len(ts)):
            changed += int(float(w[i, b]) > 0)
            changed += int(sum(float(w[i, o]) for o in range(3) if o != b) > 0)
    changed *= 2                                                  # the empty prompt and another row's prompt
    count = 0
    for seed in range(2):
        for g in range(4):
            plain = 50 * (3 if g == 0 else 2)                     # the reference, empty everywhere, the repeat
            count += 2 * 32 * (plain + changed)                   # guidance doubles the 32 rows
    assert changed == 348 and count == 235_776
    rf = band_needs.row_forwards(2, 64, 4, 128, 32, 50, 7.5, 16, 3)
    assert rf["r3"]["total"] == count
    # born reads: 64 rows x 12 probe steps a forward set; controls 4 sets, r1 7, r2 4 a seed
    assert rf["reads"]["total"] == 2 * 64 * 12 * 15 and rf["r4"]["total"] == 10 * 2 * 4 * 16


def test_group_sizes():
    assert band_needs.group_sizes(128, 32) == [32, 32, 32, 32]
    assert band_needs.group_sizes(10, 4) == [4, 4, 2]
    with pytest.raises(ValueError):
        band_needs.group_sizes(9, 4)


def test_parquet_pictures(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from PIL import Image

    def png(color, size):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return buf.getvalue()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (9, 9, 9), (200, 100, 50)]
    table = pa.table({
        "image": [{"bytes": png(c, (20 + i, 12)), "path": None} for i, c in enumerate(colors)],
        "answer": [[f"caption {i}", "another"] for i in range(5)],
        "id": list(range(100, 105)), "license": [1, 2, 2, 3, 1], "extra": ["x"] * 5})
    path = str(tmp_path / "pictures.parquet")
    pq.write_table(table, path, row_group_size=2)
    pictures, captions, meta = parquet_pictures(path, 4)
    assert len(pictures) == 4 and all(p.mode == "RGB" for p in pictures)
    assert [p.size for p in pictures] == [(20, 12), (21, 12), (22, 12), (23, 12)]
    assert pictures[2].getpixel((0, 0)) == (0, 0, 255)
    assert captions == ["caption 0", "caption 1", "caption 2", "caption 3"]
    assert meta == [{"id": 100 + i, "license": [1, 2, 2, 3][i]} for i in range(4)]        # file_name is absent
    with pytest.raises(ValueError):
        parquet_pictures(path, 6)
    plain = pa.table({"image": table["image"], "caption": ["a", "b", " ", "d", "e"]})
    pq.write_table(plain, str(tmp_path / "plain.parquet"))
    assert parquet_pictures(str(tmp_path / "plain.parquet"), 2, caption="caption")[1] == ["a", "b"]
    with pytest.raises(ValueError, match="empty caption"):
        parquet_pictures(str(tmp_path / "plain.parquet"), 3, caption="caption")
