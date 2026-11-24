"""The layout_write experiment end to end with fake writers (truth, broken, shuffled, empty), the parser writer (when
its package is importable), the chain forms, the comparison of two cells, the plain LoRA merge on a tiny module, and
one optional test of the chat writer on a real local model (off unless MD_TEST_CHAT_MODEL names a folder)."""
import importlib.util
import json
import os
import time

import pytest

from geolip.modular_diffusion import layout as L
from geolip.modular_diffusion import run
from geolip.modular_diffusion.experiments import layout_write as W

N = 6        # prompts a family in these tests
TEXTS = 3 + 1 + 4 * N        # their distinct texts: count's copies come in pairs, absent's first six share one text


def _cell(tmp_path, name, **kw):
    out = str(tmp_path / name)
    kw.setdefault("first_n", N)
    kw.setdefault("repeat_n", 4)
    return run.main("layout_write", [f"out={out}"], variant="write", **kw), out


def _rows(out):
    return W.read_jsonl(os.path.join(out, "rows.jsonl"))


@pytest.mark.parametrize("form", ["box", "cell"])
def test_fake_truth_scores_1_and_every_control_holds_but_the_random_band(tmp_path, form):
    real = tmp_path / "real.json"
    real.write_text(json.dumps(["A dog sits on a couch.", "Two cats on a bed."]), encoding="utf-8")
    r, out = _cell(tmp_path, f"truth_{form}", writer="fake:truth", form=form, examples=1, real=str(real))
    assert r["prompts"] == 6 * N + 2 and r["real_prompts"] == 2
    s = r["summary"]
    assert all(s["families"][f]["scores"]["address_right"]["mean"] == 1.0 for f in L.FAMILIES)
    assert s["all"]["per_instance_error"]["errors"] == 0 and s["real"]["parsed"]["mean"] == 1.0
    c = r["controls"]
    assert c["a"]["holds"] is True and c["a"]["prompts"] == 606 and c["a"]["writer_answers_failed"] == 0
    assert c["c"]["holds"] is True and c["d"]["holds"] is True and c["d"]["compared"] == 4
    assert c["b"]["deranged_holds"] is True
    rows, outputs = _rows(out), W.read_jsonl(os.path.join(out, "outputs.jsonl"))
    assert len(rows) == len(outputs) == 6 * N + 2
    assert all(len(o["ask_sha256"]) == 64 for o in outputs)
    assert open(os.path.join(out, "log.txt"), encoding="utf-8").read().rstrip().endswith("done")
    md = open(os.path.join(out, "result.md"), encoding="utf-8").read()
    assert "## the controls" in md and "## the load curve on combined" in md
    if form == "box":
        assert s["published"]["lmd_four_families"]["mean"] == 1.0 and r["asks"]["box"] == L.ASK_BOX


def test_fake_broken_and_empty_and_shuffled(tmp_path):
    r, out = _cell(tmp_path, "broken_box", writer="fake:broken", form="box", examples=0)
    s = r["summary"]["all"]
    assert s["scores"]["parsed"]["mean"] == 0.0 and s["scores"]["address_right"]["mean"] == 0.0
    assert s["kinds"] == {"broken format": TEXTS} and s["prompts"] == 6 * N and r["summary"]["published"]["lmd_four_families"]["mean"] == 0.0
    e, _ = _cell(tmp_path, "empty_cell", writer="fake:empty", form="cell", examples=0)
    assert e["summary"]["all"]["scores"]["parsed"]["mean"] == 0.0
    sh, _ = _cell(tmp_path, "shuffled_cell", writer="fake:shuffled", form="cell", examples=1, first_n=20)
    fams = sh["summary"]["families"]
    assert fams["absent"]["scores"]["address_right"]["mean"] == 1.0          # every absent truth is empty
    assert all(fams[f]["scores"]["address_right"]["mean"] < 0.2 for f in ("colour", "relation", "combined"))


def test_chain_text_with_a_fake_writer(tmp_path):
    r, out = _cell(tmp_path, "truth_chain_text", writer="fake:truth", form="chain_text", examples=1)
    assert r["examples"] is False and r["read_form"] == "chain"
    s = r["summary"]["families"]
    assert s["count"]["scores"]["things"]["mean"] == 1.0 and s["count"]["scores"]["counts"]["mean"] == 1.0
    assert r["controls"]["a"]["holds"] is None and r["controls"]["c"]["holds"] is True
    outputs = W.read_jsonl(os.path.join(out, "outputs.jsonl"))
    assert all(o["fallback_text"] is False for o in outputs)


def test_write_compare_on_two_fake_cells(tmp_path):
    _, a = _cell(tmp_path, "a", writer="fake:truth", form="cell", examples=1)
    _, b = _cell(tmp_path, "b", writer="fake:broken", form="cell", examples=1)
    out = str(tmp_path / "cmp")
    r = run.main("layout_write", [f"out={out}"], variant="write_compare", a=a, b=b)
    x = r["scores"]["address_right"]
    assert x["all"]["mean"] == 1.0 and x["all"]["n"] == TEXTS and x["all"]["tie"] is False
    assert x["count"]["n"] == N // 2 and r["scores"]["parsed"]["all"]["mean"] == 1.0
    same = run.main("layout_write", [f"out={out}_same"], variant="write_compare", a=a, b=a)
    assert same["scores"]["address_right"]["all"]["tie"] is True
    table = W.summarize(str(tmp_path))
    assert "| a |" in table and "| b |" in table


def test_arguments_are_checked(tmp_path):
    with pytest.raises(ValueError):
        run.main("layout_write", [f"out={tmp_path}/x"], variant="write", writer="fake:truth", form="poem")
    with pytest.raises(ValueError):
        run.main("layout_write", [f"out={tmp_path}/x"], variant="write", writer="fake:nope", form="box")
    with pytest.raises(ValueError):
        run.main("layout_write", [f"out={tmp_path}/x"], variant="write", writer="fake:truth", form="spec")
    with pytest.raises(ValueError):
        run.main("layout_write", [f"out={tmp_path}/x"], variant="write_compare")
    assert W.cell_name("base", "box", True) == "base_box_ex1" and W.cell_name("parser", "spec", False) == "parser_spec_ex0"
    assert "layout_write" in run.EXPERIMENTS


HAS_PARSER = importlib.util.find_spec("thought_seeds") is not None
HAS_SPEC_SHORT = importlib.util.find_spec("qwen_symbolic_thinking") is not None


@pytest.mark.skipif(not HAS_PARSER, reason="the parser writer needs the thought-image-seeds package (thought_seeds), "
                                           "which is not installed in this environment")
def test_the_parser_writer(tmp_path):
    r, out = _cell(tmp_path, "parser_spec_ex0", writer="parser", form="spec", examples=0)
    s = r["summary"]["families"]
    assert s["count"]["scores"]["things"]["mean"] == 1.0 and s["count"]["scores"]["counts"]["mean"] == 1.0
    assert s["absent"]["scores"]["address_right"]["mean"] == 1.0
    assert s["count"]["scores"]["placed"]["mean"] == 0.0
    outputs = W.read_jsonl(os.path.join(out, "outputs.jsonl"))
    assert json.loads(outputs[0]["answer"])["thing"]["name"] == "backpack"


@pytest.mark.skipif(not (HAS_PARSER and HAS_SPEC_SHORT), reason="the chain_records form needs the thought-image-seeds "
                                                               "and qwen-symbolic-thinking packages, not installed here")
def test_chain_records_with_a_fake_writer(tmp_path):
    r, out = _cell(tmp_path, "truth_chain_records", writer="fake:truth", form="chain_records", examples=0)
    outputs = W.read_jsonl(os.path.join(out, "outputs.jsonl"))
    assert len(outputs) == 6 * N and r["fallback_text"]["n"] == sum(o["fallback_text"] for o in outputs)


@pytest.mark.skipif(not (HAS_PARSER and HAS_SPEC_SHORT), reason="the chain_full_records form needs the "
                                                               "thought-image-seeds and qwen-symbolic-thinking packages, "
                                                               "not installed here")
def test_chain_full_records_with_a_fake_writer(tmp_path):
    r, out = _cell(tmp_path, "truth_chain_full_records", writer="fake:truth", form="chain_full_records", examples=0)
    assert r["read_form"] == "chain_full" and r["controls"]["c"]["holds"] is True
    assert r["summary"]["families"]["count"]["scores"]["counts"]["mean"] == 1.0
    outputs = W.read_jsonl(os.path.join(out, "outputs.jsonl"))
    assert all("question: thing name ->" in o["answer"] for o in outputs if o["family"] == "count")


def test_the_bands_and_the_adapter_revision():
    assert W.CONTROL_B_BANDS == {"relation": (0.2, 0.6), "place": (0.03, 0.22)}
    assert "adapter_revision" in W.DEFAULTS
    import inspect
    assert "adapter_revision" in inspect.signature(W.ChatWriter.__init__).parameters
    assert W.READ_FORM["chain_full_records"] == "chain_full"


def test_the_plain_lora_merge(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 3, bias=False)

    m = Tiny()
    before = m.proj.weight.detach().clone()
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2, 4, generator=g), torch.randn(3, 2, generator=g)
    save_file({"base_model.model.proj.lora_A.weight": a, "base_model.model.proj.lora_B.weight": b},
              str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}), encoding="utf-8")
    info = W.merge_lora(m, str(tmp_path))
    assert info == {"pairs": 1, "scale": 2.0}
    assert torch.allclose(m.proj.weight, before + 2.0 * b @ a, atol=1e-6)
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "use_dora": True}), encoding="utf-8")
    with pytest.raises(ValueError):
        W.merge_lora(m, str(tmp_path))


CHAT = os.environ.get("MD_TEST_CHAT_MODEL", "")


@pytest.mark.skipif(not (CHAT and os.path.isdir(CHAT)), reason="set MD_TEST_CHAT_MODEL to a local chat model folder")
def test_the_chat_writer_on_a_real_model():
    pytest.importorskip("transformers")
    writer = W.ChatWriter(CHAT, device="cpu", batch=1, max_new_tokens=int(os.environ.get("MD_TEST_CHAT_TOKENS", "512")))
    count = [p for p in L.make_prompts(0) if p["family"] == "count"][:2]
    asks = [L.ask("box", count[0]["text"], True), L.ask("cell", count[1]["text"], True)]
    t0 = time.time()
    first = writer.write(asks)
    t1 = time.time()
    second = writer.write(asks)
    t2 = time.time()
    print("CHAT ANSWERS", json.dumps({"box": first[0], "cell": first[1], "seconds_first": t1 - t0,
                                      "seconds_second": t2 - t1, "thinking_switch": writer.thinking_switch}))
    assert len(first) == 2 and all(isinstance(x, str) for x in first)
    assert second == first
