"""Layout writing: how often a writer writes the right addresses for a picture from a text prompt alone
(``geolip.modular_diffusion.layout``). Inference only: nothing is trained.

    python -m geolip.modular_diffusion.run layout_write variant=write writer=fake:truth form=box examples=1 out=<folder>
    from geolip.modular_diffusion.experiments import layout_write; layout_write.main(variant="write", writer=..., out=...)

| variant       | what it does                                                                                       |
|---------------|----------------------------------------------------------------------------------------------------|
| write         | one cell of the measurement: the 600 prompts with known answers (``first_n`` of each family, 0 =   |
|               | all), and the real captions of ``real`` when given, asked of one writer in one form; every answer  |
|               | read by the form's strict grammar and scored; the controls the cell can judge alone                |
| write_compare | two cell folders (``a``, ``b``) compared prompt by prompt: address_right per family and over the   |
|               | six families, and every score                                                                      |

WRITERS (``writer=``): ``fake:<mode>`` (no model: ``truth`` the right answer in the cell's form, ``broken``,
``empty``, ``shuffled`` the right answer of another prompt of the family), ``parser`` (no model: the records of
``thought_seeds.textspec.spec_from_text`` on the prompt, read as a layout; needs the thought-image-seeds package), or a
local folder or hub id of a ``transformers`` picture-and-text chat model used with text only (``ChatWriter``: one user
turn, greedy, thinking off when the chat template has that switch, left padding in batches, only the new tokens
decoded; ``adapter``: a LoRA folder or hub id merged into the weights; ``system``: empty for no system turn, ``v1``
for the trained system text of the qwen-symbolic-thinking package, or ``file:<path>``).

FORMS (``form=``): ``box`` and ``cell`` (the asks of ``layout.ASKS``, with the six worked examples when
``examples=1``); ``chain_records`` (the trained text-only ask of ``qwen_symbolic_thinking.prompts.spec_short`` at the
depth "medium" on the parser's records of the prompt; a prompt from which the parser reads no record is asked in the
chain_text form and counted apart); ``chain_full_records`` (the same with the trained whole-row ask ``spec_full``,
read by ``layout.chain_full_read``: the chain lines joined with the question lines); ``chain_text``
(``layout.ASK_CHAIN_TEXT``); ``spec`` (only with ``writer=parser``: the parser's records are the answer). The other
chain forms are read by ``layout.chain_read``.

Arguments (key=value): variant, writer, adapter, adapter_revision (a tag or commit of an adapter given as a hub id),
revision, form, examples, first_n, real (a JSON file: a list of
captions, or of {"id", "text"}), seed (the prompts' generator), repeat_n (control d: the first prompts asked again, in
whole batches, the first repeat_n answers compared), device, dtype (empty: float32; fp16 or bf16), batch,
max_new_tokens, system, local_only (1: nothing is downloaded), threads, a, b, out.

Output in ``out``: outputs.jsonl (id, family, the ask's sha256, the raw answer, the reading's failure), rows.jsonl (one
scored row a prompt), result.json (the summary, the controls, the asks' texts, the versions, the seconds, the answers a
second), result.md (the tables), log.txt (its last line 'done').

THE CONTROLS a cell judges alone: (a) the generator's layouts written by code in the cell's form (box or cell) and
read back score 1 on every applicable score of every prompt, and with ``writer=fake:truth`` the cell's own answers do;
(b) the cell's readings scored against another prompt of the same family (``layout.family_derangement``) read
address_right under 0.2 on count, colour, relation and combined, and the right things at random cells read spatial
between 0.2 and 0.6 on relation (one statement, 1 in 3 by chance) and between 0.03 and 0.22 on place (two statements,
1 in 9); (c) every broken answer of ``layout.BROKEN`` for the form gives its own failure kind (a missing bracket or
field fails PARSED; a number out of range or a word that is no place is read and gives no address); (d) the first
``repeat_n`` prompts asked again give the same text. Rates count each distinct prompt text once (``layout.distinct``);
box cells also report COPIED (``layout.copied``), which judges nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

from .. import __version__, layout
from ..config import in_notebook, parse

EXPERIMENT = "layout_write"
VARIANTS = ("write", "write_compare")
FORMS = ("box", "cell", "chain_records", "chain_full_records", "chain_text", "spec")
READ_FORM = {"box": "box", "cell": "cell", "chain_records": "chain", "chain_full_records": "chain_full",
             "chain_text": "chain", "spec": "spec"}
RECORD_FORMS = ("chain_records", "chain_full_records")
FAKE_MODES = ("truth", "broken", "empty", "shuffled")
DEPTH = "medium"
DEFAULTS = dict(variant="write", writer="", adapter="", adapter_revision="", revision="", form="box", examples=1, first_n=0, real="", seed=0,
                repeat_n=20, device="cpu", dtype="", batch=8, max_new_tokens=512, system="", local_only=1, threads=4,
                a="", b="", out="")
CONTROL_B_FAMILIES = ("count", "colour", "relation", "combined")
CONTROL_B_LIMIT = 0.2
# the right things at random cells, SPATIAL: one statement on relation (1 in 3 by chance), two on place (1 in 9)
CONTROL_B_BANDS = {"relation": (0.2, 0.6), "place": (0.03, 0.22)}
CONTROL_B_RANDOM = tuple(CONTROL_B_BANDS)
RANDOM_SEED = 1000


def cell_name(writer_tag: str, form: str, examples: bool) -> str:
    """A cell's folder name: <writer tag>_<form>_<ex1|ex0>."""
    return f"{writer_tag}_{form}_{'ex1' if examples else 'ex0'}"


# ------------------------------------------------------------------ writers
class FakeWriter:
    """A writer for tests and dry runs. ``prompts``: {ask text: prompt}; answers in ``form`` (box, cell or chain)."""

    def __init__(self, mode: str, form: str, prompts: Dict[str, dict], others: Optional[Dict[str, dict]] = None):
        if mode not in FAKE_MODES:
            raise ValueError(f"a fake writer's mode is one of {FAKE_MODES}, not {mode!r}")
        self.mode, self.form, self.prompts, self.others = mode, form, prompts, others or {}
        self.batch = 1
        self.name = f"fake:{mode}"

    def _truth(self, p: Optional[dict]) -> str:
        return layout.render(self.form, (p or {}).get("truth", []))      # a real caption has no known layout

    def write(self, asks: Sequence[str]) -> List[str]:
        out = []
        for a in asks:
            if self.mode == "truth":
                out.append(self._truth(self.prompts.get(a)))
            elif self.mode == "shuffled":
                out.append(self._truth(self.others.get(a)))
            elif self.mode == "broken":
                out.append(layout.BROKEN[self.form][0][0])
            else:
                out.append("")
        return out


def _textspec():
    try:
        from thought_seeds.textspec import spec_from_text
    except ImportError as e:
        raise ImportError("the parser writer and the chain_records form need the thought-image-seeds package "
                          "(thought_seeds); install it at its tag v0.4.0") from e
    return spec_from_text


def parser_records(text: str) -> dict:
    """The records the house parser reads from a text, as ``qwen_symbolic_thinking.prompts.records`` takes them."""
    spec, _ = _textspec()(text)
    return spec.spec()


class ParserWriter:
    """No model: the answer is the JSON of the parser's records of the prompt (read with ``layout.read("spec", ...)``).
    ``texts``: {ask: prompt text}; the ask is the prompt text itself when it is not in the map."""

    def __init__(self, texts: Optional[Dict[str, str]] = None):
        _textspec()
        self.texts = texts or {}
        self.batch = 1
        self.name = "parser"

    def write(self, asks: Sequence[str]) -> List[str]:
        return [json.dumps(parser_records(self.texts.get(a, a)), sort_keys=True) for a in asks]


def _resolve_system(system: str) -> Optional[str]:
    if not system:
        return None
    if system.startswith("file:"):
        with open(system[5:], encoding="utf-8") as fh:
            return fh.read().strip()
    try:
        from qwen_symbolic_thinking.prompts import SYSTEMS
    except ImportError as e:
        raise ImportError("system=<name> needs the qwen-symbolic-thinking package") from e
    if system not in SYSTEMS:
        raise ValueError(f"system is empty, file:<path> or one of {sorted(SYSTEMS)}, not {system!r}")
    return SYSTEMS[system]


def _local_folder(where: str, revision: Optional[str], local_only: bool) -> str:
    if os.path.isdir(where):
        return where
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=where, revision=revision or None, local_files_only=local_only)


def merge_lora(model, folder: str) -> dict:
    """Merge a LoRA adapter folder (adapter_config.json, adapter_model.safetensors) into ``model`` in place by the
    standard sum W + (alpha / r) B A over the adapter's pairs. Refuses rslora, dora, modules_to_save and per-module
    ranks or alphas. -> {"pairs": merged pairs, "scale": alpha / r}."""
    import torch
    from safetensors.torch import load_file
    with open(os.path.join(folder, "adapter_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key in ("use_rslora", "use_dora"):
        if cfg.get(key):
            raise ValueError(f"the adapter uses {key}: the plain merge does not apply")
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern"):
        if cfg.get(key):
            raise ValueError(f"the adapter has {key}: the plain merge does not apply")
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    weights = load_file(os.path.join(folder, "adapter_model.safetensors"))
    params = dict(model.named_parameters())
    pairs = 0
    for k, a in weights.items():
        if ".lora_A." not in k:
            continue
        b = weights[k.replace(".lora_A.", ".lora_B.")]
        base = k.split(".lora_A.")[0]
        name = base[len("base_model.model."):] if base.startswith("base_model.model.") else base
        target = params.get(name + ".weight")
        if target is None:
            raise KeyError(f"the adapter's module {name} is not in the model")
        with torch.no_grad():
            delta = (b.double() @ a.double()) * scale
            target.add_(delta.to(target.dtype).to(target.device))
        pairs += 1
    if not pairs:
        raise ValueError(f"no LoRA pairs in {folder}")
    return {"pairs": pairs, "scale": scale}


class ChatWriter:
    """A ``transformers`` picture-and-text chat model used with text only. One user turn (and a system turn when
    ``system`` is given); greedy; at most ``max_new_tokens``; left padding in batches of ``batch``; only the new tokens
    decoded. ``adapter``: a LoRA folder or hub id merged into the weights (through peft when it is importable, else
    ``merge_lora``)."""

    def __init__(self, model: str, adapter: str = "", revision: str = "", adapter_revision: str = "",
                 device: str = "cpu", dtype: str = "",
                 local_only: bool = True, batch: int = 8, max_new_tokens: int = 512, system: Optional[str] = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dtypes = {"": torch.float32, "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if dtype not in dtypes:
            raise ValueError(f"dtype is empty, fp32, fp16 or bf16, not {dtype!r}")
        self.device, self.batch, self.max_new_tokens, self.system = device, int(batch), int(max_new_tokens), system
        folder = _local_folder(model, revision, local_only)
        self.processor = AutoProcessor.from_pretrained(folder)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(folder, dtype=dtypes[dtype]).eval()
        self.merge = None
        if adapter:
            afolder = _local_folder(adapter, adapter_revision or None, local_only)
            try:
                from peft import PeftModel
            except ImportError:
                self.merge = dict(merge_lora(self.model, afolder), how="plain sum")
            else:
                self.model = PeftModel.from_pretrained(self.model, afolder).merge_and_unload().eval()
                self.merge = {"how": "peft merge_and_unload"}
        self.model.to(device)
        template = getattr(self.processor, "chat_template", None) or getattr(self.tokenizer, "chat_template", "") or ""
        self.thinking_switch = "enable_thinking" in str(template)
        self.name = model + (f" + {adapter}" if adapter else "")

    def prompt_text(self, ask: str) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": ask}]})
        extra = {"enable_thinking": False} if self.thinking_switch else {}
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)

    def write(self, asks: Sequence[str]) -> List[str]:
        import torch
        out: List[str] = []
        for k in range(0, len(asks), self.batch):
            texts = [self.prompt_text(a) for a in asks[k:k + self.batch]]
            enc = self.processor(text=texts, return_tensors="pt", padding=True)
            enc = {n: v.to(self.device) for n, v in enc.items()}
            with torch.no_grad():
                gen = self.model.generate(**enc, do_sample=False, max_new_tokens=self.max_new_tokens, temperature=None,
                                          top_p=None, top_k=None)
            new = gen[:, enc["input_ids"].shape[1]:]
            out += [t.strip() for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]
        return out


# ------------------------------------------------------------------ the cell
def _real_prompts(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for k, x in enumerate(data):
        text = x if isinstance(x, str) else x["text"]
        pid = f"real_{k:03d}" if isinstance(x, str) else str(x.get("id", f"real_{k:03d}"))
        out.append({"id": pid, "family": "real", "text": str(text)})
    return out


def _select(prompts: List[dict], first_n: int) -> List[dict]:
    if first_n <= 0:
        return list(prompts)
    out, seen = [], {}
    for p in prompts:
        seen[p["family"]] = seen.get(p["family"], 0) + 1
        if seen[p["family"]] <= first_n:
            out.append(p)
    return out


def _asks(form: str, examples: bool, prompts: List[dict]) -> List[dict]:
    """[{"ask", "fallback_text"}] a prompt."""
    out = []
    for p in prompts:
        if form in ("box", "cell"):
            out.append({"ask": layout.ask(form, p["text"], examples), "fallback_text": False})
        elif form == "chain_text":
            out.append({"ask": layout.ask("chain_text", p["text"]), "fallback_text": False})
        elif form == "spec":
            out.append({"ask": p["text"], "fallback_text": False})
        else:
            records = parser_records(p["text"])
            if records:
                try:
                    from qwen_symbolic_thinking.prompts import spec_full, spec_short
                except ImportError as e:
                    raise ImportError(f"the {form} form needs the qwen-symbolic-thinking package") from e
                make = spec_short if form == "chain_records" else spec_full
                out.append({"ask": make(DEPTH, records, keys="record"), "fallback_text": False, "records": records})
            else:
                out.append({"ask": layout.ask("chain_text", p["text"]), "fallback_text": True, "records": records})
    return out


def _writer(a: dict, form: str, prompts: List[dict], asks: List[dict]):
    w = a["writer"]
    if w.startswith("fake:"):
        mode = w[5:]
        render_form = {"box": "box", "cell": "cell", "chain_records": "chain", "chain_full_records": "chain_full",
                       "chain_text": "chain"}.get(form)
        if render_form is None:
            raise ValueError(f"a fake writer writes the forms box, cell and the chain forms, not {form!r}")
        by_ask = {x["ask"]: p for x, p in zip(asks, prompts)}
        syn = [p for p in prompts if p["family"] in layout.FAMILIES]
        perm = layout.family_derangement(syn, int(a["seed"])) if len(syn) > 1 else list(range(len(syn)))
        other = {id(p): syn[perm[k]] for k, p in enumerate(syn)}
        others = {x["ask"]: other.get(id(p)) for x, p in zip(asks, prompts)}
        return FakeWriter(mode, render_form, by_ask, others)
    if w == "parser":
        if form != "spec":
            raise ValueError("writer=parser takes form=spec (its records are its answer)")
        return ParserWriter()
    if form == "spec":
        raise ValueError("form=spec is the parser's form (writer=parser)")
    return ChatWriter(w, adapter=a["adapter"], revision=a["revision"], adapter_revision=a["adapter_revision"],
                      device=a["device"], dtype=a["dtype"],
                      local_only=bool(int(a["local_only"])), batch=int(a["batch"]),
                      max_new_tokens=int(a["max_new_tokens"]), system=_resolve_system(a["system"]))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _versions() -> dict:
    out = {"package": __version__, "python": sys.version.split()[0]}
    for name in ("torch", "transformers", "peft", "thought_seeds", "qwen_symbolic_thinking"):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, "__version__", "?")
    return out


def _controls(a: dict, form: str, prompts: List[dict], readings: List[dict], answers: List[str],
              repeat: Optional[dict]) -> dict:
    rf = READ_FORM[form]
    syn_idx = [k for k, p in enumerate(prompts) if p["family"] in layout.FAMILIES]
    syn = [prompts[k] for k in syn_idx]
    out: Dict[str, dict] = {}
    # (a) the scorer's own answers, over the whole generated list in this form
    if rf in ("box", "cell"):
        full = layout.make_prompts(int(a["seed"])) + layout.example_prompts()
        bad = [p["id"] for p in full if layout.score(p, layout.read(rf, layout.render(rf, p["truth"])), rf)
               ["address_right"] != 1]
        out["a"] = {"prompts": len(full), "failures": bad[:20], "failed": len(bad), "holds": not bad}
        if a["writer"] == "fake:truth":
            own = [layout.score(p, r, rf) for p, r in zip(syn, [readings[k] for k in syn_idx])]
            wrong = [s["id"] for s in own if s["address_right"] != 1 or
                     any(s[x] not in (None, 1) for x in layout.SCORES)]
            out["a"]["writer_answers_failed"] = len(wrong)
            out["a"]["holds"] = out["a"]["holds"] and not wrong
    else:
        out["a"] = {"holds": None, "note": "control (a) writes the box and cell forms"}
    # (b) the scorer can fail
    b: Dict[str, dict] = {"deranged": {}, "random_spatial": {}}
    perm = layout.family_derangement(syn, int(a["seed"])) if len(syn) > 1 else []
    for fam in layout.FAMILIES:
        idx = [k for k, p in enumerate(syn) if p["family"] == fam]
        if len(idx) < 2:
            continue
        vals = [layout.score(syn[perm[k]], readings[syn_idx[k]], rf)["address_right"] for k in idx]
        b["deranged"][fam] = layout.rate(vals)
    for fam in CONTROL_B_RANDOM:
        idx = [k for k, p in enumerate(syn) if p["family"] == fam]
        vals = [layout.score(syn[k], layout.random_reading(syn[k], RANDOM_SEED + k))["spatial"] for k in idx]
        if vals:
            b["random_spatial"][fam] = layout.rate(vals)
    der = [b["deranged"][f]["mean"] < CONTROL_B_LIMIT for f in CONTROL_B_FAMILIES if f in b["deranged"]]
    ran = [CONTROL_B_BANDS[f][0] <= b["random_spatial"][f]["mean"] <= CONTROL_B_BANDS[f][1] for f in b["random_spatial"]]
    b["bands"] = {f: list(v) for f, v in CONTROL_B_BANDS.items()}
    b["deranged_holds"] = all(der) if der else None
    b["random_holds"] = all(ran) if ran else None
    b["holds"] = None if b["deranged_holds"] is None or b["random_holds"] is None else (b["deranged_holds"] and b["random_holds"])
    out["b"] = b
    # (c) broken answers
    if rf in layout.BROKEN:
        c = layout.broken_check(rf)
        out["c"] = {"answers": c, "holds": all(x["ok"] for x in c)}
    else:
        out["c"] = {"holds": None, "note": "the spec form is code"}
    out["d"] = repeat or {"holds": None}
    return out


def _run_write(a: dict, log: Callable[[str], None], t0: float) -> dict:
    form = a["form"]
    if form not in FORMS:
        raise ValueError(f"form is one of {FORMS}, not {form!r}")
    if not a["writer"]:
        raise ValueError("writer= is required (fake:<mode>, parser, or a model folder or hub id)")
    out = a["out"]
    examples = bool(int(a["examples"])) and form in ("box", "cell")
    prompts = _select(layout.make_prompts(int(a["seed"])), int(a["first_n"]))
    if a["real"]:
        real = _real_prompts(a["real"])
        prompts += real[: int(a["first_n"])] if int(a["first_n"]) > 0 else real
    asks = _asks(form, examples, prompts)
    tw = time.time()
    writer = _writer(a, form, prompts, asks)
    load_s = time.time() - tw
    log(f"LAYOUT write: writer {writer.name}; form {form}; examples {int(examples)}; {len(prompts)} prompts; "
        f"package {__version__}; loaded in {load_s:.1f} s")
    tg = time.time()
    answers: List[str] = []
    step = max(1, int(getattr(writer, "batch", 1)))
    for k in range(0, len(asks), step * 4):
        answers += writer.write([x["ask"] for x in asks[k:k + step * 4]])
        log(f"answers {len(answers):,} of {len(asks):,}; {time.time() - tg:.0f} s")
    gen_s = time.time() - tg
    rf = READ_FORM[form]
    readings = [layout.read(rf, ans) for ans in answers]
    outputs, rows = [], []
    for p, x, ans, r in zip(prompts, asks, answers, readings):
        outputs.append({"id": p["id"], "family": p["family"], "ask_sha256": _sha(x["ask"]), "answer": ans,
                        "failure": r.get("failure"), "fallback_text": x["fallback_text"]})
        if p["family"] == "real":
            row = dict(layout.score_real(p["text"], r), id=p["id"], family="real")
        else:
            row = layout.score(p, r, rf)
            row["published"] = layout.published(p, r)
        row["fallback_text"] = x["fallback_text"]
        rows.append(row)
    # control (d): the first repeat_n prompts asked again in whole batches; the first repeat_n answers compared
    repeat = None
    n_rep = min(int(a["repeat_n"]), len(asks))
    if n_rep > 0:
        whole = min(len(asks), -(-n_rep // step) * step)
        tr = time.time()
        again = writer.write([x["ask"] for x in asks[:whole]])[:n_rep]
        same = [i for i in range(n_rep) if again[i] == answers[i]]
        repeat = {"asked": whole, "compared": n_rep, "same": len(same), "holds": len(same) == n_rep,
                  "different": [prompts[i]["id"] for i in range(n_rep) if again[i] != answers[i]],
                  "seconds": time.time() - tr}
    _write_jsonl(os.path.join(out, "outputs.jsonl"), outputs)
    _write_jsonl(os.path.join(out, "rows.jsonl"), rows)
    summary = layout.summarize(rows)
    fb = [r for r in rows if r.get("fallback_text")]
    result = {"experiment": EXPERIMENT, "variant": "write", "args": {k: a[k] for k in sorted(a)}, "writer": writer.name,
              "form": form, "read_form": rf, "examples": examples, "prompts": len(prompts),
              "real_prompts": sum(p["family"] == "real" for p in prompts), "summary": summary,
              "fallback_text": {"n": len(fb), "ids": [r["id"] for r in fb][:50],
                                "address_right": layout.rate([r.get("address_right") for r in fb
                                                              if r.get("family") != "real"])},
              "controls": _controls(a, form, prompts, readings, answers, repeat),
              "asks": {"box": layout.ASK_BOX, "cell": layout.ASK_CELL, "chain_text": layout.ASK_CHAIN_TEXT,
                       "examples": [p["text"] for p in layout.example_prompts()], "depth": DEPTH},
              "system": _resolve_system(a["system"]) if not a["writer"].startswith(("fake:", "parser")) else None,
              "merge": getattr(writer, "merge", None), "versions": _versions(),
              "seconds_loading": load_s, "seconds_writing": gen_s,
              "answers_a_second": len(answers) / gen_s if gen_s > 0 else None, "seconds": time.time() - t0}
    _write_result(a, result, _tables(result), log)
    return result


def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_f(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _tables(result: dict) -> str:
    s = result["summary"]
    head = ["family", "n", "address right", "se", "parsed", "things", "counts", "colours", "placed", "spatial",
            "no conflict", "absent", "per-instance error"]
    rows = []
    for fam, blk in list(s["families"].items()) + [("all six", s["all"])]:
        sc = blk["scores"]
        rows.append([fam, blk["n"], sc["address_right"]["mean"], sc["address_right"]["se"]]
                    + [sc[k]["mean"] for k in layout.SCORES] + [blk["per_instance_error"]["rate"]])
    text = (f"# layout_write: {result['writer']}, form {result['form']}, examples {int(result['examples'])}\n\n"
            "## the scores (mean over the prompts; se = sqrt(p (1 - p) / n))\n\n" + _table(head, rows))
    kinds = s["all"]["kinds"]
    text += ("\n## the kinds of failure (prompts)\n\n" + _table(["kind", "prompts"], [[k, v] for k, v in kinds.items()])
             + f"\nunparsed: {s['all']['unparsed']}; recovered by the lenient read: {s['all']['lenient_recovered']}; "
             f"prompts {s['all']['prompts']}, distinct texts {s['all']['texts']}, texts whose copies disagree "
             f"{s['all']['duplicates_disagree']}"
             + (f"; copied boxes (every box an example's): {_f(s['all']['copied']['mean'])}" if "copied" in s["all"] else "")
             + "\n")
    text += "\n## the load curve on combined\n\n" + _table(["instances", "address right", "se", "n"],
                                                           [[c["instances"], c["mean"], c["se"], c["n"]]
                                                            for c in s["load_curve"]])
    pub = s["published"]
    text += "\n## the published rules on the raw boxes\n\n" + _table(
        ["rule", "mean", "se", "n"], [[k, v["mean"], v["se"], v["n"]] for k, v in pub.items()])
    if "real" in s:
        text += "\n## the real captions\n\n" + _table(["score", "mean", "se", "n"],
                                                     [[k, v["mean"], v["se"], v["n"]] for k, v in s["real"].items()])
    c = result["controls"]
    text += "\n## the controls\n\n" + _table(["control", "holds"], [[k, c[k].get("holds")] for k in ("a", "b", "c", "d")])
    text += (f"\nseconds: {result['seconds']:.1f} (writing {result['seconds_writing']:.1f}); answers a second: "
             f"{_f(result['answers_a_second'])}\n")
    return text


def _write_result(a: dict, result: dict, text: str, log) -> None:
    out = a["out"]
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(out, "result.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(text.rstrip())
    log("done")


def _run_compare(a: dict, log, t0: float) -> dict:
    if not (a["a"] and a["b"]):
        raise ValueError("the write_compare variant needs a= and b= (two cell folders)")
    ra, rb = read_jsonl(os.path.join(a["a"], "rows.jsonl")), read_jsonl(os.path.join(a["b"], "rows.jsonl"))
    log(f"LAYOUT write_compare: {a['a']} against {a['b']}; package {__version__}")
    fams = list(layout.FAMILIES) + ["all"]
    scores: Dict[str, Dict[str, dict]] = {}
    for key in ("address_right",) + layout.SCORES:
        scores[key] = {f: layout.compare(ra, rb, key, None if f == "all" else f) for f in fams}
    real_a = {r["id"]: r for r in ra if r.get("family") == "real"}
    real = {}
    if real_a:
        rr = [dict(r, family="count") for r in ra if r.get("family") == "real"]
        rbb = [dict(r, family="count") for r in rb if r.get("family") == "real"]
        real = {k: layout.compare(rr, rbb, k, "count") for k in ("parsed", "placed", "no_conflict",
                                                                    "written_things_in_caption")}
    result = {"experiment": EXPERIMENT, "variant": "write_compare", "args": {k: a[k] for k in sorted(a)},
              "a": a["a"], "b": a["b"], "scores": scores, "real": real, "version": __version__,
              "seconds": time.time() - t0}
    rows = [[f, scores["address_right"][f]["mean"], scores["address_right"][f]["se"], scores["address_right"][f]["n"],
             scores["address_right"][f]["tie"]] for f in fams]
    text = (f"# layout_write: {a['a']} against {a['b']} (a - b, prompt by prompt)\n\n## address right\n\n"
            + _table(["family", "mean difference", "se", "n", "tie"], rows))
    for key in layout.SCORES:
        x = scores[key]["all"]
        text += f"\n{key} over the six: {_f(x['mean'])} (se {_f(x['se'])}, n {x['n']}, tie {_f(x['tie'])})"
    _write_result(a, result, text + "\n", log)
    return result


def main(argv=(), **overrides) -> dict:
    a = parse(DEFAULTS, argv, **overrides)
    if not a["out"]:
        raise ValueError("out= is required")
    if a["variant"] not in VARIANTS:
        raise ValueError(f"unknown variant {a['variant']!r}; known: {list(VARIANTS)}")
    os.makedirs(a["out"], exist_ok=True)
    log_path = os.path.join(a["out"], "log.txt")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    t0 = time.time()
    if not a["writer"].startswith(("fake:", "parser")) and a["variant"] == "write":
        try:
            import torch
            torch.set_num_threads(int(a["threads"]))
        except ImportError:
            pass
    if a["variant"] == "write_compare":
        return _run_compare(a, log, t0)
    return _run_write(a, log, t0)


def summarize(root: str) -> str:
    """A markdown table of the cells under ``root``: address_right per family and over the six, the parsed rate, the
    per-instance error and the controls."""
    rows = []
    for dirpath, _, files in sorted(os.walk(root)):
        if "result.json" not in files:
            continue
        with open(os.path.join(dirpath, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("experiment") != EXPERIMENT or r.get("variant") != "write":
            continue
        s = r["summary"]
        fam = s["families"]
        row = [os.path.relpath(dirpath, root)]
        row += [fam.get(f, {}).get("scores", {}).get("address_right", {}).get("mean") for f in layout.FAMILIES]
        row += [s["all"]["scores"]["address_right"]["mean"], s["all"]["scores"]["parsed"]["mean"],
                s["all"]["per_instance_error"]["rate"]]
        row += [r["controls"][k].get("holds") for k in ("a", "b", "c", "d")]
        rows.append(row)
    return _table(["cell"] + list(layout.FAMILIES) + ["all six", "parsed", "per-instance error", "a", "b", "c", "d"],
                  rows)


if __name__ == "__main__" and not in_notebook():
    main(sys.argv[1:])
