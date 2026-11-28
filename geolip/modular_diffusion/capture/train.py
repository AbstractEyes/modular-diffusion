"""One band engine trained alone against the frozen teacher's answers.

    result = train_band_engine(teacher, "base", "clean", source, steps=10000, batch=16, seed=0, out=folder)

A BATCH SOURCE is any callable ``source(step, batch) -> (x0 [B, 4, h, w], ids [B, 77])``: clean latents and token
ids. It must depend on (step, batch) only, so a resumed job sees the data an uninterrupted one would.

Per step k (counted from 0), every draw from one generator seeded from (seed, k):
    t ~ the band's training law (``schedule.sample_steps``: its step set, or the whole range with probability rho);
    eps ~ N(0, 1); z_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps;
    the prompt states from the frozen text encoder, each row replaced by the empty prompt's with probability ``drop``;
    the teacher's answer with no gradient; the engine's answer; the capture loss (``loss.capture_loss``);
    one step of PURE ADAM (betas 0.9 / 0.999, eps 1e-8, no weight decay), learning rate 5e-5 by default, no
    clipping; the gradient norm is logged.
The engine's numbers are kept in float32; on a card the forwards may run under autocast (bf16 by default there; on the
CPU everything is float32).

SMALLER GROUPS (``micro``): with 0 < micro < batch, the step's rows, noise, steps and dropped prompts are drawn for
the whole batch exactly as above; then the rows go through the forwards and the backward in consecutive groups of
``micro`` rows (the last group may be smaller). Each group's loss is weighted by (rows in the group / batch), so the
gradients added over the groups are the gradient of the whole batch's loss (the same update, up to the order of a
floating-point sum), and ONE optimizer step follows. The logged terms are the same weighted sums. ``micro`` = 0 or
micro >= batch is the one-forward path. ``micro`` changes the memory a step needs, not what it computes, so it is not
one of the settings a checkpoint is checked against: a resumed job may use another value.

An AVERAGED copy of the engine's parameters is kept: after update n (n = 1, 2, ..)
    avg <- d_n avg + (1 - d_n) w,     d_n = min(decay, (1 + n) / (10 + n))      decay 0.9999 by default
(the warm-up keeps the early average close to the recent weights). Reads use the averaged copy.

START: ``init="inherit"`` builds the engine by ``engine.build_engine`` (every number copied from the teacher);
``init="random"`` by ``engine.fresh_engine(teacher, shape, seed=seed, text=text)`` (the library's own initialization
from the job's seed, nothing inherited).

READS DURING TRAINING: with ``every`` > 0 and ``on_every`` given, after every ``every`` updates and after the job's
last update the trainer calls ``on_every(n, engine, averaged)``: n the updates made, ``engine`` the live engine (put in
eval mode for the call and back in train mode after it), ``averaged`` a copy holding the averaged parameters in eval
mode. What it returns is kept as {"step": n, "value": ...} in ``result["reads"]``, in checkpoint.pt and in train.json
(so it must be plain numbers, strings, lists and dicts). The call touches none of the trainer's generators (each step's
generator is made from (seed, step)) and no number of the engine, its average or Adam's state: the final numbers are
the same, bit for bit on the CPU, with and without it. A pause at ``until`` makes no call of its own, so a resumed
job's list is the list of a job that never stopped.

RESUME: every ``save_every`` steps (and at the end, and at ``until``) the job folder gets checkpoint.pt (the engine,
the averaged copy, Adam's state, the step, the seed the per-step generators come from, the loss log, the reads, the
measured seconds of each step, the settings), written to a side file and then renamed. A job started on a folder that
holds a checkpoint continues from it; on the CPU it reaches the same final numbers, bit for bit, as a job that never
stopped. The settings must match (a checkpoint made before ``init`` was a setting counts as "inherit").

``keep_checkpoint=False`` removes checkpoint.pt once the job has finished and its engine files are written. A job
started on a folder holding train.json and no checkpoint is finished: its stored result is returned (the engine and
its averaged copy read back from the engine files) and nothing is trained.

The trainer never downloads and writes only inside its job folder: checkpoint.pt, engine.safetensors,
engine_avg.safetensors, engine.json, train.json (the settings, the loss log, the reads, the median seconds a step over
every step of the job, the trained parameter count, the group size of the call that wrote it). It prints the job's
size, the time spent and the time left.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import time
from typing import Callable, Dict, Optional, Union

import torch
from torch import Tensor

from . import BANDS
from .engine import Engine, build_engine, fresh_engine, load_engine, param_count, save_engine
from .loss import capture_loss, drop_prompts
from .schedule import add_noise, band_index, sample_steps
from .shapes import Shape, shape_from

BETAS = (0.9, 0.999)
EPS = 1e-8
CHECKPOINT = "checkpoint.pt"
TRAIN_JSON = "train.json"
ENGINE_FILES = ("engine.safetensors", "engine_avg.safetensors", "engine.json")
INITS = ("inherit", "random")
SETTING_KEYS = ("shape", "band", "steps", "batch", "seed", "rho", "lr", "lam_out", "lam_data", "drop", "text",
                "average_decay", "init")


def step_generator(seed: int, step: int) -> torch.Generator:
    """The CPU generator of step k of a job."""
    return torch.Generator().manual_seed(1_000_003 * (int(seed) + 1) + int(step))


def average_decay_at(n: int, decay: float) -> float:
    """The averaging weight after update n (1-based): min(decay, (1 + n) / (10 + n))."""
    return min(float(decay), (1.0 + n) / (10.0 + n))


def _autocast(kind: str, device: torch.device):
    if kind == "auto":
        kind = "bf16" if device.type == "cuda" else "none"
    if kind == "none":
        return contextlib.nullcontext()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(kind)
    if dt is None:
        raise ValueError(f"autocast is auto, none, bf16 or fp16, not {kind!r}")
    return torch.autocast(device_type=device.type, dtype=dt)


def averaged_engine(engine: Engine, average: Dict[str, Tensor]) -> Engine:
    """A copy of the engine holding the averaged parameters (the copy holds no gradients)."""
    cfg = getattr(engine.unet, "config", None)
    if cfg is not None:
        from diffusers import UNet2DConditionModel
        clean_cfg = {k: v for k, v in dict(cfg).items() if not k.startswith("_")}
        with torch.device("meta"):
            unet = UNet2DConditionModel.from_config(clean_cfg)
        ctx = None if engine.text else average["context"]
        e = Engine(unet, engine.shape, text=engine.text, context=ctx)
        state = {name: average[name].detach().clone() for name, _ in engine.named_parameters()}
        for name, buf in engine.named_buffers():
            state[name] = buf.detach().clone()
        e.load_state_dict(state, strict=True, assign=True)
        for p in e.parameters():
            p.requires_grad_(True)
        e.train(engine.training)
        return e
    e = copy.deepcopy(engine)
    with torch.no_grad():
        for name, p in e.named_parameters():
            p.grad = None
            p.copy_(average[name])
    return e


def _median(values) -> Optional[float]:
    v = sorted(float(x) for x in values)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def _settings_of(ck: dict) -> dict:
    """A checkpoint's settings; one made before ``init`` was a setting counts as "inherit"."""
    s = dict(ck["settings"])
    s.setdefault("init", "inherit")
    return s


def _check_settings(old: dict, settings: dict, where: str) -> None:
    if old != settings:
        keys = sorted(set(old) | set(settings))
        diff = {k: (old.get(k), settings.get(k)) for k in keys if old.get(k) != settings.get(k)}
        raise ValueError(f"the job folder's {where} was made with other settings: {diff}")


def _finished_result(teacher, out: str, settings: dict, log: Callable[[str], None]) -> dict:
    """The stored result of a finished job whose checkpoint was removed (nothing is trained)."""
    with open(os.path.join(out, TRAIN_JSON), encoding="utf-8") as fh:
        stored = json.load(fh)
    old = dict(stored["settings"])
    old.setdefault("init", "inherit")
    _check_settings(old, settings, TRAIN_JSON)
    missing = [n for n in ENGINE_FILES if not os.path.isfile(os.path.join(out, n))]
    if missing:
        raise ValueError(f"{out} holds {TRAIN_JSON} and no checkpoint, but lacks {missing}: the job is neither "
                         f"finished nor resumable")
    engine = load_engine(teacher, out, "engine", average=False)
    engine.train()
    average = load_engine(teacher, out, "engine", average=True)
    average.train()
    steps = int(settings["steps"])
    log(f"FINISHED already at step {steps:,} of {steps:,} ({out}: {TRAIN_JSON} and the engine files, no checkpoint); "
        f"nothing trained")
    result = {"engine": engine, "average": average, "log": list(stored.get("log", [])), "step": steps, "finished": True,
              "seconds": 0.0, "settings": settings, "reads": list(stored.get("reads", [])),
              "seconds_a_step": stored.get("seconds_a_step"), "trained": stored.get("trained", param_count(engine)),
              "restored": True}
    if "micro" in stored:
        result["micro"] = int(stored["micro"])
    return result


def _save(path: str, state: dict) -> None:
    tmp = path + ".part"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _make_adam(params, lr: float, device: torch.device) -> torch.optim.Adam:
    if device.type == "cuda":
        try:
            return torch.optim.Adam(params, lr=float(lr), betas=BETAS, eps=EPS, weight_decay=0.0, fused=True)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.Adam(params, lr=float(lr), betas=BETAS, eps=EPS, weight_decay=0.0, foreach=None)


def train_band_engine(teacher, shape: Union[str, Shape], band: str, source: Callable, *, steps: int, batch: int,
                      seed: int, out: str, rho: float = 0.0, lr: float = 5e-5, lam_out: float = 1.0,
                      lam_data: float = 0.0, drop: float = 0.1, text: bool = True, average_decay: float = 0.9999,
                      save_every: int = 0, log_every: int = 10, until: Optional[int] = None, autocast: str = "auto",
                      log: Callable[[str], None] = print, init: str = "inherit", every: int = 0,
                      on_every: Optional[Callable] = None, keep_checkpoint: bool = True, micro: int = 0) -> dict:
    """Train the band engine of a shape alone (see the module's text). Returns {"engine", "average" (the engine with
    the averaged parameters), "log" (one dict per step), "step", "finished", "seconds", "settings", "reads" (the
    values ``on_every`` returned, with their steps), "seconds_a_step" (the median over the job's steps), "trained"
    (the trained parameter count), "micro" (the group size this run used; 0: the whole batch, also when the asked group
    holds the whole batch)}; a finished job read back from its files also holds "restored": True, and its "micro" is the
    one train.json holds (the group size of the call that wrote that file; the asked value when it holds none)."""
    kind = getattr(getattr(teacher, "process", None), "kind", "eps")
    if kind != "eps":
        raise ValueError(f"a band engine is trained from a noise-predicting teacher only; this source is of the kind "
                         f"{kind!r}, whose capture loss is not defined in this library")
    shape = shape_from(shape)
    band = BANDS[band_index(band)]
    steps, batch = int(steps), int(batch)
    micro = int(micro or 0)
    if micro < 0:
        raise ValueError(f"micro is 0 (the whole batch in one forward) or a positive number of rows, not {micro}")
    if micro >= batch:
        micro = 0                                           # one group holds the whole batch: the one-forward path
    if init not in INITS:
        raise ValueError(f"init is one of {list(INITS)}, not {init!r}")
    every = int(every or 0)
    if every < 0:
        raise ValueError("every is 0 (no reads during training) or a positive number of updates")
    stop = steps if until is None else min(int(until), steps)
    settings = {"shape": shape.text, "band": band, "steps": steps, "batch": batch, "seed": int(seed), "rho": float(rho),
                "lr": float(lr), "lam_out": float(lam_out), "lam_data": float(lam_data), "drop": float(drop),
                "text": bool(text), "average_decay": float(average_decay), "init": init}
    os.makedirs(out, exist_ok=True)
    ck_path = os.path.join(out, CHECKPOINT)
    if not os.path.isfile(ck_path) and os.path.isfile(os.path.join(out, TRAIN_JSON)):
        back = _finished_result(teacher, out, settings, log)
        return dict(back, micro=back.get("micro", micro))   # the recorded group size when train.json holds one
    device = next(teacher.unet.parameters()).device
    if init == "random":
        engine = fresh_engine(teacher, shape, seed=int(seed), text=text)
    else:
        engine = build_engine(teacher, shape, text=text)
    engine.train()
    named_params = list(engine.named_parameters())
    params = [p for _, p in named_params]
    opt = _make_adam(params, lr, device)
    average = {n: p.detach().clone() for n, p in named_params}
    history = []
    reads_log = []
    step_seconds = []
    done = 0
    if os.path.isfile(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        _check_settings(_settings_of(ck), settings, "checkpoint")
        engine.load_state_dict(ck["engine"])
        opt.load_state_dict(ck["adam"])
        average = {n: v.to(device) for n, v in ck["average"].items()}
        history = list(ck["log"])
        reads_log = list(ck.get("reads", []))
        step_seconds = [float(x) for x in ck.get("step_seconds", [])]
        done = int(ck["step"])
        log(f"RESUME from step {done:,} of {steps:,} ({ck_path})")
    avg_list = [average[n] for n, _ in named_params]
    ac = teacher.alphas_cumprod
    empty_f = teacher.empty_states(1).to(device=device, dtype=torch.float32)
    trained = param_count(engine)
    log(f"JOB band engine: shape {shape.name}, band {band}, init {init}, rho {rho:g}, {steps:,} steps x batch {batch}"
        f" = {steps * batch:,} rows; {trained:,} trained numbers ({trained / param_count(teacher):.3f} of the teacher's); "
        f"pure Adam lr {lr:g}; average decay {average_decay:g}; device {device}"
        + (f"; a read every {every:,} updates" if every and on_every is not None else "")
        + (f"; micro {micro} (the rows of a step in groups of {micro}, one optimizer step)" if micro > 0 else ""))
    grouped = 0 < micro < batch

    def checkpoint(step: int) -> None:
        _save(ck_path, {"engine": engine.state_dict(), "average": average, "adam": opt.state_dict(), "step": step,
                        "seed": int(seed), "generator": "step_generator(seed, step)", "log": history,
                        "reads": reads_log, "step_seconds": step_seconds, "settings": settings})

    def read_now(n: int) -> None:
        avg = averaged_engine(engine, average).eval()
        engine.eval()
        try:
            value = on_every(n, engine, avg)
        finally:
            engine.train()
        del avg
        reads_log.append({"step": n, "value": value})

    t0 = time.time()
    start = done
    zero_f64 = torch.zeros((), dtype=torch.float64, device=device)
    for k in range(done, stop):
        s0 = time.perf_counter()
        g = step_generator(seed, k)
        x0, ids = source(k, batch)
        x0 = x0.to(device=device, dtype=torch.float32)
        t_cpu = sample_steps(band, batch, rho, g)
        eps = torch.randn(tuple(x0.shape), generator=g).to(device)
        z = add_noise(x0, eps, t_cpu, ac)
        t = t_cpu.to(device)
        dropped_cpu = torch.rand(batch, generator=g, device=g.device) < float(drop)
        n_dropped = int(dropped_cpu.sum())
        if n_dropped == batch:
            states = empty_f.expand(batch, -1, -1)
        elif n_dropped == 0:
            states = teacher.text_states(ids.to(device)).float()
        else:
            keep = ~dropped_cpu
            kept = teacher.text_states(ids[keep].to(device)).float()
            states = empty_f.expand(batch, -1, -1).clone()
            states[keep.to(device)] = kept
        if grouped:
            opt.zero_grad(set_to_none=True)
            t_out = zero_f64
            t_data = zero_f64
            t_total = zero_f64
            for a in range(0, batch, micro):
                b = min(a + micro, batch)
                w = (b - a) / batch                 # this group's share of the whole batch's mean
                with _autocast(autocast, device):
                    o_t = teacher.predict(z[a:b], t[a:b], states[a:b])
                    o_s = engine(z[a:b], t[a:b], states[a:b])
                loss, part = capture_loss(o_s, o_t, eps[a:b], lam_out, lam_data, sync=False)
                (loss * w).backward()               # the gradients add up in the parameters
                t_out = t_out + w * part["out"].double()
                t_data = t_data + w * part["data"].double()
                t_total = t_total + w * part["total"].double()
                del o_t, o_s, loss
        else:
            with _autocast(autocast, device):
                o_t = teacher.predict(z, t, states)        # no gradient; the inputs cast to the teacher's precision
                o_s = engine(z, t, states)
            loss, part = capture_loss(o_s, o_t, eps, lam_out, lam_data, sync=False)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            t_out = part["out"].double()
            t_data = part["data"].double()
            t_total = part["total"].double()
            del o_t, o_s, loss
        # one host read a step: fused per-tensor norms summed on device, combined with loss terms
        grads = [p.grad for p in params if p.grad is not None]
        sq_t = torch.stack(torch._foreach_norm(grads, 2.0)).double().pow(2).sum() if grads else zero_f64
        opt.step()
        opt.zero_grad(set_to_none=True)
        n = k + 1
        d = average_decay_at(n, average_decay)
        with torch.no_grad():
            torch._foreach_mul_(avg_list, d)
            torch._foreach_add_(avg_list, params, alpha=1.0 - d)
        out_v, data_v, total_v, sq = torch.stack([t_out, t_data, t_total, sq_t]).tolist()
        row = {"step": n, "loss": total_v, "out": out_v, "data": data_v,
               "grad_norm": math.sqrt(sq), "dropped": n_dropped}
        history.append(row)
        step_seconds.append(time.perf_counter() - s0)       # the step only: the reads below are not counted
        if n == 1:
            log(f"STEP 1 terms: teacher term {out_v:.6g} (weight {lam_out:g}), noise term {data_v:.6g} "
                f"(weight {lam_data:g}), total {total_v:.6g}, gradient norm {row['grad_norm']:.4g}")
        if log_every and (n % int(log_every) == 0 or n == stop):
            spent = time.time() - t0
            left = spent / (n - start) * (steps - n)
            log(f"step {n:,}/{steps:,} loss {total_v:.6g} (teacher {out_v:.6g}, noise {data_v:.6g})"
                f" grad {row['grad_norm']:.4g}; spent {spent / 60:.1f} min, left {left / 60:.1f} min")
        if every and on_every is not None and (n % every == 0 or n == steps):
            read_now(n)                                     # before a checkpoint of the same step, so a resume has it
        if save_every and n % int(save_every) == 0 and n != stop:
            checkpoint(n)
    final = max(done, stop)
    finished = final >= steps
    if not finished or keep_checkpoint:
        checkpoint(final)
    avg_engine = averaged_engine(engine, average)
    per_step = _median(step_seconds)
    if finished:
        save_engine(engine, out, "engine", average={k: v for k, v in average.items()},
                    extra={"band": band, "rho": float(rho), "steps": steps, "init": init})
        tmp = os.path.join(out, TRAIN_JSON + ".part")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"settings": settings, "seconds_a_step": per_step, "trained": int(trained), "micro": micro,
                       "reads": reads_log, "log": history}, fh, indent=1)
            fh.write("\n")
        os.replace(tmp, os.path.join(out, TRAIN_JSON))
        if not keep_checkpoint and os.path.isfile(ck_path):
            os.remove(ck_path)
    seconds = time.time() - t0
    log(f"{'DONE' if finished else 'PAUSED'} at step {final:,} of {steps:,}; {seconds / 60:.1f} min this run"
        + (f"; {per_step:.4g} s a step (median)" if per_step is not None else ""))
    return {"engine": engine, "average": avg_engine, "log": history, "step": final, "finished": finished,
            "seconds": seconds, "settings": settings, "reads": reads_log, "seconds_a_step": per_step,
            "trained": int(trained), "micro": micro}
