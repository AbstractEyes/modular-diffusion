"""Measuring the teacher per band with no training, on the tiny random teacher: the six single changes of shape and
their born reads per band, the prompt-need read (with its exact control), and the measured forward cost."""
import math

import pytest
import torch

pytest.importorskip("diffusers")        # the capture tests need the optional extra

from geolip.modular_diffusion.capture import measure
from geolip.modular_diffusion.capture.schedule import BANDS
from geolip.modular_diffusion.capture.shapes import Shape
from geolip.modular_diffusion.capture.teacher import tiny_teacher

PROMPTS = ["a red house by a lake", "two cats asleep", "blue sky at noon"]


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


@pytest.fixture(scope="module")
def probe(teacher):
    from geolip.modular_diffusion.capture.reads import make_probe
    states = teacher.text_states(teacher.tokenize(PROMPTS[:2]))
    return make_probe(states, seed=3, per_band=2, latent_shape=(4, 16, 16), empty=teacher.empty_states(2),
                      alphas_cumprod=teacher.alphas_cumprod)


def test_the_six_single_changes():
    full = Shape((2, 2, 2, 2), True, 4)
    changes = measure.single_changes()
    assert [n for n, _ in changes] == ["layers0", "layers1", "layers2", "layers3", "mid_off", "stages3"]
    for name, s in changes:
        differs = [a != b for a, b in zip(s.layers, full.layers)] + [s.mid != full.mid, s.stages != full.stages]
        assert sum(differs) == 1, name
    assert len({s for _, s in changes}) == 6


def test_removal_scores_six_rows_a_band(teacher, probe):
    from geolip.modular_diffusion.capture.reads import born_read
    out = measure.removal_scores(teacher, probe, log=None)
    assert len(out["rows"]) == 6 * len(BANDS)
    for b in BANDS:
        assert sorted(out["table"][b]) == sorted(n for n, _ in measure.single_changes())
    for r in out["rows"]:
        assert math.isfinite(r["mse"]) and r["mse"] > 0.0, r       # every tensor of the tiny teacher takes part
        assert 0.0 < r["share"] < 1.0
    # the shortcut (the teacher's answers once a band) gives the born read's number
    one = born_read(teacher, Shape((2, 2, 2, 2), False, 4), "middle", probe)
    assert one["mse"] == out["table"]["middle"]["mid_off"]


def test_prompt_swap_control_is_exact(teacher):
    """The swap with the true prompts as their own replacement is the teacher, and inside an assembly it gives the
    teacher's sample to the bit; the guidance branch's empty rows are never swapped."""
    from geolip.modular_diffusion.capture.assemble import Assembly, sample
    states = teacher.text_states(teacher.tokenize(PROMPTS))
    empty = teacher.empty_states(3)
    swap = measure.PromptSwap(teacher, states, states.roll(1, dims=0))
    both = torch.cat([empty, states])
    got = swap.swap(both)
    assert torch.equal(got[:3], empty) and torch.equal(got[3:], states.roll(1, dims=0))
    same = measure.PromptSwap(teacher, states, states)
    kw = dict(steps=4, guidance=7.5, latent_shape=(4, 16, 16), alphas_cumprod=teacher.alphas_cumprod)
    ref = sample(teacher, states, empty, 5, **kw)
    for b in BANDS:
        parts = {o: teacher for o in BANDS}
        parts[b] = same
        assert torch.equal(sample(Assembly(parts), states, empty, 5, **kw), ref), b


def test_text_need(teacher):
    out = measure.text_need(teacher, PROMPTS, seed=1, steps=5, log=None)
    assert [r["band"] for r in out["rows"]] == list(BANDS)
    assert out["empty_everywhere"] > 0.0
    for r in out["rows"]:
        for k in ("empty", "shuffled"):
            assert math.isfinite(r[k]) and r[k] >= 0.0
    # five DDIM steps (981 .. 1) visit every band, so every band's change moves the final latent
    assert all(r["empty"] > 0.0 and r["shuffled"] > 0.0 for r in out["rows"])
    with pytest.raises(ValueError):
        measure.text_need(teacher, PROMPTS[:1], steps=2, log=None)
    with pytest.raises(ValueError):
        measure.text_need(teacher, ["", "x"], steps=2, log=None)


def test_cost(teacher):
    r = measure.cost(teacher, "small", batch=1, repeats=1)
    for k in ("params", "teacher_params", "share", "seconds", "teacher_seconds", "cost"):
        assert k in r and r[k] > 0, k
    assert r["share"] < 1.0 and r["size"] == 16
