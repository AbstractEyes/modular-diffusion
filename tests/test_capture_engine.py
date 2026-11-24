"""Capture, engines by paired block removal: the shape full is the teacher bit for bit, every shape's tensors are all
inherited and shape-exact (test 2), and an engine that ignores the prompt does not depend on the prompt's states
(test 9)."""
import pytest
import torch

pytest.importorskip("diffusers")

from geolip.modular_diffusion.capture import engine as E
from geolip.modular_diffusion.capture.shapes import SHAPES, Shape, all_shapes, shape_from
from geolip.modular_diffusion.capture.teacher import tiny_teacher


@pytest.fixture(scope="module")
def teacher():
    return tiny_teacher(0)


def _inputs(teacher, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, 4, 16, 16, generator=g)
    states = teacher.text_states(teacher.tokenize([f"prompt number {i}" for i in range(b)]))
    return z, states


def test_2_full_is_the_teacher_bit_for_bit(teacher):
    eng = E.build_engine(teacher, "full")
    a, b = eng.unet.state_dict(), teacher.unet.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    z, states = _inputs(teacher)
    for t in (5, 400, 990):
        tt = torch.full((2,), t)
        assert torch.equal(eng(z, tt, states), teacher.predict(z, tt, states))


def _expected_source(key: str, shape) -> str:
    """The teacher tensor an engine tensor must come from, by the rule of paired removal stated on its own: a down
    stage keeps its first layers; an up stage that keeps two layers keeps the teacher's first and third; an engine of
    three stages has lost the teacher's innermost down stage and its first up stage; all else keeps its name."""
    import re
    m = re.match(r"up_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.*)", key)
    if m:
        j, kind, r, rest = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        mirrored_down = shape.stages - 1 - j
        if shape.kept[mirrored_down] == 1 and r == 1:
            r = 2
        return f"up_blocks.{j + (4 - shape.stages)}.{kind}.{r}.{rest}"
    m = re.match(r"up_blocks\.(\d+)\.(.*)", key)
    if m:
        return f"up_blocks.{int(m.group(1)) + (4 - shape.stages)}.{m.group(2)}"
    return key


def _fingerprinted(seed: int = 3):
    """A tiny teacher whose every tensor is redrawn, so that no two of its tensors hold the same numbers (a fresh
    network's normalization weights are all ones): a tensor copied from the wrong place cannot then pass by
    coincidence."""
    teacher = tiny_teacher(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for p in teacher.unet.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    marks = {(tuple(v.shape), float(v.double().sum())) for v in teacher.unet.state_dict().values()}
    assert len(marks) == len(teacher.unet.state_dict())            # every tensor tells itself apart
    return teacher


def test_2_every_shape_is_inherited_exactly():
    teacher = _fingerprinted()
    shapes = all_shapes()
    assert len(shapes) == 64 and len(set(shapes)) == 64
    src = teacher.unet.state_dict()
    for shape in shapes:
        eng = E.build_engine(teacher, shape)
        dst = eng.unet.state_dict()
        mapping = E.inherit_map(teacher, shape)
        assert set(mapping) == set(dst)                        # every engine tensor has its source
        for k, v in dst.items():
            s = src[mapping[k]]
            assert s.shape == v.shape, (shape, k)
            assert torch.equal(s, v), (shape, k)               # copied, never resized or drawn afresh
            # the same tensor found by a rule written here, apart from the library's own map
            assert torch.equal(src[_expected_source(k, shape)], v), (shape, k)
        # the paired removal: a down stage keeps n layers, its mirrored up stage n + 1
        kept = shape.kept
        assert len(eng.unet.down_blocks) == shape.stages == len(eng.unet.up_blocks)
        for i, n in enumerate(kept):
            assert len(eng.unet.down_blocks[i].resnets) == n
            assert len(eng.unet.up_blocks[shape.stages - 1 - i].resnets) == n + 1
        assert (eng.unet.mid_block is not None) == shape.mid
        assert eng.unet.down_blocks[-1].downsamplers is None
        z, states = _inputs(teacher, 1)
        assert eng(z, torch.tensor([500]), states).shape == z.shape


def test_2_a_wrong_map_does_not_pass(monkeypatch):
    """The control of the test above: a library that kept an up stage's first and SECOND layers, where the rule keeps
    its first and third, must be told apart by the rule written in this file (or refused by the build itself, where
    the wrong tensor does not even fit)."""
    teacher = _fingerprinted()
    src = teacher.unet.state_dict()
    monkeypatch.setattr(E, "up_kept", lambda n: (0, 1) if n == 1 else (0, 1, 2))
    told_apart = 0
    for shape in (Shape((1, 1, 1, 1), True, 4), Shape((2, 1, 2, 1), True, 4), Shape((1, 1, 1, 1), False, 3)):
        try:
            eng = E.build_engine(teacher, shape)
        except ValueError:
            told_apart += 1
            continue
        wrong = [k for k, v in eng.unet.state_dict().items() if not torch.equal(src[_expected_source(k, shape)], v)]
        assert wrong and all(k.startswith("up_blocks.") for k in wrong), shape
        told_apart += 1
    assert told_apart == 3


def test_2_the_map_takes_first_and_third_up_layers():
    shape = Shape((1, 2, 1, 2), True, 3)
    assert E.teacher_key("down_blocks.0.resnets.0.conv1.weight", shape) == "down_blocks.0.resnets.0.conv1.weight"
    assert E.teacher_key("down_blocks.1.resnets.1.conv1.weight", shape) == "down_blocks.1.resnets.1.conv1.weight"
    # three stages: engine up block 0 is the teacher's up block 1, mirrored to down stage 2 (one layer kept)
    assert E.teacher_key("up_blocks.0.resnets.1.conv1.weight", shape) == "up_blocks.1.resnets.2.conv1.weight"
    assert E.teacher_key("up_blocks.0.attentions.1.proj_in.weight", shape) == "up_blocks.1.attentions.2.proj_in.weight"
    assert E.teacher_key("up_blocks.1.resnets.2.conv1.weight", shape) == "up_blocks.2.resnets.2.conv1.weight"
    assert E.teacher_key("up_blocks.0.upsamplers.0.conv.weight", shape) == "up_blocks.1.upsamplers.0.conv.weight"
    assert E.teacher_key("conv_in.weight", shape) == "conv_in.weight"


def test_named_shapes_and_text_forms():
    assert shape_from("full") == Shape() and SHAPES["tiny"] == Shape((1, 1, 1, 1), False, 3)
    assert Shape.parse("1211/nomid/4") == Shape((1, 2, 1, 1), False, 4)
    assert Shape.parse(Shape((2, 1, 2, 1), True, 3).text).name == "2121/mid/3"
    assert SHAPES["base"].name == "base"
    with pytest.raises(ValueError):
        Shape((3, 1, 1, 1))
    with pytest.raises(ValueError):
        Shape.parse("big")


def test_shape_report(teacher):
    r = E.shape_report(teacher, "base", batch=1, repeats=1)
    assert r["teacher_params"] == E.param_count(teacher.unet)
    assert 0 < r["share"] < 1 and r["params"] == E.param_count(E.build_engine(teacher, "base"))
    assert r["seconds"] > 0 and r["cost"] > 0


def test_9_an_engine_without_text_ignores_the_prompt(teacher):
    eng = E.build_engine(teacher, "base", text=False)
    assert eng.context is not None and eng.context.shape == (1, 77, 32) and eng.context.requires_grad
    assert torch.equal(eng.context.detach(), teacher.empty_states(1))
    z, s1 = _inputs(teacher, 2, 1)
    s2 = torch.randn_like(s1) * 3
    for t in (10, 500, 980):
        tt = torch.full((2,), t)
        a, b = eng(z, tt, s1), eng(z, tt, s2)
        assert torch.equal(a, b)
    assert torch.equal(eng(z, tt, None), a)
    # with text, the states do matter
    eng_t = E.build_engine(teacher, "base", text=True)
    assert not torch.equal(eng_t(z, tt, s1), eng_t(z, tt, s2))
    with pytest.raises(ValueError):
        E.build_engine(teacher.unet, "base", text=False)


def test_engine_files_round_trip(teacher, tmp_path):
    eng = E.build_engine(teacher, "small", text=False)
    with torch.no_grad():
        for p in eng.parameters():
            p.add_(0.01)
    avg = {n: p.detach() * 0.5 for n, p in eng.named_parameters()}
    E.save_engine(eng, str(tmp_path), average=avg)
    last = E.load_engine(teacher, str(tmp_path), average=False)
    mean = E.load_engine(teacher, str(tmp_path), average=True)
    assert last.shape == eng.shape and not last.text
    for (n, p), (_, q), (_, r) in zip(eng.named_parameters(), last.named_parameters(), mean.named_parameters()):
        assert torch.equal(p, q) and torch.equal(avg[n], r)
