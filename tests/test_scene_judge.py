"""The scene judge: its shapes, a training step, predict, the ceiling's counts, save and load."""
import torch

from geolip.modular_diffusion import scene_judge as SJ
from geolip.modular_diffusion import scenes
from geolip.modular_diffusion.scene_reads import NOTHING


def test_the_judge_shapes_and_its_reads():
    j = SJ.born_judge(0)
    lt, lc = j(torch.zeros(5, 3, 16, 16))
    assert lt.shape == (5, NOTHING + 1) and lc.shape == (5, 9)
    assert j.things.in_features == 128 * 4 * 4                      # a flatten, no pooling
    convs = [j.conv1, j.conv2, j.conv3]
    assert [c.out_channels for c in convs] == [32, 64, 128] and [c.stride[0] for c in convs] == [1, 2, 2]
    x = scenes.ScenesBed("F").train_batch("mix", 3, torch.Generator().manual_seed(0)).x
    pt, pc = SJ.predict(j, x)
    assert pt.shape == (3, 9) and pc.shape == (3, 9) and pt.dtype == torch.long
    j2 = SJ.born_judge(0)
    assert torch.equal(j.conv1.weight, j2.conv1.weight) and not torch.equal(j.conv1.weight, SJ.born_judge(1).conv1.weight)


def test_a_training_step_lowers_the_loss_on_its_batch_and_color_counts_only_things():
    bed = scenes.ScenesBed("F")
    sb = bed.train_batch("mix", 32, torch.Generator().manual_seed(1))
    crops = scenes.cell_crops(sb.x).reshape(-1, 3, 16, 16)
    things, colors = SJ.labels(sb.cells.reshape(-1), sb.colors.reshape(-1))
    assert bool((things[colors < 0] == NOTHING).all())
    j = SJ.born_judge(0)
    opt = torch.optim.Adam(j.parameters(), lr=1e-3)
    first = float(SJ.judge_loss(j, crops, things, colors))
    for _ in range(5):
        loss = SJ.judge_loss(j, crops, things, colors)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(SJ.judge_loss(j, crops, things, colors)) < first
    empty_only = things == NOTHING
    l1 = SJ.judge_loss(j, crops[empty_only], things[empty_only], colors[empty_only])
    lt, _ = j(crops[empty_only])
    assert abs(float(l1) - float(torch.nn.functional.cross_entropy(lt, things[empty_only]))) < 1e-6


def test_train_judge_writes_its_ceiling_and_loads(tmp_path):
    bed = scenes.ScenesBed("F")
    r = SJ.train_judge(bed, 0, str(tmp_path), steps=2, batch=8, log=lambda s: None)
    c = r["ceiling"]
    assert c["crops"] == 4500 and c["thing_rate"]["n"] == 4500 and 0 < c["color_rate"]["n"] < 4500
    j = SJ.load_judge(str(tmp_path))
    x = bed.heldout("mix", 4, 3).x
    j2 = SJ.born_judge(0)
    assert j.conv1.weight.shape == j2.conv1.weight.shape
    a, b = SJ.predict(j, x)
    assert a.shape == (4, 9)
