"""SinkGD, ReREX and gaussian LoRA init: the formulas, checked against BluFlow's recorded values."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_sr_sinkhorn_gives_unit_rms_rows_and_columns() -> None:
    from training.optimizers.sinkgd import sr_sinkhorn

    grad = torch.randn(64, 8) * torch.logspace(-6, 2, 64).unsqueeze(1)
    out = sr_sinkhorn(grad, 5)
    assert out.pow(2).mean().sqrt().item() == pytest.approx(1.0, abs=1e-3)
    assert out.pow(2).mean(dim=0).sqrt().numpy() == pytest.approx([1.0] * 8, abs=1e-3)
    assert out.pow(2).mean(dim=1).sqrt().numpy() == pytest.approx([1.0] * 64, abs=0.05)


def test_sr_sinkhorn_cancels_gradient_scale() -> None:
    from training.optimizers.sinkgd import sr_sinkhorn

    grad = torch.randn(32, 8)
    assert torch.allclose(sr_sinkhorn(grad * 1e-5, 5), sr_sinkhorn(grad, 5), atol=1e-4)


def test_sr_sinkhorn_keeps_zero_gradient_zero_and_handles_other_shapes() -> None:
    from training.optimizers.sinkgd import sr_sinkhorn

    assert torch.count_nonzero(sr_sinkhorn(torch.zeros(16, 8), 5)) == 0
    vector = sr_sinkhorn(torch.tensor([3.0, 4.0]), 5)
    assert vector.pow(2).mean().sqrt().item() == pytest.approx(1.0)
    assert sr_sinkhorn(torch.tensor(-2.0), 5).item() == -1.0
    assert sr_sinkhorn(torch.randn(4, 3, 2), 5).shape == (4, 3, 2)


def test_sinkgd_step_is_lr_times_normalized_gradient_per_group() -> None:
    from training.optimizers.sinkgd import SinkGD, sr_sinkhorn

    down = torch.nn.Parameter(torch.zeros(8, 16))
    up = torch.nn.Parameter(torch.zeros(16, 8))
    optimizer = SinkGD([{"params": [down]}, {"params": [up], "lr": 0.04}], lr=0.001)
    down.grad, up.grad = torch.randn(8, 16), torch.randn(16, 8)
    expected_down = -0.001 * sr_sinkhorn(down.grad, 5)
    expected_up = -0.04 * sr_sinkhorn(up.grad, 5)
    optimizer.step()
    assert torch.allclose(down.data, expected_down)
    assert torch.allclose(up.data, expected_up)


def test_sinkgd_state_dict_round_trip() -> None:
    from training.optimizers.sinkgd import SinkGD

    p = torch.nn.Parameter(torch.zeros(4, 4))
    optimizer = SinkGD([p], lr=0.1)
    optimizer.load_state_dict(optimizer.state_dict())
    assert optimizer.param_groups[0]["sinkhorn_iterations"] == 5


def test_rerex_matches_the_lr_recorded_in_a_bluflow_lora() -> None:
    """Aoki Exhibition: lr 3e-4, 1200 steps, warmup 20; its file at step 700 records this lr."""
    from training.schedulers.rerex import rerex_lambda

    factor = rerex_lambda(1200, warmup_steps=20, global_d=0.6, local_d=0.85, weight_power=1.5, num_segments=8)
    assert 3e-4 * factor(700) == pytest.approx(0.00021053338190070173, rel=1e-9)


def test_rerex_warmup_then_continuous_monotone_decay_to_floor() -> None:
    from training.schedulers.rerex import rerex_lambda

    factor = rerex_lambda(1000, warmup_steps=10)
    assert factor(0) == pytest.approx(0.1)
    assert factor(9) == 1.0
    values = [factor(s) for s in range(10, 1000)]
    assert values[0] == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))
    assert max(a - b for a, b in zip(values, values[1:])) < 0.2
    assert values[-1] < 0.1
    assert factor(5000) == 0.0
    assert rerex_lambda(1000, floor=0.25)(5000) == 0.25


def test_rerex_builder_keeps_each_group_peak() -> None:
    from training.schedulers import build_scheduler

    p, q = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([{"params": [p], "lr": 1e-3}, {"params": [q], "lr": 4e-2}], lr=1e-3)
    args = SimpleNamespace(lr_scheduler="rerex", lr_scheduler_warmup_steps=0, lr_scheduler_eta_min=0.0)
    scheduler = build_scheduler(args, optimizer, 100)
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([1e-3, 4e-2])
    for _ in range(30):
        optimizer.step()
        scheduler.step()
    low, high = [g["lr"] for g in optimizer.param_groups]
    assert high / low == pytest.approx(40.0)
    assert high < 4e-2


def test_gaussian_init_sets_down_std_one_over_rank_and_leaves_up() -> None:
    from training.families.moss_image.experimental import init_down_gaussian

    lora = SimpleNamespace(lora_down=torch.nn.Linear(4096, 8, bias=False))
    network = SimpleNamespace(loras=[lora])
    torch.manual_seed(0)
    assert init_down_gaussian(network) == 1
    weight = lora.lora_down.weight
    assert weight.std().item() == pytest.approx(0.125, abs=0.002)
    x = weight.flatten() - weight.mean()
    assert ((x ** 4).mean() / (x ** 2).mean() ** 2).item() == pytest.approx(3.0, abs=0.1)


def test_schema_accepts_the_new_options() -> None:
    from studio.schema import TrainingConfig

    cfg = TrainingConfig(optimizer_type="sinkgd", lr_scheduler="rerex", lr_scheduler_warmup_steps=20,
                         model_family="moss_image", moss_gaussian_init=True, moss_caption_variations=5)
    assert cfg.sinkgd_iterations == 5
    assert cfg.lr_scheduler_rerex_num_segments == 8
    assert math.isclose(cfg.moss_caption_tag_dropout, 0.05)
    from training.optimizers import validate_schema_consistency as optimizers
    from training.schedulers import validate_schema_consistency as schedulers
    optimizers()
    schedulers()
