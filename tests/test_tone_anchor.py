from __future__ import annotations

import torch

from training.tone_anchor import remove_error_mean

_DIMS = (2, 3, 4)


def _batch():
    torch.manual_seed(0)
    target = torch.randn(2, 4, 1, 6, 6)
    pred = target + torch.randn(2, 4, 1, 6, 6) * 0.1 + torch.tensor([0.5, -0.3, 0.2, 0.0]).view(1, 4, 1, 1, 1)
    return pred, target


def test_high_noise_sample_loses_the_mean_error_and_keeps_the_rest():
    pred, target = _batch()
    out = remove_error_mean(pred, target, torch.tensor([0.9, 0.2]), min_t=0.6)
    error = out - target
    assert error[0].mean((1, 2, 3)).abs().max() < 1e-6
    before = (pred - target)[0]
    torch.testing.assert_close(error[0], before - before.mean((1, 2, 3), keepdim=True), atol=1e-6, rtol=0)
    # the low-noise sample is untouched
    torch.testing.assert_close(out[1], pred[1])


def test_nothing_changes_below_the_threshold_or_at_zero_strength():
    pred, target = _batch()
    assert remove_error_mean(pred, target, torch.tensor([0.1, 0.5]), min_t=0.6) is pred
    assert remove_error_mean(pred, target, torch.tensor([0.9, 0.9]), min_t=0.6, strength=0.0) is pred


def test_strength_removes_a_share_of_the_mean():
    pred, target = _batch()
    out = remove_error_mean(pred, target, torch.tensor([0.9, 0.9]), min_t=0.6, strength=0.25)
    before = (pred - target).mean(_DIMS)
    torch.testing.assert_close((out - target).mean(_DIMS), before * 0.75, atol=1e-6, rtol=0)


def test_no_gradient_on_a_uniform_shift_of_the_prediction():
    pred, target = _batch()
    shift = torch.zeros(1, 4, 1, 1, 1, requires_grad=True)
    out = remove_error_mean(pred + shift, target, torch.tensor([0.9, 0.9]), min_t=0.6)
    torch.nn.functional.mse_loss(out, target).backward()
    assert shift.grad.abs().max() < 1e-7


def test_mask_limits_the_mean_to_supervised_positions():
    pred, target = _batch()
    mask = torch.zeros(2, 1, 1, 6, 6)
    mask[..., :3] = 1
    out = remove_error_mean(pred, target, torch.tensor([0.9, 0.9]), min_t=0.6, spatial_mask=mask)
    inside = ((out - target) * mask).sum(_DIMS) / mask.sum(_DIMS)
    assert inside.abs().max() < 1e-6
