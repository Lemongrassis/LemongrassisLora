"""Partial gradient checkpointing for Anima: leading blocks skip the recompute. CPU only."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training.families.anima.forward import (  # noqa: E402
    forward_with_optional_checkpoint,
    partial_checkpoint_blocks,
)


def _tiny_anima(num_blocks=3):
    from modeling.anima.anima_modeling import Anima

    torch.manual_seed(0)
    return Anima(
        max_img_h=64, max_img_w=64, max_frames=1, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=64, num_blocks=num_blocks, num_heads=2, mlp_ratio=2.0,
        crossattn_emb_channels=32, pos_emb_cls="rope3d", use_adaln_lora=True, adaln_lora_dim=16,
    ).eval()


def test_block_count_from_percent():
    assert partial_checkpoint_blocks(28, 25) == 7
    assert partial_checkpoint_blocks(28, 50) == 14
    assert partial_checkpoint_blocks(36, 25) == 9
    # swapped trailing blocks always stay checkpointed
    assert partial_checkpoint_blocks(28, 25, swapped=26) == 2
    assert partial_checkpoint_blocks(28, 95, swapped=28) == 0
    assert partial_checkpoint_blocks(28, 0) == 0


def _run(model, use_checkpoint, skip, monkeypatch):
    import torch.utils.checkpoint as tuc

    calls = []
    real = tuc.checkpoint

    def counting(fn, *args, **kw):
        calls.append(1)
        return real(fn, *args, **kw)

    monkeypatch.setattr(tuc, "checkpoint", counting)
    model.checkpoint_skip_blocks = skip
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    x = torch.randn(1, 16, 1, 8, 8)
    t = torch.full((1, 1), 0.5)
    cross = torch.randn(1, 4, 32)
    pad = torch.zeros(1, 1, 16, 16)
    out = forward_with_optional_checkpoint(model, x, t, cross, pad, use_checkpoint=use_checkpoint)
    out.square().mean().backward()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    return out.detach(), grads, len(calls)


def test_partial_matches_full_and_skips_leading_blocks(monkeypatch):
    model = _tiny_anima(num_blocks=3)
    plain, g_plain, n_plain = _run(model, False, 0, monkeypatch)
    full, g_full, n_full = _run(model, True, 0, monkeypatch)
    part, g_part, n_part = _run(model, True, 2, monkeypatch)
    assert (n_plain, n_full, n_part) == (0, 3, 1)
    for out, grads in ((full, g_full), (part, g_part)):
        torch.testing.assert_close(out, plain)
        assert len(grads) == len(g_plain)
        for a, b in zip(grads, g_plain):
            torch.testing.assert_close(a, b)


def test_setup_sets_skip_and_respects_swap_and_low_vram():
    from training.phases.models import _setup_partial_checkpoint

    def ctx(**args):
        base = dict(grad_checkpoint=True, grad_checkpoint_partial=True, grad_checkpoint_skip_pct=25, low_vram=False)
        base.update(args)
        return SimpleNamespace(
            args=SimpleNamespace(**base),
            family=SimpleNamespace(spec=SimpleNamespace(family_id="anima")),
            model=SimpleNamespace(blocks=[None] * 28),
            block_swap=None,
        )

    c = ctx()
    _setup_partial_checkpoint(c)
    assert c.model.checkpoint_skip_blocks == 7

    c = ctx()
    c.block_swap = SimpleNamespace(num_swap=24)
    _setup_partial_checkpoint(c)
    assert c.model.checkpoint_skip_blocks == 4

    for off in (dict(grad_checkpoint_partial=False), dict(grad_checkpoint=False), dict(low_vram=True)):
        c = ctx(**off)
        _setup_partial_checkpoint(c)
        assert not hasattr(c.model, "checkpoint_skip_blocks")


def test_schema_fields():
    from pydantic import ValidationError
    from studio.schema import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.grad_checkpoint_partial is False
    assert cfg.grad_checkpoint_skip_pct == 25
    with pytest.raises(ValidationError):
        TrainingConfig(grad_checkpoint_skip_pct=100)
    props = TrainingConfig.model_json_schema()["properties"]
    assert props["grad_checkpoint_partial"]["group"] == "training"
    assert "grad_checkpoint==true" in props["grad_checkpoint_skip_pct"]["show_when"]
