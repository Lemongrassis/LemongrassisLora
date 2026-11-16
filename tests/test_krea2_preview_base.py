"""Krea2 previews on Turbo: base weights swapped in for a preview and restored exactly after."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from modeling.krea2 import Krea2Config, SingleStreamDiT
from training.families.krea2.loader import load_krea2_model
from training.families.krea2.preview_base import Krea2PreviewBase, check_compatible

INT8_LAYERS = ("blocks.0.attn.wq", "blocks.1.mlp.up")


def _tiny_config() -> Krea2Config:
    return Krea2Config(
        features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=2,
        patch=2, channels=4, txtlayers=3, txtheads=4, txtkvheads=2,
    )


def _checkpoint(path: Path, config: Krea2Config, seed: int, *, int8: bool,
                float_dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    state = {
        key: value.detach().to(float_dtype).contiguous()
        for key, value in SingleStreamDiT(config).state_dict().items()
    }
    if int8:
        spec = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
        for layer in INT8_LAYERS:
            weight = state[f"{layer}.weight"]
            state[f"{layer}.weight"] = torch.randint(-127, 128, weight.shape, dtype=torch.int8)
            state[f"{layer}.weight_scale"] = torch.rand(weight.shape[0], 1) + 0.01
            state[f"{layer}.comfy_quant"] = torch.tensor(list(spec.encode()), dtype=torch.uint8)
    save_file(state, str(path))
    return state


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out = {name: p.detach().clone() for name, p in model.named_parameters()}
    for layer in INT8_LAYERS:
        module = dict(model.named_modules())[layer]
        if hasattr(module, "scale_weight"):
            out[f"{layer}.scale_weight"] = module.scale_weight.clone()
    return out


def _assert_same(model: torch.nn.Module, expected: dict[str, torch.Tensor]) -> None:
    actual = _snapshot(model)
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name


@pytest.mark.parametrize("int8", [False, True])
def test_preview_base_swaps_in_turbo_and_restores_training_weights(tmp_path: Path, int8: bool) -> None:
    config = _tiny_config()
    raw, turbo = tmp_path / "raw.safetensors", tmp_path / "turbo.safetensors"
    _checkpoint(raw, config, 1, int8=int8)
    # Turbo stores floats in another dtype (like Comfy-Org's ConvRot files)
    turbo_state = _checkpoint(turbo, config, 2, int8=int8, float_dtype=torch.bfloat16)
    model = load_krea2_model(raw, "cpu", torch.bfloat16, config=config)
    # a trained tensor (train_norm) must not be touched by the swap
    model.blocks[0].prenorm.scale.requires_grad_(True)
    before = _snapshot(model)

    preview = Krea2PreviewBase(model, raw, turbo, config=config)
    with preview.swapped():
        assert preview.active
        params = dict(model.named_parameters())
        assert torch.equal(params["first.weight"], turbo_state["first.weight"])
        assert torch.equal(params["blocks.0.prenorm.scale"], before["blocks.0.prenorm.scale"])
        if int8:
            module = dict(model.named_modules())["blocks.0.attn.wq"]
            assert torch.equal(module.weight, turbo_state["blocks.0.attn.wq.weight"])
            assert torch.equal(module.scale_weight, turbo_state["blocks.0.attn.wq.weight_scale"])
        with preview.swapped():  # nested: no second swap
            pass
    assert not preview.active
    _assert_same(model, before)


def test_preview_base_restores_when_the_body_fails(tmp_path: Path) -> None:
    config = _tiny_config()
    raw, turbo = tmp_path / "raw.safetensors", tmp_path / "turbo.safetensors"
    _checkpoint(raw, config, 1, int8=True)
    _checkpoint(turbo, config, 2, int8=True)
    model = load_krea2_model(raw, "cpu", torch.bfloat16, config=config)
    before = _snapshot(model)

    preview = Krea2PreviewBase(model, raw, turbo, config=config)
    with pytest.raises(RuntimeError, match="boom"):
        with preview.swapped():
            raise RuntimeError("boom")
    _assert_same(model, before)


def test_preview_base_unavailable_keeps_training_weights(tmp_path: Path) -> None:
    config = _tiny_config()
    raw, turbo = tmp_path / "raw.safetensors", tmp_path / "turbo.safetensors"
    _checkpoint(raw, config, 1, int8=False)
    _checkpoint(turbo, config, 2, int8=False)
    model = load_krea2_model(raw, "cpu", torch.bfloat16, config=config)
    before = _snapshot(model)

    preview = Krea2PreviewBase(model, raw, turbo, config=config)
    turbo.unlink()
    with pytest.raises(Krea2PreviewBase.Unavailable):
        with preview.swapped():
            pytest.fail("body must not run")
    _assert_same(model, before)


def test_check_compatible_rejects_a_different_storage_format(tmp_path: Path) -> None:
    config = _tiny_config()
    raw, turbo = tmp_path / "raw.safetensors", tmp_path / "turbo.safetensors"
    _checkpoint(raw, config, 1, int8=True)
    _checkpoint(turbo, config, 2, int8=False)

    with pytest.raises(ValueError, match="stores weights differently"):
        check_compatible(raw, turbo, config=config)
