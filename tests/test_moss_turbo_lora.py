"""MOSS previews through a Turbo LoRA: hooked in for a preview, gone after it, base weights untouched."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from modeling.moss_image.model import MossImage
from studio.domain.config_rules import eval_show_when
from training.families import get_family
from training.families.moss_image.loader import load_dit
from training.families.moss_image.preset import MOSS_PRESET
from training.families.moss_image.turbo_lora import MossTurboLora

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)
PROJECTIONS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
RANK, ALPHA = 4, 2.0


def _model() -> MossImage:
    torch.manual_seed(5)
    return MossImage(**TINY, attn_mode="torch").eval().requires_grad_(False)


def _lora_file(path: Path, model: MossImage, *, down="lora_down", up="lora_up", prefix="diffusion_model.",
               alpha: float = ALPHA) -> Path:
    """A file in the layout of MOSS-Image-Base-8-steps-dmd: every proj.* of every layer."""
    torch.manual_seed(9)
    state = {}
    for index in range(TINY["num_layers"]):
        for name in PROJECTIONS:
            linear = getattr(model.layers[index].proj, name)
            key = f"{prefix}layers.{index}.proj.{name}"
            state[f"{key}.{down}.weight"] = torch.randn(RANK, linear.in_features) * 0.01
            state[f"{key}.{up}.weight"] = torch.randn(linear.out_features, RANK) * 0.01
            state[f"{key}.alpha"] = torch.tensor(alpha)
    save_file(state, str(path))
    return path


def _merged(model: MossImage, lora: Path, strength: float, *, alpha: float = ALPHA) -> MossImage:
    """The same LoRA written into the weights: the result a hook must match."""
    from safetensors.torch import load_file

    state = load_file(str(lora))
    merged = copy.deepcopy(model)
    for index in range(TINY["num_layers"]):
        for name in PROJECTIONS:
            key = f"diffusion_model.layers.{index}.proj.{name}"
            delta = state[f"{key}.lora_up.weight"] @ state[f"{key}.lora_down.weight"]
            getattr(merged.layers[index].proj, name).weight.data += delta * strength * alpha / RANK
    return merged


def _forward(model: MossImage) -> torch.Tensor:
    torch.manual_seed(1)
    return model(torch.randn(1, 128, 2, 2), torch.tensor([500.]), torch.randn(1, 3, 32))


@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_preview_matches_merged_weights_and_leaves_the_model_as_it_was(tmp_path, strength):
    model = _model()
    lora = _lora_file(tmp_path / "turbo.safetensors", model)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    plain = _forward(model)

    turbo = MossTurboLora(model, lora, strength=strength)
    with turbo.swapped():
        assert turbo.active
        with turbo.swapped():  # nested: not applied twice
            actual = _forward(model)

    torch.testing.assert_close(actual, _forward(_merged(model, lora, strength)), atol=1e-5, rtol=1e-4)
    assert not torch.allclose(actual, plain)
    assert not turbo.active
    assert torch.equal(_forward(model), plain)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


@pytest.mark.parametrize("names", [
    dict(down="lora_A", up="lora_B"),
    dict(prefix="model.diffusion_model."),
])
def test_other_key_layouts_are_accepted(tmp_path, names):
    model = _model()
    lora = _lora_file(tmp_path / "turbo.safetensors", model, **names)
    expected = _forward(_merged(model, _lora_file(tmp_path / "plain.safetensors", model), 1.0))
    with MossTurboLora(model, lora).swapped():
        torch.testing.assert_close(_forward(model), expected, atol=1e-5, rtol=1e-4)


def test_hooks_are_removed_when_the_preview_fails(tmp_path):
    model = _model()
    plain = _forward(model)
    turbo = MossTurboLora(model, _lora_file(tmp_path / "turbo.safetensors", model))
    with pytest.raises(RuntimeError, match="boom"):
        with turbo.swapped():
            raise RuntimeError("boom")
    assert not turbo.active
    assert torch.equal(_forward(model), plain)


def test_unavailable_at_preview_time_keeps_the_model_untouched(tmp_path):
    model = _model()
    plain = _forward(model)
    lora = _lora_file(tmp_path / "turbo.safetensors", model)
    turbo = MossTurboLora(model, lora)
    lora.unlink()
    with pytest.raises(MossTurboLora.Unavailable):
        with turbo.swapped():
            pytest.fail("body must not run")
    assert not turbo.active
    assert torch.equal(_forward(model), plain)


def test_a_file_that_does_not_fit_fails_at_startup(tmp_path):
    model = _model()
    with pytest.raises(ValueError, match="not found"):
        MossTurboLora(model, tmp_path / "missing.safetensors")

    wrong_shape = tmp_path / "wrong_shape.safetensors"
    save_file({
        "diffusion_model.layers.0.proj.q.lora_down.weight": torch.zeros(RANK, 7),
        "diffusion_model.layers.0.proj.q.lora_up.weight": torch.zeros(64, RANK),
    }, str(wrong_shape))
    with pytest.raises(ValueError, match="layers.0.proj.q"):
        MossTurboLora(model, wrong_shape)

    other_model = tmp_path / "other_model.safetensors"
    save_file({"diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.zeros(RANK, 64)}, str(other_model))
    with pytest.raises(ValueError, match="nothing to apply"):
        MossTurboLora(model, other_model)

    one_factor = tmp_path / "one_factor.safetensors"
    save_file({"diffusion_model.layers.0.proj.q.lora_down.weight": torch.zeros(RANK, 64)}, str(one_factor))
    with pytest.raises(ValueError, match="missing a LoRA factor"):
        MossTurboLora(model, one_factor)


def test_stacks_on_the_lora_being_trained_on_int8_and_stays_out_of_its_state(tmp_path):
    """Hook on top of the LyCORIS forward: both deltas count, and the adapter never sees the Turbo LoRA."""
    from utils.lycoris_adapter import LycorisAdapter

    source = _model()
    base = tmp_path / "base.safetensors"
    save_file(source.state_dict(), str(base))
    model = load_dit(base, "cpu", torch.bfloat16, config=TINY, quantization="convrot_int8")
    adapter = LycorisAdapter(preset=MOSS_PRESET, algo="lora", rank=RANK, alpha=RANK)
    adapter.inject(model)
    for parameter in adapter.network.parameters():
        torch.nn.init.normal_(parameter, std=0.01)
    keys = set(adapter.state_dict())
    params = adapter.get_params()
    lora = _lora_file(tmp_path / "turbo.safetensors", source)
    linear = model.layers[0].proj.q
    x = torch.randn(1, 3, 64, dtype=torch.bfloat16)
    without = linear(x).float()

    with MossTurboLora(model, lora).swapped():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(torch.randn(1, 128, 2, 2), torch.tensor([500.]), torch.randn(1, 3, 32))
        stacked = linear(x).float()
        with adapter.disabled():
            turbo_only = linear(x).float()

    assert not torch.allclose(stacked, without)
    assert not torch.allclose(stacked, turbo_only)
    assert set(adapter.state_dict()) == keys
    assert adapter.get_params() == params
    assert torch.equal(linear(x).float(), without)


def test_family_builds_it_only_when_enabled(tmp_path):
    family = get_family("moss_image")
    model = _model()
    lora = _lora_file(tmp_path / "turbo.safetensors", model)
    assert family.preview_base(SimpleNamespace(), model) is None
    assert family.preview_base(SimpleNamespace(sample_turbo_lora=False, sample_turbo_lora_path=str(lora)), model) is None
    with pytest.raises(ValueError, match="sample_turbo_lora_path"):
        family.preview_base(SimpleNamespace(sample_turbo_lora=True, sample_turbo_lora_path=" "), model)
    turbo = family.preview_base(
        SimpleNamespace(sample_turbo_lora=True, sample_turbo_lora_path=str(lora), sample_turbo_lora_strength=0.5), model)
    assert isinstance(turbo, MossTurboLora)
    assert turbo.strength == 0.5


def test_form_swaps_the_sampling_fields_with_the_checkbox():
    from studio.domain.training import TrainingConfig

    def shown(field, **values):
        return eval_show_when(TrainingConfig.model_fields[field].json_schema_extra.get("show_when"), values)

    on = dict(model_family="moss_image", sample_turbo_lora=True, sample_base="train")
    off = dict(model_family="moss_image", sample_turbo_lora=False, sample_base="train")
    for field in ("sample_turbo_lora_path", "sample_turbo_lora_strength", "sample_turbo_steps", "sample_turbo_cfg"):
        assert shown(field, **on), field
        assert not shown(field, **off), field
    for field in ("sample_infer_steps", "sample_cfg_scale", "sample_sampler_name", "sample_scheduler"):
        assert not shown(field, **on), field
        assert shown(field, **off), field

    assert shown("sample_turbo_lora", **off)
    # The other families keep what they had
    anima = dict(model_family="anima", sample_turbo_lora=True, sample_base="train")
    assert not shown("sample_turbo_lora", **anima)
    assert not shown("sample_turbo_steps", **anima)
    assert shown("sample_infer_steps", **anima)
    assert shown("sample_sampler_name", **anima)
    krea_turbo = dict(model_family="krea2", sample_base="turbo", sample_turbo_lora=False)
    assert shown("sample_turbo_steps", **krea_turbo)
    assert not shown("sample_infer_steps", **krea_turbo)
    assert shown("sample_infer_steps", model_family="krea2", sample_base="train")


def test_generate_loads_this_key_layout_on_bf16_and_int8(tmp_path):
    from studio.services.inference.core import LoRASpec, apply_loras

    source = _model()
    # alpha == rank, like the real file
    lora = _lora_file(tmp_path / "turbo.safetensors", source, alpha=float(RANK))
    base = tmp_path / "base.safetensors"
    save_file(source.state_dict(), str(base))
    expected = _forward(_merged(source, lora, 1.0, alpha=float(RANK)))

    injected = load_dit(base, "cpu", torch.float32, config=TINY)
    handles = apply_loras(injected, [LoRASpec(str(lora))], "cpu", torch.float32, family_id="moss_image")
    assert handles
    torch.testing.assert_close(_forward(injected), expected, atol=1e-4, rtol=1e-3)

    quantized = load_dit(base, "cpu", torch.float32, config=TINY, quantization="convrot_int8")
    weight = quantized.layers[0].proj.q.weight.clone()
    assert apply_loras(quantized, [LoRASpec(str(lora))], "cpu", torch.float32, family_id="moss_image")
    assert not torch.equal(quantized.layers[0].proj.q.weight, weight)
