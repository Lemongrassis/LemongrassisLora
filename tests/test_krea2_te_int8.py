"""Krea2 Qwen3-VL text encoder in the Comfy-Org ConvRot INT8 single-file form."""
from __future__ import annotations

import json

import torch

from training.families.quant_int8 import requantize_convrot

GROUP = 16


def _tiny_qwen3vl_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            vocab_size=128, tie_word_embeddings=True,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=4,
            out_hidden_size=32, patch_size=4, temporal_patch_size=1,
            spatial_merge_size=1,
        ),
    )


def _blob() -> torch.Tensor:
    raw = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()
    return torch.tensor(list(raw), dtype=torch.uint8)


def _write_int8_te(dir_path, config):
    """Same layout as Comfy-Org/Krea-2 text_encoders/qwen3vl_4b_int8_convrot.safetensors:
    HF ``model.language_model.*`` keys, int8 text linears + embed_tokens, bf16 vision."""
    from safetensors.torch import save_file
    from transformers import Qwen3VLForConditionalGeneration

    torch.manual_seed(11)
    ref = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)
    out: dict[str, torch.Tensor] = {}
    int8_layers: list[str] = []
    for key, value in ref.state_dict().items():
        if key == "lm_head.weight":
            continue
        quantize = key.startswith("model.language_model.") and key.endswith(".weight") and value.ndim == 2
        if quantize:
            layer = key[: -len(".weight")]
            q, scale = requantize_convrot(value.float(), GROUP, seed=0)
            out[key] = q.contiguous()
            out[f"{layer}.weight_scale"] = scale.contiguous()
            out[f"{layer}.comfy_quant"] = _blob()
            int8_layers.append(layer)
        else:
            out[key] = value.detach().clone().contiguous()
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dir_path / "qwen3vl_4b_int8_convrot.safetensors"))
    return ref, int8_layers


def test_int8_convrot_te_loads_and_matches_reference(tmp_path, monkeypatch):
    import transformers

    import training.families.krea2.text_encoding as te_mod

    config = _tiny_qwen3vl_config()
    te_dir = tmp_path / "qwen3vl-int8"
    ref, int8_layers = _write_int8_te(te_dir, config)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", classmethod(lambda cls, *a, **k: config),
    )

    model = te_mod._default_model_loader(te_dir, torch.device("cpu"), torch.bfloat16)
    te_mod.patch_manual_cast(model, torch.float32)

    modules = dict(model.named_modules())
    assert int8_layers and all(modules[n].weight.dtype == torch.int8 for n in int8_layers)
    assert "model.language_model.embed_tokens" in int8_layers
    assert model.model.visual.blocks[0].attn.proj.weight.dtype == torch.bfloat16

    ids = torch.randint(0, 128, (1, 9))
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        got = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                          use_cache=False, return_dict=True).hidden_states[-1].float()
        want = ref.float().model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                                 use_cache=False, return_dict=True).hidden_states[-1]
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    assert cos > 0.99, cos


def test_int8_te_gets_its_own_cache_fingerprint(tmp_path):
    from training.families.krea2.text_encoding import _single_file_is_int8

    config = _tiny_qwen3vl_config()
    te_dir = tmp_path / "qwen3vl-int8"
    _write_int8_te(te_dir, config)
    assert _single_file_is_int8(te_dir / "qwen3vl_4b_int8_convrot.safetensors")
