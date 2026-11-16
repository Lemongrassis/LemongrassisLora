"""MOSS behavior adapter. The shared training loop remains rectified flow."""
from functools import lru_cache

from . import MOSS_SPEC
from .preset import MOSS_PRESET

#: VRAM left free beside the resident text encoder: prompt activations, allocator slack
_TEXT_HEADROOM_BYTES = 2 * 1024**3


@lru_cache(maxsize=1)
def _dit_param_counts():
    """(params per backbone layer, total params) of the DiT, from a meta-device build."""
    from accelerate import init_empty_weights
    from modeling.moss_image import model as moss_image_models
    with init_empty_weights():
        model = moss_image_models.MossImage(**moss_image_models.MOSS_IMAGE_CONFIG)
    per_layer = [sum(p.numel() for p in layer.parameters()) for layer in model.layers]
    return per_layer, sum(p.numel() for p in model.parameters())


@lru_cache(maxsize=4)
def _text_layer_bytes(path):
    """Resident BF16 bytes of each Qwen3.5 decoder layer and of everything else, read from the
    safetensors headers only (the vision tower / MTP / lm_head are dropped on load, as in the loader)."""
    import re
    from safetensors import safe_open
    from modeling.moss_image.weights import checkpoint_files
    layers, other = {}, 0
    for file in checkpoint_files(path):
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith(("visual.", "model.visual.", "mtp.", "lm_head.")):
                    continue
                numel = 1
                for dim in reader.get_slice(key).get_shape():
                    numel *= dim
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
                if match:
                    layers[int(match.group(1))] = layers.get(int(match.group(1)), 0) + 2 * numel
                else:
                    other += 2 * numel
    return [layers[i] for i in sorted(layers)], other


class MossFamily:
    spec = MOSS_SPEC

    def preview_base(self, args, model):
        """Previews through a distilled Turbo LoRA (``sample_turbo_lora``), else None."""
        if not getattr(args, "sample_turbo_lora", False):
            return None
        path = str(getattr(args, "sample_turbo_lora_path", "") or "").strip()
        if not path:
            raise ValueError("Previews are set to use a Turbo LoRA, but its file (sample_turbo_lora_path) is empty")
        from .turbo_lora import MossTurboLora
        return MossTurboLora(model, path, strength=getattr(args, "sample_turbo_lora_strength", 1.0))

    def swappable_blocks(self, *, checkpoint_path=""):
        from modeling.moss_image.model import MOSS_IMAGE_CONFIG
        return MOSS_IMAGE_CONFIG["num_layers"]

    def swapped_param_ratio(self, blocks_to_swap, *, checkpoint_path=""):
        """Share of the DiT's parameters in the trailing ``blocks_to_swap`` layers (VRAM budget discount)."""
        if blocks_to_swap <= 0:
            return 0.0
        per_layer, total = _dit_param_counts()
        return min(blocks_to_swap, len(per_layer)) * per_layer[0] / total if total else 0.0

    def generate_text_options(self, text_encoder_path):
        """Generate-time text encoder offload: the 9B Qwen3.5 is ~16GB in BF16, so on smaller cards
        the trailing layers stream from RAM while the prompts are encoded (the TE is released before
        the DiT loads, so only the encode itself pays for it)."""
        from training import sysmem
        layer_bytes, other_bytes = _text_layer_bytes(text_encoder_path)
        free = sysmem.gpu_free_bytes_global()
        if not layer_bytes or free is None:
            return {}
        budget = free - other_bytes - _TEXT_HEADROOM_BYTES
        resident, kept = 0, 0
        for size in layer_bytes:  # leading layers stay on the GPU, offload counts from the tail
            if resident + size > budget:
                break
            resident += size
            kept += 1
        return {"offload_layers": len(layer_bytes) - kept} if kept < len(layer_bytes) else {}

    def load_options(self, args):
        get = args.get if isinstance(args, dict) else lambda k, d: getattr(args, k, d)
        return {
            "dit": {"quantization": get("transformer_quantization", "none"),
                    "sample_flow_shift": get("moss_sample_shift", 5.0),
                    "sample_uncond_skip_layers": get("moss_uncond_skip_layers", [22])},
            "text": {"quantization": get("text_encoder_quantization", "none"),
                     "offload_layers": get("text_encoder_layers_to_offload", 0)},
        }

    def load_dit(self, path, device, dtype, *, attention_backend="none", repo_root=None,
                 purpose="train", blocks_to_swap=0, quantization="none",
                 sample_flow_shift=5.0, sample_uncond_skip_layers=(22,)):
        from .loader import load_dit
        model = load_dit(path, device, dtype, quantization=quantization,
                         blocks_to_swap=blocks_to_swap, attention_backend=attention_backend)
        model.sample_flow_shift = sample_flow_shift
        model.sample_uncond_skip_layers = tuple(sample_uncond_skip_layers)
        model.moss_comfy_forward = purpose == "generate"
        if model.moss_comfy_forward and model.is_convrot_int8:
            import torch
            from ..quant_int8 import patch_int8_linear
            for module in model.modules():
                if isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.int8:
                    patch_int8_linear(module, module.scale_weight, module._convrot_groupsize, forward="w8a8")
        return model

    def load_vae(self, path, device, dtype, *, tiling="auto"):
        from .vae import load_vae
        return load_vae(path, device, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *, t5_tokenizer_path="",
                  comfy_qwen=False, t5_fast=False, purpose="train", cache_enabled=True,
                  quantization="none", offload_layers=0, max_length=4096, tokenizer_path=""):
        from .text_encoding import MossTextStack
        return MossTextStack(text_encoder_path, device, quantization=quantization,
            offload_layers=offload_layers, max_length=max_length, tokenizer_path=tokenizer_path,
            cache_enabled=cache_enabled)

    def prepare_text_cache(self, captions, extra_prompts, *, cache_entries=(), cache_root=None,
                           text=None, device=None, dtype=None):
        text.prepare_text_cache(captions, extra_prompts, cache_entries=cache_entries, cache_root=cache_root)

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *, comfy_encoding=True, kv_trim=True):
        return text.encode_text_for_batch(captions, device=device, dtype=dtype)

    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint=False):
        if noisy.shape[0] > 1:
            import torch
            contexts = [ctx[mask.bool()] for ctx, mask in zip(cond.context, cond.attention_mask)]
            return torch.cat(dit.forward_packed(list(noisy.split(1)), t, contexts, use_checkpoint=use_checkpoint))
        from modeling.moss_image.vae import patchify_latents, unpatchify_latents
        if use_checkpoint:
            # Set once (logging on every step would obscure the training progress).
            if not dit.layers[0].gradient_checkpointing:
                dit.enable_gradient_checkpointing()
        else:
            if dit.layers[0].gradient_checkpointing:
                dit.disable_gradient_checkpointing()
        packed = patchify_latents(noisy[:, :, 0])
        pred = dit(packed, t.float() * 1000, cond.context, cond.attention_mask)
        return unpatchify_latents(pred).unsqueeze(2)

    def navit_encode_text(self, text, dit, captions, device, dtype, *, trim_padding=False, **kwargs):
        cond = self.encode_text_for_batch(text, dit, captions, device, dtype, **kwargs)
        return [ctx[mask.bool()] for ctx, mask in zip(cond.context, cond.attention_mask)]

    def navit_forward_and_loss(self, model, latents, t, contexts, loss_fn, **kwargs):
        from .navit import navit_packed_forward_and_loss
        return navit_packed_forward_and_loss(model, latents, t, contexts, loss_fn, **kwargs)

    def navit_text_token_count(self, text, caption):
        return text.token_count(caption)

    def sample_image(self, *args, **kwargs):
        from .sampling import sample_image
        return sample_image(*args, **kwargs)

    def lora_preset(self):
        return dict(MOSS_PRESET)

    def lora_preset_for_args(self, args):
        """Training: preset with the experimental targets (tone head / timestep modulation / image-only)."""
        from .experimental import adapt_preset
        return adapt_preset(self.lora_preset(), args)

    def lora_preset_for_state_dict(self, keys):
        """Inference: preset extended with the extra targets present in a LoRA file."""
        from .experimental import preset_for_state_dict
        return preset_for_state_dict(self.lora_preset(), keys)

    def adapter_reg_dims(self, args):
        from .experimental import extra_reg_dims
        return extra_reg_dims(args)

    def prepare_adapter_args(self, args):
        # Per-block norms go through the shared TrainNorm path (ComfyUI .diff export)
        if getattr(args, "moss_block_norms", False):
            args.train_norm = True

    def after_adapter_inject(self, model, injector, args):
        from .experimental import (
            BlockModulation,
            IMAGE_TOKENS_ONLY_KEY,
            init_down_from_weight_svd,
            init_down_gaussian,
            install_image_token_mask,
        )
        if getattr(args, IMAGE_TOKENS_ONLY_KEY, False):
            install_image_token_mask(model, injector.network)
            injector.metadata_extra[IMAGE_TOKENS_ONLY_KEY] = "true"
        if getattr(args, "moss_gaussian_init", False):
            init_down_gaussian(injector.network)
            injector.metadata_extra["moss_down_init"] = "gaussian"
        if getattr(args, "moss_svd_init", False):
            init_down_from_weight_svd(injector.network)
            injector.metadata_extra["moss_down_init"] = "weight_svd"
        if getattr(args, "moss_block_modulation", False):
            network = injector.network
            module = BlockModulation(
                len(model.layers), model.dim, int(getattr(args, "moss_block_modulation_rank", 4)),
                lambda: float(getattr(network, "multiplier", 1.0)),
                device=next(model.parameters()).device,
            )
            module.attach(model)
            injector.extras.append(module)

    def lora_metadata(self):
        return {"model_family": "moss_image", "ss_base_model_version": "moss_image",
                "modelspec.architecture": "moss_image/lora"}

    def convert_lora_state_dict(self, sd):
        return sd
