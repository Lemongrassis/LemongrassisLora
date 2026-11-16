"""DoRA epilogue computed the way ComfyUI applies it at generation time.

LyCORIS rescales the merged weight in the dtype of ``dora_scale`` (FP32 with
FP32 masters) and its fused kernels accumulate in FP32. ComfyUI's
``weight_decompose`` (comfy/weight_adapter/base.py) works in the model's
weight dtype when it patches on the fly (lowvram / dynamic VRAM), which is BF16
for Anima:

- ``W' = W + ΔW`` added in BF16;
- output axis: the norm is taken of the *base* weight ``W``, not ``W'``;
  input axis: the norm is taken of ``W'`` per input column;
- ``norm + finfo(bf16).eps`` (0.0078 — about 1% of a typical column norm);
- ``W' *= (m / norm).to(bf16)``;
- strength ``s != 1``: ``W + s·(W' − W)``.

``install_comfy_dora`` swaps that formula into every DoRA module of a LyCORIS
network so training sees the same weight ComfyUI will build from the file.
"""
from __future__ import annotations

import types
from typing import Any

import torch

DORA_COMPUTE_MODES = ("lycoris", "comfyui_bf16")


def comfy_weight_decompose(
    merged: torch.Tensor,
    base: torch.Tensor,
    dora_scale: torch.Tensor,
    multiplier: float = 1.0,
    *,
    wd_on_out: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """ComfyUI's DoRA rescale of ``merged = base + ΔW`` in ``dtype``."""
    base = base.to(dtype)
    calc = merged.to(dtype)
    if wd_on_out:
        norm = base.reshape(base.shape[0], -1).norm(dim=1, keepdim=True)
        norm = norm.reshape(base.shape[0], *[1] * (base.dim() - 1))
    else:
        norm = (
            calc.transpose(0, 1)
            .reshape(calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(calc.shape[1], *[1] * (calc.dim() - 1))
            .transpose(0, 1)
        )
    norm = norm + torch.finfo(dtype).eps
    calc = calc * (dora_scale.to(device=calc.device, dtype=dtype) / norm).to(dtype)
    if multiplier != 1:
        calc = base + multiplier * (calc - base)
    return calc


def comfy_dora_init(weight: torch.Tensor, *, wd_on_out: bool, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Magnitude that makes the ComfyUI rescale an identity at step 0 (norm + eps, in ``dtype``)."""
    w = weight.detach().to(dtype)
    if wd_on_out:
        norm = w.reshape(w.shape[0], -1).norm(dim=1, keepdim=True)
        norm = norm.reshape(w.shape[0], *[1] * (w.dim() - 1))
    else:
        norm = (
            w.transpose(0, 1)
            .reshape(w.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(w.shape[1], *[1] * (w.dim() - 1))
            .transpose(0, 1)
        )
    return norm.float() + torch.finfo(dtype).eps


def _apply_comfy(self, weight, multiplier=1):
    return comfy_weight_decompose(
        weight, self._current_weight(), self.dora_scale, multiplier,
        wd_on_out=bool(getattr(self, "wd_on_out", True)),
    )


def install_comfy_dora(network: Any, *, reinit: bool) -> int:
    """Route every DoRA module of ``network`` through the ComfyUI formula.

    ``reinit`` re-seeds ``dora_scale`` so the step-0 output equals the base
    model (fresh training only; a resumed or loaded file keeps its values).
    Returns the number of modules switched.
    """
    count = 0
    for lora in getattr(network, "loras", []):
        if not getattr(lora, "wd", False) or not hasattr(lora, "dora_scale"):
            continue
        lora.apply_weight_decompose = types.MethodType(_apply_comfy, lora)
        if reinit:
            with torch.no_grad():
                init = comfy_dora_init(lora._current_weight(), wd_on_out=bool(lora.wd_on_out))
                lora.dora_scale.copy_(init.to(device=lora.dora_scale.device, dtype=lora.dora_scale.dtype))
        count += 1
    return count
