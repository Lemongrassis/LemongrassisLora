"""MOSS training previews through a distilled "Turbo" LoRA (e.g. MOSS-Image-Base-8-steps-dmd).

The LoRA is applied only while a preview runs, as a forward hook on each target Linear:
``y = W x + strength * (alpha / rank) * up(down(x))``. Nothing is merged, for two reasons:
the base weights stay untouched (no restore step that could leave training on wrong weights),
and that distillation delta is smaller than BF16's rounding step, so merging it would lose it.

The hook runs on top of whatever the module's forward does, so the LoRA being trained
(LyCORIS replaces ``forward``) and quantized forwards keep working underneath it. The factors
are read from disk for each preview and freed after it: nothing stays in RAM or VRAM while training.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

_DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")
_KOHYA_PREFIX = "lora_unet_"
_KEY_RE = re.compile(r"^(?P<module>.+)\.(?P<part>lora_down\.weight|lora_up\.weight|lora_A\.weight|lora_B\.weight|alpha)$")
_PART_NAMES = {"lora_down.weight": "down", "lora_A.weight": "down",
               "lora_up.weight": "up", "lora_B.weight": "up", "alpha": "alpha"}


class TurboLoraUnavailable(RuntimeError):
    """The LoRA could not be loaded for this preview; the model is untouched."""


class _Target(NamedTuple):
    module: str
    down_key: str
    up_key: str
    alpha_key: str | None
    rank: int


def _module_name(raw: str, kohya_names: dict[str, str]) -> str | None:
    if raw.startswith(_KOHYA_PREFIX):
        return kohya_names.get(raw[len(_KOHYA_PREFIX):])
    for prefix in _DIT_PREFIXES:
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def inspect_turbo_lora(path: str | Path, model: torch.nn.Module) -> list[_Target]:
    """Match the file against the model from its header only; ValueError when any part doesn't fit."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Turbo LoRA file not found: {path}")
    linears = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    kohya_names = {name.replace(".", "_"): name for name in linears}
    grouped: dict[str, dict[str, str]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        shapes = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
    for key in shapes:
        match = _KEY_RE.match(key)
        name = _module_name(match["module"], kohya_names) if match else None
        if name not in linears:
            raise ValueError(f"{path.name} doesn't fit MOSS-Image: nothing to apply {key} to")
        grouped.setdefault(name, {})[_PART_NAMES[match["part"]]] = key
    if not grouped:
        raise ValueError(f"{path.name} has no LoRA weights")
    targets = []
    for name, parts in grouped.items():
        if "down" not in parts or "up" not in parts:
            raise ValueError(f"{path.name}: layer {name} is missing a LoRA factor")
        module, down, up = linears[name], shapes[parts["down"]], shapes[parts["up"]]
        if len(down) != 2 or len(up) != 2 or down[0] != up[1] \
                or down[1] != module.in_features or up[0] != module.out_features:
            raise ValueError(
                f"{path.name} doesn't fit MOSS-Image: layer {name} is "
                f"{module.in_features}->{module.out_features}, the LoRA has {down} / {up}"
            )
        targets.append(_Target(name, parts["down"], parts["up"], parts.get("alpha"), int(down[0])))
    return targets


class MossTurboLora:
    """Preview base for ``training.sample_runner``: a LoRA hooked in around each preview."""

    Unavailable = TurboLoraUnavailable
    #: What the LoRA is distilled for; the steps and CFG stay in sample_turbo_steps / sample_turbo_cfg
    sample_kwargs = {"sampler_name": "euler", "scheduler": "simple"}

    def __init__(self, model: torch.nn.Module, path: str | Path, *, strength: float = 1.0) -> None:
        self.model = model
        self.preview_path = Path(path)
        self.strength = float(strength)
        self.active = False
        self._targets = inspect_turbo_lora(self.preview_path, model)

    def _device(self) -> torch.device:
        """Where previews compute: block-swapped layers wait in RAM, so skip CPU parameters."""
        return next((p.device for p in self.model.parameters() if p.device.type != "cpu"),
                    torch.device("cpu"))

    def _load_hooks(self) -> list[tuple[torch.nn.Module, object]]:
        device = self._device()
        modules = dict(self.model.named_modules())
        out = []
        with safe_open(str(self.preview_path), framework="pt", device="cpu") as handle:
            for target in self._targets:
                alpha = float(handle.get_tensor(target.alpha_key)) if target.alpha_key else target.rank
                out.append((modules[target.module], _make_hook(
                    handle.get_tensor(target.down_key).to(device=device, dtype=torch.float32),
                    handle.get_tensor(target.up_key).to(device=device, dtype=torch.float32),
                    self.strength * alpha / target.rank,
                )))
        return out

    @contextmanager
    def swapped(self) -> Iterator[None]:
        """Run the body with the LoRA applied; nested use is a no-op.

        Raises ``TurboLoraUnavailable`` when it can't be loaded (a missing file, no VRAM left).
        """
        if self.active:
            yield
            return
        started = time.monotonic()
        self.active = True
        handles = []
        try:
            try:
                hooks = self._load_hooks()
            except Exception as exc:
                raise TurboLoraUnavailable(str(exc)) from exc
            handles = [module.register_forward_hook(hook) for module, hook in hooks]
            del hooks
            logger.info("Turbo LoRA: %s applied to %d layers in %.1fs",
                        self.preview_path.name, len(handles), time.monotonic() - started)
            yield
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            self.active = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from training.sysmem import trim_working_set

            trim_working_set()


def _make_hook(down: torch.Tensor, up: torch.Tensor, scale: float):
    def hook(_module, args, output):
        # In FP32 like the file: autocast would round the factors to BF16 on every call
        with torch.autocast(output.device.type, enabled=False):
            delta = args[0].float() @ down.T @ up.T
        return output + (delta * scale).to(output.dtype)

    return hook
