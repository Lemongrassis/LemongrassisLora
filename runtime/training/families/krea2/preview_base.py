"""Krea2 training previews on another base checkpoint (Turbo while training on Raw).

Raw and Turbo share one structure (same keys, same shapes), so a preview can
run on Turbo without a second model in memory: right before the preview the
frozen base weights are overwritten in place from the Turbo file, and right
after they are read back from the training file. LoRA parameters, the
optimizer and trained norms (``train_norm``) are never touched -- only frozen
tensors that come from the checkpoint.

Both files must hold every layer in the same storage form (both ConvRot INT8
with the same group size, both fp8_scaled, or both full precision); float
tensors may differ in file dtype, they are cast to the loaded dtype exactly
as the loader does. mxfp8 and plain int8_tensorwise files are not supported
(their scales are re-laid out at load time).

Safety: the training weights are fingerprinted before the first swap and
checked after every restore. A mismatch raises, stopping training rather than
letting it continue on wrong weights.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from modeling.krea2 import KREA2_CONFIG, Krea2Config

logger = logging.getLogger(__name__)

#: Bytes summed per step by the fingerprint (bounded temporary memory)
_CHUNK_WORDS = 1 << 24


class PreviewBaseUnavailable(RuntimeError):
    """The preview base could not be swapped in; the training weights are intact."""


def _storage_kinds(path: Path, n2s: dict[str, str]) -> dict[str, str]:
    """Normalized key -> storage kind (int8 / fp8 / float), from the header only."""
    kinds = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for normalized, source in n2s.items():
            dtype = handle.get_slice(source).get_dtype()
            kinds[normalized] = (
                "int8" if dtype == "I8" else "fp8" if dtype.startswith("F8") else "float"
            )
    return kinds


def check_compatible(train_path: str | Path, preview_path: str | Path, *,
                     config: Krea2Config = KREA2_CONFIG) -> None:
    """Raise ValueError unless the preview checkpoint can be swapped into the training model."""
    from training.families.krea2.loader import _expected_state, _inspect

    _, expected = _expected_state(config)
    infos = {}
    for label, path in (("training", train_path), ("preview", preview_path)):
        info, n2s, s2s = _inspect(path, config, expected_shapes=expected, allow_fp8=True)
        if info.mxfp8_layers or info.int8_tensorwise_layers:
            raise ValueError(
                f"the {label} base {Path(path).name} is mxfp8 / plain int8, which can't be "
                "swapped for previews; use ConvRot INT8, fp8_scaled or bf16 files"
            )
        infos[label] = (info, _storage_kinds(info.path, n2s), set(s2s))
    (train, train_kinds, train_scales), (preview, preview_kinds, preview_scales) = (
        infos["training"], infos["preview"],
    )
    differing = sorted(k for k in train_kinds if train_kinds[k] != preview_kinds[k])
    if differing or train_scales != preview_scales or train.convrot_groups != preview.convrot_groups:
        sample = ", ".join(differing[:3]) or "quantization scales"
        raise ValueError(
            f"the preview base {preview.path.name} stores weights differently from the training "
            f"base {train.path.name} ({sample}); pick the same format, e.g. Turbo INT8 ConvRot "
            "for a Raw INT8 ConvRot base"
        )


def _scale_buffer(module: torch.nn.Module) -> torch.Tensor:
    for name in ("scale_weight", "weight_scale"):
        buffer = getattr(module, name, None)
        if isinstance(buffer, torch.Tensor):
            return buffer
    raise ValueError(f"quantized layer {type(module).__name__} has no scale buffer")


def _checksum(tensor: torch.Tensor) -> int:
    """Sum of the tensor's bytes read as int32 words (int16 / int8 when the size needs it)."""
    flat = tensor.detach().reshape(-1).view(torch.uint8)
    for dtype, width in ((torch.int32, 4), (torch.int16, 2), (torch.int8, 1)):
        if flat.numel() % width == 0:
            words = flat.view(dtype)
            break
    total = 0
    for start in range(0, words.numel(), _CHUNK_WORDS):
        total += int(words[start:start + _CHUNK_WORDS].sum(dtype=torch.int64))
    return total


class Krea2PreviewBase:
    """Swaps the frozen base weights of a loaded Krea2 model for previews."""

    Unavailable = PreviewBaseUnavailable

    def __init__(self, model: torch.nn.Module, train_path: str | Path,
                 preview_path: str | Path, *, config: Krea2Config = KREA2_CONFIG) -> None:
        check_compatible(train_path, preview_path, config=config)
        self.config = config
        self.model = model
        self.train_path = Path(train_path)
        self.preview_path = Path(preview_path)
        self.active = False
        self._reference: list[int] | None = None

    # ------------------------------------------------------------------ helpers
    def _targets(self, path: Path) -> Iterator[tuple[torch.Tensor, str]]:
        """(destination tensor, source key) for every frozen tensor taken from ``path``."""
        from training.families.krea2.loader import _inspect

        _, n2s, s2s = _inspect(path, self.config, allow_fp8=True)
        params = dict(self.model.named_parameters())
        modules = dict(self.model.named_modules())
        for name, source in n2s.items():
            param = params[name]
            # trained tensors (train_norm) belong to the adapter, keep them
            if not param.requires_grad:
                yield param.data, source
        for layer, source in s2s.items():
            yield _scale_buffer(modules[layer]), source

    def _masters(self) -> None:
        """Point block-swapped layers at their CPU master copy, so writes land there."""
        from training.block_swap import find_block_swap

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        swap = find_block_swap(self.model)
        if swap is not None:
            swap.restore_masters()

    def _fingerprint(self) -> list[int]:
        self._masters()
        return [_checksum(tensor) for tensor, _ in self._targets(self.train_path)]

    def _write(self, path: Path) -> None:
        self._masters()
        with safe_open(str(path), framework="pt", device="cpu") as handle, torch.no_grad():
            for tensor, source in self._targets(path):
                tensor.copy_(handle.get_tensor(source).reshape(tensor.shape))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        from training.sysmem import trim_working_set

        trim_working_set()

    def _restore(self) -> None:
        started = time.monotonic()
        self._write(self.train_path)
        if self._fingerprint() != self._reference:
            raise RuntimeError(
                "Krea2 preview base: the training weights did not come back exactly after the "
                "preview; training stopped so it doesn't continue on wrong weights"
            )
        logger.info("preview base: training weights restored and verified in %.1fs",
                    time.monotonic() - started)

    # ------------------------------------------------------------------ public
    @contextmanager
    def swapped(self) -> Iterator[None]:
        """Run the body on the preview base; nested use is a no-op.

        Raises ``PreviewBaseUnavailable`` (training weights intact) when the
        swap-in fails, ``RuntimeError`` when the training weights can't be
        restored.
        """
        if self.active:
            yield
            return
        if self._reference is None:
            self._reference = self._fingerprint()
        started = time.monotonic()
        self.active = True
        try:
            try:
                self._write(self.preview_path)
            except Exception as exc:
                self._restore()
                raise PreviewBaseUnavailable(str(exc)) from exc
            logger.info("preview base: %s swapped in in %.1fs",
                        self.preview_path.name, time.monotonic() - started)
            try:
                yield
            finally:
                self._restore()
        finally:
            self.active = False
