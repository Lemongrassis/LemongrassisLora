"""LemongrassisLora card: a short, pretty summary written into every saved LoRA file.

Only the safetensors header gets it (config.yaml and config snapshots stay as they
are). Readers that rebuild the adapter look at ``ss_network_*`` only, so these keys
are purely informational: LoRA info panels (ComfyUI, SwarmUI, Forge, Civitai) show
``modelspec.title`` / ``modelspec.description`` / ``ss_training_comment``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

APP_NAME = "LemongrassisLora"
# bold_cursive Unicode letters, made with the cute-text skill's fontify.py
_TITLE = "𝓛𝓮𝓶𝓸𝓷𝓰𝓻𝓪𝓼𝓼𝓲𝓼𝓛𝓸𝓻𝓪"
_MARK = "✿"


def _fmt_num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def base_model_name(family_id: str, transformer_path: str | None, model: Any = None) -> str:
    """Human-readable base model name: "MOSS Image Base", "Krea2 Raw", "Anima 2.9B"..."""
    path = str(transformer_path or "")
    lowered = path.replace("\\", "/").lower()
    # A folder checkpoint (MOSS shards) is named by its folder, a file by its name.
    name = Path(path).parent.name.lower() if "-of-" in Path(path).name else Path(path).name.lower()
    hint = f"{name} {lowered}"
    if family_id == "moss_image":
        if "heal" in hint:
            return "MOSS Image Heal"
        if "turbo" in hint:
            return "MOSS Image Turbo"
        return "MOSS Image Base"
    if family_id == "krea2":
        return "Krea2 Turbo" if "turbo" in name else "Krea2 Raw"
    if family_id == "anima":
        blocks = getattr(model, "blocks", None)
        if (blocks is not None and len(blocks) == 40) or "2.9b" in hint:
            return "Anima 2.9B"
        for tag, label in (("preview3", "Anima Preview 3"), ("preview2", "Anima Preview 2"),
                           ("preview", "Anima Preview")):
            if tag in name:
                return label
        return "Anima"
    return family_id.replace("_", " ").title() if family_id else "unknown"


def network_label(args: Any, injector: Any) -> str:
    if getattr(args, "use_riemannian", False):
        label = "lora · riemannion"
    else:
        lora_type = str(getattr(args, "lora_type", "lora") or "lora")
        label = {"ortho": "lora · ortho", "tlora": "t-lora"}.get(lora_type, lora_type)
        if lora_type == "lokr":
            factor = getattr(injector, "factor", None)
            if factor is None:
                factor = getattr(args, "lokr_factor", None)
            if factor is not None:
                label += f" · factor {_fmt_num(factor)}"
    if getattr(args, "lora_dora", False):
        label += " · dora"
    return label


def build_card(args: Any, injector: Any, family_id: str, model: Any = None) -> dict[str, str]:
    """Metadata keys merged into the safetensors header by every adapter's save()."""
    project = str(getattr(args, "output_name", "") or "lora")
    base = base_model_name(family_id, getattr(args, "transformer_path", None), model)
    network = network_label(args, injector)
    rank = getattr(injector, "rank", None) or getattr(args, "lora_rank", None)
    alpha = getattr(injector, "alpha", None) or getattr(args, "lora_alpha", None)
    if getattr(args, "use_riemannian", False):
        alpha = rank  # the Riemann export bakes alpha == rank
    rank_dim = f"{_fmt_num(rank)}/{_fmt_num(alpha)}"
    batch = _fmt_num(getattr(args, "batch_size", 1))
    timesteps = str(getattr(args, "timestep_sampling", "") or "default")

    rows = [
        ("project", project),
        ("base model", base),
        ("network", network),
        ("rank / alpha", rank_dim.replace("/", " / ")),
        ("batch size", batch),
        ("timesteps", timesteps),
    ]
    card = "\n".join([
        f"{_MARK} ─── {_TITLE} ─── {_MARK}",
        "",
        *(f"  {_MARK} {key} · {value}" for key, value in rows),
        "",
        f"        ─── {_MARK} ───",
    ])
    return {
        "lemongrassis.card": card,
        "lemongrassis.trainer": APP_NAME,
        "lemongrassis.project": project,
        "lemongrassis.base_model": base,
        "lemongrassis.network": network,
        "lemongrassis.rank_dim": rank_dim,
        "lemongrassis.batch_size": batch,
        "lemongrassis.timestep_sampling": timesteps,
        "modelspec.title": project,
        "modelspec.description": card,
        "ss_training_comment": card,
    }
