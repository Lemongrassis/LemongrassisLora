"""ReREX: linear warmup, then a global REX curve walked in segments that each have their own REX drop.

With R(p; d) = (1 - p) / ((1 - d) + d * (1 - p)):
    g(s) = floor + (1 - floor) * R(s / D; global_d), D = steps after warmup
Segment i of n has a length proportional to (n - i) ** weight_power and falls from g(start) to
g(end) along R(p; local_d), so the curve is continuous and monotone. Early segments are longer.

Same formula as the BluFlow trainer's ``rerex`` schedule; see tests/test_sinkgd_rerex.py for a
check against a learning rate recorded in a LoRA it trained.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional


logger = logging.getLogger(__name__)


def rex_shape(progress: float, d: float) -> float:
    remaining = 1.0 - min(1.0, max(0.0, progress))
    return remaining / ((1.0 - d) + d * remaining)


def rerex_lambda(total_steps: int, *, warmup_steps: int = 0, floor: float = 0.0, global_d: float = 0.6,
                 local_d: float = 0.85, weight_power: float = 1.5, num_segments: int = 8) -> Callable[[int], float]:
    warmup = max(0, int(warmup_steps))
    decay_steps = max(1, int(total_steps) - warmup)
    n = max(1, int(num_segments))
    weights = [(n - i) ** float(weight_power) for i in range(n)]
    total_weight = sum(weights)
    lengths = [int(decay_steps * w / total_weight) for w in weights]
    lengths[-1] += decay_steps - sum(lengths)
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)

    def global_rex(step: float) -> float:
        return floor + (1.0 - floor) * rex_shape(step / decay_steps, global_d)

    ends = [(global_rex(bounds[i]), global_rex(bounds[i + 1])) for i in range(n)]

    def decay(step: int) -> float:
        for i in range(n):
            if step < bounds[i + 1]:
                span = bounds[i + 1] - bounds[i]
                start_lr, end_lr = ends[i]
                return end_lr + (start_lr - end_lr) * rex_shape((step - bounds[i]) / span, local_d)
        return floor

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return decay(step - warmup)

    return lr_lambda


def build(args, optimizer, total_steps: Optional[int]):
    if total_steps is None or int(total_steps) <= 0:
        logger.warning("rerex scheduler needs a known total_steps, falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    warmup_steps = max(0, min(int(getattr(args, "lr_scheduler_warmup_steps", 0) or 0), total_steps))
    eta_min = max(0.0, float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0))
    shape = {
        "global_d": float(getattr(args, "lr_scheduler_rerex_global_d", 0.6)),
        "local_d": float(getattr(args, "lr_scheduler_rerex_local_d", 0.85)),
        "weight_power": float(getattr(args, "lr_scheduler_rerex_weight_power", 1.5)),
        "num_segments": int(getattr(args, "lr_scheduler_rerex_num_segments", 8)),
    }
    if not (0.0 < shape["global_d"] <= 1.0 and 0.0 < shape["local_d"] <= 1.0):
        raise ValueError("rerex: global_d and local_d must be in (0, 1]")
    if shape["num_segments"] < 1:
        raise ValueError("rerex: num_segments must be positive")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
        rerex_lambda(total_steps, warmup_steps=warmup_steps, **shape,
                     floor=min(1.0, eta_min / group["lr"]) if group["lr"] > 0 else 0.0)
        for group in optimizer.param_groups
    ])
    logger.info("LR schedule: rerex (total_steps=%s, warmup_steps=%s, eta_min=%s, %s)",
                total_steps, warmup_steps, eta_min, shape)
    return scheduler
