"""SinkGD: gradient descent on a Sinkhorn-normalized gradient (arXiv 2502.06742, "SR-Sinkhorn").

For a 2-D gradient G of shape m x n, repeat L times: rescale each row to l2 norm sqrt(n),
then each column to l2 norm sqrt(m). The update is W <- W - lr * X. The result has about unit
RMS per element, so every layer moves by the same amount whatever its gradient scale is, and
the LR is on Adam's scale. There is no momentum and no per-parameter state.

Non-matrix parameters follow the same convention as the BluFlow trainer: 1-D is rescaled to
RMS 1, 0-D becomes its sign, and >2-D is flattened to (shape[0], -1) and restored. Zero rows
and columns stay zero: a zero-initialized lora_up gives lora_down a zero gradient on step 1.
"""

from __future__ import annotations

import logging
import math

import torch


logger = logging.getLogger(__name__)

_EPS = 1e-30


def sr_sinkhorn(grad: torch.Tensor, iterations: int = 5, eps: float = _EPS) -> torch.Tensor:
    """The SR-Sinkhorn normalization of ``grad``, in float32 or wider."""
    x = grad.float() if grad.dtype in (torch.bfloat16, torch.float16) else grad
    if x.numel() == 0:
        return x.clone()
    if x.ndim == 0:
        return x.sign()
    if x.ndim == 1:
        return x * (math.sqrt(x.numel()) / x.norm().clamp_min(eps))
    shape = x.shape
    x = x.reshape(shape[0], -1)
    m, n = x.shape
    for _ in range(iterations):
        x = x * (math.sqrt(n) / x.norm(dim=1, keepdim=True).clamp_min(eps))
        x = x * (math.sqrt(m) / x.norm(dim=0, keepdim=True).clamp_min(eps))
    return x.reshape(shape)


_SIGNIFICAND_BITS = {torch.bfloat16: 8, torch.float16: 11}


def stochastic_round(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round to ``dtype``, going up with a probability equal to the fractional part.

    The expected result equals ``value``, so an update smaller than the dtype's rounding step
    still counts on average instead of being lost every time.
    """
    _, exponent = torch.frexp(value)
    step = torch.ldexp(torch.ones_like(value), exponent - _SIGNIFICAND_BITS[dtype])
    scaled = value / step
    low = scaled.floor()
    return ((low + (torch.rand_like(value) < scaled - low)) * step).to(dtype)


class SinkGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0.0, sinkhorn_iterations: int = 5,
                 stochastic_rounding: bool = False):
        if not (math.isfinite(lr) and lr >= 0.0):
            raise ValueError(f"SinkGD: invalid lr {lr}")
        if not (math.isfinite(weight_decay) and weight_decay >= 0.0):
            raise ValueError(f"SinkGD: invalid weight_decay {weight_decay}")
        if int(sinkhorn_iterations) < 1:
            raise ValueError(f"SinkGD: sinkhorn_iterations must be positive, got {sinkhorn_iterations}")
        super().__init__(params, {
            "lr": lr, "weight_decay": weight_decay, "sinkhorn_iterations": int(sinkhorn_iterations),
            "stochastic_rounding": bool(stochastic_rounding),
        })

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, decay = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SinkGD does not support sparse gradients")
                update = sr_sinkhorn(p.grad, group["sinkhorn_iterations"])
                if group.get("stochastic_rounding") and p.dtype in _SIGNIFICAND_BITS:
                    target = p.float() * (1.0 - lr * decay) - lr * update.float()
                    p.copy_(stochastic_round(target, p.dtype))
                    continue
                if decay:
                    p.mul_(1.0 - lr * decay)
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss


def validate(args) -> None:
    if not getattr(args, "lora_fp32", False) and not getattr(args, "use_stochastic_rounding", False):
        logger.warning(
            "SinkGD with BF16 LoRA weights: an update of about lr per element is below BF16's rounding "
            "step on lora_down, so those updates are lost. Turn on use_stochastic_rounding or lora_fp32."
        )


def build(args, params, lr: float, weight_decay: float):
    from training.optimizers.extra_args import with_optimizer_args

    kwargs = with_optimizer_args(args, {
        "sinkhorn_iterations": int(getattr(args, "sinkgd_iterations", 5) or 5),
        "stochastic_rounding": bool(getattr(args, "use_stochastic_rounding", False)),
    })
    optimizer = SinkGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    logger.info("SinkGD (lr=%s, weight_decay=%s, sinkhorn_iterations=%s, stochastic_rounding=%s)", lr, weight_decay,
                kwargs["sinkhorn_iterations"], kwargs["stochastic_rounding"])
    return optimizer
