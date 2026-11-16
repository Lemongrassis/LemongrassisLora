"""Loss weighting schemes: min_snr / detail_inv_t / cosmap etc.

Extracted from the original runtime/anima_train.py L1898-1937 (ADR 0003 PR-A).

All schemes share the compute_loss_weight entry point; scheme-specific
parameters (e.g. detail_inv_t_min/max) are passed through as keyword args.
With only a few schemes so far this stays a single file with multiple if
branches; if the number of schemes or scheme-specific fields grows, migrate
to the plugin pattern (see runtime/training/losses/).
"""

from __future__ import annotations

import math

import torch

#: Floor of the BSMNTW bell, exp(-2 * 0.5^2): its value at t = 0 and t = 1.
_BSMNTW_FLOOR = math.exp(-0.5)


def _bsmntw_raw(t: torch.Tensor) -> torch.Tensor:
    """Unnormalized BSMNTW bell: exp(-2 (t - 0.5)^2) minus its endpoint value (0 at t = 0 and 1)."""
    return (torch.exp(-2.0 * (t - 0.5) ** 2) - _BSMNTW_FLOOR).clamp_min(0.0)


def _bsmntw_norm() -> float:
    """Mean of the raw bell over DiffSynth-Studio's Krea-2 training grid.

    DiffSynth builds 1000 training sigmas from t = linspace(1, 0, 1001)[:-1]
    shifted by exp(mu), mu = 0.8, and scales the bell to mean 1 over that
    grid (``FlowMatchScheduler.set_training_weight``). The same constant is
    used for every sampler here, so with uniform + timestep_schedule_shift
    exp(0.8) the weights equal DiffSynth's; other samplers keep the same curve.
    """
    s = math.exp(0.8)
    u = torch.linspace(1.0, 0.0, 1001, dtype=torch.float64)[:-1]
    sigmas = s / (s + (1.0 / u - 1.0))
    return float(_bsmntw_raw(sigmas).mean())


_BSMNTW_NORM = _bsmntw_norm()


def compute_loss_weight(
    t: torch.Tensor,
    scheme: str = "none",
    min_snr_gamma: float = 5.0,
    weight_cap_ratio: float = 0.0,
    detail_inv_t_min: float = 1.0,
    detail_inv_t_max: float = 5.0,
) -> torch.Tensor:
    """Return per-sample loss weights (B,). Under the Flow Matching CONST schedule: SNR(t) = ((1-t)/t)^2.

    scheme:
      none          -- all 1s, matches the original behavior
      min_snr       -- w = min(gamma/SNR, 1), downweights easy high-SNR steps (recommended baseline)
      detail_inv_t  -- w = 1/t clamped to [detail_inv_t_min, detail_inv_t_max],
                      mild detail boost, friendly to small batches + Prodigy; default [1,5]
      cosmap        -- SD3 cosmap weighting, more even across mid t (max/min ~= 1.81x)
      bsmntw        -- DiffSynth-Studio's training weight (ModelScope trainer): a bell
                      exp(-2 (t-0.5)^2) shifted to 0 at t=0 and t=1, peak ~1.69 at t=0.5,
                      mean 1 over DiffSynth's Krea-2 grid; weight_cap_ratio does not apply

    weight_cap_ratio -- cap on the batch's max/min ratio (0=disabled), prevents a single sample
                        from dominating and breaking Prodigy's d estimate
    detail_inv_t_min/max -- lower/upper bound of the detail_inv_t weighting curve.
                           Default [1, 5]: w=1 at t=1, w=5 (clamped) at t=0.2.
                           Lowering max (e.g. 3) -> steadier but hazier/less saturated look;
                           raising max (e.g. 8) -> more aggressive detail learning but Prodigy's
                           d estimate is more easily dominated by a single sample.
    """
    scheme = (scheme or "none").lower()
    if scheme == "none":
        return torch.ones_like(t)
    if scheme == "bsmntw":
        # Endpoint weights are 0 by design, so a max/min cap would zero the batch: skip it
        return (_bsmntw_raw(t.float()) / _BSMNTW_NORM).to(dtype=t.dtype)

    eps = 1e-4
    t_c = t.clamp(eps, 1 - eps)

    if scheme == "min_snr":
        snr = ((1 - t_c) / t_c) ** 2
        w = torch.minimum(torch.tensor(float(min_snr_gamma), device=t.device) / snr, torch.ones_like(t_c))
    elif scheme == "detail_inv_t":
        # min/max are already validated as min <= max by TrainingConfig.model_validator, used as-is here
        w = (1.0 / t_c).clamp(min=float(detail_inv_t_min), max=float(detail_inv_t_max))
    elif scheme == "cosmap":
        bot = (1 - 2 * t_c + 2 * t_c ** 2).clamp(min=eps)
        w = 2.0 / (math.pi * bot)
    else:
        return torch.ones_like(t)

    if weight_cap_ratio and weight_cap_ratio > 1.0:
        w_min = w.min().clamp(min=eps)
        w = w.clamp(max=w_min * float(weight_cap_ratio))

    return w
