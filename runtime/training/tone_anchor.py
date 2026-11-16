"""Tone anchor: do not learn the overall tone from high-noise steps.

Near pure noise the best MSE answer is the dataset average, and its per-channel mean is the
average brightness and colour of the whole set. A LoRA trained mostly at high noise picks
that mean up as a global shift (a pale veil when the set has many light images) and carries
it to every sampling step.

For samples with t >= ``min_t`` the per-channel spatial mean of the error (pred - target) is
removed before the loss, so those steps still teach shapes and composition but give no
gradient on the overall level of a channel. Tone is then learned only from the steps below
``min_t``, where the target is a specific image rather than an average.

The mean is left unsupervised there, not pinned to the base model (that would need a second
forward pass). Latent-space approximation of "tone".
"""

from __future__ import annotations

from typing import Optional

import torch

_EPS = 1e-6


def remove_error_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    *,
    min_t: float,
    strength: float = 1.0,
    spatial_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``pred`` with the mean error removed on high-noise samples.

    pred / target: (B, C, ...spatial); t: (B,); spatial_mask broadcastable to (B, 1, ...spatial).
    Samples with t < min_t come back unchanged.
    """
    active = t.float().view(-1) >= float(min_t)
    if strength <= 0 or not bool(active.any()):
        return pred
    dims = tuple(range(2, pred.dim()))
    error = pred.float() - target.float()
    if spatial_mask is not None:
        w = spatial_mask.float().expand(pred.shape[0], 1, *pred.shape[2:])
        mean = (error * w).sum(dims, keepdim=True) / w.sum(dims, keepdim=True).clamp_min(_EPS)
    else:
        mean = error.mean(dims, keepdim=True)
    scale = active.float().view(-1, *([1] * (pred.dim() - 1))) * float(strength)
    return pred.float() - mean * scale
