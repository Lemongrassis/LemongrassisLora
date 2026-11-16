"""Low-noise quota: a fixed share of training samples is drawn from the low-noise range.

Distributions that follow a model's pre-training (MOSS: about 2% of samples below t = 0.3)
leave the low-noise steps — rendering, texture, fine contrast — almost without signal. A LoRA
is one set of weights for every noise level, so what it learns at high noise is applied
unchanged at the low-noise steps it never saw.

This wraps any timestep sampler: every ``every``-th sample is drawn uniformly from
(0, ``max_t``], the others come from the wrapped sampler untouched. With batch 1 and gradient
accumulation N, ``every = N`` gives exactly one low-noise sample per optimizer step.
"""

from __future__ import annotations

import torch

_MIN_T = 0.001


class LowNoiseQuotaSampler:
    def __init__(self, inner, every: int = 4, max_t: float = 0.4):
        if int(every) < 2:
            raise ValueError("low_noise_quota_every must be at least 2")
        if not _MIN_T < float(max_t) <= 1.0:
            raise ValueError("low_noise_quota_max_t must be in (0.001, 1]")
        self._inner = inner
        self.every = int(every)
        self.max_t = float(max_t)
        self._seen = 0

    def __getattr__(self, name):
        # record / maybe_refresh / state_dict / capability flags belong to the wrapped sampler
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def sample(self, bs: int, device, *, token_counts=None) -> torch.Tensor:
        if token_counts is not None:
            t = self._inner.sample(bs, device, token_counts=token_counts)
        else:
            t = self._inner.sample(bs, device)
        index = self._seen + torch.arange(bs)
        self._seen += bs
        quota = (index % self.every == self.every - 1).to(t.device)
        if bool(quota.any()):
            low = torch.rand(bs, device=t.device, dtype=t.dtype) * (self.max_t - _MIN_T) + _MIN_T
            t = torch.where(quota, low, t)
        return t

    def status(self) -> dict:
        status = dict(self._inner.status() or {})
        status["low_noise_quota"] = {"every": self.every, "max_t": self.max_t}
        return status


def wrap(sampler, args):
    """Wrap ``sampler`` when args.low_noise_quota is set, otherwise return it unchanged."""
    if not getattr(args, "low_noise_quota", False):
        return sampler
    return LowNoiseQuotaSampler(
        sampler,
        int(getattr(args, "low_noise_quota_every", 4) or 4),
        float(getattr(args, "low_noise_quota_max_t", 0.4) or 0.4),
    )
