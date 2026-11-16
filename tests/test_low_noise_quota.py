from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training.timestep_samplers import build_timestep_sampler
from training.timestep_samplers.low_noise_quota import LowNoiseQuotaSampler
from training.timestep_samplers.moss import MossTruncatedTimestepSampler


class _Fixed:
    """Inner sampler that always answers 0.9 and remembers what it was asked."""

    requires_token_counts = False

    def __init__(self):
        self.recorded = []
        self.token_counts = "unset"

    def sample(self, bs, device, *, token_counts=None):
        self.token_counts = token_counts
        return torch.full((bs,), 0.9, device=device)

    def record(self, t, raw_mse):
        self.recorded.append((t, raw_mse))

    def status(self):
        return {"kind": "fixed"}


def test_every_nth_sample_is_low_noise_with_batch_one():
    sampler = LowNoiseQuotaSampler(_Fixed(), every=4, max_t=0.4)
    t = torch.cat([sampler.sample(1, "cpu") for _ in range(12)])
    quota = torch.arange(12) % 4 == 3
    assert (t[quota] <= 0.4).all() and (t[quota] > 0).all()
    assert (t[~quota] == 0.9).all()


def test_quota_counts_samples_across_batches():
    sampler = LowNoiseQuotaSampler(_Fixed(), every=4, max_t=0.4)
    t = torch.cat([sampler.sample(3, "cpu") for _ in range(4)])
    assert ((t <= 0.4) == (torch.arange(12) % 4 == 3)).all()


def test_wrapper_passes_through_hooks_flags_and_status():
    inner = _Fixed()
    sampler = LowNoiseQuotaSampler(inner, every=2, max_t=0.3)
    assert sampler.requires_token_counts is False
    sampler.record(torch.tensor([0.5]), torch.tensor([1.0]))
    assert len(inner.recorded) == 1
    sampler.sample(2, "cpu", token_counts=torch.tensor([64, 64]))
    assert inner.token_counts is not None
    assert sampler.status() == {"kind": "fixed", "low_noise_quota": {"every": 2, "max_t": 0.3}}


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        LowNoiseQuotaSampler(_Fixed(), every=1)
    with pytest.raises(ValueError):
        LowNoiseQuotaSampler(_Fixed(), max_t=0.0)


def test_build_wraps_only_when_enabled():
    base = dict(timestep_sampling="moss_truncated_logit_normal")
    plain = build_timestep_sampler(SimpleNamespace(**base), 100)
    assert isinstance(plain, MossTruncatedTimestepSampler)
    wrapped = build_timestep_sampler(
        SimpleNamespace(**base, low_noise_quota=True, low_noise_quota_every=4, low_noise_quota_max_t=0.4), 100)
    assert isinstance(wrapped, LowNoiseQuotaSampler)
    assert wrapped.status()["kind"] == "moss_truncated_logit_normal"


def test_quota_raises_the_low_noise_share_of_the_moss_distribution():
    torch.manual_seed(0)
    plain = MossTruncatedTimestepSampler()
    wrapped = LowNoiseQuotaSampler(MossTruncatedTimestepSampler(), every=4, max_t=0.4)
    share = lambda s: (s.sample(8000, "cpu") < 0.3).float().mean().item()
    assert share(plain) < 0.05
    assert 0.17 < share(wrapped) < 0.25
