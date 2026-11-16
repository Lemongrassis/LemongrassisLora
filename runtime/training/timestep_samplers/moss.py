"""MOSS timestep distributions on the 1000-step grid.

``moss_logit_normal`` is kohya's first approximation. ``moss_truncated_logit_normal`` is the
final interval of the MOSS-Image pre-training (technical report eq. 4), sampled exactly:

    S = 1 - sigmoid(Z), Z ~ N(mu, s^2);  F(x) = Pr(S <= x) = 1 - Phi((log((1 - x) / x) - mu) / s)
    sigma_j = j / T;  p_j = (1 - rho) * (F(a * j / T) - F(a * (j - 1) / T)) / F(a) + rho / T

The cutoff ``a`` drops the noisiest samples, the rest is stretched back over the whole grid,
and ``rho`` of the samples are uniform, so the low-noise end is never empty.
"""
import torch
from .baseline import BaselineTimestepSampler


class MossTimestepSampler(BaselineTimestepSampler):
    def __init__(self, mean=-1.09, std=0.95, schedule_shift=1.0):
        super().__init__()
        self.mean, self.std = float(mean), float(std)
        self.schedule_shift = float(schedule_shift)

    def sample(self, bs, device, *, token_counts=None):
        u = torch.sigmoid(torch.randn(bs, device=device) * self.std + self.mean)
        t = (1000 - (u * 1000).long()).float().div(1000)
        t = self.schedule_shift * t / (1 + (self.schedule_shift - 1) * t)
        return t.clamp(0.001, 0.999)

    def status(self):
        return {"kind": "moss_logit_normal", "mean": self.mean, "std": self.std, "schedule_shift": self.schedule_shift}


class MossTruncatedTimestepSampler(BaselineTimestepSampler):
    def __init__(self, mean=-1.09, std=0.95, cutoff=0.95, uniform=0.01, steps=1000):
        super().__init__()
        self.mean, self.std = float(mean), float(std)
        self.cutoff, self.uniform = float(cutoff), float(uniform)
        self.steps = int(steps)
        normal = torch.distributions.Normal(0.0, 1.0)
        x = (torch.arange(1, self.steps + 1, dtype=torch.float64) / self.steps * self.cutoff).clamp(max=1 - 1e-12)
        cdf = 1 - normal.cdf((torch.log((1 - x) / x) - self.mean) / self.std)
        bins = torch.diff(cdf, prepend=torch.zeros(1, dtype=torch.float64)) / cdf[-1]
        self.probs = ((1 - self.uniform) * bins + self.uniform / self.steps).float()

    def sample(self, bs, device, *, token_counts=None):
        index = torch.multinomial(self.probs.to(device), bs, replacement=True)
        return (index + 1).float().div(self.steps)

    def status(self):
        return {"kind": "moss_truncated_logit_normal", "mean": self.mean, "std": self.std,
                "cutoff": self.cutoff, "uniform": self.uniform}


def build_truncated(args, total_steps):
    return MossTruncatedTimestepSampler(getattr(args, "moss_timestep_mean", -1.09),
                                        getattr(args, "moss_timestep_std", 0.95),
                                        getattr(args, "moss_timestep_cutoff", 0.95),
                                        getattr(args, "moss_timestep_uniform", 0.01))


def build(args, total_steps):
    return MossTimestepSampler(getattr(args, "moss_timestep_mean", -1.09),
                               getattr(args, "moss_timestep_std", 0.95),
                               getattr(args, "timestep_schedule_shift", 1.0))
