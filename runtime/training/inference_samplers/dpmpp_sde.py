"""DPM-Solver++ (stochastic) for CONST flow, aligned with ComfyUI k_diffusion.sample_dpmpp_sde (GPL-3.0).

Two model calls per step, except the last one. ``dpmpp_sde`` builds the BrownianTree on the CPU and
``dpmpp_sde_gpu`` on the latent's device; for the same seed the two give different noise, as in ComfyUI.
"""

from __future__ import annotations

from typing import Optional

import torch

from training.inference_samplers.dpmpp_3m_sde import _build_noise_sampler, _offset_first_sigma_for_snr


def _half_log_snr(sigma: torch.Tensor) -> torch.Tensor:
    return sigma.logit().neg()


def _sigma_from_half_log_snr(value: torch.Tensor) -> torch.Tensor:
    return value.neg().sigmoid()


def _ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float):
    if not eta:
        return sigma_to, sigma_to.new_zeros(())
    sigma_up = torch.minimum(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    return (sigma_to ** 2 - sigma_up ** 2) ** 0.5, sigma_up


@torch.no_grad()
def sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    seed: Optional[int] = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    r: float = 0.5,
    shift: float = 3.0,
    noise_on_cpu: bool = True,
    step_callback=None,
    **_unused,
) -> torch.Tensor:
    """``denoise_fn(x, sigma)`` returns the x0 estimate; ``sigmas`` descend and end with 0."""
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    if sigmas.numel() <= 1:
        return x
    noise = _build_noise_sampler(x, sigmas, seed=seed, cpu=noise_on_cpu)
    sigmas = _offset_first_sigma_for_snr(sigmas, shift=shift)
    fac = 1 / (2 * r)

    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = denoise_fn(x, sigma)
        if step_callback is not None:
            try:
                step_callback(i, len(sigmas) - 1, denoised)
            except Exception:
                pass
        if sigma_next == 0:
            x = denoised
            continue

        lambda_s, lambda_t = _half_log_snr(sigma), _half_log_snr(sigma_next)
        h = lambda_t - lambda_s
        lambda_s_1 = lambda_s + r * h
        sigma_s_1 = _sigma_from_half_log_snr(lambda_s_1)
        alpha_s = sigma * lambda_s.exp()
        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigma_next * lambda_t.exp()

        # Step 1
        sd, su = _ancestral_step(lambda_s.neg().exp(), lambda_s_1.neg().exp(), eta)
        h_ = sd.log().neg() - lambda_s
        x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (-h_).expm1() * denoised
        if eta > 0 and s_noise > 0:
            x_2 = x_2 + alpha_s_1 * noise(float(sigma), float(sigma_s_1)) * s_noise * su
        denoised_2 = denoise_fn(x_2, sigma_s_1)

        # Step 2
        sd, su = _ancestral_step(lambda_s.neg().exp(), lambda_t.neg().exp(), eta)
        h_ = sd.log().neg() - lambda_s
        denoised_d = (1 - fac) * denoised + fac * denoised_2
        x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (-h_).expm1() * denoised_d
        if eta > 0 and s_noise > 0:
            x = x + alpha_t * noise(float(sigma), float(sigma_next)) * s_noise * su
    return x


def sample_gpu(denoise_fn, x, sigmas, **kwargs) -> torch.Tensor:
    return sample(denoise_fn, x, sigmas, **{**kwargs, "noise_on_cpu": False})
