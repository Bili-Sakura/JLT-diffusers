"""JLT-specific flow-matching helpers (schedulers come from diffusers)."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def configure_linear_flow_timesteps(
    scheduler: SchedulerMixin,
    num_inference_steps: int,
    device: Union[str, torch.device, None] = None,
) -> None:
    """Uniform sigma schedule from 1 (noise) to 0 (data), matching JLT training."""
    if isinstance(scheduler, FlowMatchHeunDiscreteScheduler):
        scheduler.num_inference_steps = num_inference_steps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps, dtype=np.float32)
        sigmas_t = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        shift = scheduler.config.shift
        sigmas_t = shift * sigmas_t / (1 + (shift - 1) * sigmas_t)
        timesteps = sigmas_t * scheduler.config.num_train_timesteps
        timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
        scheduler.timesteps = timesteps.to(device=device)
        sigmas_t = torch.cat([sigmas_t, torch.zeros(1, device=sigmas_t.device)])
        scheduler.sigmas = torch.cat([sigmas_t[:1], sigmas_t[1:-1].repeat_interleave(2), sigmas_t[-1:]])
        scheduler._step_index = None
        scheduler._begin_index = None
        scheduler.prev_derivative = None
        scheduler.dt = None
        scheduler.sample = None
        return

    sigmas = np.linspace(1.0, 0.0, num_inference_steps, dtype=np.float32).tolist()
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)


def velocity_from_prediction(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: Union[float, torch.Tensor],
    *,
    prediction_type: str = "sample",
    t_eps: float = 5e-2,
) -> torch.Tensor:
    if prediction_type == "velocity":
        return model_output

    t = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
    while t.ndim < sample.ndim:
        t = t.unsqueeze(-1)
    denom = (1.0 - t).clamp_min(t_eps)
    return (model_output - sample) / denom


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    p_mean: float = -0.8,
    p_std: float = 0.8,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * p_std + p_mean
    return torch.sigmoid(z).to(dtype=dtype)
