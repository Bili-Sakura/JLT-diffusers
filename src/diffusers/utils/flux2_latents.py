"""FLUX.2 latent encode/decode helpers using diffusers AutoencoderKLFlux2."""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLFlux2


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    # Same as diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._patchify_latents
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    # Same as diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpatchify_latents
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)


def _retrieve_latents(encoder_output):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def flux2_latent_channels(vae: AutoencoderKLFlux2) -> int:
    return int(vae.config.latent_channels) * 4


def flux2_spatial_compression(vae: AutoencoderKLFlux2) -> int:
    return 2 ** (len(vae.config.block_out_channels) - 1) * 2


def flux2_bn_stats(vae: AutoencoderKLFlux2, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(ref.device, ref.dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(ref.device, ref.dtype) + vae.config.batch_norm_eps
    )
    return mean, std


@torch.no_grad()
def encode_flux2_latents(vae: AutoencoderKLFlux2, images: torch.Tensor) -> torch.Tensor:
    vae_param = next(vae.parameters())
    images = images.to(device=vae_param.device, dtype=vae_param.dtype)
    latents = _retrieve_latents(vae.encode(images))
    latents = _patchify_latents(latents)
    mean, std = flux2_bn_stats(vae, latents)
    return (latents - mean) / std


@torch.no_grad()
def decode_flux2_latents(vae: AutoencoderKLFlux2, latents: torch.Tensor) -> torch.Tensor:
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=torch.float32)
    mean, std = flux2_bn_stats(vae, latents)
    latents = latents * std + mean
    latents = _unpatchify_latents(latents)
    return vae.decode(latents.to(vae_param.dtype), return_dict=False)[0].float()
