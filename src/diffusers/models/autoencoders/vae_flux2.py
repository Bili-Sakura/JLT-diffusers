"""FLUX.2 VAE wrapper for JLT latent diffusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def _retrieve_latents(encoder_output):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


class Flux2LatentVAE(ModelMixin, ConfigMixin):
    """Black Forest FLUX.2 VAE with patchify and BN normalization for JLT."""

    @register_to_config
    def __init__(
        self,
        model_name_or_path: str = "black-forest-labs/FLUX.2-klein-4B",
        subfolder: str = "vae",
        latent_channels: int = 128,
    ):
        super().__init__()
        try:
            from diffusers import AutoencoderKLFlux2
        except ImportError as exc:
            raise ImportError("diffusers with AutoencoderKLFlux2 is required for Flux2LatentVAE") from exc

        load_kwargs = {"subfolder": subfolder} if subfolder else {}
        vae = AutoencoderKLFlux2.from_pretrained(model_name_or_path, **load_kwargs)
        vae.requires_grad_(False)
        vae.eval()
        object.__setattr__(self, "_vae", vae)

        encoder_down = 2 ** (len(vae.config.block_out_channels) - 1)
        self.latent_channels = latent_channels
        self.spatial_compression = encoder_down * 2

    def _apply(self, fn):
        super()._apply(fn)
        self._vae._apply(fn)
        return self

    def _bn_stats(self, ref: torch.Tensor):
        vae = self._vae
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(ref.device, ref.dtype)
        std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1).to(ref.device, ref.dtype) + vae.config.batch_norm_eps
        )
        return mean, std

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        vae = self._vae
        vae_param = next(vae.parameters())
        images = images.to(device=vae_param.device, dtype=vae_param.dtype)
        latents = _retrieve_latents(vae.encode(images))
        latents = _patchify(latents)
        mean, std = self._bn_stats(latents)
        return (latents - mean) / std

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self._vae
        vae_param = next(vae.parameters())
        latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
        mean, std = self._bn_stats(latents)
        latents = latents * std + mean
        latents = _unpatchify(latents)
        return vae.decode(latents, return_dict=False)[0]
