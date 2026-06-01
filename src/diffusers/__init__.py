from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler

from .models.transformers.transformer_jlt import JLTTransformer2DModel
from .pipelines.jlt.pipeline_jlt import JLTPipeline
from .schedulers.jlt_flow import make_flow_scheduler, sample_timesteps, velocity_from_prediction
from .utils.flux2_latents import decode_flux2_latents, encode_flux2_latents

# Aliases matching common FLUX.2 naming.
Flux2VAE = AutoencoderKLFlux2
FlowMatchHeunScheduler = FlowMatchHeunDiscreteScheduler

__all__ = [
    "AutoencoderKLFlux2",
    "Flux2VAE",
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "FlowMatchHeunScheduler",
    "JLTTransformer2DModel",
    "JLTPipeline",
    "make_flow_scheduler",
    "sample_timesteps",
    "velocity_from_prediction",
    "encode_flux2_latents",
    "decode_flux2_latents",
]
