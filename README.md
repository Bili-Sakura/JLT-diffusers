# JLT: Clean-Latent Prediction in Latent Diffusion Transformers

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2605.27102-b31b1b.svg)](https://arxiv.org/abs/2605.27102)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-JLT-blue.svg)](https://akatsuki-neo.github.io/JLT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Models-dawn--neo/JLT-yellow)](https://huggingface.co/dawn-neo/JLT)

</div>

## Overview

JLT trains class-conditional diffusion transformers in FLUX.2 latent space with **clean-latent (x) prediction**, following the native [diffusers](https://github.com/huggingface/diffusers) layout used in [JiT-diffusers](https://github.com/Bili-Sakura/JiT-diffusers.git).

## Package layout

- `src/diffusers/models/transformers/transformer_jlt.py` — `JLTTransformer2DModel` (`ModelMixin` / `ConfigMixin`)
- `src/diffusers/schedulers/scheduling_jlt.py` — `JLTScheduler` (Euler / Heun, x- or v-prediction)
- `src/diffusers/models/autoencoders/vae_flux2.py` — `Flux2LatentVAE` (FLUX.2 encode/decode)
- `src/diffusers/pipelines/jlt/pipeline_jlt.py` — `JLTPipeline` (CFG + latent sampling + VAE decode)
- `examples/image_generation/train_jlt.py` — Accelerate training entrypoint
- `scripts/convert_jlt_to_diffusers.py` — legacy `.pth` → diffusers directory
- `scripts/convert_diffusers_to_jlt.py` — diffusers → legacy checkpoint
- `scripts/sample_jlt.py` — single-image sampling
- `scripts/encode_latents.py` — ImageNet → FLUX.2 latent shards

## Installation

```bash
git clone https://github.com/akatsuki-neo/JLT.git
cd JLT
pip install -e ".[train]"
# optional: pip install -e ".[flash]" for flash-attn
```

## Data preparation

Encode ImageNet to latent shards:

```bash
python scripts/encode_latents.py \
  --data_path /path/to/imagenet \
  --output_path /path/to/imagenet_latents_256 \
  --img_size 256 \
  --vae_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --batch_size 256 \
  --num_workers 8
```

## Training

### JLT-B/1 (clean-latent, patch /1)

```bash
./start_latent_jit_16.sh 0,1,2,3
```

### JLT-B/2 (clean-latent, patch /2)

```bash
./start_latent_jit_32.sh
```

### DiT-B/2 baseline (velocity / flow matching)

```bash
./start_latent_v_32.sh
```

Or launch directly:

```bash
accelerate launch examples/image_generation/train_jlt.py \
  --model JiT-B/1 \
  --vae_type flux2 \
  --use_latent_cache \
  --data_path /path/to/imagenet_latents_256 \
  --output_dir ./output_dir/jlt-b1
```

## Convert checkpoint

```bash
python scripts/convert_jlt_to_diffusers.py \
  --checkpoint_path checkpoint-last.pth \
  --output_dir jlt-diffusers \
  --weights ema1 \
  --safe_serialization
```

## Sample

```bash
python scripts/sample_jlt.py \
  --model jlt-diffusers \
  --output sample.png \
  --class-label 207 \
  --num-inference-steps 50 \
  --solver heun \
  --cfg 2.9
```

## Key arguments

| Argument | Description |
|----------|-------------|
| `--model` | `JiT-B/1`, `JiT-B/2`, etc. |
| `--vae_type` | `flux2` or `identity` |
| `--flow_matching` | Velocity prediction (DiT baseline) |
| `--use_latent_cache` | Load pre-encoded safetensor shards |
| `--async_timesteps` | Token-wise timesteps during training |

## Citation

```bibtex
@article{fu2026jlt,
  title={{JLT}: {C}lean-{L}atent {P}rediction in {L}atent {D}iffusion {T}ransformers},
  author={Fu, Funing and Wang, Tenghui and Zhou, Guanyu and Cen, Junyong and Zhu, Qichao},
  journal = {arXiv preprint arXiv:2605.27102},
  year={2026}
}
```

## Acknowledgements

- Li & He. "Back to Basics: Let Denoising Generative Models Denoise." arXiv:2511.13720, 2025.
- [JiT-diffusers](https://github.com/Bili-Sakura/JiT-diffusers) — diffusers integration pattern
- Black Forest Labs. FLUX.2 VAE.
