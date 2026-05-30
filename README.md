# JLT: Clean-Latent Prediction in Latent Diffusion Transformers

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2605.27102-b31b1b.svg)](https://arxiv.org/abs/2605.27102)
[![GitHub](https://img.shields.io/badge/GitHub-akatsuki--neo/JLT-blue.svg)](https://github.com/akatsuki-neo/JLT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Models-dawn--neo/JLT-yellow)](https://huggingface.co/dawn-neo/JLT)

</div>

## Overview

JLT investigates whether predicting clean data is better than predicting velocity in latent space. Under the same architecture, training settings, and FLUX.2 VAE representation, clean-latent prediction achieves **FID 2.50** vs. velocity prediction at **FID 6.56** — a 62% improvement on ImageNet 256×256.

This repository contains the training and evaluation code for JLT.

## Results

| Model | Target | FID-50K ↓ | IS ↑ |
|-------|--------|-----------|------|
| **JLT-B/1** | x (clean) | **2.50** | 232.51 |
| DiT-B/1 | v (velocity) | 6.56 | 132.12 |
| **JLT-B/2** | x (clean) | **14.81** | 107.29 |
| DiT-B/2 | v (velocity) | 28.71 | 58.46 |

## Installation

```bash
# Clone repository
git clone https://github.com/akatsuki-neo/JLT.git
cd JLT

# Create conda environment
conda env create -f environment.yaml
conda activate jit

# Install accelerate (required for distributed training)
pip install accelerate

# Install additional dependencies
pip install torch-fidelity  # for FID evaluation
```

## Data Preparation

### 1. Download ImageNet

Download ImageNet train/val from [image-net.org](https://image-net.org/download.php) and extract to a directory.

### 2. Encode Images to FLUX.2 Latents

Encode ImageNet to latent shards for efficient training:

```bash
python prepare_ref.py \
    --data_path /path/to/imagenet \
    --output_path /path/to/imagenet_latents_256 \
    --img_size 256 \
    --vae_type flux2 \
    --vae_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
    --batch_size 256 \
    --num_workers 8
```

This produces safetensor latent shards in `/path/to/imagenet_latents_256`.

## Running Experiments

### JLT-B/1 (Clean-Latent Prediction, /1 scale)

```bash
./start_latent_jit_16.sh [GPU_IDS]

# Example: use GPUs 0-3 only
./start_latent_jit_16.sh 0,1,2,3
```

Key settings:
- Model: JiT-B/1 (patch 1, 16x16 latent grid)
- Batch size: 256 × 8 GPUs × 2 accum = 4096 effective
- Epochs: 40 (with 5 warmup)
- Learning rate: 5e-5 base LR

### JLT-B/2 (Clean-Latent Prediction, /2 scale)

```bash
./start_latent_jit_32.sh [GPU_IDS]
```

### DiT-B/2 Baseline (Velocity Prediction)

```bash
./start_latent_v_32.sh [GPU_IDS]
```

Key difference: `--flow_matching` flag enables direct velocity prediction.

## Architecture

| Component | Specification |
|-----------|--------------|
| Transformer Blocks | 12 |
| Hidden Dimension | 768 |
| Attention Heads | 12 |
| Bottleneck Patch Embedding | 128-dim |
| Parameters | 130M |
| Tokenizer | FLUX.2 VAE (frozen) |

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Model variant: `JiT-B/1` or `JiT-B/2` |
| `--vae_type` | `flux2` for FLUX.2 latent space |
| `--flow_matching` | Enable velocity prediction (DiT baseline) |
| `--batch_size` | Micro batch per GPU |
| `--blr` | Base learning rate |
| `--epochs` | Training epochs |
| `--cfg` | Classifier-free guidance scale |
| `--data_path` | Path to pre-encoded latents |
| `--use_latent_cache` | Load pre-encoded safetensor latents |
| `--vae_model_name_or_path` | FLUX.2 VAE path or HuggingFace repo |

## Citation

```bibtex
@misc{fu2026jltcleanlatentpredictionlatent,
  title={JLT: Clean-Latent Prediction in Latent Diffusion Transformers},
  author={Funing Fu and Tenghui Wang and Guanyu Zhou and Junyong Cen and Qichao Zhu},
  year={2026},
  eprint={2605.27102},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.27102}
}
```

## Acknowledgements

- Li & He. "Back to Basics: Let Denoising Generative Models Denoise." arXiv:2511.13720, 2025.
- JiT GitHub: https://github.com/LTH14/JiT
- Black Forest Labs. FLUX.2 Small Decoder. HuggingFace, 2026.