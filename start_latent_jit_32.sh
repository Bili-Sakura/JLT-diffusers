#!/bin/bash
# JLT-B/2 on FLUX.2 latents (16x16 grid, patch 2).
set -euo pipefail

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN=latent-JLT-B-2
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/${RUN}}
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
    --num_processes="${NUM_PROCS}" \
    --mixed_precision=bf16 \
    examples/image_generation/train_jlt.py \
        --model JiT-B/2 \
        --vae_type flux2 \
        --img_size 256 --noise_scale 1.0 \
        --batch_size 128 --accum_iter 2 --blr 5e-5 \
        --epochs 40 --warmup_epochs 5 \
        --data_path "${DATA_PATH:-./data/imagenet_latents_256}" --use_latent_cache \
        --output_dir "${OUTPUT_DIR}"
