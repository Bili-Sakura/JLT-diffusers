#!/bin/bash
# JLT-B/2 with async token-wise timesteps.
set -euo pipefail
GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
RUN=latent-JLT-B-2-async
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/${RUN}}
mkdir -p "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --num_processes="${NUM_PROCS}" --mixed_precision=bf16 \
  examples/image_generation/train_jlt.py \
  --model JiT-B/2 --vae_type flux2 --use_latent_cache \
  --async_timesteps --async_timestep_drop 0.1 \
  --data_path "${DATA_PATH:-./data/imagenet_latents_256}" \
  --output_dir "${OUTPUT_DIR}"
