#!/bin/bash
# JLT pixel-space JiT-B/16.
set -euo pipefail
GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
RUN=pixel-JLT-B-16
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/${RUN}}
mkdir -p "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --num_processes="${NUM_PROCS}" --mixed_precision=bf16 \
  examples/image_generation/train_jlt.py \
  --model JiT-B/16 --vae_type identity --img_size 256 \
  --data_path "${DATA_PATH:-./data/imagenet}" --output_dir "${OUTPUT_DIR}"
