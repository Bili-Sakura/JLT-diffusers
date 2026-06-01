#!/bin/bash
# JLT-B/1 on FLUX.2 latents (16x16 grid, patch 1).
set -euo pipefail

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN=latent-JLT-B-1
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/${RUN}}
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
    --num_processes="${NUM_PROCS}" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port="${MASTER_PORT:-29500}" \
    examples/image_generation/train_jlt.py \
        --model JiT-B/1 \
        --vae_type flux2 \
        --proj_dropout 0.0 \
        --P_mean -0.8 --P_std 0.8 \
        --img_size 256 --noise_scale 1.0 \
        --batch_size 256 --accum_iter 2 --blr 5e-5 \
        --epochs 40 --warmup_epochs 5 \
        --gen_bsz 128 --num_images 50000 \
        --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
        --eval_freq 10 --online_eval \
        --num_workers 12 \
        --data_path "${DATA_PATH:-./data/imagenet_latents_256}" --use_latent_cache \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project JLT --wandb_name "${RUN}" --wandb_mode online
