"""Sample images from a diffusers JLT model directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusers import JLTPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a JLT diffusers model.")
    parser.add_argument("--model", type=str, required=True, help="Path to diffusers model directory.")
    parser.add_argument("--output", type=str, default="sample.png")
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--solver", type=str, default="heun", choices=["heun", "euler"])
    parser.add_argument("--cfg", type=float, default=2.9)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = JLTPipeline.from_pretrained(args.model)
    pipeline.to(args.device, torch.bfloat16)

    result = pipeline(
        class_labels=args.class_label,
        num_inference_steps=args.num_inference_steps,
        sampling_method=args.solver,
        guidance_scale=args.cfg,
        noise_scale=args.noise_scale,
    )
    result.images[0].save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
