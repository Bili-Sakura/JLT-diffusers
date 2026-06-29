"""Convert legacy JLT training checkpoints to diffusers model directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler

from src.diffusers import JLTTransformer2DModel
from src.diffusers.models.transformers.transformer_jlt import config_from_legacy


def parse_args():
    parser = argparse.ArgumentParser(description="Convert legacy JLT checkpoint to diffusers layout.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, default="ema1", choices=["model", "ema1", "ema2"])
    parser.add_argument("--safe_serialization", action="store_true")
    parser.add_argument("--in_channels", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_channels = args.in_channels
    if in_channels is None:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        if hasattr(ckpt_args, "__dict__"):
            ckpt_args = vars(ckpt_args)
        in_channels = 128 if ckpt_args.get("vae_type") == "flux2" else 3

    transformer, metadata = JLTTransformer2DModel.from_jlt_checkpoint(
        args.checkpoint_path,
        weights=args.weights,
        in_channels=in_channels,
    )

    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"
    transformer.save_pretrained(transformer_dir, safe_serialization=args.safe_serialization)

    source_args = metadata.get("source_args")
    solver = getattr(source_args, "sampling_method", "heun") if source_args else "heun"
    if solver == "heun":
        scheduler = FlowMatchHeunDiscreteScheduler()
        scheduler_name = "FlowMatchHeunDiscreteScheduler"
    else:
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler_name = "FlowMatchEulerDiscreteScheduler"
    scheduler.save_pretrained(scheduler_dir)
    model_index = {
        "_class_name": ["pipeline", "JLTPipeline"],
        "_diffusers_version": "0.36.0",
        "transformer": ["transformer_jlt", "JLTTransformer2DModel"],
        "scheduler": ["diffusers", scheduler_name],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
    }
    (output_dir / "model_index.json").write_text(json.dumps(model_index, indent=2), encoding="utf-8")
    (output_dir / "conversion_metadata.json").write_text(
        json.dumps({k: str(v) if not isinstance(v, (int, float, str, dict, list, bool, type(None))) else v for k, v in metadata.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved diffusers model to {output_dir}")


if __name__ == "__main__":
    main()
