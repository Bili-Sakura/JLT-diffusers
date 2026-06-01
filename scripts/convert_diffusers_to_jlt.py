"""Convert diffusers JLT model back to legacy checkpoint format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusers import JLTTransformer2DModel
from src.diffusers.models.transformers.transformer_jlt import JLT_PRESET_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Convert diffusers JLT model back to legacy .pth checkpoint.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ema_mode", type=str, default="copy_to_both", choices=["none", "copy_to_both"])
    parser.add_argument("--epoch", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    transformer_path = model_path / "transformer" if (model_path / "transformer").exists() else model_path

    transformer = JLTTransformer2DModel.from_pretrained(str(transformer_path))
    checkpoint = transformer.to_jlt_checkpoint(ema_mode=args.ema_mode)
    checkpoint["epoch"] = args.epoch

    config = transformer.config
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        for name, preset in JLT_PRESET_CONFIGS.items():
            if (
                int(config.sample_size) == int(preset["sample_size"])
                and int(config.patch_size) == int(preset["patch_size"])
            ):
                model_type = name
                break

    import argparse as argparse_module

    checkpoint["args"] = argparse_module.Namespace(
        model=model_type or "JiT-B/1",
        class_num=int(getattr(config, "num_classes", 1000)),
        vae_type="flux2" if int(config.in_channels) == 128 else "identity",
        img_size=int(config.sample_size) * 16 if int(config.in_channels) == 128 else int(config.sample_size),
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(checkpoint, args.output_path)
    print(f"Saved legacy checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
