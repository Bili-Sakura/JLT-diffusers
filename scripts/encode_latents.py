"""Encode ImageNet images to FLUX.2 latent shards for JLT training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.image_generation.train_jlt import center_crop_arr
from diffusers import AutoencoderKLFlux2

from src.diffusers.utils.flux2_latents import encode_flux2_latents


def parse_args():
    parser = argparse.ArgumentParser(description="Encode images to FLUX.2 latent safetensor shards.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--vae_model_name_or_path", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=4096)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    load_kwargs = {"subfolder": args.vae_subfolder} if args.vae_subfolder else {}
    vae = AutoencoderKLFlux2.from_pretrained(args.vae_model_name_or_path, **load_kwargs)
    vae.cuda().eval()

    latents_buf, latents_flip_buf, labels_buf = [], [], []
    shard_idx = 0
    global_idx = 0

    for images, labels in tqdm(loader):
        images = images.cuda()
        encoded = encode_flux2_latents(vae, images)
        flipped = images.flip(-1)
        encoded_flip = encode_flux2_latents(vae, flipped)

        latents_buf.append(encoded.cpu().half())
        latents_flip_buf.append(encoded_flip.cpu().half())
        labels_buf.append(labels.cpu())

        while sum(t.shape[0] for t in latents_buf) >= args.shard_size:
            latents = torch.cat(latents_buf, dim=0)
            latents_flip = torch.cat(latents_flip_buf, dim=0)
            labels_cat = torch.cat(labels_buf, dim=0)
            take = args.shard_size
            shard_latents = latents[:take]
            shard_flip = latents_flip[:take]
            shard_labels = labels_cat[:take]
            latents_buf = [latents[take:]]
            latents_flip_buf = [latents_flip[take:]]
            labels_buf = [labels_cat[take:]]

            out_path = os.path.join(args.output_path, f"latents_rank00_shard{shard_idx:03d}.safetensors")
            save_file(
                {
                    "latents": shard_latents,
                    "latents_flip": shard_flip,
                    "labels": shard_labels.to(torch.int32),
                },
                out_path,
            )
            shard_idx += 1
            global_idx += take

    if latents_buf:
        latents = torch.cat(latents_buf, dim=0)
        latents_flip = torch.cat(latents_flip_buf, dim=0)
        labels_cat = torch.cat(labels_buf, dim=0)
        out_path = os.path.join(args.output_path, f"latents_rank00_shard{shard_idx:03d}.safetensors")
        save_file(
            {"latents": latents, "latents_flip": latents_flip, "labels": labels_cat.to(torch.int32)},
            out_path,
        )

    from examples.image_generation.latent_dataset import build_shard_index

    build_shard_index(args.output_path, force=True)
    print(f"Encoded {global_idx} samples into {shard_idx + 1} shards at {args.output_path}")


if __name__ == "__main__":
    main()
