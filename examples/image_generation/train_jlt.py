"""Train JLT with a diffusers-style Accelerate loop."""

from __future__ import annotations

import argparse
import copy
import datetime
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.image_generation.latent_dataset import Flux2LatentDataset
from src.diffusers import Flux2LatentVAE, JLTScheduler, JLTTransformer2DModel
from src.diffusers.models.transformers.transformer_jlt import JLT_PRESET_CONFIGS
from src.diffusers.pipelines.jlt.pipeline_jlt import JLTPipeline


def center_crop_arr(pil_image, image_size):
    from PIL import Image

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def parse_args():
    parser = argparse.ArgumentParser(description="Train JLT (clean-latent prediction) with diffusers components.")

    parser.add_argument("--model", default="JiT-B/1", type=str, choices=sorted(JLT_PRESET_CONFIGS.keys()))
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--mask_prob", type=float, default=0.0)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--loop_indices", type=str, default="")
    parser.add_argument("--loop_count", type=int, default=0)

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)

    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--flow_matching", action="store_true")
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--async_timesteps", action="store_true")
    parser.add_argument("--async_timestep_drop", default=0.0, type=float)

    parser.add_argument("--vae_type", default="flux2", type=str, choices=["identity", "flux2"])
    parser.add_argument("--vae_model_name_or_path", default="black-forest-labs/FLUX.2-klein-4B", type=str)
    parser.add_argument("--vae_subfolder", default="vae", type=str)
    parser.add_argument("--use_latent_cache", action="store_true")
    parser.add_argument("--use_parquet", action="store_true")
    parser.add_argument("--cache_dir", default="./hf_cache", type=str)

    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")

    parser.add_argument("--output_dir", default="./output_dir", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--save_last_freq", type=int, default=5)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=2.9, type=float)
    parser.add_argument("--interval_min", default=0.1, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--eval_freq", type=int, default=40)
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--evaluate_gen", action="store_true")
    parser.add_argument("--gen_bsz", type=int, default=128)

    parser.add_argument("--wandb_project", default="JLT", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=["online", "offline", "disabled"])
    parser.add_argument("--collective_timeout_hours", default=2.0, type=float)

    parser.add_argument("--model_name_or_path", type=str, default=None, help="Resume from diffusers model directory.")
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args, data_iter_step=0, steps_per_epoch=1):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / max(args.warmup_epochs, 1)
    elif args.lr_schedule == "constant":
        lr = args.lr
    else:
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def add_weight_decay(model, weight_decay=0.0):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def build_transformer(args, in_channels: int) -> JLTTransformer2DModel:
    if args.model_name_or_path:
        return JLTTransformer2DModel.from_pretrained(args.model_name_or_path)

    config = dict(JLT_PRESET_CONFIGS[args.model])
    config["in_channels"] = in_channels
    config["num_classes"] = args.class_num
    config["model_type"] = args.model
    config["attention_dropout"] = args.attn_dropout
    config["dropout"] = args.proj_dropout
    if args.vae_type == "flux2":
        config["sample_size"] = args.img_size // 16
    if args.mask_prob > 0.0:
        config["mask_prob"] = args.mask_prob
        config["mask_ratio"] = args.mask_ratio
    if args.loop_count > 0 and args.loop_indices.strip():
        config["loop_indices"] = [int(x) for x in args.loop_indices.split(",") if x.strip()]
        config["loop_count"] = args.loop_count
    return JLTTransformer2DModel(**config)


def compute_training_loss(
    transformer: JLTTransformer2DModel,
    scheduler: JLTScheduler,
    x: torch.Tensor,
    labels: torch.Tensor,
    args,
    vae: Flux2LatentVAE | None,
) -> torch.Tensor:
    if vae is not None and not args.use_latent_cache:
        with torch.no_grad():
            x = vae.encode(x)

    model_dtype = next(transformer.parameters()).dtype
    x = x.to(dtype=model_dtype)

    if transformer.training and args.label_drop_prob > 0.0:
        drop = torch.rand(labels.shape[0], device=labels.device) < args.label_drop_prob
        labels = torch.where(drop, torch.full_like(labels, args.class_num), labels)

    if args.async_timesteps:
        patch_size = int(transformer.patch_size)
        _, _, height, width = x.shape
        token_h = height // patch_size
        token_w = width // patch_size
        t_tokens = scheduler.sample_timesteps(
            x.size(0) * token_h * token_w, x.device, x.dtype, args.P_mean, args.P_std
        ).view(x.size(0), token_h * token_w)
        if args.async_timestep_drop > 0.0:
            disable = torch.rand(x.size(0), device=x.device) < args.async_timestep_drop
            if disable.any():
                fallback = scheduler.sample_timesteps(int(disable.sum().item()), x.device, x.dtype, args.P_mean, args.P_std)
                t_tokens[disable] = fallback.unsqueeze(-1)
        t_map = t_tokens.view(x.size(0), 1, token_h, token_w)
        if patch_size > 1:
            t_map = t_map.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
    else:
        t_scalar = scheduler.sample_timesteps(x.size(0), x.device, x.dtype, args.P_mean, args.P_std)
        t_map = t_scalar.view(-1, *([1] * (x.ndim - 1)))
        t_tokens = t_scalar

    e = torch.randn_like(x) * args.noise_scale
    z = t_map * x + (1 - t_map) * e

    if args.flow_matching:
        v = x - e
    else:
        v = (x - z) / (1 - t_map).clamp_min(args.t_eps)

    net_out = transformer(z, timestep=t_tokens, class_labels=labels).sample
    if args.flow_matching:
        v_pred = net_out
    else:
        v_pred = (net_out - z) / (1 - t_map).clamp_min(args.t_eps)

    return (v - v_pred).pow(2).mean(dim=(1, 2, 3)).mean()


def update_ema(ema_params, source_params, decay):
    for targ, src in zip(ema_params, source_params):
        targ.detach().mul_(decay).add_(src.detach().float(), alpha=1 - decay)


def save_checkpoint(path, transformer, optimizer, epoch, args, ema_params1, ema_params2):
    ckpt = transformer.to_jlt_checkpoint(ema_mode="none")
    ckpt["optimizer"] = optimizer.state_dict()
    ckpt["epoch"] = epoch
    ckpt["args"] = args

    ema1 = copy.deepcopy(ckpt["model"])
    ema2 = copy.deepcopy(ckpt["model"])
    for i, (name, _) in enumerate(transformer.named_parameters()):
        ema1[name] = ema_params1[i]
        ema2[name] = ema_params2[i]
    ckpt["model_ema1"] = ema1
    ckpt["model_ema2"] = ema2

    if Path(path).parent:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, transformer, optimizer, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    from src.diffusers.models.transformers.transformer_jlt import remap_legacy_state_dict

    transformer.load_state_dict(remap_legacy_state_dict(state_dict), strict=False)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint.get("epoch", -1)

    ema_params1 = []
    ema_params2 = []
    ema1 = checkpoint.get("model_ema1", {})
    ema2 = checkpoint.get("model_ema2", {})
    for name, param in transformer.named_parameters():
        ema_params1.append(ema1.get(name, param.detach().float().clone()).to(device=device, dtype=torch.float32))
        ema_params2.append(ema2.get(name, param.detach().float().clone()).to(device=device, dtype=torch.float32))
    return epoch, ema_params1, ema_params2


@torch.no_grad()
def evaluate_generation(accelerator, transformer, vae, scheduler, args, epoch, ema_params1):
    transformer.eval()
    state_dict = copy.deepcopy(transformer.state_dict())
    for i, (name, param) in enumerate(transformer.named_parameters()):
        state_dict[name] = ema_params1[i].to(dtype=param.dtype, device=param.device)
    transformer.load_state_dict(state_dict)

    pipeline = JLTPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
    )
    pipeline.to(accelerator.device)

    labels = torch.tensor([0], device=accelerator.device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = pipeline(
            class_labels=labels,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.cfg,
            guidance_interval_min=args.interval_min,
            guidance_interval_max=args.interval_max,
            sampling_method=args.sampling_method,
            noise_scale=args.noise_scale,
            output_type="latent" if vae is None else "pt",
        )
    accelerator.print(f"[eval] smoke sample ok at epoch {epoch}")
    transformer.load_state_dict(state_dict)
    transformer.train()


def main():
    args = parse_args()

    pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=args.collective_timeout_hours))
    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[pg_kwargs],
        gradient_accumulation_steps=args.accum_iter,
    )

    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = args.accum_iter

    seed = args.seed + accelerator.process_index
    torch.manual_seed(seed)
    np.random.seed(seed)

    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={
            "wandb": {
                "name": args.wandb_name or os.path.basename(args.output_dir.rstrip("/")),
                "dir": args.output_dir,
                "mode": args.wandb_mode,
            }
        },
    )

    vae = None
    in_channels = 3
    if args.vae_type == "flux2":
        vae = Flux2LatentVAE(
            model_name_or_path=args.vae_model_name_or_path,
            subfolder=args.vae_subfolder,
        )
        in_channels = vae.latent_channels
        vae.requires_grad_(False)
        vae.eval()

    transform_train = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if args.use_latent_cache:
        dataset_train = Flux2LatentDataset(args.data_path, use_flip=True)
    elif args.use_parquet:
        from examples.image_generation.parquet_dataset import HuggingFaceImageNetDataset

        os.makedirs(args.cache_dir, exist_ok=True)
        dataset_train = HuggingFaceImageNetDataset(
            data_dir=args.data_path,
            split="train",
            transform=transform_train,
            cache_dir=args.cache_dir,
        )
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform_train)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_mem,
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True
    data_loader = DataLoader(dataset_train, **loader_kwargs)

    prediction_type = "velocity" if args.flow_matching else "sample"
    scheduler = JLTScheduler(t_eps=args.t_eps, solver=args.sampling_method, prediction_type=prediction_type)
    transformer = build_transformer(args, in_channels)
    transformer = transformer.to(dtype=torch.bfloat16)

    eff_batch_size = args.batch_size * accelerator.num_processes * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    optimizer = torch.optim.AdamW(add_weight_decay(transformer, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))

    transformer, optimizer, data_loader, vae = accelerator.prepare(transformer, optimizer, data_loader, vae)
    transformer_unwrapped = accelerator.unwrap_model(transformer)

    device = accelerator.device
    ema_params1 = [p.detach().float().clone() for p in transformer_unwrapped.parameters()]
    ema_params2 = [p.detach().float().clone() for p in transformer_unwrapped.parameters()]
    start_epoch = 0

    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, ema_params1, ema_params2 = load_checkpoint(checkpoint_path, transformer_unwrapped, optimizer, device)
        start_epoch += 1
        accelerator.print(f"Resumed from {checkpoint_path}, starting epoch {start_epoch}")

    if args.evaluate_gen:
        evaluate_generation(accelerator, transformer_unwrapped, vae, scheduler, args, start_epoch, ema_params1)
        accelerator.end_training()
        return

    updates_per_epoch = math.ceil(len(data_loader) / args.accum_iter)
    optimizer_step = start_epoch * updates_per_epoch

    for epoch in range(start_epoch, args.epochs):
        transformer.train()
        if hasattr(data_loader, "set_epoch"):
            data_loader.set_epoch(epoch)

        for data_iter_step, (x, labels) in enumerate(data_loader):
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            with accelerator.accumulate(transformer):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = compute_training_loss(
                        transformer_unwrapped,
                        scheduler,
                        x,
                        labels,
                        args,
                        accelerator.unwrap_model(vae) if vae is not None else None,
                    )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                optimizer_step += 1
                update_ema(ema_params1, transformer_unwrapped.parameters(), args.ema_decay1)
                update_ema(ema_params2, transformer_unwrapped.parameters(), args.ema_decay2)
                if optimizer_step % args.log_freq == 0:
                    accelerator.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=optimizer_step)

        if accelerator.is_main_process and (epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs):
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoint-last.pth"),
                transformer_unwrapped,
                optimizer,
                epoch,
                args,
                ema_params1,
                ema_params2,
            )
            diffusers_dir = os.path.join(args.output_dir, "diffusers")
            transformer_unwrapped.save_pretrained(os.path.join(diffusers_dir, "transformer"))
            scheduler.save_pretrained(os.path.join(diffusers_dir, "scheduler"))
            if vae is not None:
                accelerator.unwrap_model(vae).save_pretrained(os.path.join(diffusers_dir, "vae"))

        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            evaluate_generation(accelerator, transformer_unwrapped, vae, scheduler, args, epoch, ema_params1)

    accelerator.end_training()


if __name__ == "__main__":
    main()
