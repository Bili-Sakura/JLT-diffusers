# JLT Image Generation Training

Diffusers-style training for JLT (clean-latent prediction) on ImageNet or pre-encoded FLUX.2 latents.

## Train on pre-encoded latents

```bash
accelerate launch examples/image_generation/train_jlt.py \
  --model JiT-B/1 \
  --vae_type flux2 \
  --img_size 256 \
  --use_latent_cache \
  --data_path /path/to/imagenet_latents_256 \
  --output_dir ./output_dir/jlt-b1
```

## Train with velocity prediction (DiT baseline)

```bash
accelerate launch examples/image_generation/train_jlt.py \
  --model JiT-B/2 \
  --flow_matching \
  --vae_type flux2 \
  --use_latent_cache \
  --data_path /path/to/imagenet_latents_256 \
  --output_dir ./output_dir/dit-b2
```

## Encode latents

```bash
python scripts/encode_latents.py \
  --data_path /path/to/imagenet \
  --output_path /path/to/imagenet_latents_256 \
  --img_size 256
```
