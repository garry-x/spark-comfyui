#!/bin/bash
# Flux2 模型下载脚本
# 所有模型来自 Comfy-Org/flux2-dev (HF 镜像)

source /data/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=========================================="
echo "Flux2 模型下载"
echo "=========================================="

# 1. 主模型：FLUX.2-dev FP8 mixed（约 35GB）
echo ">>> [1/3] 主模型: flux2_dev_fp8mixed.safetensors (约 35GB)"
hf download Comfy-Org/flux2-dev \
  split_files/diffusion_models/flux2_dev_fp8mixed.safetensors \
  --local-dir /data/ComfyUI/ComfyUI/models/

# 2. Text Encoder：Mistral 3 Small FP8（约 17GB）
echo ">>> [2/3] Text Encoder: mistral_3_small_flux2_fp8.safetensors (约 17GB)"
hf download Comfy-Org/flux2-dev \
  split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors \
  --local-dir /data/ComfyUI/ComfyUI/models/

# 3. VAE：FLUX.2 专用 VAE（约 400MB）
echo ">>> [3/3] VAE: flux2-vae.safetensors (约 400MB)"
hf download Comfy-Org/flux2-dev \
  split_files/vae/flux2-vae.safetensors \
  --local-dir /data/ComfyUI/ComfyUI/models/

echo "=========================================="
echo "Flux2 模型下载完成！"
echo "=========================================="
