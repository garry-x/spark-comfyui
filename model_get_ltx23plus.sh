#!/bin/bash
# Flux2 模型下载脚本
# 所有模型来自 Comfy-Org/flux2-dev (HF 镜像)

source /data/.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

#hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
#  --local-dir /data/ComfyUI/ComfyUI/models/latent_upscale_models

#wget -c -O /data/ComfyUI/ComfyUI/models/frame_interpolation/rife49.pth \
#  "https://hf-mirror.com/AlexWortega/RIFE/resolve/main/rife49.pth"

#wget -c -O /data/ComfyUI/ComfyUI/models/upscale_models/RealESRGAN_x2.pth \
#  "https://hf-mirror.com/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth"
