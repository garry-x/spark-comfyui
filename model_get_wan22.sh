#!/bin/bash
# Wan 2.2 模型下载脚本
# 注: Wan 2.7 是 API-only (Partner Nodes)，无可下载权重
# Wan 2.2 是最新开源版本，ComfyUI 原生支持
# 模型来源: Comfy-Org/Wan_2.2_ComfyUI_Repackaged (5.4M 下载)

source /data/.venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
MODEL_DIR="/data/ComfyUI/ComfyUI/models"

echo "=========================================="
echo "Wan 2.2 T2V 模型下载 (FP8 + LightX2V 加速)"
echo "=========================================="

# ===== 扩散模型 (DIFFUSION MODELS) =====
echo ">>> [1/4] T2V High Noise FP8 (约 16GB)"
hf download "$REPO" \
  split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors \
  --local-dir "$MODEL_DIR/"

echo ">>> [2/4] T2V Low Noise FP8 (约 16GB)"
hf download "$REPO" \
  split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors \
  --local-dir "$MODEL_DIR/"

# ===== 文本编码器 (TEXT ENCODER) =====
echo ">>> [3/4] UMT5 XXL FP8 (约 6GB)"
hf download "$REPO" \
  split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  --local-dir "$MODEL_DIR/"

# ===== VAE + LoRA (并行下载) =====
echo ">>> [4/4] VAE + LightX2V LoRA (约 600MB)"
hf download "$REPO" \
  split_files/vae/wan2.2_vae.safetensors \
  --local-dir "$MODEL_DIR/" &

hf download "$REPO" \
  split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors \
  --local-dir "$MODEL_DIR/" &

hf download "$REPO" \
  split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors \
  --local-dir "$MODEL_DIR/" &

wait

echo ""
echo "=========================================="
echo "Wan 2.2 模型下载完成！"
echo "=========================================="
echo ""
echo "模型位置:"
echo "  diffusion_models/split_files/diffusion_models/"
echo "  text_encoders/split_files/text_encoders/"
echo "  vae/split_files/vae/"
echo "  loras/split_files/loras/"
echo ""
echo "注意: hf download 会保留仓库目录结构"
echo "如需整理到扁平目录，运行:"
echo "  find $MODEL_DIR/split_files -name '*.safetensors' -exec mv {} $MODEL_DIR/ \;"
echo "  rm -rf $MODEL_DIR/split_files/"
