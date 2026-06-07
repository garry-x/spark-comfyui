#!/bin/bash
# LTX-2.3 模型下载脚本
#
# 主模型 (Lightricks/LTX-2.3-fp8): 需要 HuggingFace 授权
# VAE + Text Projection (Kijai/LTX2.3_comfy): 公开仓库
# Gemma3 TE (Pavpif/ltx2-gemma3-text-encoder): 公开仓库
#
# 如遇 403/Gated 错误，请先访问以下链接接受许可协议：
# https://huggingface.co/Lightricks/LTX-2.3-fp8
# 然后执行：hf auth login

source /data/.venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=========================================="
echo "LTX-2.3 模型下载"
echo "=========================================="

# ---- 主模型 (需 HF 授权) ----
# 主模型 1：Distilled FP8（8步采样，速度最快，适合快速验证）
# 约 29 GB
echo ">>> [1/5] Distilled FP8: ltx-2.3-22b-distilled-fp8.safetensors (约 29GB)"
hf download Lightricks/LTX-2.3-fp8 \
  ltx-2.3-22b-distilled-fp8.safetensors \
  --local-dir /data/ComfyUI/ComfyUI/models/checkpoints/

# 主模型 2：Dev FP8（20-30步，画质最高，支持CFG和负向提示词）
# 约 29 GB
echo ">>> [2/5] Dev FP8: ltx-2.3-22b-dev-fp8.safetensors (约 29GB)"
hf download Lightricks/LTX-2.3-fp8 \
  ltx-2.3-22b-dev-fp8.safetensors \
  --local-dir /data/ComfyUI/ComfyUI/models/checkpoints/

# ---- VAE (公开仓库) ----
# VAE（Tiny AutoEncoder，所有工作流必需）
# 约 23 MB，Kijai 社区打包
echo ">>> [3/5] VAE: taeltx2_3.safetensors (约 23MB)"
hf download Kijai/LTX2.3_comfy \
  vae/taeltx2_3.safetensors \
  --local-dir /data/ComfyUI/ComfyUI/models/vae/

# ---- Text Projection (公开仓库) ----
# Text Projection 层（Gemma3 → LTX 维度映射）
# 约 2.2 GB，Kijai 社区打包
echo ">>> [4/5] Text Projection: ltx-2.3_text_projection_bf16.safetensors (约 2.2GB)"
hf download Kijai/LTX2.3_comfy \
  text_encoders/ltx-2.3_text_projection_bf16.safetensors \
  --local-dir /data/ComfyUI/ComfyUI/models/text_encoders/

# ---- Gemma 3 12B Text Encoder (公开仓库) ----
# Gemma 3 12B FP8 量化文本编码器
# 约 12 GB，社区打包
echo ">>> [5/5] Gemma3 12B FP8 TE: model_gemma_3_12B_it_fp8_e4m3fn.safetensors (约 12GB)"
hf download Pavpif/ltx2-gemma3-text-encoder \
  model_gemma_3_12B_it_fp8_e4m3fn.safetensors \
  tokenizer.model \
  --local-dir /data/ComfyUI/ComfyUI/models/text_encoders/ltx23_gemma/

echo "=========================================="
echo "LTX-2.3 模型下载完成！"
echo "=========================================="
echo ""
echo "注意：下载完成后，请将 vae/ 子目录中的文件移到上级："
echo "  mv models/vae/vae/taeltx2_3.safetensors models/vae/"
echo "  rmdir models/vae/vae/"
echo "  mv models/text_encoders/text_encoders/ltx-2.3_text_projection_bf16.safetensors models/text_encoders/"
echo "  rmdir models/text_encoders/text_encoders/"
