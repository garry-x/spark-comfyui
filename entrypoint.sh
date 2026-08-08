#!/bin/bash
set -e

LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/comfyui_$(date +%Y%m%d_%H%M%S).log"

# H3-optimized defaults:
#   --use-sage-attention  采样 ~25x 加速 (替代旧的 fp8 + pytorch-cross-attention)
#   --highvram            模型常驻 GPU
#   --reserve-vram 8      预留 VRAM 防止大分辨率时卸载模型（131GB 卡）
DEFAULT_ARGS=(
    "--listen" "0.0.0.0"
    "--port" "8188"
    "--enable-cors-header" "*"
    "--preview-method" "auto"
    "--use-sage-attention"
    "--highvram"
    "--reserve-vram" "8"
)

if [ $# -eq 0 ]; then
    set -- "${DEFAULT_ARGS[@]}"
fi

echo "=== ComfyUI H3 | $(date) ===" | tee -a "$LOG_FILE"
echo "Args: $*" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

exec python main.py "$@"
