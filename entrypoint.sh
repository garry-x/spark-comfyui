#!/bin/bash
set -e

LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/comfyui_$(date +%Y%m%d_%H%M%S).log"

# 默认参数（与你本地命令完全一致）
DEFAULT_ARGS=(
    "--listen" "0.0.0.0"
    "--port" "8188"
    "--enable-cors-header" "*"
    "--preview-method" "auto"
    "--highvram"
    "--fp8_e4m3fn-unet"
    "--fp8_e4m3fn-text-enc"
    "--reserve-vram" "1.5"
    "--dont-upcast-attention"
    "--use-pytorch-cross-attention"
)

# 如果外部没有传参，使用默认参数；否则用外部传入的
if [ $# -eq 0 ]; then
    set -- "${DEFAULT_ARGS[@]}"
fi

echo "=== ComfyUI Starting | $(date) ===" | tee -a "$LOG_FILE"
echo "Args: $@" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

exec python main.py "$@"
