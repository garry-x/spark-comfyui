FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
    WHEELS_DIR=/app/wheels

# 替换 apt 源 + 安装依赖 + 解除 PEP 668 限制
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -f /usr/lib/python3*/EXTERNALLY-MANAGED \
    && python -m pip install --upgrade pip --break-system-packages --ignore-installed pip

WORKDIR /app

# ========== 阶段 1：优先使用本地 wheels ==========
COPY wheels ${WHEELS_DIR}

RUN pip install --no-cache-dir \
    --find-links ${WHEELS_DIR} \
    torch torchvision torchaudio \
    --extra-index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --find-links ${WHEELS_DIR} \
    -r requirements.txt

# ========== 阶段 2：业务代码 & custom_nodes 依赖 ==========
COPY . .

RUN find /app/custom_nodes -name "requirements.txt" -print 2>/dev/null | \
    while IFS= read -r req; do \
        echo "Installing: $req"; \
        pip install --no-cache-dir --find-links ${WHEELS_DIR} -r "$req"; \
    done || true

# ========== 阶段 3：启动入口 ==========
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8188

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
