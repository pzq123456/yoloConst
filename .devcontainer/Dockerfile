# 基于 PyTorch 官方 CUDA 镜像，确保环境一致性
FROM pytorch/pytorch:2.11.0-cuda12.1-cudnn8-runtime

# 环境变量设置：优化高性能计算与日志输出
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    DEBIAN_FRONTEND=noninteractive \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1

# 1. 安装系统依赖
# 2. 安装 uv (极速依赖管理器)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.cargo/env && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 将 uv 加入环境变量
ENV PATH="/root/.cargo/bin:$PATH"

# 工作目录
WORKDIR /workspace

COPY pyproject.toml .
RUN uv pip install --system --no-cache ultralytics