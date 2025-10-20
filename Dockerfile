# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG UBUNTU_MIRROR=https://mirrors.edge.kernel.org/ubuntu
ARG UBUNTU_SECURITY=https://mirrors.edge.kernel.org/ubuntu

RUN set -eux; \
    sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list || true; \
    sed -i "s|http://security.ubuntu.com/ubuntu|${UBUNTU_SECURITY}|g" /etc/apt/sources.list || true; \
    printf 'Acquire::Retries "5";\nAcquire::https::Timeout "60";\nAcquire::http::Timeout "60";\nAcquire::ForceIPv4 "true";\n' >/etc/apt/apt.conf.d/99retries; \
    for i in 1 2 3 4 5; do \
        if apt-get update; then break; fi; \
        echo "apt-get update failed (attempt $i), retrying in 10s..."; \
        sleep 10; \
    done; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential git unzip libgl1 libglib2.0-0 cmake ffmpeg wget ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

ENV PATH="/usr/local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}" \
    CUDA_HOME=/usr/local/cuda \
    MPLCONFIGDIR=/workspace/.mplconfig \
    TF_CPP_MIN_LOG_LEVEL=1 \
    GRADIO_SERVER_NAME=0.0.0.0

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cuda-runtime-cu12==12.6.77 \
        nvidia-cublas-cu12==12.6.3.3 && \
    pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
        tensorflow==2.18.0 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "script_launcher.py"]
