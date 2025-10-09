# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    cmake \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY . .
# Ensure images directory (and all subfolders) is included explicitly (in case of future .dockerignore changes)

# Optional: clone remote repository again (fresh) for comparison or updates
ARG REPO_URL="https://github.com/MISTERNEGATIVE21/CUSTOM_CNN_FPGA.git"
ARG REPO_REF=""
RUN echo "Cloning repo from ${REPO_URL}" && \
    git clone --depth 1 "${REPO_URL}" /workspace/repo_remote && \
    if [ -n "${REPO_REF}" ]; then \
        echo "Checking out ref ${REPO_REF}" && \
        cd /workspace/repo_remote && git fetch --depth 1 origin "${REPO_REF}" && git checkout "${REPO_REF}" ; \
    fi && \
    rm -rf /workspace/repo_remote/.git


ENV MPLCONFIGDIR=/workspace/.mplconfig \
    TF_CPP_MIN_LOG_LEVEL=1 \
    GRADIO_SERVER_NAME=0.0.0.0

RUN mkdir -p ${MPLCONFIGDIR}

EXPOSE 7860 6006 8501

CMD ["sleep", "infinity"]
