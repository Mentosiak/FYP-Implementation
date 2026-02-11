# Base image with CUDA + cuDNN (adjust versions to your driver)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Symlink python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working dir
WORKDIR /workspace

# Copy requirements first (for better build caching)
COPY requirements.txt .

# Install Python deps (CPU+GPU wheels will be picked automatically for torch)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command: bash shell
CMD ["/bin/bash"]
