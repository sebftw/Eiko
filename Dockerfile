# Use an NVIDIA CUDA development base image (includes the nvcc compiler)
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Git, and essential tools
RUN apt-get update && apt-get install -y \
    python3-full \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create the virtual environment
RUN python3 -m venv /opt/eiko
ENV PATH="/opt/eiko/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch matching the CUDA 12.1 container
RUN python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu132

# Install Eiko (and optional dependencies)
RUN python3 -m pip install eiko[examples]
RUN python3 -m pip install eiko[jax]

# Pre-compile Eiko C++/CUDA extensions
RUN python3 -c "import eiko.eiko_torch"

# Set the working directory
WORKDIR /workspace
CMD ["/bin/bash"]