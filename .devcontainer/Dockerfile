FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    curl \
    cmake \
    git \
    ssh \
    sudo \
    mesa-utils \
    wget \
    htop \
    tmux \
    nano \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

ARG PIXI_VERSION=v0.66.0

RUN curl -L -o /usr/local/bin/pixi -fsSL --compressed "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    && chmod +x /usr/local/bin/pixi \
    && pixi info


# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Create and set proper permissions for workspace directory
RUN mkdir -p /workspaces && \
    chmod -R 755 /workspaces && \
    chown -R $USERNAME:$USERNAME /workspaces

USER vscode
WORKDIR /workspaces

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# prevent OOM: https://docs.jax.dev/en/latest/gpu_memory_allocation.html#common-causes-of-oom-failures
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false