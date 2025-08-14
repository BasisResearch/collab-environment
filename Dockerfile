# Docker multi-stage build
# syntax=docker/dockerfile:1
# Build with the following commands:
# docker build --platform=linux/amd64 --progress=plain -t tommybotch/collab-environment .

ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
# ARG NVIDIA_CUDA_VERSION=12.6.1
#11.8.0

##################################################
#           Get pre-built components             #
##################################################

# Get conda from official image
FROM continuumio/miniconda3:latest as conda-source

# For NVIDIA CUDA 11.8.0
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5 as colmap-source

# # For NVIDIA CUDA 12.3.0
# FROM colmap/colmap:20240213.23 as colmap-source

## For NVIDIA CUDA 12.6.1
# FROM colmap/colmap:20241128.1598 as colmap-source

##################################################
#           Runtime stage                        #
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ENV DEBIAN_FRONTEND=noninteractive

# Use faster apt mirror
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.math.princeton.edu/pub/ubuntu/|g' /etc/apt/sources.list

# Install additional tools useful for RunPod
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    wget \
    curl \
    unzip \
    git \
    vim \
    htop \
    tmux \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    build-essential \
    python3.10 \
    python3-pip \
    python-is-python3 \
    ffmpeg \
    libc6 \
    libgcc-s1 \
    libgl1 \
    libglew2.2 \
    libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 \
    libceres2 \
    libfreeimage3 \
    libgoogle-glog0v5 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

# Copy conda installation from conda-source
COPY --from=conda-source /opt/conda/ /opt/conda
COPY --from=colmap-source /usr/local/bin/colmap /usr/local/bin/colmap
COPY --from=colmap-source /usr/local/lib/libcolmap* /usr/local/lib/

# Copy env.yml to container
COPY env.yml /tmp/env.yml

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Set environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV TORCH_HOME="/workspace/models"
ENV HF_HOME="/workspace/models"

# Build everything in conda environment --> last step is to install buildtools
RUN /bin/bash -c \
    "source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -n collab-env -f /tmp/env.yml"

# SSH configuration
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# Add environment setup to bashrc
RUN echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc && \
    echo 'export TORCH_HOME="/workspace/models"' >> ~/.bashrc && \
    echo 'export HF_HOME="/workspace/models"' >> ~/.bashrc && \
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc && \
    echo 'export CUDA_ROOT=/usr/local/cuda' >> ~/.bashrc && \
    echo 'export PATH="/usr/local/cuda/bin:${PATH}"' >> ~/.bashrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate collab-env' >> ~/.bashrc

WORKDIR /workspace

CMD bash -c "\
apt update && \
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"