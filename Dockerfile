# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"
ARG NERFSTUDIO_VERSION=""

##################################################
#           Builder stage (for compilation)      #
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
ARG CUDA_ARCHITECTURES

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        unzip \
        build-essential \
        g++-11 \
        gcc-11 \
        cmake \
        ninja-build \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install conda in builder stage
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Copy environment files
COPY env.yml /tmp/env.yml
COPY requirements.txt /tmp/requirements.txt

# Set CUDA architectures
ENV TCNN_CUDA_ARCHITECTURES="90;89;86;80;75;70;61"

# Build everything in conda environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -n nerfstudio -f /tmp/env.yml && \
    conda activate nerfstudio && \
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    conda install -c 'nvidia/label/cuda-11.8.0' cuda-toolkit -y && \
    pip install setuptools==69.5.1 'numpy<2.0.0' && \
    pip install -v ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && \
    export TORCH_CUDA_ARCH_LIST=\"\$(echo \"${CUDA_ARCHITECTURES}\" | tr ';' '\n' | awk '\$0 > 70 {print substr(\$0,1,1)\".\"substr(\$0,2)}' | tr '\n' ' ' | sed 's/ \$//')\" && \
    pip install git+https://github.com/brian-xu/gsplat-rade.git && \
    pip install nerfstudio && \
    pip install git+https://github.com/vuer-ai/feature-splatting && \
    pip install -r /tmp/requirements.txt"

##################################################
#           Get pre-built components             #
##################################################

# Get conda from official image
FROM continuumio/miniconda3:latest as conda-source

# Get nerfstudio components
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5 as nerfstudio

##################################################
#           Runtime stage                        #
##################################################

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libc6 \
        libceres2 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v5 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        python3.10 \
        python3.10-dev \
        python-is-python3 \
        ffmpeg \
        wget \
        curl \
        unzip \
        git \
        vim \
        htop \
        tmux \
        openssh-server \
        && rm -rf /var/lib/apt/lists/*

# Copy conda installation from conda-source
COPY --from=conda-source /opt/conda/ /opt/conda

# Copy compiled conda environment from builder
COPY --from=builder /opt/conda/envs/nerfstudio/ /opt/conda/envs/nerfstudio/

# Copy colmap from nerfstudio
COPY --from=nerfstudio /usr/local/bin/colmap /usr/local/bin/
COPY --from=nerfstudio /usr/local/lib/libcolmap* /usr/local/lib/

# Set environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV TORCH_HOME="/workspace/models"
ENV HF_HOME="/workspace/models"

# SSH configuration
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# Add environment setup to bashrc
RUN echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc && \
    echo 'export TORCH_HOME="/workspace/models"' >> ~/.bashrc && \
    echo 'export HF_HOME="/workspace/models"' >> ~/.bashrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate nerfstudio' >> ~/.bashrc && \
    echo 'cd /workspace/' >> ~/.bashrc

WORKDIR /workspace

CMD bash -c "\
apt update && \
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"