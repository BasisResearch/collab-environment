#Use Nerfstudio as main base and copy what you need
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5

# Install additional tools useful for RunPod
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    git \
    vim \
    htop \
    tmux \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install all Python dependencies in a single layer
RUN python3 -m pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
    
# Needs to be run outside the requirements.txt file 
RUN python3 -m pip install --no-deps timm==1.0.13

# # Install miniconda properly
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh

# Set environment variables properly using ENV
# ENV PATH=/opt/conda/bin:$PATH
ENV TORCH_HOME=/workspace/models
ENV HF_HOME=/workspace/models

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

WORKDIR /workspace/

# Add conda initialization and environment activation to bashrc
RUN echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc && \
    echo 'export TORCH_HOME="/workspace/models"' >> ~/.bashrc &&\
    echo 'export HF_HOME="/workspace/models"' >> ~/.bashrc &&\
    # echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc &&\
    echo 'cd /workspace/' >> ~/.bashrc

CMD bash -c "\
apt update && \
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"