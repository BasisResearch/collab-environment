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

# Install miniconda properly
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PermitTTY yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# Initialize conda and install Python 3.10 + OpenCV in base environment
RUN conda init bash && \
    conda install -y -c conda-forge python=3.10 opencv && \
    conda clean -ya

# pip install git+https://github.com/vuer-ai/feature-splatting

RUN echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate base' >> ~/.bashrc

WORKDIR /workspace/

CMD bash -c "\
apt update && \
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
service ssh start && \
sleep infinity"