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
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
echo "PermitTTY yes" >> /etc/ssh/sshd_config && \
echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

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

# CMD ["/start.sh"]