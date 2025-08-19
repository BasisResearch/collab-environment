#!/bin/bash

# Check for NVIDIA GPU
echo "Checking NVIDIA GPU..."
nvidia-smi || { echo "❌ NVIDIA GPU not found. Exiting."; exit 1; }

# Download and install Miniconda
echo "Downloading Miniconda..."
curl -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3.sh -b -p $HOME/miniconda

# Add Conda to PATH
export PATH="$HOME/miniconda/bin:$PATH"

# Initialize Conda
echo "Initializing Conda..."
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init bash

# Accept Conda Terms of Service
echo "Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || { echo "❌ Failed to accept ToS for pkgs/main. Exiting."; exit 1; }
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || { echo "❌ Failed to accept ToS for pkgs/r. Exiting."; exit 1; }

# Create Conda environment
echo "Creating Conda environment..."
conda create -y -n idtrackerai python=3.10 || { echo "❌ Failed to create Conda environment. Exiting."; exit 1; }

# Install PyTorch and torchvision
echo "Installing PyTorch and torchvision..."
$HOME/miniconda/envs/idtrackerai/bin/pip install torch torchvision || { echo "❌ Failed to install PyTorch. Exiting."; exit 1; }

# Install idtrackerai
echo "Installing idtrackerai..."
$HOME/miniconda/envs/idtrackerai/bin/pip install idtrackerai || { echo "❌ Failed to install idtrackerai. Exiting."; exit 1; }

# Test idtrackerai installation
echo "Testing idtrackerai installation..."
$HOME/miniconda/envs/idtrackerai/bin/idtrackerai_test || { echo "❌ idtrackerai test failed. Check installation."; exit 1; }

echo "✅ Environment setup complete!"
echo "To activate the environment, run:"
echo "  eval \"\$($HOME/miniconda/bin/conda shell.bash hook)\""
echo "  conda activate idtrackerai"