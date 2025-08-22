#!/bin/bash
# Generic optimized training script for multi-GPU setups

# Set environment for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d ' ')
else
    NUM_GPUS=0
    GPU_NAME="CPU"
fi

echo "Starting optimized training"
echo "================================================"
echo "Detected: ${NUM_GPUS}x ${GPU_NAME}"
echo "================================================"

# Default parameters optimized for multi-GPU setup
DATASET="${1:-boid_single_species_basic}"
BATCH_SIZE="${2:-500}"
WORKERS_PER_GPU="${3:-4}"
EPOCHS="${4:-20}"
SEEDS="${5:-5}"

# Calculate total workers
if [ "$NUM_GPUS" -gt 0 ]; then
    TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
else
    TOTAL_WORKERS=4  # Default for CPU
fi

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Batch size: $BATCH_SIZE"
echo "  Workers per GPU: $WORKERS_PER_GPU"
echo "  Total workers: $TOTAL_WORKERS"
echo "  Epochs: $EPOCHS"
echo "  Seeds: $SEEDS"
echo ""

# Build command with GPU-specific optimizations
CMD="python collab_env/gnn/train_maximize_gpu.py"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --workers-per-gpu $WORKERS_PER_GPU"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --seeds $SEEDS"

# Add GPU optimizations if GPUs are available
if [ "$NUM_GPUS" -gt 0 ]; then
    # CMD="$CMD --compile"  # Disabled: compilation slower for variable graph sizes
    CMD="$CMD --memory-fraction 0.85"
    LOG_FILE="training_${NUM_GPUS}x_gpu_$(date +%Y%m%d_%H%M%S).log"
else
    CMD="$CMD --cpu-only --cpu-optimize"
    LOG_FILE="training_cpu_$(date +%Y%m%d_%H%M%S).log"
fi

# Run training with logging
echo "Executing: $CMD"
echo ""
$CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete! Logs saved to logs/ directory and results saved to results/ directory"