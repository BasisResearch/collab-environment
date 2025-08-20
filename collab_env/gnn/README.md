# GNN Multi-GPU Training Pipeline

This directory contains scripts for training Graph Neural Networks (GNNs) with multi-GPU support and hyperparameter optimization using Optuna.

## Files

- `train_cli.py` - **Main CLI script** for all training modes
- `train_gnn_optuna.py` - Core Optuna hyperparameter optimization module
- `train_gnn_simple.py` - Core module for training all combinations
- `requirements_gnn.txt` - Python dependencies

## Quick Start

### Installation

```bash
# Clone repository
cd /workspace  # For RunPod
git clone <your-repo>
cd collab-environment

# Install dependencies
pip install -r collab_env/gnn/requirements_gnn.txt
```

### Using the Python CLI

```bash
# Quick test (minimal parameters)
python collab_env/gnn/train_cli.py test

# Simple training without Optuna (all combinations)
python collab_env/gnn/train_cli.py simple --epochs 20 --seeds 5

# Hyperparameter search with Optuna
python collab_env/gnn/train_cli.py search --trials 50 --epochs 20

# Train with best parameters (after search)
python collab_env/gnn/train_cli.py train --epochs 50 --seeds 5

# Complete pipeline (search + train)
python collab_env/gnn/train_cli.py both --trials 50 --epochs 20 --seeds 5

# Get help
python collab_env/gnn/train_cli.py -h
```

### CLI Options

```
Modes:
  test    - Quick test with minimal parameters (1 dataset, 2 epochs)
  simple  - Train all parameter combinations without optimization
  search  - Optuna hyperparameter search only  
  train   - Train with best parameters (requires previous search)
  both    - Complete pipeline (search + train)

Options:
  --trials, -t    Number of Optuna trials (default: 50)
  --epochs, -e    Number of training epochs (default: 20)  
  --seeds, -s     Number of random seeds (default: 5)
  --batch-size, -b  Batch size (default: 4)
  --gpus, -g      Number of GPUs to use (default: auto-detect)
  --no-cuda       Disable CUDA even if available
```

## Parameters

The training covers these configurations:
- **Models**: `vpluspplus_a`, `lazy`
- **Noise levels**: 0, 0.005
- **Heads**: 1, 2, 3
- **Visual ranges**: 0.1, 0.5
- **Datasets**: 
  - `boid_single_species_basic`
  - `boid_single_species_independent`
  - `boid_food_basic_alignment`
  - `boid_food_basic_independent`
  - `boid_food_strong`

## Modes

- `test` - Quick test run (2 epochs, 1 seed)
- `simple` - Train all combinations without optimization
- `search` - Optuna hyperparameter search only
- `train` - Train with best parameters (requires previous search)
- `both` - Complete pipeline (search + train)

## Output

Results are saved to:
- `models/` - Trained model checkpoints
- `results/` - Training summaries and metrics
- `logs/` - Training logs
- `/workspace/` - RunPod persistent storage (if available)

## Multi-GPU Support

The scripts automatically detect and use all available GPUs. For RunPod:
- Single GPU instances: Regular training
- Multi-GPU instances: Distributed Data Parallel (DDP) training

## Monitor Training

```bash
# Watch training progress
tail -f logs/training_*.log

# Check GPU usage
watch nvidia-smi

# View results
cat results/*/training_summary.json
```

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size in the script
2. **Module not found**: Ensure you're in the project root when running scripts
3. **Permission denied**: Run `chmod +x collab_env/gnn/*.sh`

## Example RunPod Setup

```bash
# In RunPod terminal
cd /workspace
git clone <your-repo>
cd collab-environment

# Install dependencies
pip install torch torch-geometric optuna tqdm numpy matplotlib

# Run quick test
./collab_env/gnn/run_runpod.sh test

# If successful, run full training
./collab_env/gnn/run_runpod.sh both 50 20 5
```