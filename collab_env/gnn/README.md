# GNN Training - Maximize GPU Utilization

Simple, efficient GNN training that maximizes GPU utilization by running multiple training jobs concurrently across all available GPUs.

## Files

- `train_maximize_gpu.py` - Main training script
- `gnn.py` - Core GNN training functions
- `gnn_definition.py` - Model definitions (GNN, Lazy)
- `utility.py` - Utility functions
- `requirements_gnn.txt` - Dependencies

## Quick Start

```bash
# Install dependencies
pip install -r collab_env/gnn/requirements_gnn.txt

# Quick test
python collab_env/gnn/train_maximize_gpu.py --test

# Full training on single dataset
python collab_env/gnn/train_maximize_gpu.py \
    --dataset boid_single_species_basic \
    --workers-per-gpu 3 \
    --seeds 5
```

## Key Features

- **Automatic GPU distribution** - Jobs automatically assigned to least loaded GPU
- **Multiple jobs per GPU** - Run 3-4 training jobs per GPU to maximize utilization
- **Simple and minimal** - No complex frameworks, just concurrent.futures
- **Single dataset focus** - Hyperparameter search on one dataset at a time

## Parameters

Training explores these hyperparameters:
- **Models**: `vpluspplus_a`, `lazy`
- **Noise levels**: 0, 0.005
- **Heads**: 1, 2, 3
- **Visual ranges**: 0.1, 0.5

## Usage Examples

```bash
# Basic usage (auto-detects GPUs)
python collab_env/gnn/train_maximize_gpu.py

# Specify dataset
python collab_env/gnn/train_maximize_gpu.py --dataset boid_food_strong

# Increase GPU utilization (4 jobs per GPU)
python collab_env/gnn/train_maximize_gpu.py --workers-per-gpu 4

# Multiple seeds for robustness
python collab_env/gnn/train_maximize_gpu.py --seeds 10

# Quick test
python collab_env/gnn/train_maximize_gpu.py --test
```

## Available Datasets

- `boid_single_species_basic`
- `boid_single_species_independent`
- `boid_food_basic_alignment`
- `boid_food_basic_independent`
- `boid_food_strong`

## Output

Results are saved as JSON files:
- `results_<dataset>_<timestamp>.json`

Models are saved in the current directory with naming:
- `<dataset>_<model>_n<noise>_h<heads>_vr<visual_range>_s<seed>.pt`

## GPU Utilization Tips

If each training uses ~30% of GPU:
- `--workers-per-gpu 3` → ~90% utilization
- `--workers-per-gpu 4` → may cause slight slowdown but more throughput

Monitor with `nvidia-smi` to find optimal workers-per-gpu for your setup.