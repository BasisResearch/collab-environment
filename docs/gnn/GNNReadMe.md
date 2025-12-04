# GNN Training and Rollouts

All GNN scripts are in `collab_env/gnn/`.

## Files

- `train.py` - Main training script
- `gnn.py` - Core GNN training functions
- `gnn_definition.py` - Model definitions
- `utility.py` - Utility functions

## Quick Start

**Quick test:**

```bash
python collab_env/gnn/train.py --test
```

**Full training on single dataset:**

```bash
python collab_env/gnn/train.py \
    --dataset boid_single_species_basic \
    --workers-per-gpu 3 \
    --seeds 5
```

## Key Features

- **Automatic GPU distribution** - Jobs automatically assigned to least loaded GPU
- **Multiple jobs per GPU** - Run 3-4 training jobs per GPU to maximize utilization
- **Simple and minimal** - No complex frameworks, just concurrent.futures
- **Single dataset focus** - Hyperparameter search on one dataset at a time

## Hyperparameters

Training explores these hyperparameters:
- **Noise levels**: 0, 0.005
- **Heads**: 1, 2, 3
- **Visual ranges**: 0.1, 0.5

## Setup

See the main [README](../../README.rst) for environment setup instructions.

## Folder Structure

The training pipeline uses specific folder conventions:

```text
project_root/
├── simulated_data/
│   └── runpod/
│       ├── <dataset>.pt           # Input: Dataset file
│       └── <dataset>_config.pt    # Input: Species configuration
│
├── trained_models/
│   └── runpod/
│       └── <dataset>/
│           ├── trained_models/    # Output: Trained model files
│           │   ├── <name>.pt
│           │   ├── <name>_model_spec.pt
│           │   └── <name>_train_spec.pt
│           └── rollouts/          # Output: Rollout results
│               └── <name>.pkl
│
├── logs/                          # Training logs
└── results/                       # JSON results files
```

**File naming convention:**

```text
<dataset>_<model>_n<noise>_h<heads>_vr<visual_range>_s<seed>[_selfloops][_rp].pt
```

Example: `boid_food_basic_vpluspplus_a_n0_h1_vr0.1_s0.pt`

### Download data from Google Cloud

Requires **rclone** configured for GCS access - see the main [README](../../README.rst) for setup.

```bash
# Copy dataset and config to simulated_data/runpod/
rclone copy collab_simulated_data:boid_food_basic.pt simulated_data/runpod/ --gcs-bucket-policy-only
rclone copy collab_simulated_data:boid_food_basic_config.pt simulated_data/runpod/ --gcs-bucket-policy-only
```

## Training

Run training with 10 seeds for each hyperparameter combination:

```bash
python collab_env/gnn/train.py --dataset boid_food_basic --batch-size 50 --train-size 0.7 --seeds 10
```

Increase GPU utilization (4 jobs per GPU):

```bash
python collab_env/gnn/train.py --workers-per-gpu 4
```

### Output Files

The training code saves 3 files per model in `trained_models/runpod/<dataset>/trained_models/`:

| File | Description |
|------|-------------|
| `<name>.pt` | Trained model weights |
| `<name>_model_spec.pt` | Model specification (heads, architecture) |
| `<name>_train_spec.pt` | Training specification (visual_range, noise, epochs) |

## Rollouts

Run a rollout starting from frame 5 for 100 total frames:

```bash
python collab_env/gnn/train.py --dataset boid_food_basic --batch-size 50 --rollout 5 --total_rollout 100
```

Rollout results are saved to `trained_models/runpod/<dataset>/rollouts/` as `.pkl` files.

## Available Datasets

- `boid_single_species_basic`
- `boid_single_species_independent`
- `boid_food_basic_alignment`
- `boid_food_basic_independent`
- `boid_food_strong`


## Kill running processes spawned

```bash
ps -Af | grep multiproc | grep '[p]ython -c from multiprocessing.spawn' | awk '{print $2}' | xargs -r kill
```
