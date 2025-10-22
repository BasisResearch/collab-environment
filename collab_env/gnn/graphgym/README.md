# GraphGym Integration for Boids Trajectory Prediction

This module integrates [GraphGym](https://github.com/snap-stanford/GraphGym) for systematic GNN architecture search on boids trajectory data. GraphGym provides a modular framework to explore the GNN design space and find optimal architectures for your specific task.

## Overview

**Task**: Node-level regression to predict agent accelerations from trajectories

**Metrics**:
- MSE (Mean Squared Error) - Primary metric
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

**GNN Architectures Supported**:
- **GCNConv**: Graph Convolutional Network
- **GATv2Conv**: Graph Attention Network v2 (improved attention mechanism)
- **GINConv**: Graph Isomorphism Network
- **SAGEConv**: GraphSAGE
- **GeneralConv**: General GNN layer combining multiple mechanisms

## Quick Start

### 1. Test Installation

Run a quick test with minimal configuration:

```bash
python collab_env/gnn/run_graphgym_experiments.py --test
```

### 2. Single Experiment

Run a single experiment with GATv2:

```bash
python collab_env/gnn/run_graphgym_experiments.py \
    --config configs/graphgym/base/boids_trajectory.yaml \
    --layer-type gatv2conv \
    --visual-range 0.1
```

### 3. Architecture Search

Run grid search over multiple architectures (quick test):

```bash
python collab_env/gnn/run_graphgym_experiments.py \
    --config configs/graphgym/base/boids_trajectory.yaml \
    --grid configs/graphgym/grids/quick_test.txt \
    --max-workers 4
```

Run full architecture search:

```bash
python collab_env/gnn/run_graphgym_experiments.py \
    --config configs/graphgym/base/boids_trajectory.yaml \
    --grid configs/graphgym/grids/architecture_search.txt \
    --max-workers 8
```

## GraphGym Design Space

Based on the [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843) (NeurIPS 2020), GraphGym recommends exploring these dimensions:

### Recommended Search Space (Adapted for Boids)

```yaml
gnn:
  # Depth
  layers_pre_mp: [1]              # Pre-message passing layers
  layers_mp: [2, 3, 4]            # Message passing layers
  layers_post_mp: [1, 2]          # Post-message passing layers

  # Architecture
  dim_inner: [64, 128, 256]       # Hidden dimensions
  layer_type: ['gcnconv', 'gatv2conv', 'ginconv', 'sageconv']

  # Attention (for GATv2)
  heads: [1, 2, 3]                # Number of attention heads

  # Connections
  stage_type: ['stack', 'skipsum', 'skipconcat']
  agg: ['add', 'mean', 'max']

  # Regularization
  batchnorm: [True, False]
  dropout: [0.0, 0.1]

  # Task-specific
  visual_range: [0.1, 0.5]        # For edge construction
```

This gives ~100-200 configurations to compare systematically.

## Why GATv2 Instead of GAT?

**Important**: We use **GATv2Conv** instead of the original GATConv because:

- **GAT v1**: `LeakyReLU(a^T [Wh_i || Wh_j])` - Attention is essentially **static**
- **GAT v2**: `a^T LeakyReLU(W [h_i || h_j])` - Attention is **truly dynamic**

GATv2 is strictly more expressive and consistently outperforms GAT v1. There's no reason to use GAT v1 anymore.

## Configuration Files

### Base Config: `configs/graphgym/base/boids_trajectory.yaml`

Defines the task, dataset, training parameters, and default architecture:

```yaml
dataset:
  format: BoidsTrajectory
  name: boid_single_species_basic
  task: node
  task_type: regression
  visual_range: 0.1

model:
  loss_fun: mse

gnn:
  layer_type: gatv2conv
  dim_inner: 128
  heads: 1
  layers_mp: 2
```

### Grid Search: `configs/graphgym/grids/architecture_search.txt`

Specifies which parameters to search:

```
gnn.layer_type layer ['gcnconv','gatv2conv','ginconv','sageconv']
gnn.dim_inner dim [64,128,256]
gnn.layers_mp l_mp [2,3,4]
gnn.heads heads [1,2,3]
```

## Module Structure

```
collab_env/gnn/graphgym/
├── __init__.py          # Module exports
├── config.py            # Configuration system
├── dataset.py           # Data adapter for boids trajectories
├── models.py            # GNN model registry
├── trainer.py           # Training loop and metrics
└── README.md            # This file

configs/graphgym/
├── base/
│   └── boids_trajectory.yaml    # Base configuration
└── grids/
    ├── quick_test.txt           # Small grid for testing
    └── architecture_search.txt  # Full architecture search
```

## Data Format

The data adapter converts boids trajectories into temporal graphs:

**Input**: Trajectories `[time_steps, n_agents, 2]` with species labels

**Output**: PyG Data objects with:
- **Nodes**: Agents at timestep t
- **Node features**: `[pos_x, pos_y, vel_x, vel_y, species_one_hot]`
- **Node labels**: `[acc_x, acc_y]` (regression target)
- **Edges**: Based on visual range (spatial proximity)
- **Edge attributes**: Distance between agents

## Results

Results are saved in `results/graphgym/` with the following structure:

```
results/graphgym/
├── <exp_name>/
│   ├── best_model.pt
│   ├── metrics.json
│   └── checkpoint_*.pt
└── grid_search/
    ├── results_<timestamp>.json
    └── configs/
        └── config_*.yaml
```

## Comparing with Existing GNN

To compare GraphGym results with the existing custom GNN:

1. **Run existing GNN** (from `collab_env/gnn/train.py`):
```bash
python collab_env/gnn/train.py \
    --dataset boid_single_species_basic \
    --workers-per-gpu 1 \
    --seeds 5
```

2. **Run GraphGym search**:
```bash
python collab_env/gnn/run_graphgym_experiments.py \
    --config configs/graphgym/base/boids_trajectory.yaml \
    --grid configs/graphgym/grids/architecture_search.txt
```

3. **Compare metrics**:
   - Both use MSE loss on acceleration prediction
   - Both use visual range for edge construction
   - GraphGym provides systematic comparison across architectures

## Advanced Usage

### Custom Architecture

Create a custom config:

```python
from collab_env.gnn.graphgym.config import create_config

config = create_config(
    dataset_name="boid_single_species_basic",
    visual_range=0.1,
    layer_type="gatv2conv",
    dim_inner=256,
    heads=3,
    **{
        'gnn.layers_mp': 4,
        'gnn.dropout': 0.1,
        'gnn.stage_type': 'skipsum',
    }
)
```

### Programmatic Grid Search

```python
from collab_env.gnn.graphgym.config import load_config, generate_grid_configs
from collab_env.gnn.graphgym.trainer import train_model
from collab_env.gnn.graphgym.dataset import create_boids_datasets

# Load base config
base_config = load_config("configs/graphgym/base/boids_trajectory.yaml")

# Generate all configurations
configs = generate_grid_configs(
    base_config,
    grid_file="configs/graphgym/grids/quick_test.txt"
)

# Run experiments
for i, config in enumerate(configs):
    train_dataset, val_dataset, test_dataset = create_boids_datasets(
        dataset_name=config.dataset.name,
        visual_range=config.dataset.visual_range,
    )

    model, history = train_model(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
```

## Key Findings from GraphGym Paper

From the [NeurIPS 2020 paper](https://arxiv.org/abs/2011.08843) analyzing 10M model-task combinations:

1. **Depth matters**: 4-8 message passing layers work well
2. **Skip connections**: Significantly improve performance
3. **Batch normalization**: Always beneficial
4. **Activation**: PReLU > ReLU
5. **Architecture varies by task**: Best GNN design differs across tasks

Our adapted search space reflects these findings while being tailored to trajectory prediction.

## Citation

If you use GraphGym in your research, please cite:

```bibtex
@inproceedings{you2020design,
  title={Design Space for Graph Neural Networks},
  author={You, Jiaxuan and Ying, Rex and Leskovec, Jure},
  booktitle={NeurIPS},
  year={2020}
}
```

## Troubleshooting

**Out of memory**: Reduce batch size or hidden dimension:
```bash
python collab_env/gnn/run_graphgym_experiments.py \
    --config configs/graphgym/base/boids_trajectory.yaml \
    --layer-type gcnconv  # GCN uses less memory than GAT
```

**Slow training**: Use fewer message passing layers:
```yaml
gnn:
  layers_mp: 2  # Instead of 4
```

**Poor performance**: Try different visual ranges:
```bash
python collab_env/gnn/run_graphgym_experiments.py \
    --visual-range 0.5  # Increase connectivity
```
