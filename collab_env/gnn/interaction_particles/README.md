# InteractionParticle Model Training on 2D Boids Data

This module implements training of the InteractionParticle model on 2D boids trajectory data from `docs/gnn/0a-Simulate_Boid_2D.ipynb`. The InteractionParticle model learns interaction functions between particles as a function of their relative distances and velocities.

## Quick Start

Train on existing 2D boids data:

```bash
# Run the quick start script (10 epochs)
./collab_env/gnn/interaction_particles/train_2d_boids.sh

# Or call run_training directly
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_basic.pt \
    --epochs 100
```

Available 2D datasets:
- `boid_single_species_basic.pt` - Clean flocking behavior
- `boid_single_species_noisy.pt` - Noisy trajectories
- `boid_single_species_high_cluster_high_speed.pt` - High clustering
- `boid_single_species_short.pt` - Short trajectories

## Model Overview

The InteractionParticle model is based on the work from:
- Paper: [Interaction Networks for Learning about Objects, Relations and Physics (NeurIPS 2016)](https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html)
- Implementation adapted from: https://github.com/saalfeldlab/decomp-gnn

### Key Features

1. **Message Passing Architecture**: Uses PyTorch Geometric's message passing framework to learn particle interactions
2. **Learnable Interaction Functions**: MLPs learn the force functions based on:
   - Relative positions between particles
   - Relative velocities
   - Distance between particles
   - Learnable particle embeddings
3. **Graph Construction**: Builds dynamic graphs based on visual range (neighborhood distance)

### Model Components

- **Edge MLP** (`lin_edge`): Multi-layer perceptron that learns the interaction function
- **Particle Embeddings** (`a`): Learnable embeddings for each particle type
- **Message Function**: Computes interactions based on relative positions, velocities, and distances
- **Aggregation**: Sums messages from all neighbors to compute total force

## Installation

The module is part of the `collab_env` package. Ensure you have the following dependencies:

```bash
pip install torch torch_geometric loguru pyyaml matplotlib scipy
```

Or install the full environment:
```bash
pip install -e .
```

## Usage

### Quick Start Script

The easiest way to get started:

```bash
# Edit train_2d_boids.sh to uncomment the command you want
./collab_env/gnn/interaction_particles/train_2d_boids.sh
```

The script includes examples for:
- Quick test (10 epochs)
- Full training (100 epochs)
- Different datasets (noisy, high_cluster, short)
- High capacity models
- Custom parameters
- Evaluation only

### Direct Training

Call run_training.py directly:

```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_basic.pt \
    --config simulated_data/boid_single_species_basic_config.pt \
    --epochs 100 \
    --batch-size 32 \
    --visual-range 0.104 \
    --save-dir trained_models/my_model
```

### Command Line Arguments

**Data Arguments:**
- `--dataset`: Path to boids dataset (.pt file)
- `--config`: Path to boids config.yaml (for comparison with true rules)

**Training Arguments:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--train-split`: Train/validation split ratio (default: 0.8)
- `--visual-range`: Visual range for edge construction in normalized units (default: 0.3)
- `--seed`: Random seed (default: 42)

**Model Arguments:**
- `--hidden-dim`: Hidden dimension for MLP (default: 128)
- `--embedding-dim`: Particle embedding dimension (default: 16)
- `--n-layers`: Number of MLP layers (default: 3)
- `--n-particles`: Number of particles in the dataset (default: 20)

**Output Arguments:**
- `--save-dir`: Directory to save models and plots
- `--no-plot`: Skip generating plots
- `--device`: Device to use (cpu/cuda/cuda:0)

**Evaluation:**
- `--eval-only`: Only evaluate a saved model
- `--model-path`: Path to saved model checkpoint for evaluation

### Examples

See `train_2d_boids.sh` for ready-to-run examples, or run directly:

**Quick Test:**
```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_basic.pt \
    --epochs 10 \
    --visual-range 0.104
```

**Different Datasets:**
```bash
# Noisy trajectories
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_noisy.pt \
    --epochs 100

# High clustering
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_high_cluster_high_speed.pt \
    --epochs 100
```

**High Capacity Model:**
```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --hidden-dim 256 \
    --embedding-dim 32 \
    --n-layers 4 \
    --epochs 200
```

**Evaluate Saved Model:**
```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --eval-only \
    --model-path trained_models/interaction_particle_2d_*/best_model.pt
```

## Output

The training script generates the following outputs in `--save-dir`:

### Model Checkpoints
- `best_model.pt`: Best model (lowest validation loss)
- `final_model.pt`: Final model after all epochs

### Plots
- `training_history.png`: Training and validation loss curves
- `interaction_functions.png`: Learned interaction functions (force vs distance)
- `comparison_with_boids.png`: Comprehensive comparison with true boid rules

### Logs and Metadata
- `training.log`: Detailed training log
- `model_info.yaml`: Model configuration and final metrics

## Comparison with True Boid Rules

The module compares learned interaction functions with the true 2D boid rules implemented in `collab_env/sim/boids_gnn_temp/boid.py`.

### 2D Boid Rules

1. **Separation** (`avoid_others`, line 144-160):
   - Distance threshold: `min_distance` = 15 pixels
   - Force: `avoid_factor * (self_pos - other_pos)` = 0.05 * displacement
   - Effect: **Linear repulsion** at close range

2. **Alignment** (`match_velocity`, line 163-184):
   - Distance threshold: `visual_range` = 50 pixels
   - Force: `matching_factor * (avg_velocity - self_velocity)` = 0.5 * vel_diff
   - Effect: **Step function** - match velocity with neighbors in visual range

3. **Cohesion** (`fly_towards_center`, line 120-141):
   - Distance threshold: `visual_range` = 50 pixels
   - Force: `centering_factor * (center_of_mass - self_pos)` = 0.005 * displacement
   - Effect: **Linear attraction** towards center of neighbors

### Comparison Visualization

The `comparison_with_boids.png` plot shows:
- **Top Row**: Learned interaction force magnitude and components
- **Middle Row**: True separation and alignment rules
- **Bottom Row**: True cohesion rule and normalized comparison

The model should learn:
- **Short range**: Strong repulsive forces (separation)
- **Medium range**: Attractive forces (cohesion) and alignment
- **Long range**: Diminishing interactions

## Data Format

The model expects 2D boids data in `AnimalTrajectoryDataset` format:

```python
# Each sample is a tuple: (positions, species)
positions: torch.Tensor  # Shape: [steps, N, 2]
species: torch.Tensor    # Shape: [N]

# Positions are normalized to [0, 1] by scene size (480x480)
# steps = number of frames (e.g., 800)
# N = number of particles (e.g., 20)
```

The training script automatically:
- Loads all samples from the dataset
- Computes velocities and accelerations using finite differences
- Builds graphs based on visual range

## Model Architecture Details

### Node Features
For each particle:
```
[particle_id, pos_x, pos_y, vel_x, vel_y]
```

### Edge Features (Message Function)
For each edge from particle j to i:
```python
[
    delta_pos_x, delta_pos_y,  # Normalized relative position
    distance,                   # Normalized distance
    vel_i_x, vel_i_y,          # Normalized velocity of i
    vel_j_x, vel_j_y,          # Normalized velocity of j
    embedding_i                 # Learnable particle embedding
]
```

### Output
Predicted 2D acceleration for each particle:
```
[acc_x, acc_y]
```

## Rollout Evaluation

The model supports multi-step autoregressive rollout evaluation to test its ability to predict trajectories over time:

```bash
# Train with rollout evaluation
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_basic.pt \
    --epochs 100 \
    --evaluate-rollout \
    --n-rollout-steps 50 \
    --save-dir trained_models/with_rollout
```

### What is Rollout?

Instead of predicting just one step ahead, rollout uses the model's own predictions as input for the next step, generating entire trajectories. This tests whether the model has learned dynamics that are stable and accurate over time.

### Rollout Outputs

When `--evaluate-rollout` is enabled, the following are generated in `rollout_evaluation/`:

1. **Side-by-side trajectory comparisons** (`rollout_comparison_*.png`):
   - Left: Ground truth trajectories
   - Right: Model predicted trajectories
   - Shows first 3 validation samples
   - Green markers = start, Red markers = end

2. **Error over time** (`rollout_error_over_time.png`):
   - Position error as a function of time step
   - Shows mean and ±1 standard deviation across all validation trajectories
   - Helps identify if error accumulates over time

3. **Metrics** (logged to console):
   - Mean position error: Average Euclidean distance between predicted and true positions
   - Mean velocity error: Average velocity deviation
   - Statistics computed over all validation trajectories

### Expected Results

For well-trained models on 2D boids:
- Position error should remain relatively stable over ~50 timesteps
- Trajectories should maintain flock structure (not drift apart or collapse)
- Mean position error typically < 0.05 for first 20 steps, then gradually increases

### Validation Dataset

Rollout evaluation uses the **validation split** (last 20% of the dataset by default). This ensures the model is tested on trajectories it hasn't seen during training.

The split matches the training validation split (80/20 by default, configurable via `--train-split`).

## API Usage

You can also use the module programmatically:

```python
from collab_env.gnn.interaction_particles import (
    InteractionParticle,
    train_interaction_particle,
    evaluate_model,
    evaluate_rollout,
    generate_rollout,
    plot_interaction_functions,
    compare_with_true_boids,
    create_rollout_report
)

# Training
model, history = train_interaction_particle(
    dataset_path='collab_env/data/boids/boid_single_species_basic.pt',
    epochs=100,
    batch_size=32,
    save_dir='trained_models/my_model'
)

# Plotting
plot_interaction_functions(model, save_path='interaction.png')
compare_with_true_boids(model, save_path='comparison.png')

# One-step evaluation
metrics = evaluate_model(model, 'test_data.pt')
print(f"Test RMSE: {metrics['rmse']}")

# Multi-step rollout evaluation
rollout_results = evaluate_rollout(
    model,
    dataset_path='test_data.pt',
    visual_range=0.104,
    n_rollout_steps=50
)
print(f"Mean position error: {rollout_results['metrics']['mean_position_error']:.6f}")

# Create rollout visualizations
create_rollout_report(rollout_results, save_dir='rollout_results')

# Generate single rollout
import torch
initial_pos = torch.randn(20, 2)  # 20 particles
initial_vel = torch.randn(20, 2)
pred_pos, pred_vel = generate_rollout(model, initial_pos, initial_vel, n_steps=100)
```

## Performance Tips

1. **Visual Range**: Adjust `--visual-range` based on your data normalization. Too small = sparse graphs, too large = dense graphs
2. **Batch Size**: Larger batches can speed up training but require more memory
3. **Model Capacity**: Increase `--hidden-dim` and `--n-layers` for more complex interaction functions
4. **Training Time**: Expect ~2-5 minutes per epoch on GPU for typical boids datasets

## Troubleshooting

**Issue**: Model not learning / high loss
- Check data normalization
- Try increasing model capacity (`--hidden-dim 256`)
- Adjust visual range (`--visual-range 0.5`)
- Increase training epochs

**Issue**: Out of memory
- Reduce `--batch-size`
- Reduce `--visual-range` (fewer edges)
- Use CPU (`--device cpu`)

**Issue**: Learned functions don't match boid rules
- Check that boids config matches the data generation parameters
- Ensure sufficient training epochs
- Verify data quality (no NaNs, proper normalization)

## References

1. Battaglia, P. W., et al. (2016). Interaction networks for learning about objects, relations and physics. NeurIPS.
2. Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. SIGGRAPH.
3. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

## License

This code is part of the `collab_env` package and follows the same license (Apache-2.0).
