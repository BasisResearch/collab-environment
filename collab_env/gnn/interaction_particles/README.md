# InteractionParticle Model Training on Boids Data

This module implements training of the InteractionParticle model on boids trajectory data. The InteractionParticle model learns interaction functions between particles as a function of their relative distances and velocities.

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

### Basic Training

Train the model on boids data:

```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset collab_env/data/boids/boid_single_species_basic.pt \
    --epochs 100 \
    --batch-size 32 \
    --save-dir trained_models/interaction_particle
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

**Quick Training (50 epochs):**
```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --epochs 50 \
    --batch-size 64
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
    --model-path trained_models/interaction_particle/best_model.pt
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

The module compares learned interaction functions with the true boid rules implemented in the simulator:

### True Boid Rules (from `collab_env/sim/boids/boidsAgents.py`)

1. **Separation** (line 318-322):
   - Distance threshold: `min_separation` = 20.0
   - Force: `separation_weight / distance²` = 15.0 / d²
   - Effect: Inverse square repulsion at close range

2. **Alignment** (line 331-334):
   - Distance threshold: `neighborhood_dist` = 80.0
   - Force: Steer towards average velocity of neighbors
   - Weight: `alignment_weight` = 1.0

3. **Cohesion** (line 331-334, 378-387):
   - Distance threshold: `neighborhood_dist` = 80.0
   - Force: Steer towards center of mass of neighbors
   - Weight: `cohesion_weight` = 0.5

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

The model expects boids trajectory data in PyTorch format:

```python
# Input: Position tensor
position: torch.Tensor  # Shape: [B, F, N, D]
# B = number of trajectories
# F = number of frames
# N = number of particles
# D = spatial dimension (2)
```

The training script automatically computes velocities and accelerations using finite differences or spline fitting.

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

## API Usage

You can also use the module programmatically:

```python
from collab_env.gnn.interaction_particles import (
    InteractionParticle,
    train_interaction_particle,
    evaluate_model,
    plot_interaction_functions,
    compare_with_true_boids
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

# Evaluation
metrics = evaluate_model(model, 'test_data.pt')
print(f"Test RMSE: {metrics['rmse']}")
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
