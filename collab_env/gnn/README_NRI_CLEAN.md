# Neural Relational Inference (NRI) - Clean Implementation

A consolidated, modern PyTorch implementation of Neural Relational Inference for learning interaction patterns in boids data.

## Overview

Neural Relational Inference learns to infer dynamic interaction graphs between agents by predicting future trajectories. The model uses a variational autoencoder approach with graph neural networks to discover which agents interact with each other.

## File Structure

```
collab_env/gnn/
├── nri_main.py          # Single entry point script
├── nri_modules.py       # Core NRI neural network modules  
├── nri_model.py         # Complete NRI model and boids integration
├── nri_training.py      # Training utilities and data loading
├── nri_visualization.py # Rollout generation and visualization
└── README_NRI_CLEAN.md  # This file
```

## Usage

### Quick Check (Fast Testing)
Test the implementation with a small subset of data:

```bash
python collab_env/gnn/nri_main.py \
    --num-sequences 10 \
    --epochs 1 \
    --batch-size 16 \
    --rollout-steps 50
```

**What this does:**
- Uses only 10 training sequences (very fast)
- Trains for 1 epoch (~30 seconds)
- Generates 50-step rollout predictions
- Creates visualization showing learned interactions

### Full Training (Production Results)
Train on complete dataset for best results:

```bash
python collab_env/gnn/nri_main.py \
    --num-sequences 100 \
    --epochs 50 \
    --batch-size 32 \
    --hidden-dim 128 \
    --rollout-steps 100
```

**What this does:**
- Uses 100 training sequences (good balance of speed/performance)
- Trains for 50 epochs (5-10 minutes)
- Uses stable model architecture (128 hidden dimensions)
- Generates longer rollout predictions

### Visualization Only
If you already have a trained model and just want to generate new visualizations:

```bash
python collab_env/gnn/nri_main.py --visualize-only
```

## Outputs

All outputs are saved to `trained_models/` subdirectories:

1. **Trained model**: `trained_models/nri_models/nri_model.pt`
2. **Static plot**: `trained_models/nri_outputs/nri_trajectories.png`  
3. **Animation**: `trained_models/nri_outputs/nri_rollout.mp4`

## What the Model Learns

The NRI model discovers:
- **Agent interactions**: Which boids influence each other's movement
- **Interaction types**: Different categories of influence (attraction, repulsion, etc.)
- **Dynamic patterns**: How interactions change over time
- **Predictive capability**: Future trajectories based on learned interactions

## Example Output

After training, you'll see results like:
```
Model parameters: 262,658
Training sequences: 50
Rollout steps: 100

Edge Statistics:
  Interaction probability: 0.153
  No interaction probability: 0.847
```

The static visualization shows:
- **Left panel**: Ground truth boid trajectories
- **Center panel**: Model's rollout predictions
- **Right panel**: Inferred interaction matrix (who interacts with whom)

## Advanced Options

### Custom Data Path
```bash
python collab_env/gnn/nri_main.py --data-path path/to/your/dataset.pt
```

### Model Architecture
```bash
python collab_env/gnn/nri_main.py \
    --hidden-dim 256 \
    --n-edge-types 3 \
    --dropout 0.2
```

### Training Parameters  
```bash
python collab_env/gnn/nri_main.py \
    --batch-size 64 \
    --lr 5e-4 \
    --seq-len 15 \
    --pred-len 1
```

## Programmatic Usage

You can also import and use the modules directly in your own code:

```python
from collab_env.gnn.nri_training import load_boids_dataset, train_model
from collab_env.gnn.nri_model import create_nri_model_for_boids
from collab_env.gnn.nri_visualization import generate_rollout

# Load your boids data
positions, velocities, species = load_boids_dataset('simulated_data/boid_single_species_basic.pt')

# Create NRI model
model, rel_rec, rel_send = create_nri_model_for_boids(num_boids=20)

# Generate predictions
rollout_pos, rollout_vel, edge_probs = generate_rollout(
    model, rel_rec, rel_send, positions[0:1], velocities[0:1], species[0:1]
)
```

## Implementation Notes

- **Modern PyTorch**: Uses current PyTorch APIs and best practices
- **Single entry point**: All functionality accessible through `nri_main.py`
- **Modular design**: Clean separation between training, modeling, and visualization
- **Proper normalization**: Handles boids data coordinate conversion correctly
- **GPU support**: Automatically detects and uses CUDA when available
- **Basic architecture**: Simplified encoder/decoder following original NRI paper for stable training
- **Stable gradients**: Uses gradient clipping and proper weight initialization to prevent training issues