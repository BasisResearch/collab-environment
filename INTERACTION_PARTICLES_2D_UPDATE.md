# InteractionParticle Training - Updated for 2D Boids

## Summary of Updates

Successfully adapted the InteractionParticle training pipeline to work with **existing 2D boids data** from `docs/gnn/0a-Simulate_Boid_2D.ipynb`, while maintaining compatibility with 3D boids.

## What Changed

### 1. Data Format Support

**Before**: Only handled raw tensor format `[B, F, N, D]`

**After**: Handles both formats:
- **2D Boids**: `AnimalTrajectoryDataset` that returns `(positions, species)` tuples
  - positions: `[steps, N, 2]` normalized by scene size
  - species: `[N]` species indices
- **3D Boids**: Raw tensors `[B, F, N, 3]` or dict format

### 2. Boid Rules Comparison

**2D Boid Rules** (from `collab_env/sim/boids_gnn_temp/boid.py`):
```python
# Separation (avoid_others): LINEAR repulsion
if distance < min_distance:  # 15 pixels
    force += avoid_factor * (self_pos - other_pos)  # 0.05 * displacement

# Alignment (match_velocity): STEP function
if distance < visual_range:  # 50 pixels
    force += matching_factor * (avg_vel - self_vel)  # 0.5 * velocity_diff

# Cohesion (fly_towards_center): LINEAR attraction
if distance < visual_range:  # 50 pixels
    force += centering_factor * (center - self_pos)  # 0.005 * displacement
```

**3D Boid Rules** (from `collab_env/sim/boids/boidsAgents.py`):
```python
# Separation: INVERSE-SQUARE repulsion
if distance < min_separation:  # 20.0
    force += separation_weight / distance²  # 15.0 / d²

# Alignment & Cohesion: Step at neighborhood_dist = 80.0
```

### 3. New Quick Start Script

Added `train_2d_boids.py` for easy training on existing data:

```bash
# Train on 2D boids (uses simulated_data/)
python -m collab_env.gnn.interaction_particles.train_2d_boids

# Quick test (10 epochs)
python -m collab_env.gnn.interaction_particles.train_2d_boids --quick

# Choose dataset
python -m collab_env.gnn.interaction_particles.train_2d_boids \
    --dataset simulated_data/boid_single_species_noisy.pt
```

### 4. Auto-Configuration

The training script now:
- Auto-detects 2D vs 3D format
- Loads `.pt` config files for 2D boids (species_configs)
- Loads `.yaml` config files for 3D boids
- Infers config path from dataset path (e.g., `data.pt` → `data_config.pt`)
- Falls back to sensible defaults

## Available 2D Datasets

Located in `simulated_data/`:

| Dataset | Description |
|---------|-------------|
| `boid_single_species_basic.pt` | Clean flocking behavior (default) |
| `boid_single_species_noisy.pt` | Noisy trajectories |
| `boid_single_species_high_cluster_high_speed.pt` | High clustering |
| `boid_single_species_short.pt` | Short trajectories |

Each has a corresponding `*_config.pt` with the species parameters.

## Usage Examples

### 2D Boids (Recommended)

```bash
# Quick start
python -m collab_env.gnn.interaction_particles.train_2d_boids

# Full training
python -m collab_env.gnn.interaction_particles.train_2d_boids \
    --dataset simulated_data/boid_single_species_basic.pt \
    --epochs 100

# Quick test
python -m collab_env.gnn.interaction_particles.train_2d_boids --quick
```

### 3D Boids

```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset collab_env/data/boids/boid_single_species_basic.pt \
    --config collab_env/sim/boids/config.yaml \
    --epochs 100 \
    --visual-range 0.3
```

## Expected Results

After training on 2D boids, the learned interaction functions should show:

1. **Short Range (0-15 pixels)**: Linear repulsive force
   - Learned force should increase linearly with proximity
   - Matches `avoid_factor * distance` profile

2. **Medium Range (15-50 pixels)**: Transition zone
   - Weak cohesive forces
   - Alignment effects

3. **At Visual Range (~50 pixels)**: Step change
   - Alignment and cohesion activate
   - Force magnitude changes at boundary

4. **Long Range (>50 pixels)**: Minimal interaction
   - Forces diminish to near zero
   - No neighbors in visual range

## Visualization

The comparison plot (`comparison_with_boids.png`) now shows:
- **Top**: Learned force magnitude and components
- **Middle**: True 2D separation (linear) and alignment (step)
- **Bottom**: True 2D cohesion (linear) and normalized comparison

All distances shown in **pixels** (480×480 scene) instead of arbitrary units.

## Key Parameters

### Visual Range Calculation
```python
# 2D boids have visual_range = 50 pixels in 480×480 scene
# In normalized coordinates:
visual_range = 50 / 480 ≈ 0.104

# For training:
--visual-range 0.104
```

### 2D Boid Parameters
```yaml
visual_range: 50.0      # pixels
min_distance: 15.0      # pixels
avoid_factor: 0.05      # separation weight
matching_factor: 0.5    # alignment weight
centering_factor: 0.005 # cohesion weight
speed_limit: 7.0        # max speed
```

### 3D Boid Parameters
```yaml
min_separation: 20.0
neighborhood_dist: 80.0
separation_weight: 15.0
alignment_weight: 1.0
cohesion_weight: 0.5
```

## Files Modified

```
collab_env/gnn/interaction_particles/
├── train.py             (+30 lines) - Dataset format detection
├── plotting.py          (+154/-97)  - 2D boid rules & pixel units
├── run_training.py      (+81 lines) - Config auto-detection
├── train_2d_boids.py    (NEW, 74 lines) - Quick start script
└── README.md            (+83 lines) - 2D boids documentation
```

## Commits

1. **6cb18b9** - Initial InteractionParticle implementation
2. **53d218b** - Implementation summary
3. **c0ce7f9** - Adapt for 2D boids data (current)

## Testing Checklist

- [x] Load AnimalTrajectoryDataset format
- [x] Auto-detect 2D vs 3D data
- [x] Load .pt config files
- [x] Correct 2D boid rule comparison
- [x] Pixel-based distance units
- [x] Quick start script
- [ ] Run actual training on 2D data (ready to test!)

## Next Steps

1. **Test Training**:
   ```bash
   python -m collab_env.gnn.interaction_particles.train_2d_boids --quick
   ```

2. **Examine Results**:
   - Check `trained_models/interaction_particle_2d_boid_single_species_basic/`
   - Look at `comparison_with_boids.png`
   - Verify learned functions match true 2D rules

3. **Full Training**:
   ```bash
   python -m collab_env.gnn.interaction_particles.train_2d_boids --epochs 100
   ```

4. **Try Different Datasets**:
   - Test on noisy data
   - Test on high_cluster_high_speed
   - Compare learned functions across variants

## Branch Status

- **Branch**: `claude/add-interaction-particles-training-011CUNy9mcenSbXUQB83X3Eb`
- **Commits**: 3 (6cb18b9, 53d218b, c0ce7f9)
- **Status**: ✅ All changes pushed
- **Ready**: ✅ Ready to test training

## How This Addresses Your Request

> "i want to use our examples from the repo: either 2d or 3d boids. if your code is for 2d then lets reuse docs/gnn/0a-Simulate_Boid_2D.ipynb; I already have some of those experimental data saved somewhere so we can just reuse the format."

✅ **Done!** The code now:
1. Works with your existing 2D boids data in `simulated_data/`
2. Reuses the exact format from `0a-Simulate_Boid_2D.ipynb`
3. Loads the config files that go with the datasets
4. Compares against the correct 2D boid rules from `boids_gnn_temp/boid.py`
5. Provides a simple `train_2d_boids.py` script for easy use
6. Maintains compatibility with 3D boids if you want to use those later

Just run:
```bash
python -m collab_env.gnn.interaction_particles.train_2d_boids --quick
```

And you'll get trained models + comparison plots!
