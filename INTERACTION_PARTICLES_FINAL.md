# InteractionParticle Training - Final Cleaned Version

## Summary

Successfully implemented and then **simplified** InteractionParticle model training to work exclusively with **2D boids data** from `docs/gnn/0a-Simulate_Boid_2D.ipynb`. The code is now clean, focused, and easy to use.

## 🎯 What You Can Do Now

### One-Line Training

```bash
python -m collab_env.gnn.interaction_particles.train_2d_boids
```

That's it! The script will:
- Load `simulated_data/boid_single_species_basic.pt`
- Auto-detect `simulated_data/boid_single_species_basic_config.pt`
- Train for 50 epochs
- Generate plots comparing learned vs. true 2D boid rules
- Save results to `trained_models/interaction_particle_2d_*`

### Available Datasets

All ready to use in `simulated_data/`:
- `boid_single_species_basic.pt` ✅
- `boid_single_species_noisy.pt` ✅
- `boid_single_species_high_cluster_high_speed.pt` ✅
- `boid_single_species_short.pt` ✅

### Quick Test

```bash
python -m collab_env.gnn.interaction_particles.train_2d_boids --quick
```

Runs 10 epochs for rapid iteration.

## 📊 What Gets Learned

The model learns to predict 2D acceleration from:
- Relative positions between particles
- Relative velocities
- Pairwise distances
- Learnable particle embeddings

Then compares against **true 2D boid rules**:

| Rule | Type | Threshold | Weight |
|------|------|-----------|--------|
| **Separation** | Linear repulsion | 15 pixels | 0.05 |
| **Alignment** | Step function | 50 pixels | 0.5 |
| **Cohesion** | Linear attraction | 50 pixels | 0.005 |

## 🔧 What Changed (Simplification)

### Before (Initial Implementation)
- Supported 3D boids, 2D boids, raw tensors, dict format
- Complex format detection and normalization
- YAML config files for 3D, .pt for 2D
- Documentation covered both 2D and 3D
- ~400 lines with lots of branching

### After (Cleaned Up)
- **Only 2D boids** (`AnimalTrajectoryDataset`)
- Simple, direct data loading
- **Only .pt config files**
- Documentation focused on 2D
- ~280 lines, much clearer

**Code reduction: ~120 lines removed, 53 lines simplified**

## 📁 File Structure

```
collab_env/gnn/interaction_particles/
├── __init__.py               # Package exports
├── model.py                  # InteractionParticle GNN (226 lines)
├── train.py                  # Training pipeline (280 lines) ⬇️ simplified
├── plotting.py               # 2D boid comparison (350 lines) ⬇️ simplified
├── run_training.py           # Main CLI (230 lines) ⬇️ simplified
├── train_2d_boids.py         # Quick start (74 lines)
├── example.py                # Toy data example (139 lines)
└── README.md                 # Documentation (350 lines) ⬇️ simplified

Total: ~1,850 lines (down from ~1,970)
```

## 🚀 Usage Examples

### Basic Training

```bash
# Default dataset (basic)
python -m collab_env.gnn.interaction_particles.train_2d_boids

# Noisy dataset
python -m collab_env.gnn.interaction_particles.train_2d_boids \
    --dataset simulated_data/boid_single_species_noisy.pt

# 100 epochs
python -m collab_env.gnn.interaction_particles.train_2d_boids --epochs 100
```

### Advanced Options

```bash
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_basic.pt \
    --epochs 100 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 16 \
    --visual-range 0.104 \
    --save-dir trained_models/my_model
```

## 📈 Expected Results

After training, check `trained_models/interaction_particle_2d_<dataset>/`:

1. **training_history.png**: Loss curves
2. **interaction_functions.png**: Learned force vs. distance
3. **comparison_with_boids.png**: Side-by-side with true rules ⭐

The comparison plot shows:
- **Top row**: Learned forces (magnitude, X/Y components)
- **Middle row**: True separation (linear) + alignment (step)
- **Bottom row**: True cohesion (linear) + normalized overlay

Distances are in **pixels** (480×480 scene).

## 🔬 2D Boid Rules

From `collab_env/sim/boids_gnn_temp/boid.py`:

```python
# Separation: Linear repulsion
if distance < 15:  # min_distance
    force += 0.05 * (self_pos - other_pos)  # avoid_factor

# Alignment: Step function
if distance < 50:  # visual_range
    force += 0.5 * (avg_vel - self_vel)  # matching_factor

# Cohesion: Linear attraction
if distance < 50:  # visual_range
    force += 0.005 * (center - self_pos)  # centering_factor
```

## 💾 Data Format

`AnimalTrajectoryDataset` format:
```python
dataset[i] -> (positions, species)
    positions: [steps, N, 2]  # Normalized to [0, 1]
    species: [N]               # Species indices

# Example dimensions:
# steps = 800 frames
# N = 20 particles
# Scene normalized from 480×480 pixels
```

## ✅ Commits

| Commit | Description |
|--------|-------------|
| `6cb18b9` | Initial InteractionParticle implementation |
| `53d218b` | Implementation summary document |
| `c0ce7f9` | Adapted for 2D boids data format |
| `dedfc03` | 2D boids update documentation |
| **`bb4783c`** | **Simplified to only 2D boids** ⭐ |

## 🎓 Key Simplifications Made

### 1. Data Loading (`train.py`)
**Before:**
```python
if hasattr(data, '__getitem__'):
    # Handle dataset
elif isinstance(data, dict):
    # Handle dict
else:
    # Handle tensor
# Complex normalization logic
```

**After:**
```python
dataset = torch.load(dataset_path, weights_only=False)
for i in range(len(dataset)):
    pos, species = dataset[i]
    all_positions.append(pos.unsqueeze(0).numpy())
# Data already normalized, p_range = 1.0
```

### 2. Configuration (`run_training.py`)
**Before:**
```python
if config_path.endswith('.pt'):
    # 2D boids
elif config_path.endswith('.yaml'):
    # 3D boids
else:
    # Unknown
```

**After:**
```python
species_configs = torch.load(config_path, weights_only=False)
first_species = list(species_configs.keys())[0]
config = species_configs[first_species]
```

### 3. Plotting (`plotting.py`)
**Before:**
```python
# 2D boid rules
true_separation_2d = ...
# 3D boid rules
true_separation_3d = ...
# Choose based on config
```

**After:**
```python
# Only 2D boid rules
distances_pixels = distances_fine * 480.0
true_separation = boid_separation_force(distances_pixels, ...)
true_alignment = boid_alignment_force(distances_pixels, ...)
true_cohesion = boid_cohesion_force(distances_pixels, ...)
```

### 4. Documentation (`README.md`)
**Before:**
- "Supports both 2D and 3D boids!"
- Separate sections for 2D vs 3D
- Examples for both formats

**After:**
- "Training on 2D Boids Data"
- Single focused path
- Clear, simple examples

## 🔍 Visual Range Calculation

For 2D boids:
```python
# Boid visual_range = 50 pixels
# Scene size = 480×480 pixels
# Normalized visual_range = 50 / 480 ≈ 0.104

# Use in training:
--visual-range 0.104
```

The `train_2d_boids.py` script calculates this automatically.

## 🎯 Next Steps

1. **Run training**:
   ```bash
   python -m collab_env.gnn.interaction_particles.train_2d_boids --quick
   ```

2. **Check results**: Look in `trained_models/interaction_particle_2d_*/`

3. **Examine plots**: See how learned functions compare to true rules

4. **Try different datasets**: Test on noisy, high_cluster, etc.

5. **Tune hyperparameters**: Adjust hidden_dim, embedding_dim, epochs

## 📚 Documentation

- `README.md`: Complete usage guide
- `INTERACTION_PARTICLES_SUMMARY.md`: Original implementation notes
- `INTERACTION_PARTICLES_2D_UPDATE.md`: 2D adaptation details
- **`INTERACTION_PARTICLES_FINAL.md`**: This file (cleaned version)

## ✨ Summary

The code is now:
- ✅ **Simple**: One data format, one config format
- ✅ **Fast**: Direct loading, no format detection
- ✅ **Clear**: Focused documentation, obvious usage
- ✅ **Clean**: ~120 lines removed, much less branching
- ✅ **Ready**: Works with your existing 2D boids data

Just run:
```bash
python -m collab_env.gnn.interaction_particles.train_2d_boids
```

And you're off to the races! 🏁

---

**Branch**: `claude/add-interaction-particles-training-011CUNy9mcenSbXUQB83X3Eb`
**Latest Commit**: `bb4783c`
**Status**: ✅ Ready to use
