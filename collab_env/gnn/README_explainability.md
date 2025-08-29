# GNN Explainability

Feature attribution analysis for GNN predictions using Captum's IntegratedGradients and Saliency methods.

## Quick Start

```bash
# Fast analysis with Saliency
python collab_env/gnn/explain_gnn_integrated_gradients.py --data-name boid_single_species_weakalignment_large --method saliency --max-frames 10

# Accurate analysis with IntegratedGradients
python collab_env/gnn/explain_gnn_integrated_gradients.py --data-name boid_single_species_weakalignment_large --method integrated_gradients --n-steps 50

# Analyze all files with caching
python collab_env/gnn/explain_gnn_integrated_gradients.py --data-name boid_single_species_weakalignment_large --file-id -1 --method saliency --max-frames 5

# Run tests
python collab_env/gnn/test_explainability.py
```

## Attribution Matrix

- **Shape**: `[2*N, N*F]` where N=agents, F=features/agent
- **Rows**: Output accelerations (x,y) for each agent
- **Columns**: Input features for all agents

### Feature Structure (per agent)
1. **Velocity**: 6D (3 timesteps × 2D)
2. **Position**: 6D (3 timesteps × 2D)  
3. **Boundary**: 6D (3 timesteps × 2D)
4. **Species**: 1D (no-food) or 2D (food models)

## Methods

- **IntegratedGradients**: Accurate, ~1-2 sec/frame with 50 steps
- **Saliency**: Fast, ~0.1 sec/frame

## Key Features

- Automatic food/no-food model detection
- Comprehensive caching for expensive computations
- Violin plots showing feature importance distributions
- Temporal evolution with confidence intervals (multi-file)
- Agent-wise sensitivity analysis

## Files

- `explain_gnn_integrated_gradients.py` - Main analysis script
- `test_explainability.py` - Test suite
- `example_usage.py` - Usage examples