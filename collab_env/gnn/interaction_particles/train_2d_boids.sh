#!/bin/bash
#
# Quick start training script for InteractionParticle on 2D boids data
#
# This script provides example commands for training on existing 2D boids datasets.
# Uncomment the command you want to run, or modify as needed.

# Default: Quick test (10 epochs) on basic dataset
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 10 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --save-dir trained_models/interaction_particle_2d_quick \
#     --device auto

# Full training (100 epochs) on basic dataset
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_basic.pt \
    --epochs 300 \
    --batch-size 32 \
    --visual-range 0.2 \
    --plot-every 50 \
    --evaluate-rollout \
    --n-rollout-steps 80 \
    --learning-rate 1e-3 \
    --n-layers 2 \
    --hidden-dim 64 \
    --embedding-dim 16 \
    --save-dir trained_models/interaction_particle_2d_basic_embed16 \
    --device cpu

# weak alignment dataset
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_weakalignment_large.pt \
    --epochs 50 \
    --batch-size 32 \
    --visual-range 0.5 \
    --plot-every 2 \
    --evaluate-rollout \
    --n-rollout-steps 80 \
    --learning-rate 1e-4 \
    --n-layers 3 \
    --hidden-dim 256 \
    --save-dir trained_models/interaction_particle_2d_weakalignment \
    --device cpu


# Training on LARGE runpod dataset (1000 samples, T=800 timesteps)
# NOTE: Requires much smaller batch size due to long trajectories
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/runpod/boid_single_species_basic.pt \
    --epochs 5 \
    --batch-size 32 \
    --visual-range 0.2 \
    --plot-every 1 \
    --evaluate-rollout \
    --n-rollout-steps 50 \
    --save-dir trained_models/interaction_particle_2d_runpod \
    --device cpu \
    --learning-rate 1e-4 \
    --n-layers 3 \
    --embedding-dim 0 \
    --hidden-dim 256

# weak alignment dataset
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/runpod/boid_single_species_weakalignment_large.pt \
    --epochs 5 \
    --batch-size 32 \
    --visual-range 0.2 \
    --plot-every 1 \
    --evaluate-rollout \
    --n-rollout-steps 50 \
    --save-dir trained_models/interaction_particle_2d_runpod_weakalignment \
    --device cpu \
    --learning-rate 1e-4 \
    --n-layers 3 \
    --embedding-dim 0 \
    --hidden-dim 256

# independent dataset
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/runpod/boid_single_species_independent.pt \
    --epochs 5 \
    --batch-size 32 \
    --visual-range 0.2 \
    --plot-every 1 \
    --evaluate-rollout \
    --n-rollout-steps 50 \
    --save-dir trained_models/interaction_particle_2d_runpod_independent \
    --device cpu \
    --learning-rate 1e-4 \
    --n-layers 3 \
    --embedding-dim 0 \
    --hidden-dim 256

# Use specific device (cpu, cuda, mps, or auto)
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --device mps \
#     --save-dir trained_models/interaction_particle_2d_mps

# Notes:
# - visual_range: 0.104 ≈ 50px / 480px (boid visual_range / scene_size)
# - All datasets are in simulated_data/ directory
# - Config files are auto-detected (e.g., *_config.pt)
# - Results saved to trained_models/interaction_particle_2d_*/
# - --device auto: Auto-detects cuda > mps > cpu
# - --plot-every N: Generates force decomposition and rollout plots every N epochs
#   Plots saved in epoch_XXXX/ subdirectories
# - --evaluate-rollout: Enables multi-step rollout evaluation
#   Results saved in rollout_evaluation/ subdirectory
# - Final plots always generated:
#   * training_history.png - loss curves
#   * learned_force_decomposition.png - learned forces (2x3 grid)
#   * true_boid_force_decomposition.png - ground truth forces (2x3 grid)
