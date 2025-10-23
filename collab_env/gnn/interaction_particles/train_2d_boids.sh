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
    --epochs 100 \
    --batch-size 32 \
    --visual-range 0.104 \
    --save-dir trained_models/interaction_particle_2d_basic \
    --device auto

# Training on noisy dataset
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_noisy.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --save-dir trained_models/interaction_particle_2d_noisy

# Training on high cluster, high speed dataset
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_high_cluster_high_speed.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --save-dir trained_models/interaction_particle_2d_high_cluster

# Training on short dataset
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_short.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --save-dir trained_models/interaction_particle_2d_short

# High capacity model (more parameters)
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 200 \
#     --batch-size 32 \
#     --hidden-dim 256 \
#     --embedding-dim 32 \
#     --n-layers 4 \
#     --visual-range 0.104 \
#     --save-dir trained_models/interaction_particle_2d_large

# Custom visual range (if your data has different normalization)
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.2 \
#     --save-dir trained_models/interaction_particle_2d_custom

# Evaluation only (no training)
# python -m collab_env.gnn.interaction_particles.run_training \
#     --eval-only \
#     --model-path trained_models/interaction_particle_2d_basic/best_model.pt \
#     --dataset simulated_data/boid_single_species_basic.pt

# Full training with rollout evaluation (multi-step prediction)
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --evaluate-rollout \
#     --n-rollout-steps 50 \
#     --save-dir trained_models/interaction_particle_2d_with_rollout \
#     --device auto

# Training with periodic plotting (plot every 10 epochs)
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --plot-every 10 \
#     --save-dir trained_models/interaction_particle_2d_with_plots \
#     --device auto

# Full training with all features: periodic plots and rollout evaluation
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --plot-every 10 \
#     --evaluate-rollout \
#     --n-rollout-steps 20 \
#     --save-dir trained_models/interaction_particle_2d_full \
#     --device mps

# Training on LARGE runpod dataset (1000 samples, T=800 timesteps)
# NOTE: Requires much smaller batch size due to long trajectories
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/runpod/boid_single_species_basic.pt \
    --epochs 50 \
    --batch-size 256 \
    --visual-range 1.0 \
    --plot-every 2 \
    --evaluate-rollout \
    --n-rollout-steps 80 \
    --save-dir trained_models/interaction_particle_2d_runpod \
    --device cpu \
    --learning-rate 1e-5 \
    --n-layers 5 \
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
