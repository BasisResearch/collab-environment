#!/bin/bash
#
# Quick start training script for InteractionParticle on 2D boids data
#
# This script provides example commands for training on existing 2D boids datasets.
# Uncomment the command you want to run, or modify as needed.

# Default: Quick test (10 epochs) on basic dataset
python -m collab_env.gnn.interaction_particles.run_training \
    --dataset simulated_data/boid_single_species_basic.pt \
    --epochs 10 \
    --batch-size 32 \
    --visual-range 0.104 \
    --save-dir trained_models/interaction_particle_2d_quick

# Full training (100 epochs) on basic dataset
# python -m collab_env.gnn.interaction_particles.run_training \
#     --dataset simulated_data/boid_single_species_basic.pt \
#     --epochs 100 \
#     --batch-size 32 \
#     --visual-range 0.104 \
#     --save-dir trained_models/interaction_particle_2d_basic

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
#     --save-dir trained_models/interaction_particle_2d_with_rollout

# Notes:
# - visual_range: 0.104 ≈ 50px / 480px (boid visual_range / scene_size)
# - All datasets are in simulated_data/ directory
# - Config files are auto-detected (e.g., *_config.pt)
# - Results saved to trained_models/interaction_particle_2d_*/
# - Check comparison_with_boids.png for learned vs true rules
