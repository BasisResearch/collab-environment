#!/usr/bin/env python
"""
Example training on 2D boids data from docs/gnn/0a-Simulate_Boid_2D.ipynb

This script demonstrates training InteractionParticle model on the 2D boids
trajectories that already exist in the repo.
"""

import argparse
import sys
from pathlib import Path

# Just call the main training script with appropriate defaults for 2D boids
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train InteractionParticle on 2D boids data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='simulated_data/boid_single_species_basic.pt',
        choices=[
            'simulated_data/boid_single_species_basic.pt',
            'simulated_data/boid_single_species_noisy.pt',
            'simulated_data/boid_single_species_high_cluster_high_speed.pt',
            'simulated_data/boid_single_species_short.pt',
        ],
        help='Which 2D boids dataset to use'
    )
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 epochs)')

    args = parser.parse_args()

    # Prepare command for run_training
    from collab_env.gnn.interaction_particles.run_training import main as train_main

    # Build sys.argv for the training script
    dataset_name = Path(args.dataset).stem
    save_dir = f'trained_models/interaction_particle_2d_{dataset_name}'

    # Calculate appropriate visual range
    # 2D boids have visual_range=50 pixels in a 480x480 scene
    # In normalized coordinates: 50/480 ≈ 0.104
    visual_range = 0.104

    epochs = 10 if args.quick else args.epochs

    sys.argv = [
        'run_training.py',
        '--dataset', args.dataset,
        '--epochs', str(epochs),
        '--batch-size', '32',
        '--visual-range', str(visual_range),
        '--save-dir', save_dir,
        '--hidden-dim', '128',
        '--embedding-dim', '16',
        '--n-layers', '3',
    ]

    print("=" * 70)
    print("Training InteractionParticle on 2D Boids Data")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {epochs}")
    print(f"Visual range: {visual_range} (normalized)")
    print(f"Save directory: {save_dir}")
    print("=" * 70)
    print()

    # Run training
    train_main()
