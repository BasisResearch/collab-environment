"""
Example script demonstrating InteractionParticle model usage.

This script trains a small model quickly to verify the implementation works.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import sys

# Add parent directory to path for imports
from collab_env.gnn.interaction_particles import (
    InteractionParticle,
    train_interaction_particle,
    plot_interaction_functions,
    compare_with_true_boids
)


def create_toy_dataset(n_trajectories=10, n_frames=50, n_particles=10):
    """
    Create a simple toy dataset for testing.

    Generates random particle trajectories with some simple dynamics.
    """
    logger.info(f"Creating toy dataset: {n_trajectories} trajectories, {n_frames} frames, {n_particles} particles")

    positions = []

    for _ in range(n_trajectories):
        # Initialize random positions
        pos = np.random.rand(n_particles, 2)
        vel = np.random.randn(n_particles, 2) * 0.01

        trajectory = [pos.copy()]

        # Simple dynamics: particles move with velocity and slight attraction to center
        for _ in range(n_frames - 1):
            center = pos.mean(axis=0)

            # Update velocities with simple cohesion-like force
            for i in range(n_particles):
                to_center = center - pos[i]
                vel[i] += to_center * 0.01

                # Add some random noise
                vel[i] += np.random.randn(2) * 0.001

            # Update positions
            pos = pos + vel

            # Keep particles in bounds
            pos = np.clip(pos, 0, 1)

            trajectory.append(pos.copy())

        positions.append(np.array(trajectory))

    # Convert to expected format: [B, F, N, D]
    positions = np.array(positions)
    positions = np.transpose(positions, (0, 1, 2, 3))

    logger.info(f"Created dataset with shape: {positions.shape}")

    return torch.tensor(positions, dtype=torch.float32)


def main():
    """Run example training."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=" * 60)
    logger.info("InteractionParticle Model - Example Training")
    logger.info("=" * 60)

    # Create toy dataset
    logger.info("\nCreating toy dataset...")
    dataset = create_toy_dataset(n_trajectories=20, n_frames=50, n_particles=10)

    # Save toy dataset
    toy_data_path = '/tmp/toy_boids.pt'
    torch.save(dataset, toy_data_path)
    logger.info(f"Saved toy dataset to {toy_data_path}")

    # Model configuration
    config = {
        'n_particles': 10,
        'n_particle_types': 1,
        'max_radius': 1.0,
        'hidden_dim': 64,
        'embedding_dim': 8,
        'n_mp_layers': 2,
        'input_size': 7,
        'output_size': 2,
    }

    logger.info("\nModel configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Train model
    logger.info("\nStarting training...")
    model, history = train_interaction_particle(
        dataset_path=toy_data_path,
        config=config,
        epochs=10,  # Just 10 epochs for quick test
        batch_size=16,
        learning_rate=1e-3,
        train_split=0.8,
        visual_range=0.4,
        save_dir='/tmp/interaction_particle_example',
        seed=42
    )

    logger.info("\nTraining completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")

    # Plot results
    logger.info("\nGenerating plots...")

    fig1 = plot_interaction_functions(model, save_path='/tmp/interaction_particle_example/interaction_fn.png')
    logger.info("Created interaction function plot")

    # Create comparison plot with default boid parameters
    boids_config = {
        'min_separation': 20.0,
        'neighborhood_dist': 80.0,
        'separation_weight': 15.0,
        'alignment_weight': 1.0,
        'cohesion_weight': 0.5,
    }

    fig2 = compare_with_true_boids(
        model,
        save_path='/tmp/interaction_particle_example/comparison.png',
        config=boids_config
    )
    logger.info("Created comparison plot")

    # Display plots
    plt.show()

    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")
    logger.info("=" * 60)
    logger.info("\nResults saved to: /tmp/interaction_particle_example/")
    logger.info("  - Model checkpoints: best_model.pt, final_model.pt")
    logger.info("  - Plots: interaction_fn.png, comparison.png")


if __name__ == '__main__':
    main()
