#!/usr/bin/env python
"""
Exploratory Data Analysis (EDA) for InteractionParticle datasets.

This script analyzes the distribution of input features and output accelerations
in boid trajectory datasets to understand the data characteristics.

Generates:
    - eda_distributions.png: 3x3 grid showing distributions of positions, velocities,
      distances, relative positions/velocities, and accelerations
    - true_boid_force_fields.png: 2x3 grid showing ground truth boid force decomposition
      (if config file is available)

Usage:
    python analyze_dataset.py <dataset_path> [--save-dir <output_dir>]

Example:
    python analyze_dataset.py simulated_data/boid_single_species_basic.pt --save-dir analysis_output
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_pairwise_features(positions, velocities):
    """
    Compute pairwise features (relative positions, distances, relative velocities).

    Parameters
    ----------
    positions : np.ndarray
        Positions [N, 2]
    velocities : np.ndarray
        Velocities [N, 2]

    Returns
    -------
    dict
        Dictionary with 'delta_pos', 'distances', 'delta_vel', 'pos_i', 'pos_j'
    """
    N = positions.shape[0]

    # Compute all pairwise features
    delta_pos = positions[None, :, :] - positions[:, None, :]  # [N, N, 2]
    distances = np.linalg.norm(delta_pos, axis=-1)  # [N, N]
    delta_vel = velocities[None, :, :] - velocities[:, None, :]  # [N, N, 2]

    # Flatten and remove self-interactions (diagonal)
    mask = ~np.eye(N, dtype=bool)

    delta_pos_flat = delta_pos[mask]  # [N*(N-1), 2]
    distances_flat = distances[mask]  # [N*(N-1)]
    delta_vel_flat = delta_vel[mask]  # [N*(N-1), 2]

    # Also get absolute positions for both particles
    # Create indices for i and j
    i_indices = np.repeat(np.arange(N), N)  # [0,0,...,1,1,...,N-1,N-1,...]
    j_indices = np.tile(np.arange(N), N)    # [0,1,...,N-1,0,1,...,N-1]

    # Remove self-interactions
    mask_flat = mask.flatten()
    i_indices_filtered = i_indices[mask_flat]
    j_indices_filtered = j_indices[mask_flat]

    pos_i = positions[i_indices_filtered]  # [N*(N-1), 2]
    pos_j = positions[j_indices_filtered]  # [N*(N-1), 2]

    return {
        'delta_pos': delta_pos_flat,
        'distances': distances_flat,
        'delta_vel': delta_vel_flat,
        'pos_i': pos_i,
        'pos_j': pos_j
    }


def analyze_dataset(dataset_path, save_dir=None, device='cpu', scene_size=480.0):
    """
    Perform exploratory data analysis on a boid trajectory dataset.

    Parameters
    ----------
    dataset_path : str
        Path to dataset .pt file
    save_dir : str, optional
        Directory to save analysis plots
    device : str, optional
        Device to use for computations (default: 'cpu')
    scene_size : float, optional
        Scene size in pixels for normalizing config parameters (default: 480.0)
    """
    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = torch.load(dataset_path, weights_only=False, map_location=device)
    logger.info(f"Dataset type: {type(dataset)}")
    logger.info(f"Number of samples: {len(dataset)}")

    # Extract first sample to check structure
    positions, species = dataset[0]
    logger.info(f"Sample shape: positions={positions.shape}, species={species.shape}")
    T, N, D = positions.shape
    logger.info(f"Timesteps (T): {T}, Particles (N): {N}, Dimensions (D): {D}")

    # Create save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving analysis to {save_dir}")

    # Collect all data
    all_positions = []
    all_velocities = []
    all_accelerations = []
    all_delta_pos = []
    all_distances = []
    all_delta_vel = []
    all_pos_i = []
    all_pos_j = []

    logger.info("Computing features from all samples...")
    for idx, (positions_traj, _) in enumerate(dataset):
        if idx % 10 == 0:
            logger.info(f"Processing sample {idx}/{len(dataset)}")

        positions_np = positions_traj.numpy()  # [T, N, 2]

        # Compute velocities (finite differences)
        velocities_np = np.diff(positions_np, axis=0)  # [T-1, N, 2]
        velocities_np = np.concatenate([velocities_np, velocities_np[-1:]], axis=0)  # [T, N, 2]

        # Compute accelerations
        accelerations_np = np.diff(velocities_np, axis=0)  # [T-1, N, 2]

        # Collect frame-by-frame
        for t in range(T-1):  # -1 because acceleration is one shorter
            pos_t = positions_np[t]  # [N, 2]
            vel_t = velocities_np[t]  # [N, 2]
            acc_t = accelerations_np[t]  # [N, 2]

            all_positions.append(pos_t)
            all_velocities.append(vel_t)
            all_accelerations.append(acc_t)

            # Compute pairwise features
            pairwise = compute_pairwise_features(pos_t, vel_t)
            all_delta_pos.append(pairwise['delta_pos'])
            all_distances.append(pairwise['distances'])
            all_delta_vel.append(pairwise['delta_vel'])
            all_pos_i.append(pairwise['pos_i'])
            all_pos_j.append(pairwise['pos_j'])

    # Concatenate all data
    positions_all = np.concatenate(all_positions, axis=0)  # [num_frames*N, 2]
    velocities_all = np.concatenate(all_velocities, axis=0)
    accelerations_all = np.concatenate(all_accelerations, axis=0)
    delta_pos_all = np.concatenate(all_delta_pos, axis=0)  # [num_pairs, 2]
    distances_all = np.concatenate(all_distances, axis=0)  # [num_pairs]
    delta_vel_all = np.concatenate(all_delta_vel, axis=0)  # [num_pairs, 2]
    pos_i_all = np.concatenate(all_pos_i, axis=0)
    pos_j_all = np.concatenate(all_pos_j, axis=0)

    logger.info(f"Total particles-frames: {len(positions_all)}")
    logger.info(f"Total pairs: {len(distances_all)}")

    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    print(f"\nPositions (absolute):")
    print(f"  Range: [{positions_all.min():.4f}, {positions_all.max():.4f}]")
    print(f"  Mean: {positions_all.mean(axis=0)}")
    print(f"  Std: {positions_all.std(axis=0)}")

    print(f"\nVelocities (absolute):")
    print(f"  Range: [{velocities_all.min():.4f}, {velocities_all.max():.4f}]")
    print(f"  Mean: {velocities_all.mean(axis=0)}")
    print(f"  Std: {velocities_all.std(axis=0)}")
    print(f"  Speed mean: {np.linalg.norm(velocities_all, axis=1).mean():.4f}")
    print(f"  Speed std: {np.linalg.norm(velocities_all, axis=1).std():.4f}")

    print(f"\nAccelerations (targets):")
    print(f"  Range: [{accelerations_all.min():.4f}, {accelerations_all.max():.4f}]")
    print(f"  Mean: {accelerations_all.mean(axis=0)}")
    print(f"  Std: {accelerations_all.std(axis=0)}")
    print(f"  Magnitude mean: {np.linalg.norm(accelerations_all, axis=1).mean():.4f}")
    print(f"  Magnitude std: {np.linalg.norm(accelerations_all, axis=1).std():.4f}")

    print(f"\nRelative positions (delta_pos):")
    print(f"  Range: [{delta_pos_all.min():.4f}, {delta_pos_all.max():.4f}]")
    print(f"  Mean: {delta_pos_all.mean(axis=0)}")
    print(f"  Std: {delta_pos_all.std(axis=0)}")

    print(f"\nDistances (between particles):")
    print(f"  Range: [{distances_all.min():.4f}, {distances_all.max():.4f}]")
    print(f"  Mean: {distances_all.mean():.4f}")
    print(f"  Std: {distances_all.std():.4f}")
    print(f"  Median: {np.median(distances_all):.4f}")

    print(f"\nRelative velocities (delta_vel):")
    print(f"  Range: [{delta_vel_all.min():.4f}, {delta_vel_all.max():.4f}]")
    print(f"  Mean: {delta_vel_all.mean(axis=0)}")
    print(f"  Std: {delta_vel_all.std(axis=0)}")
    print(f"  Magnitude mean: {np.linalg.norm(delta_vel_all, axis=1).mean():.4f}")
    print(f"  Magnitude std: {np.linalg.norm(delta_vel_all, axis=1).std():.4f}")

    print("="*80 + "\n")

    # Create visualizations
    logger.info("Generating visualizations...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Dataset EDA: {os.path.basename(dataset_path)}', fontsize=16, fontweight='bold')

    # Row 1: Velocities
    ax = axes[0, 0]
    ax.scatter(positions_all[:, 0], positions_all[:, 1], alpha=0.1, s=1)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Absolute Positions')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist2d(velocities_all[:, 0], velocities_all[:, 1], bins=50, cmap='viridis', cmin=1)
    ax.set_xlabel('Velocity X')
    ax.set_ylabel('Velocity Y')
    ax.set_title('Velocity Vector Distribution')
    ax.set_aspect('equal')
    plt.colorbar(ax.collections[0], ax=ax, label='Count')

    ax = axes[0, 2]
    speeds = np.linalg.norm(velocities_all, axis=1)
    ax.hist(speeds, bins=50, alpha=0.7, color='green')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Frequency')
    ax.set_title('Speed Distribution')
    ax.grid(True, alpha=0.3)
    ax.axvline(speeds.mean(), color='red', linestyle='--', label=f'Mean: {speeds.mean():.3f}')
    ax.legend()

    # Row 2: Pairwise features
    ax = axes[1, 0]
    ax.hist(distances_all, bins=50, alpha=0.7, color='purple')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('Pairwise Distance Distribution')
    ax.grid(True, alpha=0.3)
    ax.axvline(distances_all.mean(), color='red', linestyle='--', label=f'Mean: {distances_all.mean():.3f}')
    ax.legend()

    ax = axes[1, 1]
    ax.hist2d(delta_pos_all[:, 0], delta_pos_all[:, 1], bins=50, cmap='viridis', cmin=1)
    ax.set_xlabel('Δx (relative position)')
    ax.set_ylabel('Δy (relative position)')
    ax.set_title('Relative Position Distribution')
    ax.set_aspect('equal')
    plt.colorbar(ax.collections[0], ax=ax, label='Count')

    ax = axes[1, 2]
    delta_vel_mag = np.linalg.norm(delta_vel_all, axis=1)
    ax.hist(delta_vel_mag, bins=50, alpha=0.7, color='orange')
    ax.set_xlabel('|Δv| (relative velocity magnitude)')
    ax.set_ylabel('Frequency')
    ax.set_title('Relative Velocity Magnitude')
    ax.grid(True, alpha=0.3)
    ax.axvline(delta_vel_mag.mean(), color='red', linestyle='--', label=f'Mean: {delta_vel_mag.mean():.3f}')
    ax.legend()

    # Row 3: Accelerations (targets)
    ax = axes[2, 0]
    acc_mag = np.linalg.norm(accelerations_all, axis=1)
    ax.hist(acc_mag, bins=50, alpha=0.7, color='red')
    ax.set_xlabel('Acceleration Magnitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Acceleration Magnitude Distribution')
    ax.grid(True, alpha=0.3)
    ax.axvline(acc_mag.mean(), color='blue', linestyle='--', label=f'Mean: {acc_mag.mean():.4f}')
    ax.legend()

    ax = axes[2, 1]
    ax.hist2d(accelerations_all[:, 0], accelerations_all[:, 1], bins=50, cmap='coolwarm', cmin=1)
    ax.set_xlabel('Acceleration X')
    ax.set_ylabel('Acceleration Y')
    ax.set_title('Acceleration Vector Distribution')
    ax.set_aspect('equal')
    plt.colorbar(ax.collections[0], ax=ax, label='Count')

    ax = axes[2, 2]
    # Relative velocity vector distribution (sampled for performance)
    sample_indices = np.random.choice(len(delta_vel_all), size=min(50000, len(delta_vel_all)), replace=False)
    ax.hist2d(delta_vel_all[sample_indices, 0], delta_vel_all[sample_indices, 1],
              bins=50, cmap='plasma', cmin=1)
    ax.set_xlabel('Δv_x (relative velocity)')
    ax.set_ylabel('Δv_y (relative velocity)')
    ax.set_title('Relative Velocity Vector Distribution')
    ax.set_aspect('equal')
    plt.colorbar(ax.collections[0], ax=ax, label='Count')

    plt.tight_layout()

    if save_dir:
        plot_path = os.path.join(save_dir, 'eda_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution plots to {plot_path}")
    else:
        plt.show()

    # Generate ground truth force field plots (if config is available)
    logger.info("Generating ground truth force field plots...")

    # Try to load config
    config_path = dataset_path.replace('.pt', '_config.pt')
    if os.path.exists(config_path):
        logger.info(f"Found config at {config_path}")

        try:
            # Import plotting functions
            try:
                from .plotting import evaluate_true_boid_forces, plot_force_decomposition
            except ImportError:
                # If relative import fails, try absolute import
                from collab_env.gnn.interaction_particles.plotting import (
                    evaluate_true_boid_forces, plot_force_decomposition
                )

            boids_config = torch.load(config_path, weights_only=False)
            logger.info(f"Using config: {boids_config}")

            # Get the first species config
            species_key = list(boids_config.keys())[0] if boids_config else None
            if species_key and species_key != 'scene_size':
                logger.info(f"Using config for species '{species_key}'")

                # Extract species-specific config
                species_config = boids_config[species_key]

                # Evaluate ground truth forces
                # Determine plot boundary based on which forces are active
                # For independent configs, use min_distance; otherwise use visual_range
                if species_config.get('independent', False):
                    # Independent: only avoidance active (within min_distance)
                    relevant_range = species_config['min_distance'] / scene_size
                else:
                    # Normal: cohesion/alignment active (within visual_range)
                    relevant_range = species_config['visual_range'] / scene_size

                plot_max_dist = relevant_range * 1.5  # 50% cushion

                true_forces = evaluate_true_boid_forces(
                    species_config,  # Pass species-specific config
                    grid_size=50,
                    max_dist=plot_max_dist,
                    scene_size=scene_size
                )

                # Plot ground truth forces
                # Use visual_range and min_distance from config (normalized to [0,1] space)
                visual_range_normalized = species_config['visual_range'] / scene_size
                min_distance_normalized = None
                if 'min_distance' in species_config:
                    min_distance_normalized = species_config['min_distance'] / scene_size

                if save_dir:
                    true_plot_path = os.path.join(save_dir, 'true_boid_force_fields.png')
                    plot_force_decomposition(
                        true_forces,
                        save_path=true_plot_path,
                        title_prefix="Ground Truth Boid",
                        visual_range=visual_range_normalized,
                        min_distance=min_distance_normalized
                    )
                    logger.info(f"Saved force field plot to {true_plot_path}")
                else:
                    plot_force_decomposition(
                        true_forces,
                        save_path=None,
                        title_prefix="Ground Truth Boid",
                        visual_range=visual_range_normalized,
                        min_distance=min_distance_normalized
                    )
                    plt.show()
            else:
                logger.warning("Could not find species config in loaded config file")

        except Exception as e:
            logger.warning(f"Could not generate force field plots: {e}")
    else:
        logger.info(f"Config file not found at {config_path}, skipping force field plots")

    logger.info("Analysis complete!")

    return {
        'positions': positions_all,
        'velocities': velocities_all,
        'accelerations': accelerations_all,
        'delta_pos': delta_pos_all,
        'distances': distances_all,
        'delta_vel': delta_vel_all
    }


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory Data Analysis for InteractionParticle datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_dataset.py simulated_data/boid_single_species_basic.pt
    python analyze_dataset.py simulated_data/runpod/boid_single_species_basic.pt --save-dir analysis_runpod
    python analyze_dataset.py simulated_data/boid_single_species_basic.pt --save-dir analysis --device mps
        """
    )

    parser.add_argument('dataset', type=str, help='Path to dataset .pt file')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save analysis plots (default: show plots)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for computations (default: cpu)')
    parser.add_argument('--scene-size', type=float, default=480.0,
                       help='Scene size in pixels (for normalizing config parameters)')

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    analyze_dataset(args.dataset, save_dir=args.save_dir, device=args.device, scene_size=args.scene_size)

    return 0


if __name__ == '__main__':
    sys.exit(main())
