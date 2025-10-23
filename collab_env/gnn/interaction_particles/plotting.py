"""
Plotting and comparison utilities for InteractionParticle model.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from loguru import logger


def boid_separation_force(distances, min_distance=15.0, avoid_factor=0.05):
    """
    Compute true 2D boid separation force as a function of distance.

    From boid.py avoid_others():
    - For each neighbor within min_distance:
      force += (self_pos - other_pos) * avoid_factor
    - This is linear in displacement, not inverse-square

    Parameters
    ----------
    distances : np.ndarray
        Array of distances
    min_distance : float
        Minimum distance threshold (default: 15 from 2D boids)
    avoid_factor : float
        Avoidance weight (default: 0.05 from 2D boids)

    Returns
    -------
    forces : np.ndarray
        Separation forces
    """
    forces = np.zeros_like(distances)
    mask = distances < min_distance
    # Linear repulsion: force = avoid_factor * distance
    forces[mask] = avoid_factor * distances[mask]
    return forces


def boid_alignment_force(distances, visual_range=50.0, matching_factor=0.5):
    """
    Compute true 2D boid alignment force as a function of distance.

    From boid.py match_velocity():
    - For neighbors within visual_range:
      force += (avg_velocity - self_velocity) * matching_factor
    - This is a step function based on visual range

    Note: Actual alignment depends on relative velocities.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances
    visual_range : float
        Visual range for alignment (default: 50 from 2D boids)
    matching_factor : float
        Alignment weight (default: 0.5 from 2D boids)

    Returns
    -------
    forces : np.ndarray
        Alignment indicator (1 if within visual range, 0 otherwise)
    """
    forces = np.zeros_like(distances)
    mask = distances < visual_range
    forces[mask] = matching_factor
    return forces


def boid_cohesion_force(distances, visual_range=50.0, centering_factor=0.005):
    """
    Compute true 2D boid cohesion force as a function of distance.

    From boid.py fly_towards_center():
    - For neighbors within visual_range:
      force += (center_of_mass - self_pos) * centering_factor
    - This is linear in distance to center of mass

    Note: Actual cohesion is to center of neighbors, not pairwise.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances
    visual_range : float
        Visual range for cohesion (default: 50 from 2D boids)
    centering_factor : float
        Cohesion weight (default: 0.005 from 2D boids)

    Returns
    -------
    forces : np.ndarray
        Cohesion forces (proportional to distance)
    """
    forces = np.zeros_like(distances)
    mask = distances < visual_range
    # Cohesion force pulls towards center, so it increases with distance
    forces[mask] = centering_factor * distances[mask]
    return forces


def plot_interaction_functions(
    model,
    save_path=None,
    distances=None,
    particle_idx=0,
    config=None
):
    """
    Plot learned interaction functions from the model.

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    save_path : str, optional
        Path to save the plot
    distances : np.ndarray, optional
        Distances to evaluate (if None, uses default range)
    particle_idx : int
        Particle index to use for embedding
    config : dict, optional
        Configuration dict with boid parameters

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if distances is None:
        # Use normalized distances from 0 to 1 (will be multiplied by max_radius)
        distances = np.linspace(0.01, 1.0, 200)

    # Get interaction function from model
    interaction_fn = model.get_interaction_function(embedding_idx=particle_idx)
    forces = interaction_fn(distances * model.max_radius)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot X component
    axes[0].plot(distances * model.max_radius, forces[:, 0], 'b-', linewidth=2, label='Learned Force (X)')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Distance', fontsize=12)
    axes[0].set_ylabel('Force (X component)', fontsize=12)
    axes[0].set_title('Learned Interaction Function (X)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot Y component
    axes[1].plot(distances * model.max_radius, forces[:, 1], 'r-', linewidth=2, label='Learned Force (Y)')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Distance', fontsize=12)
    axes[1].set_ylabel('Force (Y component)', fontsize=12)
    axes[1].set_title('Learned Interaction Function (Y)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved interaction function plot to {save_path}")

    return fig


def compare_with_true_boids(
    model,
    save_path=None,
    config=None,
    particle_idx=0,
    scene_size=480.0
):
    """
    Compare learned interaction functions with true 2D boid rules.

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    save_path : str, optional
        Path to save the plot
    config : dict, optional
        2D boid configuration dict with parameters:
        - visual_range: 50.0 (pixels)
        - min_distance: 15.0 (pixels)
        - avoid_factor: 0.05
        - matching_factor: 0.5
        - centering_factor: 0.005
    particle_idx : int
        Particle index to use for embedding
    scene_size : float
        Scene size in pixels (default: 480 for 2D boids)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Default 2D boid parameters from boids_gnn_temp/boid.py
    if config is None:
        config = {
            'visual_range': 50.0,         # Visual range for alignment and cohesion (pixels)
            'min_distance': 15.0,         # Separation distance threshold (pixels)
            'avoid_factor': 0.05,         # Separation weight
            'matching_factor': 0.5,       # Alignment weight
            'centering_factor': 0.005,    # Cohesion weight
        }

    # Create distance arrays
    distances_fine = np.linspace(0.01, 1.0, 500)  # Normalized distances
    distances_real = distances_fine * model.max_radius  # Real distances

    # Get learned forces
    interaction_fn = model.get_interaction_function(embedding_idx=particle_idx)
    learned_forces = interaction_fn(distances_real)

    # Compute magnitude of learned forces
    learned_magnitude = np.sqrt(learned_forces[:, 0]**2 + learned_forces[:, 1]**2)

    # Convert distances to pixels
    # 2D boids data is normalized by scene size (typically 480x480)
    distances_pixels = distances_fine * scene_size

    # Compute true 2D boid forces
    true_separation = boid_separation_force(
        distances_pixels,
        min_distance=config['min_distance'],
        avoid_factor=config['avoid_factor']
    )
    true_alignment = boid_alignment_force(
        distances_pixels,
        visual_range=config['visual_range'],
        matching_factor=config['matching_factor']
    )
    true_cohesion = boid_cohesion_force(
        distances_pixels,
        visual_range=config['visual_range'],
        centering_factor=config['centering_factor']
    )

    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Learned force magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(distances_pixels, learned_magnitude, 'b-', linewidth=2, label='Learned Force Magnitude')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Distance (pixels)', fontsize=11)
    ax1.set_ylabel('Force Magnitude', fontsize=11)
    ax1.set_title('Learned Interaction Force Magnitude', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Learned force components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distances_pixels, learned_forces[:, 0], 'b-', linewidth=2, label='X Component', alpha=0.7)
    ax2.plot(distances_pixels, learned_forces[:, 1], 'r-', linewidth=2, label='Y Component', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Distance (pixels)', fontsize=11)
    ax2.set_ylabel('Force', fontsize=11)
    ax2.set_title('Learned Force Components', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: True boid separation (linear repulsion)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(distances_pixels, true_separation, 'g-', linewidth=2, label='Separation Force')
    ax3.axvline(x=config['min_distance'], color='g', linestyle='--', alpha=0.5,
                label=f"min_distance = {config['min_distance']:.0f}px")
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Distance (pixels)', fontsize=11)
    ax3.set_ylabel('Force', fontsize=11)
    ax3.set_title('True Boid Separation (Linear Repulsion)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim([0, config['visual_range'] * 1.5])

    # Plot 4: True boid alignment (step function)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(distances_pixels, true_alignment, 'm-', linewidth=2, label='Alignment Weight')
    ax4.axvline(x=config['visual_range'], color='m', linestyle='--', alpha=0.5,
                label=f"visual_range = {config['visual_range']:.0f}px")
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Distance (pixels)', fontsize=11)
    ax4.set_ylabel('Weight', fontsize=11)
    ax4.set_title('True Boid Alignment (Step Function)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim([0, config['visual_range'] * 1.5])

    # Plot 5: True boid cohesion (linear attraction)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(distances_pixels, true_cohesion, 'c-', linewidth=2, label='Cohesion Force')
    ax5.axvline(x=config['visual_range'], color='c', linestyle='--', alpha=0.5,
                label=f"visual_range = {config['visual_range']:.0f}px")
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Distance (pixels)', fontsize=11)
    ax5.set_ylabel('Force', fontsize=11)
    ax5.set_title('True Boid Cohesion (Linear Attraction)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim([0, config['visual_range'] * 1.5])

    # Plot 6: Combined comparison - overlay learned on true
    ax6 = fig.add_subplot(gs[2, 1])

    # Normalize learned force for comparison
    learned_norm = learned_magnitude / (np.max(learned_magnitude) + 1e-8)

    # Normalize true forces for comparison
    sep_norm = true_separation / (np.max(true_separation) + 1e-8)
    coh_norm = true_cohesion / (np.max(true_cohesion) + 1e-8)

    ax6.plot(distances_pixels, learned_norm, 'b-', linewidth=2.5, label='Learned (normalized)', alpha=0.8)
    ax6.plot(distances_pixels, sep_norm, 'g--', linewidth=2, label='Separation (normalized)', alpha=0.7)
    ax6.plot(distances_pixels, coh_norm, 'c--', linewidth=2, label='Cohesion (normalized)', alpha=0.7)
    ax6.axvline(x=config['min_distance'], color='g', linestyle=':', alpha=0.5)
    ax6.axvline(x=config['visual_range'], color='c', linestyle=':', alpha=0.5)
    ax6.set_xlabel('Distance (pixels)', fontsize=11)
    ax6.set_ylabel('Normalized Force', fontsize=11)
    ax6.set_title('Comparison: Learned vs True 2D Boid Rules', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_xlim([0, config['visual_range'] * 1.5])

    # Add main title
    fig.suptitle('InteractionParticle Model: Learned vs True 2D Boid Interaction Functions',
                 fontsize=14, fontweight='bold', y=0.995)

    # Add text box with parameters
    param_text = (
        f"2D Boid Parameters:\n"
        f"visual_range = {config['visual_range']:.1f} px\n"
        f"min_distance = {config['min_distance']:.1f} px\n"
        f"avoid_factor = {config['avoid_factor']:.3f}\n"
        f"matching_factor = {config['matching_factor']:.2f}\n"
        f"centering_factor = {config['centering_factor']:.4f}"
    )
    fig.text(0.02, 0.98, param_text, transform=fig.transFigure,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")

    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    history : dict
        Training history with 'train_loss' and 'val_loss' keys
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")

    return fig


def plot_rollout_comparison(ground_truth, predicted, trajectory_idx=0, n_particles=None, save_path=None):
    """
    Create side-by-side comparison of ground truth vs predicted trajectories.

    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth positions [T, N, 2]
    predicted : np.ndarray
        Predicted positions [T, N, 2]
    trajectory_idx : int
        Index identifier for the trajectory
    n_particles : int, optional
        Number of particles to plot (default: all)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    T, N, D = ground_truth.shape

    if n_particles is None:
        n_particles = N
    else:
        n_particles = min(n_particles, N)

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot ground truth
    ax = axes[0]
    for i in range(n_particles):
        ax.plot(ground_truth[:, i, 0], ground_truth[:, i, 1],
                alpha=0.6, linewidth=1.5)
        # Mark start and end
        ax.scatter(ground_truth[0, i, 0], ground_truth[0, i, 1],
                  c='green', s=100, marker='o', zorder=5, edgecolors='black', linewidth=1)
        ax.scatter(ground_truth[-1, i, 0], ground_truth[-1, i, 1],
                  c='red', s=100, marker='s', zorder=5, edgecolors='black', linewidth=1)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot predicted
    ax = axes[1]
    for i in range(n_particles):
        ax.plot(predicted[:, i, 0], predicted[:, i, 1],
                alpha=0.6, linewidth=1.5)
        # Mark start and end
        ax.scatter(predicted[0, i, 0], predicted[0, i, 1],
                  c='green', s=100, marker='o', zorder=5, edgecolors='black', linewidth=1)
        ax.scatter(predicted[-1, i, 0], predicted[-1, i, 1],
                  c='red', s=100, marker='s', zorder=5, edgecolors='black', linewidth=1)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Predicted (Model)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Start'),
        Patch(facecolor='red', edgecolor='black', label='End')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11)

    plt.suptitle(f'Trajectory Comparison (Sample {trajectory_idx})',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved rollout comparison to {save_path}")

    return fig


def plot_rollout_error_over_time(ground_truth_list, predicted_list, save_path=None):
    """
    Plot position error as a function of time for multiple rollouts.

    Parameters
    ----------
    ground_truth_list : list of np.ndarray
        List of ground truth trajectories, each [T, N, 2]
    predicted_list : list of np.ndarray
        List of predicted trajectories, each [T, N, 2]
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Compute errors for each trajectory
    errors_over_time = []

    for gt, pred in zip(ground_truth_list, predicted_list):
        # Compute per-particle error at each timestep
        error = np.linalg.norm(gt - pred, axis=-1)  # [T, N]
        # Average over particles
        mean_error = np.mean(error, axis=1)  # [T]
        errors_over_time.append(mean_error)

    # Convert to array
    errors_over_time = np.array(errors_over_time)  # [n_trajectories, T]

    # Compute statistics
    mean_error = np.mean(errors_over_time, axis=0)
    std_error = np.std(errors_over_time, axis=0)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    timesteps = np.arange(len(mean_error))

    # Plot mean and std
    ax.plot(timesteps, mean_error, 'b-', linewidth=2, label='Mean Error')
    ax.fill_between(timesteps,
                     mean_error - std_error,
                     mean_error + std_error,
                     alpha=0.3, color='b', label='±1 Std')

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Position Error', fontsize=12)
    ax.set_title('Rollout Position Error Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error over time plot to {save_path}")

    return fig


def create_rollout_report(results, save_dir=None):
    """
    Create comprehensive rollout evaluation report with all visualizations.

    Parameters
    ----------
    results : dict
        Results from evaluate_rollout()
    save_dir : str, optional
        Directory to save figures

    Returns
    -------
    figures : dict
        Dictionary of matplotlib figures
    """
    import os

    figures = {}

    # Extract data
    gt_positions = results['ground_truth_positions']
    pred_positions = results['predicted_positions']
    metrics = results['metrics']

    logger.info(f"Creating rollout report for {len(gt_positions)} trajectories")

    # 1. Plot first few trajectory comparisons
    n_examples = min(3, len(gt_positions))
    for i in range(n_examples):
        fig = plot_rollout_comparison(
            gt_positions[i],
            pred_positions[i],
            trajectory_idx=i,
            save_path=os.path.join(save_dir, f'rollout_comparison_{i}.png') if save_dir else None
        )
        figures[f'comparison_{i}'] = fig
        plt.close(fig)

    # 2. Plot error over time
    fig = plot_rollout_error_over_time(
        gt_positions,
        pred_positions,
        save_path=os.path.join(save_dir, 'rollout_error_over_time.png') if save_dir else None
    )
    figures['error_over_time'] = fig
    plt.close(fig)

    # 3. Print metrics summary
    logger.info("=" * 60)
    logger.info("ROLLOUT EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"Number of trajectories: {metrics['n_trajectories']}")
    logger.info(f"Mean position error: {metrics['mean_position_error']:.6f} ± {metrics['std_position_error']:.6f}")
    logger.info(f"Mean velocity error: {metrics['mean_velocity_error']:.6f} ± {metrics['std_velocity_error']:.6f}")
    logger.info("=" * 60)

    return figures
