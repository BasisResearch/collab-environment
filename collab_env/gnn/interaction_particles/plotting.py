"""
Plotting and comparison utilities for InteractionParticle model.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from loguru import logger


def boid_separation_force(distances, min_separation=20.0, separation_weight=15.0):
    """
    Compute true boid separation force as a function of distance.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances
    min_separation : float
        Minimum separation distance
    separation_weight : float
        Separation weight

    Returns
    -------
    forces : np.ndarray
        Separation forces
    """
    forces = np.zeros_like(distances)
    mask = (distances > 0) & (distances < min_separation)
    forces[mask] = separation_weight / (distances[mask] ** 2)
    return forces


def boid_alignment_force(distances, neighborhood_dist=80.0, alignment_weight=1.0):
    """
    Compute true boid alignment force as a function of distance.
    Note: This is a simplified version - actual alignment depends on relative velocities.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances
    neighborhood_dist : float
        Neighborhood distance
    alignment_weight : float
        Alignment weight

    Returns
    -------
    forces : np.ndarray
        Alignment indicator (1 if within neighborhood, 0 otherwise)
    """
    forces = np.zeros_like(distances)
    mask = distances < neighborhood_dist
    forces[mask] = alignment_weight
    return forces


def boid_cohesion_force(distances, neighborhood_dist=80.0, cohesion_weight=0.5):
    """
    Compute true boid cohesion force as a function of distance.
    Note: This is a simplified version - actual cohesion is proportional to distance to center of mass.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances
    neighborhood_dist : float
        Neighborhood distance
    cohesion_weight : float
        Cohesion weight

    Returns
    -------
    forces : np.ndarray
        Cohesion forces (proportional to distance)
    """
    forces = np.zeros_like(distances)
    mask = distances < neighborhood_dist
    # Cohesion force pulls towards center, so it increases with distance
    forces[mask] = cohesion_weight * distances[mask]
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
    particle_idx=0
):
    """
    Compare learned interaction functions with true boid rules.

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    save_path : str, optional
        Path to save the plot
    config : dict, optional
        Configuration dict with boid parameters (from config.yaml)
    particle_idx : int
        Particle index to use for embedding

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Default boid parameters from config.yaml
    if config is None:
        config = {
            'min_separation': 20.0,
            'neighborhood_dist': 80.0,
            'separation_weight': 15.0,
            'alignment_weight': 1.0,
            'cohesion_weight': 0.5,
        }

    # Create distance arrays
    distances_fine = np.linspace(0.01, 1.0, 500)  # Normalized distances
    distances_real = distances_fine * model.max_radius  # Real distances

    # Get learned forces
    interaction_fn = model.get_interaction_function(embedding_idx=particle_idx)
    learned_forces = interaction_fn(distances_real)

    # Compute magnitude of learned forces
    learned_magnitude = np.sqrt(learned_forces[:, 0]**2 + learned_forces[:, 1]**2)

    # Compute true boid forces (in real distance units)
    # Need to unnormalize distances for boid calculations
    # Assume normalization was done by dividing by p_range
    # For plotting purposes, we'll scale distances appropriately

    # Let's assume the data was normalized such that the scene size is ~1.0
    # and the boid parameters are in the original units
    # We need to scale the distances back
    scale_factor = 100.0  # Rough estimate based on config (box_size=1500, scene_scale=300)
    distances_boid_units = distances_fine * scale_factor

    true_separation = boid_separation_force(
        distances_boid_units,
        min_separation=config['min_separation'],
        separation_weight=config['separation_weight']
    )
    true_alignment = boid_alignment_force(
        distances_boid_units,
        neighborhood_dist=config['neighborhood_dist'],
        alignment_weight=config['alignment_weight']
    )
    true_cohesion = boid_cohesion_force(
        distances_boid_units,
        neighborhood_dist=config['neighborhood_dist'],
        cohesion_weight=config['cohesion_weight']
    )

    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Learned force magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(distances_boid_units, learned_magnitude, 'b-', linewidth=2, label='Learned Force Magnitude')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Distance (original units)', fontsize=11)
    ax1.set_ylabel('Force Magnitude', fontsize=11)
    ax1.set_title('Learned Interaction Force Magnitude', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Learned force components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distances_boid_units, learned_forces[:, 0], 'b-', linewidth=2, label='X Component', alpha=0.7)
    ax2.plot(distances_boid_units, learned_forces[:, 1], 'r-', linewidth=2, label='Y Component', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Distance (original units)', fontsize=11)
    ax2.set_ylabel('Force', fontsize=11)
    ax2.set_title('Learned Force Components', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: True boid separation
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(distances_boid_units, true_separation, 'g-', linewidth=2, label='Separation Force')
    ax3.axvline(x=config['min_separation'], color='g', linestyle='--', alpha=0.5,
                label=f"min_separation = {config['min_separation']}")
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Distance (original units)', fontsize=11)
    ax3.set_ylabel('Force', fontsize=11)
    ax3.set_title('True Boid Separation Rule', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim([0, config['neighborhood_dist'] * 1.2])

    # Plot 4: True boid alignment
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(distances_boid_units, true_alignment, 'm-', linewidth=2, label='Alignment Indicator')
    ax4.axvline(x=config['neighborhood_dist'], color='m', linestyle='--', alpha=0.5,
                label=f"neighborhood_dist = {config['neighborhood_dist']}")
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Distance (original units)', fontsize=11)
    ax4.set_ylabel('Alignment Weight', fontsize=11)
    ax4.set_title('True Boid Alignment Rule', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim([0, config['neighborhood_dist'] * 1.2])

    # Plot 5: True boid cohesion
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(distances_boid_units, true_cohesion, 'c-', linewidth=2, label='Cohesion Force')
    ax5.axvline(x=config['neighborhood_dist'], color='c', linestyle='--', alpha=0.5,
                label=f"neighborhood_dist = {config['neighborhood_dist']}")
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Distance (original units)', fontsize=11)
    ax5.set_ylabel('Force', fontsize=11)
    ax5.set_title('True Boid Cohesion Rule', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim([0, config['neighborhood_dist'] * 1.2])

    # Plot 6: Combined comparison - overlay learned on true
    ax6 = fig.add_subplot(gs[2, 1])

    # Normalize learned force for comparison
    learned_norm = learned_magnitude / (np.max(learned_magnitude) + 1e-8)

    # Normalize true forces for comparison
    sep_norm = true_separation / (np.max(true_separation) + 1e-8)
    coh_norm = true_cohesion / (np.max(true_cohesion) + 1e-8)

    ax6.plot(distances_boid_units, learned_norm, 'b-', linewidth=2.5, label='Learned (normalized)', alpha=0.8)
    ax6.plot(distances_boid_units, sep_norm, 'g--', linewidth=2, label='Separation (normalized)', alpha=0.7)
    ax6.plot(distances_boid_units, coh_norm, 'c--', linewidth=2, label='Cohesion (normalized)', alpha=0.7)
    ax6.axvline(x=config['min_separation'], color='g', linestyle=':', alpha=0.5)
    ax6.axvline(x=config['neighborhood_dist'], color='c', linestyle=':', alpha=0.5)
    ax6.set_xlabel('Distance (original units)', fontsize=11)
    ax6.set_ylabel('Normalized Force', fontsize=11)
    ax6.set_title('Comparison: Learned vs True Boid Rules', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_xlim([0, config['neighborhood_dist'] * 1.2])

    # Add main title
    fig.suptitle('InteractionParticle Model: Learned vs True Boid Interaction Functions',
                 fontsize=14, fontweight='bold', y=0.995)

    # Add text box with parameters
    param_text = (
        f"Boid Parameters:\n"
        f"min_separation = {config['min_separation']}\n"
        f"neighborhood_dist = {config['neighborhood_dist']}\n"
        f"separation_weight = {config['separation_weight']}\n"
        f"alignment_weight = {config['alignment_weight']}\n"
        f"cohesion_weight = {config['cohesion_weight']}"
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
