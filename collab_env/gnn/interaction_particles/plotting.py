"""
Plotting and comparison utilities for InteractionParticle model.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from loguru import logger

# Import actual boid functions for ground truth comparison
from collab_env.sim.boids_gnn_temp.boid import (
    fly_towards_center,
    avoid_others,
    match_velocity
)


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
    config=None,
    velocity_slice='stationary'
):
    """
    Plot learned 2D interaction functions from the model as vector field and heatmaps.

    Since boid separation/cohesion depend only on position (not velocity), we visualize
    the interaction function for a specific velocity configuration (slice through 4D space).

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
    velocity_slice : str
        Which velocity configuration to visualize:
        - 'stationary': Both particles stationary (v=0)
        - 'aligned': Both particles moving right →
        - 'opposing': Particles moving toward each other
        - 'perpendicular': Velocities at right angles

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import torch

    # Create a 2D grid of relative positions
    grid_size = 50
    max_dist = model.max_radius
    x = np.linspace(-max_dist, max_dist, grid_size)
    y = np.linspace(-max_dist, max_dist, grid_size)
    X, Y = np.meshgrid(x, y)

    # Flatten grid for batch processing
    positions = np.stack([X.flatten(), Y.flatten()], axis=-1)  # [N, 2]
    n_points = len(positions)

    # Compute distances
    distances = np.linalg.norm(positions, axis=1)

    # Evaluate interaction function with specified velocity configuration
    with torch.no_grad():
        # Convert to tensor
        delta_pos_t = torch.tensor(positions, dtype=torch.float32, device=model.device)

        # Compute distances
        distances_t = torch.sqrt(torch.sum(delta_pos_t ** 2, dim=1, keepdim=True))

        # Set velocity configuration based on slice
        # vel_i = 0 (particle i is stationary)
        # vel_j depends on the slice
        vel_j = torch.zeros((n_points, 2), device=model.device)

        if velocity_slice == 'stationary':
            # Both stationary - vel_j stays zero
            pass
        elif velocity_slice == 'aligned':
            # vel_j moving right
            vel_j[:, 0] = 1.0
        elif velocity_slice == 'opposing':
            # vel_j pointing toward i (opposite of position vector)
            vel_j = -delta_pos_t / (distances_t + 1e-8)
        elif velocity_slice == 'perpendicular':
            # vel_j perpendicular to position vector
            vel_j[:, 0] = -delta_pos_t[:, 1]
            vel_j[:, 1] = delta_pos_t[:, 0]
            vel_j = vel_j / (torch.norm(vel_j, dim=1, keepdim=True) + 1e-8)

        # Compute delta_vel = vel_j - vel_i (vel_i = 0, so delta_vel = vel_j)
        delta_vel = vel_j

        # Use model's evaluate_interaction method
        forces = model.evaluate_interaction(delta_pos_t, delta_vel, embedding_idx=particle_idx).cpu().numpy()

    # Reshape forces back to grid
    force_x = forces[:, 0].reshape(grid_size, grid_size)
    force_y = forces[:, 1].reshape(grid_size, grid_size)
    force_magnitude = np.sqrt(force_x**2 + force_y**2)

    # Compute radial and tangential components
    # Radial: component in direction of delta_pos
    # Tangential: component perpendicular to delta_pos
    radial_force = np.zeros_like(force_magnitude)
    tangential_force = np.zeros_like(force_magnitude)

    for i in range(grid_size):
        for j in range(grid_size):
            dx, dy = X[i, j], Y[i, j]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-8:
                # Unit vector in radial direction
                r_hat = np.array([dx, dy]) / dist
                # Unit vector in tangential direction (perpendicular)
                t_hat = np.array([-dy, dx]) / dist

                force_vec = np.array([force_x[i, j], force_y[i, j]])
                radial_force[i, j] = np.dot(force_vec, r_hat)
                tangential_force[i, j] = np.dot(force_vec, t_hat)

    # Create figure with 6 subplots (2 rows x 3 cols)
    fig = plt.figure(figsize=(18, 11))

    # Velocity mode description
    vel_desc = {
        'stationary': 'Stationary particles (v≈0)',
        'aligned': 'Both particles moving right →',
        'opposing': 'Particles moving toward each other ← →',
        'perpendicular': 'Perpendicular velocities (→ and ↑)'
    }

    # 1. Vector field plot
    ax1 = plt.subplot(2, 3, 1)
    skip = 3
    quiver = ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                        force_x[::skip, ::skip], force_y[::skip, ::skip],
                        force_magnitude[::skip, ::skip],
                        cmap='viridis', scale=None, scale_units='xy')
    ax1.set_xlabel('Relative Position X', fontsize=10)
    ax1.set_ylabel('Relative Position Y', fontsize=10)
    ax1.set_title('Force Vector Field', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(quiver, ax=ax1, label='Magnitude')

    # 2. Force magnitude heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.contourf(X, Y, force_magnitude, levels=20, cmap='hot')
    ax2.set_xlabel('Relative Position X', fontsize=10)
    ax2.set_ylabel('Relative Position Y', fontsize=10)
    ax2.set_title('Force Magnitude', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Magnitude')

    # 3. Radial component (repulsion/attraction)
    ax3 = plt.subplot(2, 3, 3)
    vmax = np.max(np.abs(radial_force))
    im3 = ax3.contourf(X, Y, radial_force, levels=20, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
    ax3.set_xlabel('Relative Position X', fontsize=10)
    ax3.set_ylabel('Relative Position Y', fontsize=10)
    ax3.set_title('Radial Component\n(+: repulsion, -: attraction)', fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='Force')

    # 4. Force X component heatmap
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.contourf(X, Y, force_x, levels=20, cmap='RdBu_r',
                       vmin=-np.max(np.abs(force_x)), vmax=np.max(np.abs(force_x)))
    ax4.set_xlabel('Relative Position X', fontsize=10)
    ax4.set_ylabel('Relative Position Y', fontsize=10)
    ax4.set_title('Force X Component', fontsize=11, fontweight='bold')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='Force X')

    # 5. Force Y component heatmap
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.contourf(X, Y, force_y, levels=20, cmap='RdBu_r',
                       vmin=-np.max(np.abs(force_y)), vmax=np.max(np.abs(force_y)))
    ax5.set_xlabel('Relative Position X', fontsize=10)
    ax5.set_ylabel('Relative Position Y', fontsize=10)
    ax5.set_title('Force Y Component', fontsize=11, fontweight='bold')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5, label='Force Y')

    # 6. Tangential component
    ax6 = plt.subplot(2, 3, 6)
    vmax_tan = np.max(np.abs(tangential_force))
    im6 = ax6.contourf(X, Y, tangential_force, levels=20, cmap='RdBu_r',
                       vmin=-vmax_tan, vmax=vmax_tan)
    ax6.set_xlabel('Relative Position X', fontsize=10)
    ax6.set_ylabel('Relative Position Y', fontsize=10)
    ax6.set_title('Tangential Component\n(perpendicular to radial)', fontsize=11, fontweight='bold')
    ax6.set_aspect('equal')
    plt.colorbar(im6, ax=ax6, label='Force')

    plt.suptitle(f'Learned 2D Interaction Function (Particle {particle_idx})\nVelocity slice: {vel_desc.get(velocity_slice, velocity_slice)}',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved interaction function plot to {save_path}")

    return fig


def plot_true_boid_rules_2d(config=None, scene_size=480.0, save_path=None):
    """
    Visualize true 2D boid rules as 2D heatmaps.

    Separation and cohesion depend only on position (radially symmetric).
    Alignment depends on relative velocity (not visualized here as it's velocity-dependent).

    Parameters
    ----------
    config : dict, optional
        Boid configuration dict
    scene_size : float
        Scene size in pixels
    save_path : str, optional
        Path to save plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Default config
    if config is None:
        config = {
            'visual_range': 50.0,
            'min_distance': 15.0,
            'avoid_factor': 0.05,
            'matching_factor': 0.5,
            'centering_factor': 0.005,
        }

    # Create 2D grid in pixel space
    grid_size = 100
    visual_range_normalized = config['visual_range'] / scene_size
    x = np.linspace(-visual_range_normalized, visual_range_normalized, grid_size)
    y = np.linspace(-visual_range_normalized, visual_range_normalized, grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute distances in pixels
    distances_px = np.sqrt(X**2 + Y**2) * scene_size

    # Compute true boid forces
    separation = np.zeros_like(distances_px)
    cohesion = np.zeros_like(distances_px)

    # Separation (repulsion within min_distance)
    mask_sep = distances_px < config['min_distance']
    separation[mask_sep] = config['avoid_factor'] * distances_px[mask_sep]

    # Cohesion (attraction within visual_range)
    mask_coh = distances_px < config['visual_range']
    cohesion[mask_coh] = config['centering_factor'] * distances_px[mask_coh]

    # Convert to force vectors (radial direction)
    # Separation: pushes away (positive radial)
    # Cohesion: pulls toward (negative radial)
    sep_force_x = np.zeros_like(X)
    sep_force_y = np.zeros_like(Y)
    coh_force_x = np.zeros_like(X)
    coh_force_y = np.zeros_like(Y)

    for i in range(grid_size):
        for j in range(grid_size):
            dist = np.sqrt(X[i,j]**2 + Y[i,j]**2)
            if dist > 1e-8:
                # Unit vector in radial direction
                r_hat_x = X[i,j] / dist
                r_hat_y = Y[i,j] / dist

                # Separation pushes away
                sep_force_x[i,j] = separation[i,j] * r_hat_x * scene_size
                sep_force_y[i,j] = separation[i,j] * r_hat_y * scene_size

                # Cohesion pulls toward (negative)
                coh_force_x[i,j] = -cohesion[i,j] * r_hat_x * scene_size
                coh_force_y[i,j] = -cohesion[i,j] * r_hat_y * scene_size

    # Combined force
    total_force_x = sep_force_x + coh_force_x
    total_force_y = sep_force_y + coh_force_y
    total_magnitude = np.sqrt(total_force_x**2 + total_force_y**2)
    sep_magnitude = np.sqrt(sep_force_x**2 + sep_force_y**2)
    coh_magnitude = np.sqrt(coh_force_x**2 + coh_force_y**2)

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    # 1. Separation force
    ax1 = plt.subplot(2, 3, 1)
    skip = 5
    quiver1 = ax1.quiver(X[::skip, ::skip]*scene_size, Y[::skip, ::skip]*scene_size,
                         sep_force_x[::skip, ::skip], sep_force_y[::skip, ::skip],
                         sep_magnitude[::skip, ::skip],
                         cmap='Reds', scale=None, scale_units='xy')
    circle1 = plt.Circle((0, 0), config['min_distance'], fill=False,
                         edgecolor='red', linestyle='--', linewidth=2, label='min_distance')
    ax1.add_patch(circle1)
    ax1.set_xlabel('Relative Position X (px)', fontsize=10)
    ax1.set_ylabel('Relative Position Y (px)', fontsize=10)
    ax1.set_title('Separation Force\n(Repulsion within min_distance)', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(quiver1, ax=ax1, label='Force')

    # 2. Separation magnitude heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.contourf(X*scene_size, Y*scene_size, sep_magnitude, levels=20, cmap='Reds')
    circle2 = plt.Circle((0, 0), config['min_distance'], fill=False,
                         edgecolor='darkred', linestyle='--', linewidth=2)
    ax2.add_patch(circle2)
    ax2.set_xlabel('Relative Position X (px)', fontsize=10)
    ax2.set_ylabel('Relative Position Y (px)', fontsize=10)
    ax2.set_title('Separation Magnitude', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Force')

    # 3. Cohesion force
    ax3 = plt.subplot(2, 3, 3)
    quiver3 = ax3.quiver(X[::skip, ::skip]*scene_size, Y[::skip, ::skip]*scene_size,
                         coh_force_x[::skip, ::skip], coh_force_y[::skip, ::skip],
                         coh_magnitude[::skip, ::skip],
                         cmap='Blues_r', scale=None, scale_units='xy')
    circle3 = plt.Circle((0, 0), config['visual_range'], fill=False,
                         edgecolor='blue', linestyle='--', linewidth=2, label='visual_range')
    ax3.add_patch(circle3)
    ax3.set_xlabel('Relative Position X (px)', fontsize=10)
    ax3.set_ylabel('Relative Position Y (px)', fontsize=10)
    ax3.set_title('Cohesion Force\n(Attraction within visual_range)', fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.colorbar(quiver3, ax=ax3, label='Force')

    # 4. Cohesion magnitude heatmap
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.contourf(X*scene_size, Y*scene_size, coh_magnitude, levels=20, cmap='Blues_r')
    circle4 = plt.Circle((0, 0), config['visual_range'], fill=False,
                         edgecolor='darkblue', linestyle='--', linewidth=2)
    ax4.add_patch(circle4)
    ax4.set_xlabel('Relative Position X (px)', fontsize=10)
    ax4.set_ylabel('Relative Position Y (px)', fontsize=10)
    ax4.set_title('Cohesion Magnitude', fontsize=11, fontweight='bold')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='Force')

    # 5. Combined force
    ax5 = plt.subplot(2, 3, 5)
    quiver5 = ax5.quiver(X[::skip, ::skip]*scene_size, Y[::skip, ::skip]*scene_size,
                         total_force_x[::skip, ::skip], total_force_y[::skip, ::skip],
                         total_magnitude[::skip, ::skip],
                         cmap='viridis', scale=None, scale_units='xy')
    circle5a = plt.Circle((0, 0), config['min_distance'], fill=False,
                          edgecolor='red', linestyle='--', linewidth=1.5, alpha=0.7)
    circle5b = plt.Circle((0, 0), config['visual_range'], fill=False,
                          edgecolor='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.add_patch(circle5a)
    ax5.add_patch(circle5b)
    ax5.set_xlabel('Relative Position X (px)', fontsize=10)
    ax5.set_ylabel('Relative Position Y (px)', fontsize=10)
    ax5.set_title('Combined Force\n(Separation + Cohesion)', fontsize=11, fontweight='bold')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(quiver5, ax=ax5, label='Force')

    # 6. Combined magnitude heatmap
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.contourf(X*scene_size, Y*scene_size, total_magnitude, levels=20, cmap='viridis')
    circle6a = plt.Circle((0, 0), config['min_distance'], fill=False,
                          edgecolor='red', linestyle='--', linewidth=1.5, alpha=0.7)
    circle6b = plt.Circle((0, 0), config['visual_range'], fill=False,
                          edgecolor='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6.add_patch(circle6a)
    ax6.add_patch(circle6b)
    ax6.set_xlabel('Relative Position X (px)', fontsize=10)
    ax6.set_ylabel('Relative Position Y (px)', fontsize=10)
    ax6.set_title('Combined Magnitude', fontsize=11, fontweight='bold')
    ax6.set_aspect('equal')
    plt.colorbar(im6, ax=ax6, label='Force')

    plt.suptitle('True 2D Boid Rules (Position-Dependent Forces)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved true boid rules plot to {save_path}")

    return fig


def evaluate_forces_on_grid(model, grid_size=60, max_dist=50.0, particle_idx=0, speed_magnitude=0.015):
    """
    Evaluate learned forces on a 2D grid for two velocity configurations.

    Setup:
    - Node i at square center (0.5, 0.5), stationary (vel_i = 0)
    - Node j at grid positions relative to i (delta_pos varies)

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    grid_size : int
        Grid resolution
    max_dist : float
        Maximum distance to evaluate (in same units as training data)
    particle_idx : int
        Particle embedding index
    speed_magnitude : float
        Magnitude of relative velocity to use (should match typical training data speed)
        Default: 0.015 (typical for normalized boid data)

    Returns
    -------
    dict with keys:
        - X, Y: meshgrid coordinates
        - away_total, away_pos, away_vel: forces when j moves away from i
        - towards_total, towards_pos, towards_vel: forces when j moves towards i
    """
    # Create 2D grid
    x = np.linspace(-max_dist, max_dist, grid_size)
    y = np.linspace(-max_dist, max_dist, grid_size)
    X, Y = np.meshgrid(x, y)

    # Flatten for batch processing
    positions = np.stack([X.flatten(), Y.flatten()], axis=-1)  # [N, 2]
    n_points = len(positions)

    # Convert to tensors
    delta_pos_t = torch.tensor(positions, dtype=torch.float32, device=model.device)

    # Set absolute position to square center (0.5, 0.5)
    # This is important because the model uses absolute position as a feature
    pos_i = torch.full((n_points, 2), 0.5, dtype=torch.float32, device=model.device)

    # Compute distances and velocity directions with REALISTIC MAGNITUDES
    distances = torch.sqrt(torch.sum(delta_pos_t ** 2, dim=1, keepdim=True))
    r_hat = delta_pos_t / (distances + 1e-8)  # Unit vector away from i

    # Use realistic speed magnitude (not unit vectors!)
    vel_away = r_hat * speed_magnitude  # Realistic velocity away from i
    vel_towards = -r_hat * speed_magnitude  # Realistic velocity towards i
    vel_zero = torch.zeros_like(delta_pos_t)

    # Evaluate forces for all configurations
    # NOTE: Model computes force ON i (at square center) DUE TO j (at grid position)
    # But we want to plot force ON j DUE TO i (Newton's 3rd law: F_j = -F_i)
    # So we negate the model output to match the true boid force convention

    # 1. Away: j moving away from i
    away_total = -model.evaluate_interaction(delta_pos_t, vel_away, pos_i=pos_i, embedding_idx=particle_idx).cpu().numpy()
    away_pos = -model.evaluate_interaction(delta_pos_t, vel_zero, pos_i=pos_i, embedding_idx=particle_idx).cpu().numpy()
    away_vel = away_total - away_pos

    # 2. Towards: j moving towards i
    towards_total = -model.evaluate_interaction(delta_pos_t, vel_towards, pos_i=pos_i, embedding_idx=particle_idx).cpu().numpy()
    towards_pos = -model.evaluate_interaction(delta_pos_t, vel_zero, pos_i=pos_i, embedding_idx=particle_idx).cpu().numpy()
    towards_vel = towards_total - towards_pos

    # Reshape to grid
    def reshape_to_grid(arr):
        return {
            'x': arr[:, 0].reshape(grid_size, grid_size),
            'y': arr[:, 1].reshape(grid_size, grid_size),
            'mag': np.sqrt(arr[:, 0]**2 + arr[:, 1]**2).reshape(grid_size, grid_size)
        }

    return {
        'X': X,
        'Y': Y,
        'away_total': reshape_to_grid(away_total),
        'away_pos': reshape_to_grid(away_pos),
        'away_vel': reshape_to_grid(away_vel),
        'towards_total': reshape_to_grid(towards_total),
        'towards_pos': reshape_to_grid(towards_pos),
        'towards_vel': reshape_to_grid(towards_vel),
    }


def evaluate_velocity_forces_on_grid(model, grid_size=60, max_vel=0.05, particle_idx=0):
    """
    Evaluate learned forces on a 2D velocity grid (symmetric decomposition).

    Setup:
    - Particles at SAME position (delta_pos = 0) at square center (0.5, 0.5)
    - Vary relative velocity on a 2D grid

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    grid_size : int
        Grid resolution
    max_vel : float
        Maximum relative velocity magnitude to evaluate
    particle_idx : int
        Particle embedding index

    Returns
    -------
    dict with keys:
        - VX, VY: meshgrid coordinates (relative velocity components)
        - forces: force vectors [grid_size, grid_size, 2]
        - force_mag: force magnitudes [grid_size, grid_size]
    """
    # Create 2D velocity grid
    vx = np.linspace(-max_vel, max_vel, grid_size)
    vy = np.linspace(-max_vel, max_vel, grid_size)
    VX, VY = np.meshgrid(vx, vy)

    # Flatten for batch processing
    delta_vels = np.stack([VX.flatten(), VY.flatten()], axis=-1)  # [N, 2]
    n_points = len(delta_vels)

    # Convert to tensors
    delta_vel_t = torch.tensor(delta_vels, dtype=torch.float32, device=model.device)

    # Particles at same position: delta_pos = 0
    delta_pos_t = torch.zeros_like(delta_vel_t)

    # Set absolute position to square center (0.5, 0.5)
    pos_i = torch.full((n_points, 2), 0.5, dtype=torch.float32, device=model.device)

    # Evaluate forces
    # NOTE: Model computes force ON i DUE TO j
    # With delta_pos=0, both particles are at same location (square center)
    # Force should depend only on relative velocity
    forces = -model.evaluate_interaction(delta_pos_t, delta_vel_t, pos_i=pos_i, embedding_idx=particle_idx).cpu().numpy()

    # Reshape to grid
    forces_x = forces[:, 0].reshape(grid_size, grid_size)
    forces_y = forces[:, 1].reshape(grid_size, grid_size)
    forces_mag = np.sqrt(forces_x**2 + forces_y**2)

    return {
        'VX': VX,
        'VY': VY,
        'forces': forces.reshape(grid_size, grid_size, 2),
        'forces_x': forces_x,
        'forces_y': forces_y,
        'force_mag': forces_mag
    }


def plot_force_decomposition(forces_dict, save_path=None, title_prefix="Learned", visual_range=None, min_distance=None):
    """
    Plot force decomposition for both velocity slices.

    Creates a 2x3 grid:
    - Row 1: Away scenario (Total | Position-Only | Velocity Residual)
    - Row 2: Towards scenario (Total | Position-Only | Velocity Residual)

    Parameters
    ----------
    forces_dict : dict
        Output from evaluate_forces_on_grid or evaluate_true_boid_forces
    save_path : str, optional
        Path to save figure
    title_prefix : str
        Prefix for title (e.g., "Learned" or "True Boid")
    visual_range : float, optional
        Visual range radius to draw as reference circle (in normalized units)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    X = forces_dict['X']
    Y = forces_dict['Y']

    fig = plt.figure(figsize=(18, 12))

    # Common settings
    skip = 3  # For quiver plots

    def plot_force_field(ax, force_dict, title):
        """Helper to plot a single force field."""
        fx, fy, mag = force_dict['x'], force_dict['y'], force_dict['mag']

        # Filter out near-zero vectors to help with autoscaling
        # Define threshold as 1% of max magnitude
        max_mag = np.max(mag)
        if max_mag > 1e-10:
            threshold = max_mag * 0.01
        else:
            threshold = 1e-10

        # Apply skip and create mask for non-zero vectors
        X_skip = X[::skip, ::skip]
        Y_skip = Y[::skip, ::skip]
        fx_skip = fx[::skip, ::skip]
        fy_skip = fy[::skip, ::skip]
        mag_skip = mag[::skip, ::skip]

        # Mask for vectors above threshold
        mask = mag_skip > threshold

        # Only plot non-trivial vectors
        if np.sum(mask) > 0:
            X_plot = X_skip[mask]
            Y_plot = Y_skip[mask]
            fx_plot = fx_skip[mask]
            fy_plot = fy_skip[mask]
            mag_plot = mag_skip[mask]

            # Vector field with autoscaling (filtering fixed the autoscale issue)
            quiver = ax.quiver(X_plot, Y_plot, fx_plot, fy_plot, mag_plot,
                              cmap='viridis', scale=None, scale_units='xy', alpha=0.8)
        else:
            # Fallback: create empty quiver
            quiver = ax.quiver([], [], [], [], [], cmap='viridis')

        # Mark origin (node i position)
        ax.scatter([0], [0], c='red', s=200, marker='o', edgecolors='black',
                  linewidth=2, zorder=10, label='Node i (origin)')

        # Draw visual range circle if provided
        if visual_range is not None:
            circle = plt.Circle((0, 0), visual_range, fill=False,
                              edgecolor='gray', linestyle='--', linewidth=1.5,
                              alpha=0.6, zorder=5, label=f'Visual range ({visual_range:.3f})')
            ax.add_patch(circle)

        # Draw min_distance circle if provided (for true boid forces)
        if min_distance is not None:
            min_circle = plt.Circle((0, 0), min_distance, fill=False,
                                   edgecolor='blue', linestyle='--', linewidth=1.5,
                                   alpha=0.6, zorder=5, label=f'Min distance ({min_distance:.3f})')
            ax.add_patch(min_circle)

        ax.set_xlabel('Relative x position (j - i)', fontsize=10)
        ax.set_ylabel('Relative y position (j - i)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        return quiver

    # Row 1: Away scenario
    ax1 = plt.subplot(2, 3, 1)
    q1 = plot_force_field(ax1, forces_dict['away_total'], 'I) Total Force\n(vel_j away from i)')
    plt.colorbar(q1, ax=ax1, label='Force mag')

    ax2 = plt.subplot(2, 3, 2)
    q2 = plot_force_field(ax2, forces_dict['away_pos'], 'II) Position-Only\n(vel=0)')
    plt.colorbar(q2, ax=ax2, label='Force mag')

    ax3 = plt.subplot(2, 3, 3)
    q3 = plot_force_field(ax3, forces_dict['away_vel'], 'III) Velocity Residual\n(I - II)')
    plt.colorbar(q3, ax=ax3, label='Force mag')

    # Row 2: Towards scenario
    ax4 = plt.subplot(2, 3, 4)
    q4 = plot_force_field(ax4, forces_dict['towards_total'], 'I) Total Force\n(vel_j towards i)')
    plt.colorbar(q4, ax=ax4, label='Force mag')

    ax5 = plt.subplot(2, 3, 5)
    q5 = plot_force_field(ax5, forces_dict['towards_pos'], 'II) Position-Only\n(vel=0)')
    plt.colorbar(q5, ax=ax5, label='Force mag')

    ax6 = plt.subplot(2, 3, 6)
    q6 = plot_force_field(ax6, forces_dict['towards_vel'], 'III) Velocity Residual\n(I - II)')
    plt.colorbar(q6, ax=ax6, label='Force mag')

    plt.suptitle(f'{title_prefix} Force on j due to i at origin (arrows show force on j at that position)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved force decomposition plot to {save_path}")

    return fig


def evaluate_true_boid_forces(config, grid_size=60, max_dist=50.0, scene_size=480.0):
    """
    Evaluate true 2D boid forces on a grid using actual boid.py functions.

    Computes the force experienced by particle j (at grid positions) due to
    particle i (at origin). This matches the natural interpretation: arrows show
    the force on a particle at that position relative to a reference particle at origin.

    Computes forces from the 3 boid rules:
    1. Cohesion (fly_towards_center): attraction within visual_range
    2. Separation (avoid_others): repulsion within min_distance
    3. Alignment (match_velocity): velocity matching within visual_range

    Parameters
    ----------
    config : dict
        Boid config with parameters in PIXEL SPACE:
        - min_distance: Separation threshold (pixels)
        - visual_range: Cohesion and alignment threshold (pixels)
        - avoid_factor: Separation weight
        - centering_factor: Cohesion weight
        - matching_factor: Alignment weight
        - speed_limit: Maximum speed (pixels/timestep)
        - independent: Whether to skip cohesion/alignment (default: False)
    grid_size : int
        Grid resolution
    max_dist : float
        Maximum distance in normalized space (recommend: visual_range * 1.5 / scene_size)
    scene_size : float
        Scene size in pixels (used to normalize config parameters)

    Returns
    -------
    dict with same structure as evaluate_forces_on_grid
    """
    # Create 2D grid in normalized space
    x = np.linspace(-max_dist, max_dist, grid_size)
    y = np.linspace(-max_dist, max_dist, grid_size)
    X, Y = np.meshgrid(x, y)

    # Normalize config parameters from pixel space to normalized space
    # The grid is in normalized space [-max_dist, max_dist], so we need to
    # convert pixel-space thresholds to normalized space
    normalized_config = config.copy()
    if 'min_distance' in config:
        normalized_config['min_distance'] = config['min_distance'] / scene_size
    if 'visual_range' in config:
        normalized_config['visual_range'] = config['visual_range'] / scene_size

    # Flatten for iteration
    positions = np.stack([X.flatten(), Y.flatten()], axis=-1)
    n_points = len(positions)

    # Compute radial unit vectors (direction from i to j)
    distances = np.sqrt(X**2 + Y**2).flatten()
    r_hat = np.zeros_like(positions)
    mask_nonzero = distances > 1e-8
    r_hat[mask_nonzero] = positions[mask_nonzero] / distances[mask_nonzero, np.newaxis]

    # Add 'independent' flag if not present in normalized config
    if 'independent' not in normalized_config:
        normalized_config['independent'] = False

    # Function to compute forces using actual boid functions
    def compute_boid_forces(vel_j_array):
        """
        Compute forces for all grid points using actual boid.py functions.

        vel_j_array: [n_points, 2] array of velocities for particle j
        """
        forces_total = np.zeros_like(positions)
        forces_pos_only = np.zeros_like(positions)

        for idx in range(n_points):
            # Create boid i at origin (stationary reference point)
            boid_i = {
                'x': 0.0,
                'y': 0.0,
                'dx': 0.0,
                'dy': 0.0,
                'species': 'A'
            }

            # Create boid j at grid position (this is the boid we compute forces FOR)
            pos_j = positions[idx]
            vel_j = vel_j_array[idx]

            boid_j = {
                'x': pos_j[0],
                'y': pos_j[1],
                'dx': vel_j[0],
                'dy': vel_j[1],
                'species': 'A'
            }

            # Compute force ON boid_j (at grid position) due TO boid_i (at origin)
            # Put boid_j in the list and call force functions on it
            boids = [boid_j, boid_i]

            fly_towards_center(boid_j, boids, normalized_config)  # Rule 1: Cohesion
            avoid_others(boid_j, boids, normalized_config)        # Rule 2: Separation
            match_velocity(boid_j, boids, normalized_config)      # Rule 3: Alignment

            # Force is change in velocity (from initial velocity)
            forces_total[idx] = [boid_j['dx'] - vel_j[0], boid_j['dy'] - vel_j[1]]

            # Compute position-only force (set velocities to zero)
            boid_j_zero = {
                'x': pos_j[0],
                'y': pos_j[1],
                'dx': 0.0,
                'dy': 0.0,
                'species': 'A'
            }
            boid_i_zero = {
                'x': 0.0,
                'y': 0.0,
                'dx': 0.0,
                'dy': 0.0,
                'species': 'A'
            }
            boids_zero = [boid_j_zero, boid_i_zero]

            fly_towards_center(boid_j_zero, boids_zero, normalized_config)
            avoid_others(boid_j_zero, boids_zero, normalized_config)
            match_velocity(boid_j_zero, boids_zero, normalized_config)

            forces_pos_only[idx] = [boid_j_zero['dx'], boid_j_zero['dy']]

        return forces_total, forces_pos_only

    # Use realistic velocity magnitude from config's speed_limit instead of unit vectors
    # The speed_limit is in pixel space, so normalize it
    if 'speed_limit' in config:
        speed_magnitude = config['speed_limit'] / scene_size
    else:
        # Fallback: use a typical speed (mean from basic datasets ~0.014)
        speed_magnitude = 0.015

    # Scenario 1: vel_j moving away from i (with realistic speed)
    vel_away = r_hat.copy() * speed_magnitude
    forces_away_total, forces_away_pos = compute_boid_forces(vel_away)
    forces_away_vel = forces_away_total - forces_away_pos

    # Scenario 2: vel_j moving towards i (with realistic speed)
    vel_towards = -r_hat.copy() * speed_magnitude
    forces_towards_total, forces_towards_pos = compute_boid_forces(vel_towards)
    forces_towards_vel = forces_towards_total - forces_towards_pos

    # Reshape to grid
    def make_force_dict(forces):
        return {
            'x': forces[:, 0].reshape(grid_size, grid_size),
            'y': forces[:, 1].reshape(grid_size, grid_size),
            'mag': np.linalg.norm(forces, axis=1).reshape(grid_size, grid_size)
        }

    return {
        'X': X,
        'Y': Y,
        'away_total': make_force_dict(forces_away_total),
        'away_pos': make_force_dict(forces_away_pos),
        'away_vel': make_force_dict(forces_away_vel),
        'towards_total': make_force_dict(forces_towards_total),
        'towards_pos': make_force_dict(forces_towards_pos),
        'towards_vel': make_force_dict(forces_towards_vel),
    }


def evaluate_true_boid_velocity_forces(config, grid_size=60, max_vel=0.05, scene_size=480.0):
    """
    Evaluate true boid forces on a 2D velocity grid (symmetric decomposition).

    Setup:
    - Two boids at SAME position (delta_pos = 0) at square center
    - Vary relative velocity on a 2D grid

    Parameters
    ----------
    config : dict
        Boid config with parameters in PIXEL SPACE (same as evaluate_true_boid_forces)
    grid_size : int
        Grid resolution
    max_vel : float
        Maximum relative velocity magnitude to evaluate (in normalized space)
    scene_size : float
        Scene size in pixels (used to normalize config parameters)

    Returns
    -------
    dict with keys:
        - VX, VY: meshgrid coordinates (relative velocity components)
        - forces: force vectors [grid_size, grid_size, 2]
        - force_mag: force magnitudes [grid_size, grid_size]
    """
    # Create 2D velocity grid in normalized space
    vx = np.linspace(-max_vel, max_vel, grid_size)
    vy = np.linspace(-max_vel, max_vel, grid_size)
    VX, VY = np.meshgrid(vx, vy)

    # Normalize config parameters from pixel space to normalized space
    normalized_config = config.copy()
    if 'min_distance' in config:
        normalized_config['min_distance'] = config['min_distance'] / scene_size
    if 'visual_range' in config:
        normalized_config['visual_range'] = config['visual_range'] / scene_size

    # Add 'independent' flag if not present
    if 'independent' not in normalized_config:
        normalized_config['independent'] = False

    # Flatten for iteration
    delta_vels = np.stack([VX.flatten(), VY.flatten()], axis=-1)  # [N, 2]
    n_points = len(delta_vels)

    # Compute forces at each velocity grid point
    forces = np.zeros_like(delta_vels)

    for idx in range(n_points):
        # Both boids at the same position (square center: 0.5, 0.5 in normalized space)
        # Boid i: stationary reference
        boid_i = {
            'x': 0.5,
            'y': 0.5,
            'dx': 0.0,
            'dy': 0.0,
            'species': 'A'
        }

        # Boid j: has relative velocity
        vel_j = delta_vels[idx]
        boid_j = {
            'x': 0.5,  # Same position as i
            'y': 0.5,
            'dx': vel_j[0],
            'dy': vel_j[1],
            'species': 'A'
        }

        # Initial velocity of boid_j
        initial_vel = np.array([boid_j['dx'], boid_j['dy']])

        # Compute force ON boid_j due TO boid_i
        # Put boids in a list and call force functions
        boids = [boid_j, boid_i]

        fly_towards_center(boid_j, boids, normalized_config)  # Rule 1: Cohesion
        avoid_others(boid_j, boids, normalized_config)        # Rule 2: Separation
        match_velocity(boid_j, boids, normalized_config)      # Rule 3: Alignment

        # Force is change in velocity
        final_vel = np.array([boid_j['dx'], boid_j['dy']])
        forces[idx] = final_vel - initial_vel

    # Reshape to grid
    forces_x = forces[:, 0].reshape(grid_size, grid_size)
    forces_y = forces[:, 1].reshape(grid_size, grid_size)
    forces_mag = np.sqrt(forces_x**2 + forces_y**2)

    return {
        'VX': VX,
        'VY': VY,
        'forces': forces.reshape(grid_size, grid_size, 2),
        'forces_x': forces_x,
        'forces_y': forces_y,
        'force_mag': forces_mag
    }


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

    # Get learned forces using evaluate_interaction
    # Create delta_pos array along x-axis (particle j to the right of particle i)
    with torch.no_grad():
        delta_pos = torch.zeros((len(distances_fine), 2), dtype=torch.float32, device=model.device)
        delta_pos[:, 0] = torch.tensor(distances_fine, dtype=torch.float32)  # Position along x-axis

        # Use zero relative velocity (position-only forces)
        delta_vel = torch.zeros_like(delta_pos)

        # Evaluate interaction
        learned_forces = model.evaluate_interaction(delta_pos, delta_vel, embedding_idx=particle_idx).cpu().numpy()

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


def plot_symmetric_force_decomposition(position_forces, velocity_forces, save_path=None,
                                       title_prefix="Learned", visual_range=None):
    """
    Plot symmetric force decomposition: position-dependent and velocity-dependent forces.

    Creates a 1x2 grid:
    - Left: Position-dependent forces (velocity=0, vary position)
    - Right: Velocity-dependent forces (position=0, vary velocity)

    Parameters
    ----------
    position_forces : dict
        Output from evaluate_forces_on_grid (with delta_vel=0)
    velocity_forces : dict
        Output from evaluate_velocity_forces_on_grid (with delta_pos=0)
    save_path : str, optional
        Path to save figure
    title_prefix : str
        Prefix for title (e.g., "Learned" or "True Boid")
    visual_range : float, optional
        Visual range radius to draw as reference circle

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT: Position-dependent forces (velocity = 0)
    ax = axes[0]
    X, Y = position_forces['X'], position_forces['Y']

    # Plot force magnitude as heatmap
    im = ax.pcolormesh(X, Y, position_forces['away_pos']['mag'],
                       cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='Force Magnitude')

    # Overlay force vectors (subsample for clarity)
    step = max(1, len(X) // 15)
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              position_forces['away_pos']['x'][::step, ::step],
              position_forces['away_pos']['y'][::step, ::step],
              color='white', alpha=0.7, scale=None, scale_units='xy')

    ax.set_xlabel('Δx (relative position)')
    ax.set_ylabel('Δy (relative position)')
    ax.set_title(f'{title_prefix}: Position-Dependent Forces\n(Δv = 0)')
    ax.set_aspect('equal')
    ax.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.5)

    # Add visual range circle if provided
    if visual_range is not None:
        circle = plt.Circle((0, 0), visual_range, fill=False,
                           edgecolor='red', linestyle='--', linewidth=2,
                           label=f'Visual range = {visual_range:.2f}')
        ax.add_patch(circle)
        ax.legend(loc='upper right')

    # RIGHT: Velocity-dependent forces (position = 0)
    ax = axes[1]
    VX, VY = velocity_forces['VX'], velocity_forces['VY']

    # Plot force magnitude as heatmap
    im = ax.pcolormesh(VX, VY, velocity_forces['force_mag'],
                       cmap='plasma', shading='auto')
    plt.colorbar(im, ax=ax, label='Force Magnitude')

    # Overlay force vectors (subsample for clarity)
    step = max(1, len(VX) // 15)
    ax.quiver(VX[::step, ::step], VY[::step, ::step],
              velocity_forces['forces_x'][::step, ::step],
              velocity_forces['forces_y'][::step, ::step],
              color='white', alpha=0.7, scale=None, scale_units='xy')

    ax.set_xlabel('Δv_x (relative velocity)')
    ax.set_ylabel('Δv_y (relative velocity)')
    ax.set_title(f'{title_prefix}: Velocity-Dependent Forces\n(Δpos = 0)')
    ax.set_aspect('equal')
    ax.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved symmetric force decomposition plot to {save_path}")

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

        # Add initial velocity vector
        if T >= 2:
            initial_vel = ground_truth[1, i] - ground_truth[0, i]
            # Scale for visibility (multiply by 3 for better visualization)
            scale = 3.0
            ax.arrow(ground_truth[0, i, 0], ground_truth[0, i, 1],
                    initial_vel[0] * scale, initial_vel[1] * scale,
                    head_width=0.02, head_length=0.015, fc='blue', ec='blue',
                    alpha=0.7, linewidth=1.5, zorder=4)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([-0.3, 1.3])

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

        # Add initial velocity vector (same initial velocity as ground truth)
        if T >= 2:
            initial_vel = ground_truth[1, i] - ground_truth[0, i]
            # Scale for visibility (multiply by 3 for better visualization)
            scale = 3.0
            ax.arrow(predicted[0, i, 0], predicted[0, i, 1],
                    initial_vel[0] * scale, initial_vel[1] * scale,
                    head_width=0.02, head_length=0.015, fc='blue', ec='blue',
                    alpha=0.7, linewidth=1.5, zorder=4)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Predicted (Model)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([-0.3, 1.3])

    # Add legend
    from matplotlib.patches import Patch, FancyArrow
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Start'),
        Patch(facecolor='red', edgecolor='black', label='End'),
        Patch(facecolor='blue', edgecolor='blue', label='Initial Velocity (3x scaled)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11)

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
