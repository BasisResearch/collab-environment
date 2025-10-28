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


