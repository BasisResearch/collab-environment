"""Visualization utilities for NRI model."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from loguru import logger


def generate_rollout(model, rel_rec, rel_send, initial_positions, initial_velocities,
                    species, rollout_steps=100, context_len=10, device='cpu'):
    """
    Generate rollout predictions using trained NRI model.
    
    Args:
        model: Trained NRI model
        rel_rec, rel_send: Relation matrices
        initial_positions: [1, agents, timesteps, 2] initial positions
        initial_velocities: [1, agents, timesteps, 2] initial velocities  
        species: [1, agents] species labels
        rollout_steps: Number of steps to rollout
        context_len: Length of context window for predictions
        device: Device to run on
        
    Returns:
        rollout_positions: [agents, context_len+rollout_steps, 2] context + predictions 
        rollout_velocities: [agents, context_len+rollout_steps, 2] context + predictions
        edge_probs: [edges, n_edge_types] edge type probabilities
    """
    model.eval()
    
    # gt_frames defaults to 0 - use pure predictions after context
    
    # Use initial context - start with exactly what the model needs
    pos = initial_positions[:, :, :context_len].clone().to(device)
    vel = initial_velocities[:, :, :context_len].clone().to(device)
    spec = species.to(device)
    
    # No additional frames needed! The model should use the context_len frames
    # The gt_frames parameter controls how many *predictions* we replace with GT,
    # but we don't need to extend the context before starting predictions
    
    # Now generate predictions - these are what we'll return
    predicted_positions = []
    predicted_velocities = []
    edge_probs = None
    
    with torch.no_grad():
        for step in range(rollout_steps):
            # Always use model predictions (gt_frames=0)
            recent_pos = pos[:, :, -context_len:]
            recent_vel = vel[:, :, -context_len:]
            
            # Prepare NRI input
            nri_data = model.prepare_boids_data(recent_pos, recent_vel, spec)
            
            # Predict next step
            output, prob, _ = model(nri_data, rel_rec, rel_send)
            
            # Store edge probabilities (only once)
            if edge_probs is None:
                edge_probs = prob[0]
            
            # Extract predictions [batch, agents, 1, 4]
            pred = output[:, :, 0, :]
            next_pos = pred[:, :, :2].unsqueeze(2)
            next_vel = pred[:, :, 2:4].unsqueeze(2)
            
            # Check if predictions are accelerations (very small values)
            if torch.abs(next_pos).mean() < 0.01:
                # Integrate accelerations
                next_vel = vel[:, :, -1:] + next_pos * 0.01
                next_pos = pos[:, :, -1:] + next_vel * 0.01
            
            # Append to sequences for next prediction
            pos = torch.cat([pos, next_pos], dim=2)
            vel = torch.cat([vel, next_vel], dim=2)
            
            # Store positions/velocities for return
            predicted_positions.append(next_pos[0, :, 0])
            predicted_velocities.append(next_vel[0, :, 0])
    
    # Stack predicted positions and velocities
    pred_pos_tensor = torch.stack(predicted_positions, dim=1)  # [agents, rollout_steps]
    pred_vel_tensor = torch.stack(predicted_velocities, dim=1)
    
    # Return full context + predictions
    # Extract the full context that was used
    context_frames = initial_positions[0, :, :context_len, :2]  # [agents, context_len, 2] 
    context_vel = initial_velocities[0, :, :context_len, :2]
    
    # Concatenate: [context_frames, predicted_frames]
    full_rollout_pos = torch.cat([
        context_frames,  # [agents, context_len, 2]
        pred_pos_tensor  # [agents, rollout_steps, 2]
    ], dim=1)  # [agents, context_len + rollout_steps, 2]
    
    full_rollout_vel = torch.cat([
        context_vel,  # [agents, context_len, 2]
        pred_vel_tensor  # [agents, rollout_steps, 2]
    ], dim=1)  # [agents, context_len + rollout_steps, 2]
    
    return full_rollout_pos, full_rollout_vel, edge_probs


def plot_trajectories_and_interactions(ground_truth_pos, predicted_pos, edge_probs=None,
                                       save_path='nri_visualization.png',
                                       xlim=(0, 1), ylim=(0, 1), skip_frames=0):
    """
    Create static visualization of trajectories and interaction matrix.
    
    Args:
        ground_truth_pos: [agents, timesteps, 2] ground truth positions
        predicted_pos: [agents, timesteps, 2] predicted positions
        edge_probs: [edges, n_edge_types] edge type probabilities
        save_path: Path to save figure
        xlim: X-axis limits (min, max)
        ylim: Y-axis limits (min, max)
        skip_frames: Number of initial frames to skip in visualization
    """
    n_agents = ground_truth_pos.shape[0]
    colors = plt.cm.tab20(np.linspace(0, 1, n_agents))
    
    n_cols = 3 if edge_probs is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
    
    # Plot ground truth trajectories
    ax = axes[0]
    for i in range(n_agents):
        traj = ground_truth_pos[i].cpu() if hasattr(ground_truth_pos, 'cpu') else ground_truth_pos[i]
        traj = traj[skip_frames:]  # Skip initial frames
        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], c=colors[i], alpha=0.7, linewidth=1.5)
            ax.scatter(traj[0, 0], traj[0, 1], c=[colors[i]], s=50, marker='o', edgecolor='black')
    ax.set_title('Ground Truth Trajectories')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Plot predicted trajectories
    ax = axes[1]
    for i in range(n_agents):
        traj = predicted_pos[i].cpu() if hasattr(predicted_pos, 'cpu') else predicted_pos[i]
        traj = traj[skip_frames:]  # Skip initial frames
        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], c=colors[i], alpha=0.7, linewidth=1.5)
            ax.scatter(traj[0, 0], traj[0, 1], c=[colors[i]], s=50, marker='o', edgecolor='black')
    ax.set_title('NRI Predicted Trajectories')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Plot interaction matrix if available
    if edge_probs is not None:
        ax = axes[2]
        
        # Convert edge probabilities to adjacency matrix
        adj_matrix = edge_probs_to_adjacency(edge_probs, n_agents)
        
        im = ax.imshow(adj_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title('Inferred Interaction Matrix')
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Agent ID')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Interaction Probability')
        
        # Add grid
        ax.set_xticks(np.arange(n_agents))
        ax.set_yticks(np.arange(n_agents))
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()


def edge_probs_to_adjacency(edge_probs, n_agents):
    """Convert edge probabilities to adjacency matrix."""
    adj_matrix = np.zeros((n_agents, n_agents))
    
    edge_idx = 0
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                # Use interaction probability (typically edge type 1)
                if edge_probs.shape[1] > 1:
                    prob = edge_probs[edge_idx, 1].item() if hasattr(edge_probs, 'item') else edge_probs[edge_idx, 1]
                else:
                    prob = edge_probs[edge_idx, 0].item() if hasattr(edge_probs, 'item') else edge_probs[edge_idx, 0]
                adj_matrix[i, j] = prob
                edge_idx += 1
    
    return adj_matrix


def create_animation(ground_truth_pos, predicted_pos, save_path='nri_rollout.mp4', fps=20, 
                    xlim=(0, 1), ylim=(0, 1)):
    """
    Create animation comparing ground truth and predicted trajectories.
    
    Args:
        ground_truth_pos: [agents, timesteps, 2] ground truth positions
        predicted_pos: [agents, timesteps, 2] predicted positions
        save_path: Path to save animation
        fps: Frames per second for animation
        xlim: X-axis limits (min, max)
        ylim: Y-axis limits (min, max)
    """
    n_agents = ground_truth_pos.shape[0]
    n_timesteps = min(ground_truth_pos.shape[1], predicted_pos.shape[1])
    colors = plt.cm.tab20(np.linspace(0, 1, n_agents))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Setup plots
    ax1.set_title('Ground Truth')
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.set_title('NRI Prediction')
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Initialize scatter plots with dummy data
    dummy_pos = np.zeros((n_agents, 2))
    scatter1 = ax1.scatter(dummy_pos[:, 0], dummy_pos[:, 1], c=colors, s=50)
    scatter2 = ax2.scatter(dummy_pos[:, 0], dummy_pos[:, 1], c=colors, s=50)
    
    # Initialize trails
    trail_length = 20
    trails1 = [ax1.plot([], [], c=colors[i], alpha=0.3, linewidth=1)[0] for i in range(n_agents)]
    trails2 = [ax2.plot([], [], c=colors[i], alpha=0.3, linewidth=1)[0] for i in range(n_agents)]
    
    def animate(frame):
        # Update ground truth
        if frame < ground_truth_pos.shape[1]:
            gt_pos = ground_truth_pos[:, frame]
            if hasattr(gt_pos, 'cpu'):
                gt_pos = gt_pos.cpu().numpy()
            scatter1.set_offsets(gt_pos)
            
            # Update trails
            trail_start = max(0, frame - trail_length)
            for i in range(n_agents):
                trail_data = ground_truth_pos[i, trail_start:frame+1]
                if hasattr(trail_data, 'cpu'):
                    trail_data = trail_data.cpu().numpy()
                if len(trail_data) > 0:
                    trails1[i].set_data(trail_data[:, 0], trail_data[:, 1])
        
        # Update prediction
        if frame < predicted_pos.shape[1]:
            pred_pos = predicted_pos[:, frame]
            if hasattr(pred_pos, 'cpu'):
                pred_pos = pred_pos.cpu().numpy()
            scatter2.set_offsets(pred_pos)
            
            # Update trails
            trail_start = max(0, frame - trail_length)
            for i in range(n_agents):
                trail_data = predicted_pos[i, trail_start:frame+1]
                if hasattr(trail_data, 'cpu'):
                    trail_data = trail_data.cpu().numpy()
                if len(trail_data) > 0:
                    trails2[i].set_data(trail_data[:, 0], trail_data[:, 1])
        
        ax1.set_title(f'Ground Truth (Frame {frame})')
        ax2.set_title(f'NRI Prediction (Frame {frame})')
        
        return [scatter1, scatter2] + trails1 + trails2
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_timesteps, 
                                  interval=1000/fps, blit=True)
    
    # Save animation
    try:
        anim.save(save_path, writer='ffmpeg', fps=fps)
        logger.info(f"Saved animation to {save_path}")
    except Exception as e:
        logger.warning(f"Could not save animation: {e}")
        logger.info("Saving as GIF instead...")
        save_path = save_path.replace('.mp4', '.gif')
        anim.save(save_path, writer='pillow', fps=fps)
        logger.info(f"Saved animation as GIF to {save_path}")
    
    plt.close()