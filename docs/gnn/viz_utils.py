import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_tracks(tracks, episode_id, figsize=(6, 4)):

    # Plot XY trajectories for first 5 agents
    fig, ax = plt.subplots(figsize=figsize)

    agent_ids = tracks["agent_id"].unique()#[:5]
    for agent_id in agent_ids:
        agent_tracks = tracks[tracks["agent_id"] == agent_id]
        ax.plot(agent_tracks["x"], agent_tracks["y"], "-", alpha=0.7, label=f"Agent {agent_id}")
        # Mark start position
        ax.scatter(agent_tracks["x"].iloc[0], agent_tracks["y"].iloc[0], s=50, marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Agent Trajectories - {episode_id}")
    # ax.legend()
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.show()
    


def create_animation_2d(times, trajectories, frame_skip=5, figsize=(8, 8)):
    """Create 2D animation of particle motion."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set axis limits
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    scatter = ax.scatter([], [], s=50, c='blue', alpha=0.6)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    frames = range(0, len(times), frame_skip)
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter, time_text
    
    def update(frame):
        positions = trajectories[frame, :, :2]
        scatter.set_offsets(positions)
        time_text.set_text(f't = {times[frame]:.2f}')
        return scatter, time_text
    
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, 
                         interval=50, blit=True)
    plt.close(fig)
    return anim

def convert_tracks_to_trajectories(tracks):
    # return 3d tensor: time, particle, dimension
    # need to pivot this
    n_agents = tracks['agent_id'].nunique()
    trajectories = tracks[['agent_id','time_index','x','y']].pivot(index='time_index', columns='agent_id', values=['x','y']).values.reshape(
		-1, 2, n_agents
	).transpose(0, 2, 1) # this should fail if n_agents is not constant
    
    times = tracks['time_index'].unique()
    
    return times, trajectories

