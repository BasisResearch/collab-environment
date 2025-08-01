import torch
from torch.utils.data import Dataset
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AnimalTrajectoryDataset(Dataset):
    def __init__(
        self,
        init_fn,
        update_fn,  # initialization of the animals, # updating of the animals
        species_configs,
        width,
        height,
        steps=5,
        num_samples=1000,
        velocity_scale=10,
        seed=2025,
    ):
        """
        species_configs: dictionary of dictionary. 1st layer of keys: spieces; 2nd layer: parameters
            different spieces shall have the same set of parameters but with different numerical values.
        species_counts: dictionary of species:count

        Returns:
        initial position, N (num of total boids across all spieces) x 2
        initial speed, N (num of total boids across all spieces) x 2
        future positions, step x N (num of total boids across all spieces) x 2
        spieces labels, N
        """
        self.width = width
        self.height = height
        self.sequences = []

        species_counts = {
            i: species_configs[i]["counts"] for i in species_configs.keys()
        }
        self.species_to_idx = {s: i for i, s in enumerate(species_counts.keys())}
        self.N = sum(species_counts.values())

        for _ in range(num_samples):  # loop through number of videos
            boids = init_fn(
                species_configs, species_counts, width, height, velocity_scale, seed
            )
            traj = []

            for _ in range(steps):
                pos = np.array([[b["x"], b["y"]] for b in boids], dtype=np.float32)
                vel = np.array([[b["dx"], b["dy"]] for b in boids], dtype=np.float32)

                traj.append((pos, vel))
                update_fn(boids, width, height, species_configs)

            positions = np.stack([p for (p, _) in traj]) / [width, height]
            species = [self.species_to_idx[b["species"]] for b in boids]
            self.sequences.append((positions, species))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        p, sp = self.sequences[idx]
        return (
            torch.tensor(p, dtype=torch.float32),  # [N, 2]
            torch.tensor(sp, dtype=torch.long),  # [N]
        )


def visualize_pair(p1, p2, v1=None, v2=None, starting_frame=0, label=None):
    """
    Takes 2 sets of positions and plot it on top of each other
    Make cartoons
    """

    # Create the figure and axes
    if label is not None:
        classes = len(np.unique(label))
        fig, axes = plt.subplots(1, classes, figsize=(classes * 4, 4))
    else:
        classes = 1
        label = np.zeros(p1.shape[1])
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        axes = [axes]

    # Set plot limits
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Initialization function: plot the background of each frame
    def init():
        for ax_ind in range(len(axes)):
            ax = axes[ax_ind]
            agent_ind = label == ax_ind

            ax.scatter(p1[0, agent_ind, 0], p1[0, agent_ind, 1], c="C0", alpha=0.5)

            ax.scatter(
                p2[0, agent_ind, 0], p2[0, agent_ind, 1], c="C1", alpha=0.5, marker="*"
            )
            ax.set_title("group " + str(ax_ind))
            if v1 is not None:
                ax.quiver(
                    p1[0, agent_ind, 0],
                    p1[0, agent_ind, 1],
                    v1[0, agent_ind, 0],
                    v1[0, agent_ind, 1],
                    c="C0",
                )
            if v2 is not None:
                ax.quiver(
                    p2[0, agent_ind, 0],
                    p2[0, agent_ind, 1],
                    v2[0, agent_ind, 0],
                    v2[0, agent_ind, 1],
                    c="C1",
                )

    def animate(i):
        print("animate", i)
        for ax_ind in range(len(axes)):
            ax = axes[ax_ind]
            agent_ind = label == ax_ind

            ax.scatter(p1[i, agent_ind, 0], p1[i, agent_ind, 1], c="C0", alpha=0.5)
            ax.scatter(
                p2[i, agent_ind, 0], p2[i, agent_ind, 1], c="C1", alpha=0.5, marker="*"
            )
            if v1 is not None:
                ax.quiver(
                    p1[i, agent_ind, 0],
                    p1[i, agent_ind, 1],
                    v1[i, agent_ind, 0],
                    v1[i, agent_ind, 1],
                    c="C0",
                )
            if v2 is not None:
                ax.quiver(
                    p2[i, agent_ind, 0],
                    p2[i, agent_ind, 1],
                    v2[i, agent_ind, 0],
                    v2[i, agent_ind, 1],
                    c="C1",
                )
            ax.set_title("Frame" + str(i))

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=np.arange(starting_frame + 1, p1.shape[0]),
        interval=20,
        repeat=False,
        blit=False,
    )

    plt.show()
    return ani, axes


def visualize_graph(batch, starting_frame=0, file_id=0, model=None, device=None):
    """ """
    model = model.to(device) if model else None

    p, species = batch
    p0 = p[
        file_id,
        starting_frame:,
    ]
    N = p0.shape[1]
    colors = ["C" + str(n % 10) for n in range(N)]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Set plot limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Initialization function: plot the background of each frame
    def init():
        ax.scatter(p0[file_id, :, 0], p0[file_id, :, 1], c=colors)
        # ax.plot([pos[0]+vel[0],pos[0]+vel[0]],[pos[0]+vel[0],pos[0]+vel[0]])

    def animate(i):
        ax.scatter(p[file_id, i, :, 0], p[file_id, i, :, 1], c=colors)
        ax.set_title("Frame" + str(i))

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=np.arange(starting_frame + 1, p.shape[1]),
        interval=5,
        repeat=False,
        blit=False,
    )

    # To save the animation as a GIF (requires ImageMagick or Pillow)
    # ani.save('flocking_test.gif', writer='imagemagick', fps=30)

    plt.show()
    return ani, ax
