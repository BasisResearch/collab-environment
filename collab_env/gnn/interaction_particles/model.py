"""
InteractionParticle model adapted from:
https://github.com/saalfeldlab/decomp-gnn/blob/main/src/ParticleGraph/models/Interaction_Particle.py

Model learning the acceleration of particles as a function of their relative distance and relative velocities.
"""

import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import numpy as np


class MLP(nn.Module):
    """Multi-Layer Perceptron for edge interaction function."""

    def __init__(self, input_size, output_size, nlayers, hidden_size, device):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.device = device
        self.to(device)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))  # tanh activation for hidden layers
        x = self.layers[-1](x)  # No activation on output
        return x


class InteractionParticle(pyg.nn.MessagePassing):
    """
    Interaction Network as proposed in:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html

    Model learning the acceleration of particles as a function of their relative distance and relative velocities.
    The interaction function is defined by a MLP self.lin_edge
    The particle embedding is defined by a table self.a

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - n_particles: number of particles
        - n_particle_types: number of particle types
        - max_radius: maximum interaction radius
        - hidden_dim: hidden dimension for MLP
        - embedding_dim: dimension of particle embeddings
        - n_mp_layers: number of message passing layers
        - input_size: input size for MLP
        - output_size: output size (typically 2 for 2D acceleration)
    device : torch.device
        Device to run the model on
    aggr_type : str, optional
        Aggregation type for message passing (default: 'add')
    dimension : int, optional
        Spatial dimension (default: 2)
    """

    def __init__(self, config, device, aggr_type='add', dimension=2):
        super(InteractionParticle, self).__init__(aggr=aggr_type)

        self.device = device
        self.input_size = config.get('input_size', 7)  # delta_pos(2) + r(1) + velocities(4)
        self.output_size = config.get('output_size', 2)  # 2D acceleration
        self.hidden_dim = config.get('hidden_dim', 128)
        self.n_layers = config.get('n_mp_layers', 3)
        self.n_particles = config.get('n_particles', 20)
        self.n_particle_types = config.get('n_particle_types', 1)
        self.max_radius = config.get('max_radius', 1.0)
        self.embedding_dim = config.get('embedding_dim', 16)
        self.dimension = dimension

        # Edge interaction function (MLP)
        self.lin_edge = MLP(
            input_size=self.input_size + self.embedding_dim,
            output_size=self.output_size,
            nlayers=self.n_layers,
            hidden_size=self.hidden_dim,
            device=self.device
        )

        # Learnable particle embeddings
        # Shape: [n_particles, embedding_dim]
        self.a = nn.Parameter(
            torch.randn(self.n_particles, self.embedding_dim, device=self.device) * 0.1,
            requires_grad=True
        )

    def forward(self, data):
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data with attributes:
            - x: node features [N, feature_dim]
                Expected format: [particle_id, pos_x, pos_y, vel_x, vel_y, ...]
            - edge_index: edge connectivity [2, E]

        Returns
        -------
        pred : torch.Tensor
            Predicted accelerations [N, 2]
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Extract features from node features
        # Assuming x format: [particle_id, pos_x, pos_y, vel_x, vel_y, ...]
        particle_id = x[:, 0].long()
        pos = x[:, 1:1+self.dimension]
        vel = x[:, 1+self.dimension:1+2*self.dimension]

        # Get particle embeddings
        embedding = self.a[particle_id]

        # Propagate messages
        pred = self.propagate(edge_index, pos=pos, vel=vel, embedding=embedding)

        return pred

    def message(self, pos_i, pos_j, vel_i, vel_j, embedding_i):
        """
        Construct messages from j to i.

        Parameters
        ----------
        pos_i, pos_j : torch.Tensor
            Positions of nodes i and j
        vel_i, vel_j : torch.Tensor
            Velocities of nodes i and j
        embedding_i : torch.Tensor
            Embeddings of nodes i

        Returns
        -------
        out : torch.Tensor
            Messages to send
        """
        # Compute relative position and distance
        delta_pos = pos_j - pos_i
        r = torch.sqrt(torch.sum(delta_pos ** 2, dim=1, keepdim=True)) / self.max_radius
        delta_pos_norm = delta_pos / self.max_radius

        # Normalize velocities
        vel_i_norm = vel_i / (torch.norm(vel_i, dim=1, keepdim=True) + 1e-8)
        vel_j_norm = vel_j / (torch.norm(vel_j, dim=1, keepdim=True) + 1e-8)

        # Construct input features
        # Format: [delta_pos(2), r(1), vel_i(2), vel_j(2), embedding_i(embedding_dim)]
        in_features = torch.cat([
            delta_pos_norm,  # Normalized relative position
            r,  # Normalized distance
            vel_i_norm,  # Normalized velocity of i
            vel_j_norm,  # Normalized velocity of j
            embedding_i  # Particle embedding
        ], dim=-1)

        # Apply edge MLP
        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):
        """Update node features (identity function)."""
        return aggr_out

    def get_interaction_function(self, embedding_idx=0):
        """
        Get the learned interaction function for a specific particle embedding.

        Parameters
        ----------
        embedding_idx : int
            Index of particle embedding to use

        Returns
        -------
        function : callable
            Function that takes (distance, relative_velocity) and returns force
        """
        def interaction_fn(distances, rel_vel=None):
            """
            Evaluate interaction function.

            Parameters
            ----------
            distances : np.ndarray
                Array of distances
            rel_vel : np.ndarray, optional
                Relative velocities (not used in basic version)

            Returns
            -------
            forces : np.ndarray
                Interaction forces
            """
            with torch.no_grad():
                # Convert to tensor
                distances_t = torch.tensor(distances, dtype=torch.float32, device=self.device)
                n = len(distances_t)

                # Create dummy features
                delta_pos = torch.zeros((n, 2), device=self.device)
                delta_pos[:, 0] = distances_t / self.max_radius  # Put distance in x direction
                r = distances_t.unsqueeze(1) / self.max_radius

                # Dummy velocities (unit vector in x direction)
                vel_i = torch.zeros((n, 2), device=self.device)
                vel_j = torch.zeros((n, 2), device=self.device)
                vel_i[:, 0] = 1.0
                vel_j[:, 0] = 1.0

                # Get embedding
                embedding = self.a[embedding_idx].unsqueeze(0).repeat(n, 1)

                # Construct input
                in_features = torch.cat([
                    delta_pos,
                    r,
                    vel_i,
                    vel_j,
                    embedding
                ], dim=-1)

                # Evaluate MLP
                forces = self.lin_edge(in_features)

                return forces.cpu().numpy()

        return interaction_fn
