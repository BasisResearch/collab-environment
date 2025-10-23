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
    Interaction Network for learning particle dynamics.

    Based on: https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html

    This model learns pairwise interaction forces between particles using edge features.
    The interaction function is a MLP (self.lin_edge) that maps edge features to forces.
    Each particle has a learnable embedding (self.a).

    Edge Features (NOT normalized):
    --------------------------------
    For an edge from particle j to particle i:
    - delta_pos: Relative position (pos_j - pos_i) [2D vector]
    - r: Distance ||delta_pos|| [scalar]
    - delta_vel: Relative velocity (vel_j - vel_i) [2D vector]
    - embedding_i: Particle i's learnable embedding [embedding_dim vector]

    Total input size: 2 (delta_pos) + 1 (r) + 2 (delta_vel) + embedding_dim = 5 + embedding_dim

    Output:
    -------
    - Force from j acting on i [2D vector]

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - n_particles: Number of particles
        - n_particle_types: Number of particle types
        - max_radius: Maximum interaction radius (not used for normalization)
        - hidden_dim: Hidden dimension for MLP
        - embedding_dim: Dimension of particle embeddings
        - n_mp_layers: Number of MLP layers
        - input_size: Input size for MLP (should be 5 for delta_pos + r + delta_vel)
        - output_size: Output size (2 for 2D forces)
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
        # New input features: delta_pos(2) + r(1) + delta_vel(2) = 5
        self.input_size = config.get('input_size', 5)
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

        Edge features (NOT normalized):
        - Relative position: delta_pos = pos_j - pos_i
        - Relative distance: r = ||delta_pos||
        - Relative velocity: delta_vel = vel_j - vel_i

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
            Messages to send (force from j to i)
        """
        # Compute relative position and distance (NO normalization)
        delta_pos = pos_j - pos_i
        r = torch.sqrt(torch.sum(delta_pos ** 2, dim=1, keepdim=True))

        # Compute relative velocity (NO normalization)
        delta_vel = vel_j - vel_i

        # Construct input features
        # Format: [delta_pos(2), r(1), delta_vel(2), embedding_i(embedding_dim)]
        in_features = torch.cat([
            delta_pos,    # Relative position (2D)
            r,            # Distance (1D)
            delta_vel,    # Relative velocity (2D)
            embedding_i   # Particle embedding
        ], dim=-1)

        # Apply edge MLP
        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):
        """Update node features (identity function)."""
        return aggr_out

    def evaluate_interaction(self, delta_pos, delta_vel, embedding_idx=0):
        """
        Evaluate the learned interaction function.

        Parameters
        ----------
        delta_pos : torch.Tensor
            Relative positions [N, 2]
        delta_vel : torch.Tensor
            Relative velocities [N, 2]
        embedding_idx : int
            Index of particle embedding to use

        Returns
        -------
        forces : torch.Tensor
            Interaction forces [N, 2]
        """
        with torch.no_grad():
            n = delta_pos.shape[0]

            # Compute distance
            r = torch.sqrt(torch.sum(delta_pos ** 2, dim=1, keepdim=True))

            # Get embedding
            embedding = self.a[embedding_idx].unsqueeze(0).repeat(n, 1)

            # Construct input features
            in_features = torch.cat([
                delta_pos,   # Relative position (2D)
                r,           # Distance (1D)
                delta_vel,   # Relative velocity (2D)
                embedding    # Embedding
            ], dim=-1)

            # Evaluate MLP
            forces = self.lin_edge(in_features)

            return forces
