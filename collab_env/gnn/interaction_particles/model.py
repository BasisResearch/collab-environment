"""
InteractionParticle model adapted from:
https://github.com/saalfeldlab/decomp-gnn/blob/main/src/ParticleGraph/models/Interaction_Particle.py

Model learning the acceleration of particles as a function of their relative distance and relative velocities.
"""

import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.models import MLP
import numpy as np


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
    - pos_i: Absolute position of particle i [2D vector] (NEW)
    - embedding_i: Particle i's learnable embedding [embedding_dim vector] (only if n_particle_types > 1)

    Total input size: 2 (delta_pos) + 1 (r) + 2 (delta_vel) + 2 (pos_i) + [embedding_dim if n_particle_types > 1]
                    = 7 + [embedding_dim if n_particle_types > 1]

    Output:
    -------
    - Force from j acting on i [2D vector]

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - n_particles: Number of particles
        - n_particle_types: Number of particle types (embeddings only used if > 1)
        - max_radius: Maximum interaction radius (not used for normalization)
        - hidden_dim: Hidden dimension for MLP
        - embedding_dim: Dimension of particle embeddings (ignored if n_particle_types=1)
        - n_mp_layers: Number of MLP layers
        - input_size: Input size for MLP (should be 7 + embedding_dim if n_particle_types > 1, else 7)
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
        # Input features: delta_pos(2) + r(1) + delta_vel(2) + pos_i(2) = 7
        self.input_size = config.get('input_size', 7)
        self.output_size = config.get('output_size', 2)  # 2D acceleration
        self.hidden_dim = config.get('hidden_dim', 128)
        self.n_layers = config.get('n_mp_layers', 3)
        self.n_particles = config.get('n_particles', 20)
        self.n_particle_types = config.get('n_particle_types', 1)
        self.max_radius = config.get('max_radius', 1.0)
        self.embedding_dim = config.get('embedding_dim', 16)
        self.dimension = dimension

        # Only use embeddings if we have multiple particle types
        self.use_embeddings = True #self.n_particle_types > 1

        # Edge interaction function (MLP)
        # Input size includes embeddings only if n_particle_types > 1
        mlp_input_size = self.input_size + (self.embedding_dim if self.use_embeddings else 0)

        # Use PyTorch Geometric's standardized MLP implementation
        # Benefits: Better maintained, supports various activations/norms, consistent with PyG ecosystem
        # Create channel list: [input_size, hidden_dim, ..., hidden_dim, output_size]
        channel_list = [mlp_input_size] + [self.hidden_dim] * (self.n_layers - 1) + [self.output_size]

        self.lin_edge = MLP(
            channel_list=channel_list,
            act='tanh',           # Tanh activation for hidden layers
            norm='layer_norm',    # Layer normalization
            plain_last=True,      # No activation on output layer (linear force prediction)
            dropout=0.0           # No dropout (can tune if overfitting)
        ).to(self.device)

        # Learnable particle embeddings (only if multiple particle types)
        if self.use_embeddings:
            # Shape: [n_particles, embedding_dim]
            self.a = nn.Parameter(
                torch.randn(self.n_particles, self.embedding_dim, device=self.device) * 0.1,
                requires_grad=True
            )
        else:
            self.a = None

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

        # Get particle embeddings (only if using embeddings)
        if self.use_embeddings:
            embedding = self.a[particle_id]
            pred = self.propagate(edge_index, pos=pos, vel=vel, embedding=embedding)
        else:
            pred = self.propagate(edge_index, pos=pos, vel=vel, embedding=None)

        return pred

    def message(self, pos_i, pos_j, vel_i, vel_j, embedding_i):
        """
        Construct messages from j to i.

        Edge features (NOT normalized):
        - Relative position: delta_pos = pos_j - pos_i
        - Relative distance: r = ||delta_pos||
        - Relative velocity: delta_vel = vel_j - vel_i
        - Absolute position: pos_i (NEW)
        - Particle embedding: embedding_i (only if n_particle_types > 1)

        Parameters
        ----------
        pos_i, pos_j : torch.Tensor
            Positions of nodes i and j
        vel_i, vel_j : torch.Tensor
            Velocities of nodes i and j
        embedding_i : torch.Tensor or None
            Embeddings of nodes i (None if n_particle_types=1)

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
        # Format: [delta_pos(2), r(1), delta_vel(2), pos_i(2), [embedding_i(embedding_dim)]]
        feature_list = [
            delta_pos,    # Relative position (2D)
            r,            # Distance (1D)
            delta_vel,    # Relative velocity (2D)
            pos_i         # Absolute position (2D) - NEW
        ]

        # Add embeddings only if using them
        if self.use_embeddings and embedding_i is not None:
            feature_list.append(embedding_i)

        in_features = torch.cat(feature_list, dim=-1)

        # Apply edge MLP
        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):
        """Update node features (identity function)."""
        return aggr_out

    def evaluate_interaction(self, delta_pos, delta_vel, pos_i=None, embedding_idx=0):
        """
        Evaluate the learned interaction function.

        Parameters
        ----------
        delta_pos : torch.Tensor
            Relative positions [N, 2]
        delta_vel : torch.Tensor
            Relative velocities [N, 2]
        pos_i : torch.Tensor, optional
            Absolute positions of particle i [N, 2]. If None, uses origin (0, 0)
        embedding_idx : int
            Index of particle embedding to use (ignored if n_particle_types=1)

        Returns
        -------
        forces : torch.Tensor
            Interaction forces [N, 2]
        """
        with torch.no_grad():
            n = delta_pos.shape[0]

            # Compute distance
            r = torch.sqrt(torch.sum(delta_pos ** 2, dim=1, keepdim=True))

            # Default pos_i to origin if not provided
            if pos_i is None:
                pos_i = torch.zeros((n, 2), device=delta_pos.device)

            # Construct input features
            feature_list = [
                delta_pos,   # Relative position (2D)
                r,           # Distance (1D)
                delta_vel,   # Relative velocity (2D)
                pos_i        # Absolute position (2D) - NEW
            ]

            # Add embedding only if using embeddings
            if self.use_embeddings:
                embedding = self.a[embedding_idx].unsqueeze(0).repeat(n, 1)
                feature_list.append(embedding)

            in_features = torch.cat(feature_list, dim=-1)

            # Evaluate MLP
            forces = self.lin_edge(in_features)

            return forces
