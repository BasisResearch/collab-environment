"""
GNN model registry for GraphGym architecture search.

Supports multiple GNN layer types:
- GCNConv: Graph Convolutional Network
- GATv2Conv: Graph Attention Network v2 (improved)
- GINConv: Graph Isomorphism Network
- SAGEConv: GraphSAGE
- GeneralConv: General GNN layer (combines multiple mechanisms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    GINConv,
    SAGEConv,
    GENConv,
    BatchNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from typing import Optional, Callable
from loguru import logger


class GraphGymGNN(nn.Module):
    """
    Flexible GNN architecture following GraphGym's design space.

    Architecture:
        Pre-MP layers -> Message Passing layers -> Post-MP layers -> Output

    Each stage can have skip connections (skipsum or skipconcat).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers_pre: int = 1,
        num_layers_mp: int = 2,
        num_layers_post: int = 1,
        layer_type: str = "gatv2conv",
        stage_type: str = "stack",  # 'stack', 'skipsum', 'skipconcat'
        agg: str = "add",
        heads: int = 1,
        dropout: float = 0.0,
        batchnorm: bool = True,
        act: str = "prelu",
        **kwargs
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers_pre = num_layers_pre
        self.num_layers_mp = num_layers_mp
        self.num_layers_post = num_layers_post
        self.layer_type = layer_type.lower()
        self.stage_type = stage_type.lower()
        self.agg = agg
        self.heads = heads
        self.dropout = dropout
        self.use_batchnorm = batchnorm

        # Activation function
        self.act = self._get_activation(act)

        # For multi-head attention (GAT), adjust hidden dim
        if self.layer_type == "gatv2conv":
            self.mp_hidden_dim = hidden_dim // heads
        else:
            self.mp_hidden_dim = hidden_dim

        # Pre-MP layers (MLPs)
        self.pre_mp = nn.ModuleList()
        if num_layers_pre > 0:
            self.pre_mp.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers_pre - 1):
                self.pre_mp.append(nn.Linear(hidden_dim, hidden_dim))

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        self.mp_batchnorms = nn.ModuleList() if batchnorm else None

        for i in range(num_layers_mp):
            # First MP layer takes pre-MP output
            in_channels = hidden_dim if i == 0 else self.mp_hidden_dim * heads

            # Create GNN layer
            layer = self._create_gnn_layer(in_channels, self.mp_hidden_dim, layer_type, heads)
            self.mp_layers.append(layer)

            # Batch norm
            if batchnorm:
                self.mp_batchnorms.append(BatchNorm(self.mp_hidden_dim * heads))

        # Post-MP layers (MLPs)
        self.post_mp = nn.ModuleList()
        post_in_dim = self._get_post_mp_input_dim()

        if num_layers_post > 0:
            self.post_mp.append(nn.Linear(post_in_dim, hidden_dim))
            for _ in range(num_layers_post - 1):
                self.post_mp.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        final_dim = hidden_dim if num_layers_post > 0 else post_in_dim
        self.output_layer = nn.Linear(final_dim, out_dim)

        logger.info(f"Created GraphGymGNN: {layer_type}, {num_layers_mp} MP layers, "
                   f"hidden_dim={hidden_dim}, heads={heads}, stage={stage_type}")

    def _get_activation(self, act_name: str) -> Callable:
        """Get activation function."""
        act_name = act_name.lower()
        if act_name == "relu":
            return F.relu
        elif act_name == "prelu":
            return nn.PReLU()
        elif act_name == "leakyrelu":
            return F.leaky_relu
        elif act_name == "elu":
            return F.elu
        else:
            return F.relu

    def _create_gnn_layer(self, in_channels: int, out_channels: int, layer_type: str, heads: int):
        """Create a GNN layer based on type."""
        layer_type = layer_type.lower()

        if layer_type == "gcnconv":
            return GCNConv(in_channels, out_channels, add_self_loops=True)

        elif layer_type == "gatv2conv":
            return GATv2Conv(
                in_channels,
                out_channels,
                heads=heads,
                dropout=self.dropout,
                add_self_loops=True,
                edge_dim=1,  # Support edge attributes
            )

        elif layer_type == "ginconv":
            # GIN uses an MLP
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels * heads),
                nn.ReLU(),
                nn.Linear(out_channels * heads, out_channels * heads),
            )
            return GINConv(mlp, train_eps=True)

        elif layer_type == "sageconv":
            return SAGEConv(in_channels, out_channels * heads, aggr=self.agg)

        elif layer_type == "generalconv" or layer_type == "genconv":
            return GENConv(
                in_channels,
                out_channels * heads,
                aggr=self.agg,
                num_layers=2,
                norm="batch",
            )

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def _get_post_mp_input_dim(self) -> int:
        """Get input dimension for post-MP layers based on stage type."""
        if self.stage_type == "skipconcat":
            # Concatenate all MP layer outputs
            return self.hidden_dim + self.num_layers_mp * self.mp_hidden_dim * self.heads
        else:
            # Just the last MP layer output
            return self.mp_hidden_dim * self.heads

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional, for graph-level tasks)

        Returns:
            Output predictions [num_nodes, out_dim]
        """
        # Pre-MP layers
        h = x
        for layer in self.pre_mp:
            h = layer(h)
            if callable(self.act):
                h = self.act(h)
            else:
                h = self.act
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Store for skip connections
        pre_mp_out = h
        mp_outputs = []

        # Message passing layers
        for i, layer in enumerate(self.mp_layers):
            # Apply GNN layer
            if self.layer_type == "gatv2conv" and edge_attr is not None:
                h = layer(h, edge_index, edge_attr=edge_attr)
            else:
                h = layer(h, edge_index)

            # Batch norm
            if self.use_batchnorm:
                h = self.mp_batchnorms[i](h)

            # Activation
            if callable(self.act):
                h = self.act(h)
            else:
                h = self.act

            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Store for skip connections
            mp_outputs.append(h)

        # Apply skip connections based on stage type
        if self.stage_type == "skipsum":
            # Sum all MP outputs with pre-MP output
            h = pre_mp_out
            for mp_out in mp_outputs:
                # Handle dimension mismatch
                if mp_out.shape[1] != h.shape[1]:
                    # Pad or project to match dimensions
                    if mp_out.shape[1] > h.shape[1]:
                        padding = mp_out.shape[1] - h.shape[1]
                        h = F.pad(h, (0, padding))
                    else:
                        mp_out = F.pad(mp_out, (0, h.shape[1] - mp_out.shape[1]))
                h = h + mp_out

        elif self.stage_type == "skipconcat":
            # Concatenate all outputs
            h = torch.cat([pre_mp_out] + mp_outputs, dim=1)

        # else: stage_type == "stack", just use the last MP output (h already set)

        # Post-MP layers
        for layer in self.post_mp:
            h = layer(h)
            if callable(self.act):
                h = self.act(h)
            else:
                h = self.act
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Output layer
        out = self.output_layer(h)

        return out


def create_model_from_config(config) -> GraphGymGNN:
    """
    Create a GNN model from a configuration object.

    Args:
        config: ExperimentConfig object

    Returns:
        GraphGymGNN model
    """
    # Determine input dimension based on dataset
    # Features: [pos_x, pos_y, vel_x, vel_y, species_one_hot]
    # For single species: 4 + 1 = 5
    # For multi-species: 4 + num_species
    # For now, assume we'll get this from the dataset
    # Default to a reasonable value
    in_dim = 19  # 4 (pos+vel) + 15 (species, placeholder)

    model = GraphGymGNN(
        in_dim=in_dim,
        hidden_dim=config.gnn.dim_inner,
        out_dim=config.gnn.out_dim,
        num_layers_pre=config.gnn.layers_pre_mp,
        num_layers_mp=config.gnn.layers_mp,
        num_layers_post=config.gnn.layers_post_mp,
        layer_type=config.gnn.layer_type,
        stage_type=config.gnn.stage_type,
        agg=config.gnn.agg,
        heads=config.gnn.heads,
        dropout=config.gnn.dropout,
        batchnorm=config.gnn.batchnorm,
        act=config.gnn.act,
    )

    return model
