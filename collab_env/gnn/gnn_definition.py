"""Defines the GNN architecture."""

import torch
import torch.nn.functional as functional
from torch_geometric.nn import GATConv, GCNConv


class GNN(torch.nn.Module):
    def __init__(
        self,
        model_name,
        in_node_dim,
        node_feature_function,
        node_prediction,
        input_differentiation="finite",  # or "spline"
        prediction_integration="Euler",  # or "Leapfrog"
        start_frame=0,
        heads=1,
        hidden_dim=128,
        output_dim=2,
    ):
        super().__init__()

        self.name = model_name
        self.node_feature_function = node_feature_function
        self.node_prediction = node_prediction
        self.input_differentiation = input_differentiation
        self.prediction_integration = prediction_integration
        self.start_frame = start_frame
        # the frame to start training.
        # For example, in the event of encoding frame i's feature,
        #  we could also use position/velocity/etc from frame i-2, i-1, i.
        # In this case, we start training at frame 3.

        # Two graph convolutional layers
        self.gcn1 = GCNConv(in_node_dim, hidden_dim, add_self_loops=False)
        """
        GAT v2 is supposedly better. Instead of sigma(AHW), it is Asigma(HW)
        Here we use GAT just because GATv2Conv have not been fully tested out by Shijie.
        However, preliminary testing shows that simply swapping GATConv to GATv2Conv works though!
        """
        self.gatn = GATConv(
            in_node_dim, hidden_dim, edge_dim=1, heads=heads, add_self_loops=False
        )
        # self.gatn = GATv2Conv(in_node_dim, hidden_dim, edge_dim=1,
        #                      heads = heads, add_self_loops = False)
        self.gcn2 = GCNConv(hidden_dim * heads, hidden_dim, add_self_loops=False)

        # Final linear layer to predict 2D acceleration
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        """
        x:          [B*N, in_node_dim] node features (velocities + node position + species)
        edge_index: [2, E]             edge list built from visual neighborhood
        Returns:    [B*N, 2]           acceleration per boid
        """
        # the 1st layer is a graph attention network.
        x = x.float()
        edge_weight = edge_weight.float()  # all input needs to be the same precision.
        h_tmp, W = self.gatn(
            x, edge_index, edge_attr=edge_weight, return_attention_weights=True
        )
        (edge_index, edge_weight) = W
        h = functional.relu(h_tmp)

        # the 2nd layer is simple convolutional later.
        # Note that we use the updated graph from the attention network
        h = functional.relu(self.gcn2(h, edge_index, edge_weight))
        return self.out(h), W


class Lazy(torch.nn.Module):
    """In this lazy network,
    The network simply uses previous frame's velocity to model movement,
    adjacency matrix also does not change."""

    def __init__(
        self,
        model_name,
        in_node_dim=3,
        heads=1,
        hidden_dim=30,
        input_differentiation="finite",  # or "spline"
        prediction_integration="Euler",
    ):  # or "Leapfrog")
        super().__init__()
        self.name = model_name

        self.node_feature_function = "vel"
        self.node_prediction = "acc"
        self.input_differentiation = input_differentiation
        self.prediction_integration = prediction_integration

        self.start_frame = 3

    def forward(self, x, edge_index, edge_weight=None):
        """
        x:          [B*N, in_node_dim] node features (velocities + species)
        edge_index: [2, E]             edge list built from visual neighborhood
        Returns:    [B*N, 2]           acceleration per boid
        """
        W = (edge_index, edge_weight)

        return torch.zeros((x.shape[0], 2)), W
