"""Defines the GNN architecture."""

import torch
import torch.nn.functional as functional
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter


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
        edge_dim=1,
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
        # self.gcn1 = GCNConv(in_node_dim, hidden_dim, add_self_loops=False)
        """
        GAT v2 is supposedly better. Instead of sigma(AHW), it is Asigma(HW)
        Here we use GAT just because GATv2Conv have not been fully tested out by Shijie.
        However, preliminary testing shows that simply swapping GATConv to GATv2Conv works though!
        """
        # self.gatn1 = GATConv(
        #     in_node_dim, hidden_dim, edge_dim=1, heads=heads, add_self_loops=False
        # )
        """ 
        TOC -- 111225 11:01AM
        With relative positions, we need to pass those in as edge features, so the edge dim will be 
        the dimension of the physical space. If we also want relative velocities, then we will need to 
        double the edge dim. Start with relative positions and see how that goods.   
        
        For the self loops, the relative positions are [0,0], so we have the fill value set to that. 
        This needs to be fixed when we move to 3D so that it depends on the dimension of the space. 
        
        For the second layer, we won't add the self loops since we rely on the attention weights to
        determine the self loopiness. Presumably, the attention layer will have self loops if paying 
        attention to yourself is useful. 
        
        """
        self.gatn = GATv2Conv(
            in_node_dim,
            hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            add_self_loops=True,
            fill_value=torch.tensor([0.0] * edge_dim, dtype=torch.float32),
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
        if edge_weight is not None:
            edge_weight = (
                edge_weight.float()
            )  # all input needs to be the same precision.

        h_tmp, W = self.gatn(
            x, edge_index, edge_attr=edge_weight, return_attention_weights=True
        )
        (edge_index, edge_weight) = W
        h = functional.relu(h_tmp)

        # the 2nd layer is simple convolutional layer.
        # Note that we use the updated graph from the attention network
        """
        TOC 
        
        Note: This take the average of the attention weights over multiple heads  
        """
        h = self.gcn2(h, edge_index, torch.mean(edge_weight, 1))
        h = functional.relu(h)
        return self.out(h), W

    def ablate_attention(self):
        """copy reset_parameters"""
        print("self.gatn.att_src", self.gatn.att_src)
        glorot(self.gatn.att_src)
        print("self.gatn.att_src, post permute", self.gatn.att_src)
        glorot(self.gatn.att_dst)
        glorot(self.gatn.att_edge)

    def permute_attention(self):
        """permute parameters"""
        current_seed = torch.cuda.initial_seed()
        print("self.gatn.att_src", self.gatn.att_src)
        self.gatn.att_src = permute_second_dim(self.gatn.att_src)
        print("self.gatn.att_src, post permute", self.gatn.att_src)

        torch.cuda.manual_seed(current_seed + 1)
        self.gatn.att_dst = permute_second_dim(self.gatn.att_dst)

        torch.cuda.manual_seed(current_seed + 2)
        self.gatn.att_edge = permute_second_dim(self.gatn.att_edge)

    def uni_attention(self):
        """zero out att, due to softmax, output will be uniform"""
        print("self.gatn.att_src", self.gatn.att_src)
        zeros(self.gatn.att_src)
        print("self.gatn.att_src, post permute", self.gatn.att_src)

        zeros(self.gatn.att_dst)
        zeros(self.gatn.att_edge)


def permute_second_dim(t):
    permuted_indices = torch.randperm(t.shape[2])
    t_perm = t[:, :, permuted_indices]
    t_perm = Parameter(t_perm)
    return t_perm


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
        start_frame=None,
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
