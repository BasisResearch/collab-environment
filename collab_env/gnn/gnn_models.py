import torch
import torch.nn.functional as functional
from torch_geometric.nn import GCNConv, GATv2Conv


class GNN_Attention(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        in_node_dim: int,
        heads: int = 1,
        edge_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 2,
        self_loops: bool = False,
        fill_value: object = None,
        self_loops_layer_2: bool = False,
        dropout_p: float = 0.0,
    ):
        """

        Args:
            model_name (str):
            in_node_dim (int):
            heads (int):
            edge_dim (int):
            hidden_dim (int):
            output_dim (int):
            self_loops (bool):
            fill_value (object):
            self_loops_layer_2 (bool):

        Returns:
            output (torch.tensor of shape [output_dim]:
                activation of the output layer
            attention_weights (torch.tensor of shape [2, number of edges]:
                cattention weights computed by the attention layer
        """
        super().__init__()

        self.name = model_name

        """
        TOC -- 111225
        GAT will add self loops if they don't exist. We can specify the value to put 
        on the edge with the fill_value parameter. For relative positions, this should be 
        a vector of 0's whose length is the dimension of the physical space. When we are 
        not using relative positions, it is unclear what we should use as the fill value. 
        The other edges have weight 1/degree. The [0] specified here for the fill value 
        was not intentional but I am not sure it matters. I am uncertain as to how the edge 
        weights being 1/degree matter for the other nodes.  

        The self loops just allow the GNN to push the bulk of the total attention weight  
        onto the node itself. If we don't do that, then the attention weights seem like  
        they will end up mostly uniform since the adjacent nodes themselves are all the  
        same.


        TOC -- 121025
        The fill value of 0 for the self loop without relative positions seems wrong. It
        seems like this should be 1/degree where degree includes the self loop. Perhaps this
        should be handled in the build edges function so that the self loops never come 
        """

        self.attention_layer = GATv2Conv(
            in_node_dim,
            hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            add_self_loops=self_loops,
            fill_value=fill_value,  # torch.tensor([0.0] * edge_dim, dtype=torch.float32),
        )

        """
        TOC -- 111925 10:08AM 
        It is unclear how self loops help down here, so leave it off by default
        """
        self.convolution_layer = GCNConv(
            hidden_dim * heads, hidden_dim, add_self_loops=self_loops_layer_2
        )

        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=dropout_p)

        # Final linear layer to perform prediction
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    """
    TOC -- 111225 1:30PM
    The edge weight here should be the relative position when we want to try that, so 
    edge_weight is probably not the right name of the parameter. 
    """

    def forward(self, data):
        """
        x:          [B*N, in_node_dim] node features (velocities + node position + species)
        edge_index: [2, num edges]             edge list built from visual neighborhood
        Returns:    [B*N, 2]           acceleration per boid
        """
        # the 1st layer is a graph attention network.
        x = data.x.float()
        # if edge_feature is not None:
        #     edge_feature = (
        #         edge_feature.float()
        #     )  # all input needs to be the same precision.

        attention_layer_activation, attention_weights = self.attention_layer(
            x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights=True
        )
        h = functional.relu(attention_layer_activation)
        (edge_index, edge_weight) = attention_weights

        # the 2nd layer is simple convolutional layer.
        # Note that we use the updated graph from the attention network

        """
        TOC -- 111325 8:38AM
        The mean here is over the heads -- each head computes a different attention weight and we are 
        taking the average as the weight in the graph passed to the convolutional layer. 
        """
        h = functional.relu(
            self.convolution_layer(h, edge_index, torch.mean(edge_weight, 1))
        )  # + attention_layer_activation # add residual connection just for kicks

        if self.dropout_p > 0:
            h = self.dropout(h)

        return self.output_layer(h), attention_weights
