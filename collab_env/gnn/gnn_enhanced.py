"""
Simple GAT model with rich edge features for direct pairwise rule extraction.

This module provides:
- Enhanced edge feature construction (9D relative features)
- Simple GAT model for direct pairwise interactions (a_i = Σ_j α_ij * f(rel_features))
- Training functions optimized for rule extraction
- PyG batch construction with rich edge features

The Simple GAT approach is optimal for extracting interpretable local rules similar to boids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))


def build_relative_edge_features(
    positions: torch.Tensor, 
    velocities: torch.Tensor,
    edge_index: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Build comprehensive edge features including relative positions and velocities.
    
    This enables the GNN to learn rules similar to boids which depend on:
    - Relative positions (for separation/cohesion)
    - Relative velocities (for alignment)
    - Distances and directions
    
    Args:
        positions: [N, 2] positions of nodes
        velocities: [N, 2] velocities of nodes
        edge_index: [2, E] edge connectivity
        normalize: Whether to normalize features
        
    Returns:
        edge_features: [E, 9] where features include:
            - relative position (2D)
            - relative velocity (2D)
            - distance (1D)
            - unit direction (2D)
            - velocity alignment (1D)
            - speed difference (1D)
    """
    # Get source and target nodes for each edge
    src_idx = edge_index[0]
    tgt_idx = edge_index[1]
    
    # Compute relative vectors
    rel_pos = positions[tgt_idx] - positions[src_idx]  # [E, 2]
    rel_vel = velocities[tgt_idx] - velocities[src_idx]  # [E, 2]
    
    # Compute distances
    distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
    distances_safe = torch.clamp(distances, min=1e-6)
    
    # Unit direction vectors
    unit_dir = rel_pos / distances_safe  # [E, 2]
    
    # Velocity alignment (dot product)
    vel_src = velocities[src_idx]  # [E, 2]
    vel_tgt = velocities[tgt_idx]  # [E, 2]
    alignment = torch.sum(vel_src * vel_tgt, dim=-1, keepdim=True)  # [E, 1]
    
    # Speed differences - reuse velocity computations
    speed_src = torch.norm(vel_src, dim=-1, keepdim=True)
    speed_tgt = torch.norm(vel_tgt, dim=-1, keepdim=True)
    speed_diff = speed_tgt - speed_src  # [E, 1]
    
    # Concatenate all features
    edge_features = torch.cat([
        rel_pos,        # Relative position (2D)
        rel_vel,        # Relative velocity (2D)
        distances,      # Distance (1D)
        unit_dir,       # Unit direction (2D)
        alignment,      # Velocity alignment (1D)
        speed_diff      # Speed difference (1D)
    ], dim=-1)  # Total: 9D edge features
    
    if normalize:
        # Normalize each feature type separately using std for stability
        if edge_features.shape[0] > 1:  # Need at least 2 samples for std
            edge_features[:, :2] /= (torch.std(edge_features[:, :2]) + 1e-6)  # rel_pos
            edge_features[:, 2:4] /= (torch.std(edge_features[:, 2:4]) + 1e-6)  # rel_vel
            # distances already in [0, visual_range]
            # unit_dir already normalized
            edge_features[:, 7:8] /= (torch.std(edge_features[:, 7:8]) + 1e-6)  # alignment
            edge_features[:, 8:9] /= (torch.std(edge_features[:, 8:9]) + 1e-6)  # speed_diff
    
    return edge_features



class SimpleGATModel(nn.Module):
    """
    Simplified GNN using only GAT layer for direct pairwise interaction learning.
    This is much closer to the boid rule formulation: a_i = Σ_j f(relative_features_ij)
    """
    
    def __init__(
        self,
        in_node_dim: int,
        edge_dim: int = 9,  # Our enhanced edge features
        output_dim: int = 2,  # Direct acceleration prediction
        heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Single GAT layer that outputs accelerations directly
        self.gat = GATConv(
            in_node_dim,
            output_dim,  # Direct to acceleration space!
            edge_dim=edge_dim,
            heads=heads,
            concat=False,  # Average across heads instead of concatenating
            dropout=dropout,
            add_self_loops=False
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, in_node_dim] node features
        edge_index: [2, E] edges  
        edge_attr: [E, edge_dim] relative edge features
        
        Returns: [N, 2] accelerations, attention weights
        """
        # Direct GAT to acceleration space
        acc, (edge_idx, att_weights) = self.gat(
            x, edge_index, 
            edge_attr=edge_attr,
            return_attention_weights=True
        )
        
        return acc, (edge_idx, att_weights)


class SimpleGATModelWithCompatibility(SimpleGATModel):
    """
    Simple GAT model with compatibility attributes for existing training code.
    """
    
    def __init__(
        self,
        model_name: str,
        in_node_dim: int,
        node_feature_function: str,
        node_prediction: str,
        edge_dim: int = 9,
        heads: int = 4,
        **kwargs
    ):
        super().__init__(
            in_node_dim=in_node_dim,
            edge_dim=edge_dim,
            output_dim=2,
            heads=heads
        )
        
        # Compatibility attributes
        self.name = model_name
        self.node_feature_function = node_feature_function
        self.node_prediction = node_prediction
        self.input_differentiation = kwargs.get('input_differentiation', 'finite')
        self.prediction_integration = kwargs.get('prediction_integration', 'Euler')
        self.start_frame = kwargs.get('start_frame', 3)


class EnhancedGNN(nn.Module):
    """
    Enhanced GNN that reuses original architecture (GAT→GCN→Linear) but with rich edge features.
    This isolates the effect of rich edge features vs architecture changes.
    """
    
    def __init__(
        self,
        model_name,
        in_node_dim,
        node_feature_function,
        node_prediction,
        input_differentiation="finite",
        prediction_integration="Euler",
        start_frame=0,
        heads=1,
        hidden_dim=128,
        output_dim=2,
        edge_dim=9  # Rich edge features
    ):
        super().__init__()
        
        # Copy attributes from original GNN for compatibility
        self.name = model_name
        self.node_feature_function = node_feature_function
        self.node_prediction = node_prediction
        self.input_differentiation = input_differentiation
        self.prediction_integration = prediction_integration
        self.start_frame = start_frame
        
        # Use same architecture as original but with edge_dim=9
        from torch_geometric.nn import GCNConv, GATConv
        import torch.nn.functional as F
        
        self.gcn1 = GCNConv(in_node_dim, hidden_dim, add_self_loops=False)
        self.gatn = GATConv(
            in_node_dim, hidden_dim, edge_dim=edge_dim, heads=heads, add_self_loops=False
        )
        self.gcn2 = GCNConv(hidden_dim * heads, hidden_dim, add_self_loops=False)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass using original architecture but with rich edge features.
        """
        import torch.nn.functional as F
        
        # GAT layer with enhanced edge features
        h1, (edge_idx, att_weights) = self.gatn(
            x, edge_index, 
            edge_attr=edge_attr,
            return_attention_weights=True
        )
        
        # GCN layer (same as original)
        h2 = F.relu(self.gcn2(h1, edge_index))
        
        # Linear output layer (same as original)
        out = self.out(h2)
        
        return out, (edge_idx, att_weights)


def build_pyg_batch_enhanced(past_p, past_v, past_a, species_idx, species_dim, visual_range, node_feature_function):
    """Build PyTorch Geometric batch with enhanced edge features."""
    from collab_env.gnn.gnn import build_single_graph_edges
    
    data_list = []
    S = past_p.shape[0]
    device = past_p.device
    
    for b in range(S):
        # Get node features using existing function
        node_features = node_feature_function(
            past_p[b:b+1], past_v[b:b+1], past_a[b:b+1], 
            species_idx[b:b+1], species_dim
        )
        
        # Get current positions and velocities for edge feature computation
        current_pos = past_p[b, -1]  # [N, 2] - most recent positions
        current_vel = past_v[b, -1]  # [N, 2] - most recent velocities
        
        # Build edge index using visual range
        edge_index = build_single_graph_edges(current_pos, visual_range)
        
        # Build enhanced edge features
        if edge_index.shape[1] > 0:
            edge_features = build_relative_edge_features(
                current_pos, current_vel, edge_index, normalize=False
            )
        else:
            # No edges - create empty edge features with correct device
            edge_features = torch.zeros(0, 9, device=device, dtype=past_p.dtype)  # 9D edge features
        
        # Create Data object with enhanced edge features
        data = Data(
            x=node_features.squeeze(0) if node_features.dim() > 2 else node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            batch_idx=b,
            init_pos=current_pos,  # reuse already computed tensor
            init_vel=current_vel,  # reuse already computed tensor
        )
        data_list.append(data)
    
    return Batch.from_data_list(data_list)


def forward_enhanced(model, pyg_batch, device):
    """Forward pass for enhanced model with relative edge features."""
    pyg_batch = pyg_batch.to(device)
    
    # Forward pass - enhanced model expects rich edge features
    pred, W = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr)
    
    pred = pred.to(device)
    
    # Reconstruct batch dimensions from PyG batch
    B = pyg_batch.batch.max().item() + 1 if pyg_batch.batch.numel() > 0 else 1
    N = pyg_batch.x.shape[0] // B
    
    # Reshape prediction back to [B, N, output_dim]
    pred_reshaped = pred.view(B, N, -1)
    
    return pred_reshaped, W


def get_node_feature_function(model):
    """Get the appropriate node feature function based on model configuration"""
    from collab_env.gnn.gnn import node_feature_vel, node_feature_vel_pos, node_feature_vel_pos_plus, node_feature_vel_plus_pos_plus
    
    node_feature_function_name = model.node_feature_function
    if node_feature_function_name == "vel":
        return node_feature_vel
    elif node_feature_function_name == "vel_pos":
        return node_feature_vel_pos
    elif node_feature_function_name == "vel_pos_plus":
        return node_feature_vel_pos_plus
    elif node_feature_function_name == "vel_plus_pos_plus":
        return node_feature_vel_plus_pos_plus
    else:
        raise ValueError(f"Unknown node feature function: {node_feature_function_name}")




