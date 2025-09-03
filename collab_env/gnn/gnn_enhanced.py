"""
Enhanced GNN with rich edge features for improved learning.

This module provides:
- Enhanced edge feature construction (9D relative features)
- Enhanced GNN model using original architecture with rich edge features
- PyG batch construction with rich edge features

The Enhanced GNN maintains the proven architecture while using richer edge features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
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
    edge_index: torch.Tensor
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
    
    # No normalization - let the model learn appropriate scaling
    # This avoids instability from batch-wise statistics and computational overhead
    
    return edge_features



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
        gat_only=False,
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
        self.gat_only = gat_only
        # Use same architecture as original but with edge_dim=9
        # self.gcn1 = GCNConv(in_node_dim, hidden_dim, add_self_loops=False)
        # GATv2 is supposedly better: Asigma(HW) instead of sigma(AHW)
        if self.gat_only:
            # Output per head, then learn weighted combination
            self.gatn = GATv2Conv(
                    in_node_dim, output_dim, edge_dim=edge_dim, heads=heads, 
                    concat=True, add_self_loops=False
                )
            # Learnable weights for combining heads
            self.head_weights = nn.Parameter(torch.ones(heads) / heads)
            self.heads = heads
            self.output_dim = output_dim
        else:
            # For hidden layer, concatenate heads (default behavior)
            self.gatn = GATv2Conv(
                in_node_dim, hidden_dim, edge_dim=edge_dim, heads=heads, 
                concat=True, add_self_loops=False
            )
            self.gcn2 = GCNConv(hidden_dim * heads, hidden_dim, add_self_loops=False)
            self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass using original architecture but with rich edge features.
        """
        # GAT layer with enhanced edge features
        h1, (edge_idx, att_weights) = self.gatn(
            x, edge_index, 
            edge_attr=edge_attr,
            return_attention_weights=True
        )
        
        # Process through additional layers
        if not self.gat_only:
            # Original architecture: GAT -> GCN -> Linear
            h2 = F.relu(self.gcn2(h1, edge_index))
            out = self.out(h2)
        else:
            # GAT-only: Weighted combination of heads
            # h1 shape: [N, heads * output_dim]
            # Reshape to [N, heads, output_dim]
            h1_reshaped = h1.view(-1, self.heads, self.output_dim)
            
            # Apply softmax to head weights for normalized combination
            normalized_weights = F.softmax(self.head_weights, dim=0)
            
            # Weighted sum across heads: [N, heads, output_dim] -> [N, output_dim]
            out = torch.einsum('nhd,h->nd', h1_reshaped, normalized_weights)
        
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
                current_pos, current_vel, edge_index
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






