#!/usr/bin/env python3
"""
GNN Feature Importance Explainer Script

This script uses PyTorch Geometric's explainer interface to visualize the importance
of different node features for GNN predictions on rollout data.

Usage:
    python gnn_feature_importance_explainer.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric import fidelity

# Add the project root to the path (go up two levels from collab_env/gnn to project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.gnn import (
    load_model, 
    build_pyg_batch,
    node_feature_vel,
    node_feature_vel_pos,
    node_feature_vel_pos_plus,
    node_feature_vel_plus_pos_plus
)
from collab_env.gnn.plotting_utility import DeviceUnpickler


class GNNFeatureExplainer:
    """Class to handle GNN feature importance analysis using PyTorch Geometric explainers."""
    
    def __init__(self, model, device=None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained GNN model
            device: Device to run computations on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Create a wrapper model that's compatible with PyTorch Geometric explainers
        self.explainer_model = self._create_explainer_compatible_model()
        
        # Map node feature function names to actual functions
        self.node_feature_functions = {
            'vel': node_feature_vel,
            'vel_pos': node_feature_vel_pos,
            'vel_pos_plus': node_feature_vel_pos_plus,
            'vel_plus_pos_plus': node_feature_vel_plus_pos_plus
        }
        
        # Feature names for visualization (based on node_feature_vel_plus_pos_plus)
        # For boid_food_independent: 20 features (19 for single species + 1 extra species)
        self.feature_names = [
            'vel_x_t-2', 'vel_y_t-2', 'vel_x_t-1', 'vel_y_t-1', 'vel_x_t', 'vel_y_t',
            'pos_x_t-2', 'pos_y_t-2', 'pos_x_t-1', 'pos_y_t-1', 'pos_x_t', 'pos_y_t',
            'boundary_x_t-2', 'boundary_y_t-2', 'boundary_x_t-1', 'boundary_y_t-1', 'boundary_x_t', 'boundary_y_t',
            'species_boids', 'species_food'
        ]
    
    def _create_explainer_compatible_model(self):
        """Create a wrapper model that's compatible with PyTorch Geometric explainers."""
        class ExplainerCompatibleModel(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
            
            def forward(self, x, edge_index, edge_weight=None):
                # Call the original model
                predictions, attention_info = self.original_model(x, edge_index, edge_weight)
                # Return the full predictions - CaptumExplainer will handle multiple outputs
                if isinstance(predictions, torch.Tensor):
                    return predictions
                else:
                    # If predictions is not a tensor, convert it
                    return torch.tensor(predictions, dtype=torch.float32)
        
        return ExplainerCompatibleModel(self.model)
        
    # Removed unused load_rollout_data method
    
    def prepare_data_for_explanation(self, rollout_data, epoch_num=0, batch_num=0, frame_num=0):
        """
        Prepare data from rollout for explanation.
        
        Args:
            rollout_data: Rollout data dictionary
            epoch_num: Epoch number to analyze
            batch_num: Batch number to analyze
            frame_num: Frame number to analyze
            
        Returns:
            PyG Data object and target predictions
        """
        # Extract data from rollout
        batch_data = rollout_data[epoch_num][batch_num]
        
        # Get positions, velocities, and accelerations
        actual_pos = torch.tensor(batch_data['actual'][frame_num], dtype=torch.float32)  # [B, N, 2]
        predicted_pos = torch.tensor(batch_data['predicted'][frame_num], dtype=torch.float32)  # [B, N, 2]
        
        # Get the adjacency matrix information
        edge_index, edge_weight = batch_data['W'][frame_num]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.clone().detach()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        if edge_weight is not None:
            if isinstance(edge_weight, torch.Tensor):
                edge_weight = edge_weight.clone().detach()
            else:
                edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        else:
            edge_weight = None
        
        # For simplicity, let's use the first batch
        B, N, D = actual_pos.shape
        
        if B > 1:
            actual_pos = actual_pos[0:1]  # Take first batch
            predicted_pos = predicted_pos[0:1]
        
        # Create species index - boid_food_independent has 2 species (0=boids, 1=food)
        species_idx = torch.zeros(B, N, dtype=torch.long)  # [B, N] shape
        # Set some nodes as food (species 1) for demonstration
        species_idx[:, N//2:] = 1  # Half boids, half food
        species_dim = 2  # 2 species: boids and food
        
        # Create dummy past data for node features
        # We'll use the current frame data repeated for past frames
        past_p = actual_pos.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, N, 2]
        past_v = torch.zeros_like(past_p)  # Dummy velocities
        past_a = torch.zeros_like(past_p)  # Dummy accelerations
        
        # Build PyG batch
        node_feature_func = self.node_feature_functions.get(
            self.model.node_feature_function, 
            node_feature_vel_plus_pos_plus
        )
        
        pyg_batch = build_pyg_batch(
            past_p, past_v, past_a, species_idx, species_dim, 
            visual_range=0.5, node_feature_function=node_feature_func
        )
        
        return pyg_batch, predicted_pos
    
    def compute_feature_importance(self, pyg_data, target_predictions, method='gnn_explainer'):
        """
        Compute feature importance using PyTorch Geometric explainers.
        
        Args:
            pyg_data: PyG Data object
            target_predictions: Target predictions for the data
            method: Explanation method ('gnn_explainer', 'pg_explainer', 'captum_explainer', or 'gradient')
            
        Returns:
            Dictionary with feature importance scores
        """
        num_nodes = pyg_data.x.size(0)
        num_features = pyg_data.x.size(1)
        num_edges = pyg_data.edge_index.size(1)
        
        if method == 'gnn_explainer':
            # Use GNNExplainer from PyTorch Geometric
            feature_importance = self._compute_gnn_explainer_importance(pyg_data, target_predictions)


        elif method == 'gradient':
            # Use gradient-based feature importance as fallback
            feature_importance = self._compute_gradient_importance(pyg_data, target_predictions)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create edge importance based on node feature importance
        edge_importance = self._compute_edge_importance(pyg_data, feature_importance)
        
        return {
            'node_mask': feature_importance,
            'edge_mask': edge_importance,
            'method': method
        }
    
    def _compute_gnn_explainer_importance(self, pyg_data, target_predictions):
        """Compute feature importance using GNNExplainer from PyTorch Geometric."""
        try:
            # Create GNNExplainer
            explainer = Explainer(
                model=self.explainer_model,  # Use the compatible wrapper
                algorithm=GNNExplainer(epochs=100, lr=0.01),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='regression',
                    task_level='node',
                    return_type='raw',
                ),
            )
            
            # Generate explanation
            explanation = explainer(
                x=pyg_data.x,
                edge_index=pyg_data.edge_index,
                edge_weight=pyg_data.edge_attr if hasattr(pyg_data, 'edge_attr') else None,
            )
            
            # Extract node mask (feature importance)
            if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
                feature_importance = explanation.node_mask.detach()
            else:
                # Fallback to gradient method if no node mask
                print("    Warning: GNNExplainer didn't return node_mask, falling back to gradient method")
                return self._compute_gradient_importance(pyg_data, target_predictions)
            
            # Normalize to [0, 1] range
            if feature_importance.max() > 0:
                feature_importance = feature_importance / feature_importance.max()
            
            return feature_importance
            
        except Exception as e:
            print(f"    Warning: GNNExplainer failed: {e}, falling back to gradient method")
            return self._compute_gradient_importance(pyg_data, target_predictions)
    

    

    

    
    def _compute_gradient_importance(self, pyg_data, target_predictions):
        """Compute feature importance using gradients w.r.t. input features."""
        self.model.zero_grad()
        
        # Forward pass
        x = pyg_data.x.clone().detach().requires_grad_(True)
        edge_index = pyg_data.edge_index
        edge_weight = pyg_data.edge_attr if hasattr(pyg_data, 'edge_attr') else None
        
        # Get model predictions
        predictions, _ = self.model(x, edge_index, edge_weight)
        
        # Compute loss (MSE between predictions and targets)
        if target_predictions is not None:
            target = target_predictions.view(-1, 2)  # Reshape to [N, 2]
            loss = torch.nn.functional.mse_loss(predictions, target)
        else:
            # If no targets, use predictions as targets (for demonstration)
            loss = torch.nn.functional.mse_loss(predictions, torch.zeros_like(predictions))
        
        # Backward pass
        loss.backward()
        
        # Get gradients w.r.t. input features
        gradients = x.grad.abs()  # [N, F]
        
        # Normalize gradients to [0, 1] range
        if gradients.max() > 0:
            gradients = gradients / gradients.max()
        
        return gradients.detach()
    

    
    def _compute_edge_importance(self, pyg_data, node_importance):
        """Compute edge importance based on node feature importance."""
        num_edges = pyg_data.edge_index.size(1)
        edge_importance = torch.zeros(num_edges, device=pyg_data.x.device)
        
        # Edge importance is the average of source and target node importance
        for i in range(num_edges):
            src, dst = pyg_data.edge_index[:, i]
            src_importance = node_importance[src].mean()
            dst_importance = node_importance[dst].mean()
            edge_importance[i] = (src_importance + dst_importance) / 2
        
        return edge_importance
    
    def analyze_feature_importance(self, rollout_data, epoch_num=0, batch_num=0, 
                                 start_frame=0, end_frame=None, method='gnn_explainer'):
        """
        Analyze feature importance across multiple frames in a rollout.
        
        Args:
            rollout_data: Rollout data dictionary
            epoch_num: Epoch number to analyze
            batch_num: Batch number to analyze
            start_frame: Starting frame number
            end_frame: Ending frame number (if None, analyze all available frames)
            method: Explanation method
            
        Returns:
            Dictionary containing feature importance analysis results
        """
        batch_data = rollout_data[epoch_num][batch_num]
        
        if end_frame is None:
            end_frame = len(batch_data['actual'])
        
        feature_importance_results = {
            'frames': [],
            'feature_importance': [],
            'node_importance': [],
            'edge_importance': []
        }
        
        print(f"Analyzing frames {start_frame} to {end_frame}...")
        
        for frame_num in range(start_frame, min(end_frame, len(batch_data['actual']))):
            print(f"  Processing frame {frame_num}...")
            
            try:
                # Prepare data for this frame
                pyg_data, target_predictions = self.prepare_data_for_explanation(
                    rollout_data, epoch_num, batch_num, frame_num
                )
                
                # Compute feature importance
                print(f"    Using method: {method}")
                explanation = self.compute_feature_importance(
                    pyg_data, target_predictions, method
                )
                

                # Extract importance scores
                if isinstance(explanation, dict) and 'node_mask' in explanation and explanation['node_mask'] is not None:
                    node_importance = explanation['node_mask'].detach().cpu().numpy()
                else:
                    node_importance = np.ones((pyg_data.x.size(0), pyg_data.x.size(1)))
                
                if isinstance(explanation, dict) and 'edge_mask' in explanation and explanation['edge_mask'] is not None:
                    edge_importance = explanation['edge_mask'].detach().cpu().numpy()
                else:
                    edge_importance = np.ones(pyg_data.edge_index.size(1))
                
                # Store results
                feature_importance_results['frames'].append(frame_num)
                feature_importance_results['feature_importance'].append(node_importance)
                feature_importance_results['node_importance'].append(node_importance)
                feature_importance_results['edge_importance'].append(edge_importance)
                
            except Exception as e:
                print(f"    Error processing frame {frame_num}: {e}")
                continue
        
        return feature_importance_results
    
    def visualize_feature_importance(self, feature_importance_results, save_path=None):
        """
        Visualize feature importance results.
        
        Args:
            feature_importance_results: Results from analyze_feature_importance
            save_path: Path to save the visualization (if None, display only)
        """
        if not feature_importance_results['frames']:
            print("No feature importance results to visualize.")
            return
        
        # Create subplots with extra space for legend
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('GNN Feature Importance Analysis', fontsize=16)
        
        frames = feature_importance_results['frames']
        feature_importance = feature_importance_results['feature_importance']
        
        # 1. Average feature importance over time
        avg_feature_importance = np.mean(feature_importance, axis=0)  # Average over nodes
        feature_names = self.feature_names[:avg_feature_importance.shape[1]]
        
        # Create bar plot with feature names
        x_pos = np.arange(len(feature_names))
        bars = axes[0, 0].bar(x_pos, np.mean(avg_feature_importance, axis=0))
        axes[0, 0].set_title('Average Feature Importance Across All Frames')
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Importance Score')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[0, 0].tick_params(axis='x', labelsize=8)  # Smaller font for long names
        
        # 2. Feature importance over time for top features
        top_features = np.argsort(np.mean(avg_feature_importance, axis=0))[-5:]  # Top 5 features
        
        for i, feature_idx in enumerate(top_features):
            feature_importance_over_time = [fi[:, feature_idx].mean() for fi in feature_importance]
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx}'
            axes[0, 1].plot(frames, feature_importance_over_time, 
                           label=feature_name, marker='o')
        
        axes[0, 1].set_title('Top 5 Features: Importance Over Time')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Importance Score')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside plot
        axes[0, 1].grid(True)
        
        # 3. Node-level importance heatmap
        if feature_importance:
            node_importance_matrix = np.array([fi.mean(axis=1) for fi in feature_importance]).T
            im = axes[1, 0].imshow(node_importance_matrix, aspect='auto', cmap='viridis')
            axes[1, 0].set_title('Node-Level Feature Importance Heatmap')
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Node')
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Feature importance distribution
        all_importances = np.concatenate([fi.flatten() for fi in feature_importance])
        axes[1, 1].hist(all_importances, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Feature Importance Scores')
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        # Add feature importance summary table
        plt.figtext(0.02, 0.02, 'Feature Importance Summary (Top 10):', fontsize=12, fontweight='bold')
        
        # Sort features by average importance
        feature_avg_importance = np.mean(avg_feature_importance, axis=0)
        sorted_indices = np.argsort(feature_avg_importance)[::-1]  # Descending order
        
        summary_text = ""
        for i, idx in enumerate(sorted_indices[:10]):
            feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'
            importance = feature_avg_importance[idx]
            summary_text += f"{i+1:2d}. {feature_name:<20} : {importance:.4f}\n"
        
        plt.figtext(0.02, 0.01, summary_text, fontsize=9, fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, feature_importance_results, save_path):
        """
        Save feature importance results to a file.
        
        Args:
            feature_importance_results: Results from analyze_feature_importance
            save_path: Path to save the results
        """
        with open(save_path, 'wb') as f:
            pickle.dump(feature_importance_results, f)
        print(f"Results saved to {save_path}")


def main():
    """Main function to run the GNN feature importance analysis."""
    
    # Configuration
    model_name = "vpluspplus_a"
    # model_file = "boid_food_independent_vpluspplus_a_n0_h1_vr0.5_s0"
    model_file = "boid_single_species_basic_vpluspplus_a_n0_h1_vr0.5_s0"
    data_name = "boid_single_species_basic"
    
    print("Loading trained GNN model...")
    try:
        # Load the trained model
        model, model_spec, train_spec = load_model(model_name, model_file, root_path="trained_models/boid_single_species_basic/trained_models")
        print(f"Model loaded successfully: {model.name}")
        print(f"Model spec: {model_spec}")
        print(f"Training spec: {train_spec}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize the explainer
    explainer = GNNFeatureExplainer(model)
    
    print("Loading rollout data...")
    try:
        # Extract seed from model file name
        seed = int(model_file.split("_s")[-1])
        
        # Create rollout specification
        rollout_spec = {
            "noise": train_spec["sigma"],
            "head": model_spec["heads"],
            "visual_range": train_spec["visual_range"],
            "seed": seed,
            "rollout_starting_frame": 5,
            "rollout_frames": 300,
        }
        print(f"Rollout spec: {rollout_spec}")
        
        # Load existing rollout file directly using DeviceUnpickler
        rollout_path = expand_path(
            "./trained_models/boid_single_species_basic/rollouts/boid_single_species_basic_vpluspplus_a_n0_h1_vr0.5_s0_rollout5.pkl",
            get_project_root()
        )
        
        if not os.path.exists(rollout_path):
            print(f"Rollout file not found: {rollout_path}")
            print("Please ensure the rollout file exists or modify the path.")
            return
        
        # Use DeviceUnpickler to handle CUDA/CPU compatibility
        with open(rollout_path, "rb") as f:
            rollout_data = DeviceUnpickler(f, device='cpu').load()
        print(f"Rollout data loaded successfully")
        print(f"Available epochs: {list(rollout_data.keys())}")
        print(f"Available batches in epoch 0: {list(rollout_data[0].keys())}")
    except Exception as e:
        print(f"Error loading rollout data: {e}")
        return
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    try:
        # Analyze first few frames to avoid long computation
        feature_importance_results = explainer.analyze_feature_importance(
            rollout_data,
            epoch_num=0,
            batch_num=0,
            start_frame=0,
            end_frame=min(100, len(rollout_data[0][0]['actual'])),  
            method='gnn_explainer'  # Use PyTorch Geometric GNNExplainer
        )
        
        print(f"Feature importance analysis completed for {len(feature_importance_results['frames'])} frames")
        
    except Exception as e:
        print(f"Error during feature importance analysis: {e}")
        return
    
    # Visualize results
    print("\nVisualizing results...")
    try:
        explainer.visualize_feature_importance(feature_importance_results)
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    # Save results
    print("\nSaving results...")
    try:
        results_save_path = "gnn_feature_importance_results.pkl"
        explainer.save_results(feature_importance_results, results_save_path)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
