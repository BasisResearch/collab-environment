#!/usr/bin/env python3
"""
GNN Explainability using IntegratedGradients and Saliency from Captum
Computes feature importance for predicted accelerations across rollout frames
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import sys
import os
import hashlib
import time
from rich import print
from captum.attr import IntegratedGradients, Saliency
from typing import Dict, List, Tuple, Optional

from collab_env.gnn.gnn import (
    load_model, 
    debug_result2prediction,
    build_pyg_batch,
    run_gnn_frame_pyg,
    node_feature_vel,
    node_feature_vel_pos,
    node_feature_vel_pos_plus,
    node_feature_vel_plus_pos_plus,
    handle_discrete_data,
    v_function_2_vminushalf
)
from collab_env.gnn.plotting_utility import load_rollout
from collab_env.data.file_utils import expand_path, get_project_root


def get_node_feature_function(model):
    """Get the appropriate node feature function based on model configuration"""
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


def create_gnn_forward_wrapper(model, pyg_batch, v_minushalf, delta_t, device):
    """
    Create a forward function wrapper for IntegratedGradients.
    Returns a function that takes node features and returns predicted accelerations.
    """
    def forward_fn(node_features):
        # Update the PyG batch with new node features
        pyg_batch_copy = pyg_batch.clone()
        pyg_batch_copy.x = node_features
        
        # Run GNN forward pass
        pred, W = model(pyg_batch_copy.x, pyg_batch_copy.edge_index, 
                        pyg_batch_copy.edge_attr.squeeze(-1))
        
        # For acceleration prediction, pred is already the acceleration
        if model.node_prediction == "acc":
            return pred  # Shape: [B*N, 2] for 2D acceleration
        else:
            raise NotImplementedError(f"Only 'acc' prediction is supported, got {model.node_prediction}")
    
    return forward_fn


def compute_attributions_single_frame(
    model, 
    pyg_batch, 
    v_minushalf, 
    delta_t, 
    device,
    method='integrated_gradients',
    n_steps=50,
    baseline=None
):
    """
    Compute attributions for a single frame using either IntegratedGradients or Saliency.
    
    Args:
        method: 'integrated_gradients' or 'saliency'
    
    Returns:
        attributions: Shape [2*N, N, F] where F is the feature dimension
    """
    model.eval()
    
    # Create forward wrapper
    forward_fn = create_gnn_forward_wrapper(model, pyg_batch, v_minushalf, delta_t, device)
    
    # Get input features
    input_features = pyg_batch.x.requires_grad_(True)
    
    # Get dimensions
    B = pyg_batch.batch.max().item() + 1  # Number of graphs in batch
    N = pyg_batch.batch.bincount()[0].item()  # Nodes per graph
    F = input_features.shape[1]  # Feature dimension
    
    if method == 'saliency':
        # Fast saliency method - compute gradients for each output
        all_attributions = []
        
        for node_idx in range(N):
            for dim_idx in range(2):  # x and y acceleration
                # Create target selector for this specific output
                def target_forward_fn(x):
                    pyg_batch_copy = pyg_batch.clone()
                    pyg_batch_copy.x = x
                    pred, _ = model(pyg_batch_copy.x, pyg_batch_copy.edge_index, 
                                  pyg_batch_copy.edge_attr.squeeze(-1))
                    return pred[node_idx, dim_idx].unsqueeze(0)
                
                # Initialize Saliency for this target
                saliency = Saliency(target_forward_fn)
                
                # Compute attribution
                attribution = saliency.attribute(input_features)
                
                all_attributions.append(attribution.detach().cpu())
        
        # Stack attributions: Shape [2*N, N, F]
        attributions = torch.stack(all_attributions, dim=0)
        
    else:  # integrated_gradients
        # Create baseline if not provided (zeros)
        if baseline is None:
            baseline = torch.zeros_like(input_features)
        
        # Store attributions for each output
        all_attributions = []
        
        # Compute attribution for each output separately
        for node_idx in range(N):
            for dim_idx in range(2):  # x and y acceleration
                # Create target selector for this specific output
                def target_forward_fn(x):
                    pyg_batch_copy = pyg_batch.clone()
                    pyg_batch_copy.x = x
                    pred, _ = model(pyg_batch_copy.x, pyg_batch_copy.edge_index, 
                                  pyg_batch_copy.edge_attr.squeeze(-1))
                    return pred[node_idx, dim_idx].unsqueeze(0)
                
                # Initialize IG for this target
                ig = IntegratedGradients(target_forward_fn)
                
                # Compute attribution
                attribution = ig.attribute(
                    input_features,
                    baselines=baseline,
                    n_steps=n_steps,
                    return_convergence_delta=False
                )
                
                all_attributions.append(attribution.detach().cpu())
        
        # Stack attributions: Shape [2*N, N, F]
        attributions = torch.stack(all_attributions, dim=0)
    
    return attributions


def compute_rollout_attributions(
    model,
    rollout_result,
    model_spec,
    train_spec,
    file_id=0,
    device=None,
    method='integrated_gradients',
    n_steps=50,
    max_frames=None
):
    """
    Compute attributions for all frames in a rollout using specified method.
    
    Args:
        file_id: Which file in batch to analyze (use -1 for ALL files)
        method: 'integrated_gradients' or 'saliency'
    
    Returns:
        If file_id >= 0: List of attribution matrices, each of shape [2*N, N*F]
        If file_id == -1: Dict with keys as file_ids, values as attribution lists
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Handle analyzing all files
    if file_id == -1:
        # Determine how many files are available in the rollout
        predicted = rollout_result[0][0]['predicted'][0]
        batch_size = predicted.shape[0]
        total_batches = len(rollout_result[0])
        total_files = batch_size * total_batches
        
        print(f"Analyzing ALL files: {total_files} total files ({total_batches} batches × {batch_size} files/batch)")
        
        all_results = {}
        for current_file_id in range(total_files):
            print(f"\n--- Processing file {current_file_id + 1}/{total_files} ---")
            try:
                # Recursively call this function for each file
                file_attributions = compute_rollout_attributions(
                    model, rollout_result, model_spec, train_spec,
                    file_id=current_file_id, device=device, method=method, 
                    n_steps=n_steps, max_frames=max_frames
                )
                all_results[current_file_id] = file_attributions
            except Exception as e:
                print(f"Warning: Failed to process file {current_file_id}: {e}")
                continue
        
        return all_results
    
    # Extract data from rollout for single file
    actual_pos, actual_vel, actual_acc, gnn_pos, gnn_vel, gnn_acc, frame_sets = \
        debug_result2prediction(rollout_result, file_id=file_id, epoch_num=0)
    
    # Get simulation parameters
    visual_range = train_spec.get("visual_range", 0.5)
    species_dim = 2 if "food" in str(model_spec.get("in_node_dim", 19)) else 1
    
    # Determine feature dimension and handle different species configurations
    in_node_dim = model_spec.get("in_node_dim", 19)
    
    # Auto-detect species dimension based on model input dimension
    if in_node_dim == 20:
        species_dim = 2  # Food model with 2 species
    else:
        species_dim = 1  # No-food model with 1 species
    
    print(f"Model input dimension: {in_node_dim}, species_dim: {species_dim}")
    
    # Get node feature function
    node_feature_fn = get_node_feature_function(model)
    
    # Process frames
    frame_attributions = []
    total_frames = actual_pos.shape[1]
    num_frames = total_frames if max_frames is None else min(max_frames + 3, total_frames)
    
    print(f"Computing attributions for frames 3 to {num_frames-1} (total: {num_frames-3} frames)...")
    print(f"Actual data shape: {actual_pos.shape}")
    
    for frame_idx in range(3, num_frames):  # Start from frame 3 (need history)
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{num_frames}")
        
        # Prepare past data (using actual data)
        past_p = actual_pos[:, max(0, frame_idx-2):frame_idx+1, :, :]  # [1, 3, N, 2]
        past_v = actual_vel[:, max(0, frame_idx-2):frame_idx+1, :, :]
        past_a = actual_acc[:, max(0, frame_idx-2):frame_idx+1, :, :]
        
        # Pad if necessary
        if past_p.shape[1] < 3:
            pad_size = 3 - past_p.shape[1]
            past_p = torch.cat([past_p[:, :1].repeat(1, pad_size, 1, 1), past_p], dim=1)
            past_v = torch.cat([past_v[:, :1].repeat(1, pad_size, 1, 1), past_v], dim=1)
            past_a = torch.cat([past_a[:, :1].repeat(1, pad_size, 1, 1), past_a], dim=1)
        
        # Create species index
        N = actual_pos.shape[2]
        if species_dim == 2 and N == 21:
            # Food model: 20 boids (species 0) + 1 food (species 1)
            species_idx = torch.zeros(1, N, dtype=torch.long)
            species_idx[0, -1] = 1  # Last agent is food
        else:
            # No-food model: all agents are same species
            species_idx = torch.zeros(1, N, dtype=torch.long)
        
        # Handle v_minushalf for leapfrog integration
        v_minushalf = actual_vel[:, frame_idx, :, :] if frame_idx > 0 else None
        
        # Build PyG batch
        pyg_batch = build_pyg_batch(
            past_p, past_v, past_a, species_idx, species_dim,
            visual_range, node_feature_fn
        )
        pyg_batch = pyg_batch.to(device)
        
        # Compute attributions for this frame
        try:
            attributions = compute_attributions_single_frame(
                model, pyg_batch, v_minushalf, delta_t=1.0, 
                device=device, method=method, n_steps=n_steps
            )
            
            # Reshape to [2*N, N*F] matrix
            # attributions shape: [2*N, N, F]
            N = pyg_batch.batch.bincount()[0].item()
            F = attributions.shape[-1]
            attribution_matrix = attributions.reshape(2*N, N*F)
            
            frame_attributions.append(attribution_matrix.numpy())
            
        except Exception as e:
            print(f"Error at frame {frame_idx}: {e}")
            # Add zero matrix on error
            frame_attributions.append(np.zeros((2*N, N*in_node_dim)))
    
    return frame_attributions


def visualize_attribution_statistics(frame_attributions, save_path=None, data_name=None, file_info=None):
    """
    Visualize statistics of the attribution matrices across frames.
    
    Args:
        frame_attributions: List of attribution matrices
        save_path: Path to save the figure
        data_name: Dataset name to display in title
        file_info: Optional list of (file_id, frame_idx) tuples for multi-file analysis
    """
    # Convert list to numpy array: [num_frames, 2*N, N*F]
    if len(frame_attributions) == 0:
        print("⚠️  No attribution data to visualize!")
        return None, None, None
    
    attributions_array = np.array(frame_attributions)
    
    if attributions_array.ndim != 3:
        print(f"Warning: Expected 3D array, got shape {attributions_array.shape}")
        if attributions_array.ndim == 2:
            # Single frame case
            attributions_array = attributions_array[np.newaxis, :, :]
        else:
            print("Error: Cannot handle attribution data with this shape")
            return None, None, None
    
    num_frames, output_dim, input_dim = attributions_array.shape
    
    print(f"Attribution array shape: {attributions_array.shape}")
    print(f"Frames: {num_frames}, Outputs: {output_dim}, Inputs: {input_dim}")
    
    # Compute statistics
    mean_attr = np.mean(attributions_array, axis=0)
    std_attr = np.std(attributions_array, axis=0)
    max_attr = np.max(np.abs(attributions_array), axis=0)
    
    # Create visualization with smaller figure size
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Add data name to suptitle if provided
    if data_name:
        fig.suptitle(f'GNN Attribution Analysis - {data_name}', fontsize=16, y=0.95)
    
    # Mean attribution
    im1 = axes[0, 0].imshow(mean_attr, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Mean Attribution')
    axes[0, 0].set_xlabel('Input Features (N*F)')
    axes[0, 0].set_ylabel('Output (2*N)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Std attribution
    im2 = axes[0, 1].imshow(std_attr, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Std Attribution')
    axes[0, 1].set_xlabel('Input Features (N*F)')
    axes[0, 1].set_ylabel('Output (2*N)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Max absolute attribution
    im3 = axes[0, 2].imshow(max_attr, cmap='hot', aspect='auto')
    axes[0, 2].set_title('Max |Attribution|')
    axes[0, 2].set_xlabel('Input Features (N*F)')
    axes[0, 2].set_ylabel('Output (2*N)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Feature importance distribution by groups
    # Compute importance coefficients for all frames and output dimensions
    frame_importance_all = np.abs(attributions_array)  # [num_frames, 2*N, N*F]
    
    # Determine number of agents and features per agent
    N = output_dim // 2  # Number of agents (2 outputs per agent)
    F_per_agent = input_dim // N  # Features per agent
    
    # Group features by type
    # Features are organized per-agent: [agent0_features, agent1_features, ...]
    # For vel_plus_pos_plus: each agent has [vel(6D), pos(6D), boundary(6D), species(1D/2D)]
    feature_groups = {}
    group_indices = {}
    
    if F_per_agent == 19:  # No-food model: vel(6) + pos(6) + boundary(6) + species(1)
        for agent_id in range(N):
            agent_offset = agent_id * 19
            group_indices.setdefault('Velocity (3t×2D)', []).extend(range(agent_offset, agent_offset + 6))
            group_indices.setdefault('Position (3t×2D)', []).extend(range(agent_offset + 6, agent_offset + 12))
            group_indices.setdefault('Boundary (3t×2D)', []).extend(range(agent_offset + 12, agent_offset + 18))
            group_indices.setdefault('Species (1D)', []).extend(range(agent_offset + 18, agent_offset + 19))
    elif F_per_agent == 20:  # Food model: vel(6) + pos(6) + boundary(6) + species(2)
        for agent_id in range(N):
            agent_offset = agent_id * 20
            group_indices.setdefault('Velocity (3t×2D)', []).extend(range(agent_offset, agent_offset + 6))
            group_indices.setdefault('Position (3t×2D)', []).extend(range(agent_offset + 6, agent_offset + 12))
            group_indices.setdefault('Boundary (3t×2D)', []).extend(range(agent_offset + 12, agent_offset + 18))
            group_indices.setdefault('Species (2D)', []).extend(range(agent_offset + 18, agent_offset + 20))
    else:
        # Fallback for other feature configurations
        for agent_id in range(N):
            agent_offset = agent_id * F_per_agent
            group_indices[f'Agent{agent_id} Features'] = list(range(agent_offset, agent_offset + F_per_agent))
    
    # Convert index lists to boolean masks for efficient extraction
    feature_groups = {name: np.array(indices) for name, indices in group_indices.items()}
    
    # Collect importance distributions for each feature group
    group_distributions = []
    group_labels = []
    total_samples = 0
    
    for name, feature_indices in feature_groups.items():
        # Extract coefficients for this feature group across all frames and outputs
        group_coeffs = frame_importance_all[:, :, feature_indices]  # [frames, outputs, group_features]
        group_flat = group_coeffs.flatten()  # Flatten across all dimensions
        
        group_distributions.append(group_flat)
        total_samples += len(group_flat)
        group_labels.append(name)
    
    # Create violin plot
    violin_parts = axes[1, 0].violinplot(group_distributions, positions=range(len(group_labels)), 
                                        showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    
    axes[1, 0].set_xticks(range(len(group_labels)))
    axes[1, 0].set_xticklabels(group_labels, rotation=45, ha='right')
    axes[1, 0].set_title(f'Feature Group Importance Distribution (n={total_samples:,} samples)')
    axes[1, 0].set_ylabel('|Attribution Coefficient|')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Use log scale only if the dynamic range is very large (>1000x)
    all_values = np.concatenate(group_distributions)
    value_range = np.max(np.abs(all_values)) / (np.min(np.abs(all_values[all_values > 0])) + 1e-12)
    if value_range > 1e4:
        axes[1, 0].set_yscale('log')
    
    # Output sensitivity by agent and dimension
    output_sensitivity = np.mean(np.abs(mean_attr), axis=1)
    
    # Reshape for visualization: [N, 2] where N is number of agents
    N = output_dim // 2
    output_reshaped = output_sensitivity.reshape(N, 2)
    
    # Create agent-wise visualization
    agent_indices = np.arange(N)
    x_pos = agent_indices - 0.2
    x_pos_y = agent_indices + 0.2
    
    bars_x = axes[1, 1].bar(x_pos, output_reshaped[:, 0], 0.4, 
                           label='X-acceleration', color='#1f77b4', alpha=0.7)
    bars_y = axes[1, 1].bar(x_pos_y, output_reshaped[:, 1], 0.4,
                           label='Y-acceleration', color='#ff7f0e', alpha=0.7)
    
    axes[1, 1].set_title('Output Sensitivity by Agent')
    axes[1, 1].set_xlabel('Agent Index')
    axes[1, 1].set_ylabel('Sensitivity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Highlight food agent if present (last agent in food models)
    if F_per_agent == 20 and N == 21:  # Food model
        # Highlight the last agent (food) with different color
        bars_x[-1].set_color('#d62728')
        bars_y[-1].set_color('#ff69b4')
        axes[1, 1].axvline(x=N-1, color='red', linestyle='--', alpha=0.5, 
                          label='Food Agent')
        axes[1, 1].legend()
    
    # Create different visualization based on data type
    if file_info is not None:
        # Multi-file analysis: show mean temporal evolution with confidence intervals
        # Reorganize data by frame index across files
        
        # Group attributions by frame index
        frame_groups = {}
        for i, (file_id, frame_idx) in enumerate(file_info):
            if frame_idx not in frame_groups:
                frame_groups[frame_idx] = []
            # Compute overall attribution magnitude for this frame
            frame_magnitude = np.mean(np.abs(attributions_array[i]))
            frame_groups[frame_idx].append(frame_magnitude)
        
        # Sort frames and compute statistics
        frame_indices = sorted(frame_groups.keys())
        frame_means = []
        frame_stds = []
        frame_cis_lower = []
        frame_cis_upper = []
        
        for frame_idx in frame_indices:
            magnitudes = np.array(frame_groups[frame_idx])
            mean_val = np.mean(magnitudes)
            std_val = np.std(magnitudes)
            n_samples = len(magnitudes)
            
            # 95% CI for the mean
            ci_margin = 1.96 * std_val / np.sqrt(n_samples)
            
            frame_means.append(mean_val)
            frame_stds.append(std_val)
            frame_cis_lower.append(mean_val - ci_margin)
            frame_cis_upper.append(mean_val + ci_margin)
        
        frame_means = np.array(frame_means)
        frame_cis_lower = np.array(frame_cis_lower)
        frame_cis_upper = np.array(frame_cis_upper)
        
        # Plot mean temporal evolution
        axes[1, 2].plot(frame_indices, frame_means, 'b-', linewidth=2, marker='o', 
                       label='Mean Attribution')
        
        # Add confidence interval as shaded region
        axes[1, 2].fill_between(frame_indices, frame_cis_lower, frame_cis_upper, 
                               alpha=0.3, color='lightblue', label='95% CI')
        
        # Determine unique files for title
        unique_files = len(set(info[0] for info in file_info))
        
        axes[1, 2].set_title(f'Mean Temporal Evolution of Attribution Magnitude\n({unique_files} files)')
        axes[1, 2].set_xlabel('Frame Index')
        axes[1, 2].set_ylabel('Mean |Attribution|')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend(fontsize=8, loc='upper right')
        
    else:
        # Single-file analysis: temporal evolution or single-frame distribution
        temporal_importance = np.mean(np.abs(attributions_array), axis=(1, 2))
        
        if len(temporal_importance) > 1:
            # Normal temporal plot for multiple frames
            axes[1, 2].plot(temporal_importance, 'b-', linewidth=2, marker='o')
            axes[1, 2].set_title('Temporal Evolution of Attribution Magnitude')
            axes[1, 2].set_xlabel('Frame')
            axes[1, 2].set_ylabel('Mean |Attribution|')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # Single frame - show distribution instead
            attrs_flat = np.abs(attributions_array).flatten()
            axes[1, 2].hist(attrs_flat, bins=50, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
            axes[1, 2].set_title(f'Attribution Magnitude Distribution\n(Single Frame)')
            axes[1, 2].set_xlabel('|Attribution| Value')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            
            # Add statistics lines
            mean_val = np.mean(attrs_flat)
            std_val = np.std(attrs_flat)
            axes[1, 2].axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                              label=f'Mean: {mean_val:.2g}')
            axes[1, 2].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.8, 
                              label=f'Mean+Std: {mean_val+std_val:.2g}')
            axes[1, 2].legend(fontsize=8, loc='upper right')
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return mean_attr, std_attr, max_attr


def generate_cache_filename(args, model_spec, train_spec, rollout_spec):
    """
    Generate a unique cache filename based on key parameters.
    """
    # Create a string with all the key parameters that affect attribution computation
    cache_params = {
        'model_name': args.model_name,
        'data_name': args.data_name,
        'method': args.method,
        'file_id': args.file_id,
        'n_steps': args.n_steps,
        'max_frames': args.max_frames,
        'seed': args.seed,
        'model_spec': {k: v for k, v in model_spec.items() if k in ['node_feature_function', 'in_node_dim', 'heads']},
        'train_spec': {k: v for k, v in train_spec.items() if k in ['sigma', 'visual_range']},
        'rollout_spec': rollout_spec
    }
    
    # Create hash from parameters
    param_str = str(sorted(cache_params.items()))
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    # Generate descriptive filename
    file_suffix = "allfiles" if args.file_id == -1 else f"file{args.file_id}"
    cache_filename = f"attribution_cache_{args.data_name}_{args.method}_{file_suffix}_{param_hash}.pkl"
    
    return cache_filename


def save_attribution_cache(cache_filename, attribution_results, metadata):
    """
    Save computed attributions to cache file.
    """
    os.makedirs("attribution_cache", exist_ok=True)
    cache_path = os.path.join("attribution_cache", cache_filename)
    
    cache_data = {
        'attribution_results': attribution_results,
        'metadata': metadata,
        'cache_version': '1.0'
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"💾 Saved attribution cache to {cache_path}")


def load_attribution_cache(cache_filename):
    """
    Load computed attributions from cache file if it exists.
    Returns None if cache doesn't exist or is invalid.
    """
    cache_path = os.path.join("attribution_cache", cache_filename)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"🚀 Loaded attribution cache from {cache_path}")
        return cache_data['attribution_results'], cache_data['metadata']
    
    except (pickle.PickleError, KeyError, FileNotFoundError) as e:
        print(f"⚠️ Failed to load cache {cache_path}: {e}")
        return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNN Explainability Analysis using IntegratedGradients or Saliency',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data arguments
    parser.add_argument('--model-name', type=str, default='vpluspplus_a',
                        help='Model architecture name')
    parser.add_argument('--data-name', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Full path to model (if provided, overrides model-name and data-name)')
    parser.add_argument('--rollout-path', type=str, default=None,
                        help='Path to rollout data (if provided, overrides default path)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Model seed')
    
    # Analysis parameters
    parser.add_argument('--method', type=str, default='integrated_gradients',
                        choices=['integrated_gradients', 'saliency'],
                        help='Attribution method to use')
    parser.add_argument('--file-id', type=int, default=5,
                        help='Which file in the batch to analyze (use -1 to analyze ALL files)')
    parser.add_argument('--n-steps', type=int, default=50,
                        help='Number of integration steps (for IntegratedGradients)')
    parser.add_argument('--max-frames', type=int, default=20,
                        help='Maximum number of frames to process')
    
    # Output arguments
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output files (default: auto-generated)')
    parser.add_argument('--save-data', action='store_true', default=False,
                        help='Save attribution data as pickle file')
    parser.add_argument('--no-save-data', dest='save_data', action='store_false',
                        help='Do not save attribution data')
    
    # Caching arguments
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='Use cached attribution results if available')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false',
                        help='Disable caching and recompute attributions')
    parser.add_argument('--force-recompute', action='store_true', default=False,
                        help='Force recomputation even if cache exists')
    
    return parser.parse_args()


def main():
    """Main function to run the explainability analysis"""
    
    args = parse_arguments()
    
    # Generate model path
    if args.model_path:
        model_rel_path = args.model_path
    else:
        # Auto-generate model path based on naming convention
        if args.data_name == 'boid_food_basic':
            noise_str = f"n{0}"  # Food model typically uses n0
        else:
            noise_str = f"n{0.005}"  # Standard model uses n0.005
        
        model_rel_path = f"trained_models/{args.data_name}/trained_models/{args.data_name}_{args.model_name}_{noise_str}_h1_vr0.5_s{args.seed}"
    
    # Load model
    print(f"Loading model from {model_rel_path}...")
    model, model_spec, train_spec = load_model(args.model_name, model_rel_path, ".")
    print(f"Model spec: {model_spec}")
    print(f"Train spec: {train_spec}")
    
    # Generate rollout path
    if args.rollout_path:
        rollout_root_path = args.rollout_path
    else:
        rollout_root_path = f"trained_models/{args.data_name}/rollouts"
    
    # Load rollout
    rollout_spec = {
        "noise": train_spec["sigma"],
        "head": model_spec["heads"],
        "visual_range": train_spec["visual_range"],
        "seed": args.seed,
        "rollout_starting_frame": 5,
        "rollout_frames": 300,
    }
    
    print(f"Loading rollout from {rollout_root_path} with spec: {rollout_spec}")
    rollout_result = load_rollout(
        args.model_name, args.data_name,
        root_path=rollout_root_path,
        **rollout_spec
    )
    
    # Generate cache filename and check for existing cache
    method_str = "IntegratedGradients" if args.method == 'integrated_gradients' else "Saliency"
    cache_filename = generate_cache_filename(args, model_spec, train_spec, rollout_spec)
    
    attribution_results = None
    cache_metadata = None
    
    # Try to load from cache if enabled and not forcing recomputation
    if args.use_cache and not args.force_recompute:
        cache_result = load_attribution_cache(cache_filename)
        if cache_result is not None:
            attribution_results, cache_metadata = cache_result
            print(f"✅ Using cached {method_str} attributions")
    
    # Compute attributions if not loaded from cache
    if attribution_results is None:
        print(f"Computing {method_str} attributions...")
        attribution_results = compute_rollout_attributions(
            model,
            rollout_result,
            model_spec,
            train_spec,
            file_id=args.file_id,
            method=args.method,
            n_steps=args.n_steps,
            max_frames=args.max_frames
        )
        
        # Save to cache if caching is enabled
        if args.use_cache:
            cache_metadata = {
                'method': args.method,
                'file_id': args.file_id,
                'max_frames': args.max_frames,
                'n_steps': args.n_steps,
                'data_name': args.data_name,
                'model_name': args.model_name,
                'seed': args.seed,
                'computed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            save_attribution_cache(cache_filename, attribution_results, cache_metadata)
    
    # Generate output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        if args.file_id == -1:
            output_prefix = f"{args.method}_analysis_{args.model_name}_{args.data_name}_s{args.seed}_allfiles"
        else:
            output_prefix = f"{args.method}_analysis_{args.model_name}_{args.data_name}_s{args.seed}"
    
    # Handle multi-file vs single-file results
    if isinstance(attribution_results, dict):
        # Multi-file analysis (file_id = -1)
        print(f"Analysis complete for {len(attribution_results)} files")
        print("Creating combined visualization across all files...")
        
        # Combine all attributions into one list for visualization
        all_frame_attributions = []
        file_info = []
        for file_id, file_attributions in attribution_results.items():
            all_frame_attributions.extend(file_attributions)
            file_info.extend([(file_id, frame_idx) for frame_idx in range(len(file_attributions))])
        
        # Visualize combined results
        save_path = f"{output_prefix}.png"
        mean_attr, std_attr, max_attr = visualize_attribution_statistics(
            all_frame_attributions, 
            save_path=save_path,
            data_name=args.data_name,
            file_info=file_info
        )
        
        frame_attributions = attribution_results  # For data saving
        
    else:
        # Single-file analysis
        frame_attributions = attribution_results
        
        # Visualize results
        print("Visualizing attribution statistics...")
        save_path = f"{output_prefix}.png"
        mean_attr, std_attr, max_attr = visualize_attribution_statistics(
            frame_attributions, 
            save_path=save_path,
            data_name=args.data_name
        )
    
    # Save attribution data
    if args.save_data:
        attribution_data = {
            'frame_attributions': frame_attributions,
            'mean_attribution': mean_attr,
            'std_attribution': std_attr,
            'max_attribution': max_attr,
            'model_spec': model_spec,
            'train_spec': train_spec,
            'rollout_spec': rollout_spec,
            'method': args.method,
            'args': vars(args)
        }
        
        save_file = f"{output_prefix}_data.pkl"
        with open(save_file, 'wb') as f:
            pickle.dump(attribution_data, f)
        print(f"Saved attribution data to {save_file}")
    
    print(f"\n✅ Analysis complete!")
    print(f"Method: {method_str}")
    
    # Handle different return types for summary
    if isinstance(frame_attributions, dict):
        total_frames = sum(len(file_attrs) for file_attrs in frame_attributions.values())
        print(f"Processed {len(frame_attributions)} files with {total_frames} total frames")
        # Get shape from first file's first frame
        first_file_attrs = next(iter(frame_attributions.values()))
        if len(first_file_attrs) > 0:
            print(f"Attribution matrix shape per frame: {first_file_attrs[0].shape}")
    else:
        print(f"Processed {len(frame_attributions)} frames")
        if len(frame_attributions) > 0:
            print(f"Attribution matrix shape per frame: {frame_attributions[0].shape}")
    
    return frame_attributions, mean_attr, std_attr, max_attr


if __name__ == "__main__":
    try:
        # Run main analysis
        frame_attributions, mean_attr, std_attr, max_attr = main()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)