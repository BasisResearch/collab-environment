#!/usr/bin/env python
"""Main entry point for NRI training and visualization on boids data."""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from loguru import logger

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from collab_env.gnn.nri_model import create_nri_model_for_boids
from collab_env.gnn.nri_training import (
    load_boids_dataset,
    prepare_data_loaders, 
    train_model,
    save_model,
    load_model
)
from collab_env.gnn.nri_visualization import (
    generate_rollout,
    plot_trajectories_and_interactions,
    create_animation
)


def main():
    parser = argparse.ArgumentParser(description='NRI for Boids - Training and Visualization')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, 
                       default='simulated_data/boid_single_species_basic.pt',
                       help='Path to boids dataset')
    parser.add_argument('--num-sequences', type=int, default=100,
                       help='Number of sequences to use from dataset')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension for NRI model')
    parser.add_argument('--n-edge-types', type=int, default=2,
                       help='Number of edge types to learn')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=10,
                       help='Input sequence length')
    parser.add_argument('--pred-len', type=int, default=1,
                       help='Prediction length')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Train/validation split ratio')
    
    # Loss weights
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for KL divergence loss')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Weight for sparsity regularization')
    parser.add_argument('--gradient-clipping', action='store_true',
                       help='Enable gradient clipping (default: disabled)')
    parser.add_argument('--clip-max-norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Rollout arguments
    parser.add_argument('--rollout-steps', type=int, default=100,
                       help='Number of rollout steps for visualization')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Skip training and only visualize with existing model')
    parser.add_argument('--val-seq-idx', type=int, default=0,
                       help='Validation sequence index to use for visualization (default: 0, use -1 for random)')
    
    # Output arguments
    parser.add_argument('--model-dir', type=str, default='trained_models/nri_models',
                       help='Directory for saving/loading models')
    parser.add_argument('--output-dir', type=str, default='trained_models/nri_outputs',
                       help='Directory for visualization outputs')
    parser.add_argument('--vis-limits', type=float, nargs=4, default=[0, 1, 0, 1],
                       help='Visualization axis limits as 4-tuple: xmin xmax ymin ymax')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    
    # Create output directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading dataset from {args.data_path}")
    positions, velocities, species = load_boids_dataset(
        args.data_path, 
        num_sequences=args.num_sequences,
        device=device
    )
    
    n_agents = positions.shape[1]
    n_species = len(torch.unique(species))
    
    logger.info(f"Dataset shape: {positions.shape}")
    logger.info(f"Agents: {n_agents}, Species: {n_species}")
    
    # Create model and relation matrices
    model, rel_rec, rel_send = create_nri_model_for_boids(
        num_boids=n_agents,
        n_species=n_species,
        use_species_features=True,
        n_edge_types=args.n_edge_types,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device
    )
    
    model_path = Path(args.model_dir) / 'nri_model.pt'
    
    if not args.visualize_only:
        # Prepare data loaders
        logger.info("Preparing data loaders...")
        train_loader, val_loader, val_data = prepare_data_loaders(
            positions, velocities, species,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            train_split=args.train_split
        )
        
        # Train model
        logger.info(f"Training model for {args.epochs} epochs...")
        model = train_model(
            model, train_loader, val_loader, val_data,
            rel_rec, rel_send,
            epochs=args.epochs,
            lr=args.lr,
            beta=args.beta,
            alpha=args.alpha,
            device=device,
            use_rollout_validation=False,
            rollout_start=5,
            rollout_steps=min(50, args.rollout_steps),
            gradient_clipping=args.gradient_clipping,
            clip_max_norm=args.clip_max_norm,
            save_best_model_path=model_path  # Save best model at each improvement
        )
        
        # Save model
        save_model(model, rel_rec, rel_send, model_path, args)
        logger.info(f"Model saved to {model_path}")
    else:
        # Load existing model
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            
            # First load checkpoint to get saved args
            checkpoint = torch.load(model_path, map_location=device)
            saved_args = checkpoint.get('args', None)
            
            if saved_args:
                # Use saved training parameters for data loading to ensure compatibility
                logger.info("Using saved training parameters for data compatibility")
                positions, velocities, species = load_boids_dataset(
                    args.data_path, 
                    num_sequences=saved_args.num_sequences,  # Use saved parameter
                    device=device
                )
                
            model, rel_rec, rel_send = load_model(model_path, device)
        else:
            logger.error(f"No model found at {model_path}. Train a model first.")
            return
        
        # For visualize-only mode, still split data to get validation sequences
        logger.info("Preparing data split for validation sequences...")
        
        # Use saved training parameters for data preparation
        seq_len = saved_args.seq_len if saved_args else args.seq_len
        pred_len = saved_args.pred_len if saved_args else args.pred_len
        batch_size = saved_args.batch_size if saved_args else args.batch_size
        train_split = saved_args.train_split if saved_args else args.train_split
        
        _, _, val_data = prepare_data_loaders(
            positions, velocities, species,
            seq_len=seq_len,
            pred_len=pred_len,
            batch_size=batch_size,
            train_split=train_split
        )
    
    # Generate rollout and visualize
    logger.info("Generating rollout...")
    
    # Use validation sequence for visualization (ensures unseen data)
    val_positions, val_velocities, val_species = val_data
    
    # Select which validation sequence to use
    num_val_seqs = val_positions.shape[0]
    if args.val_seq_idx == -1:
        # Random selection
        import random
        seq_idx = random.randint(0, num_val_seqs - 1)
        logger.info(f"Randomly selected validation sequence {seq_idx} out of {num_val_seqs}")
    else:
        # Use specified index
        seq_idx = args.val_seq_idx
        if seq_idx >= num_val_seqs:
            logger.warning(f"Requested sequence {seq_idx} >= {num_val_seqs} available, using sequence 0")
            seq_idx = 0
        else:
            logger.info(f"Using validation sequence {seq_idx} out of {num_val_seqs}")
    
    initial_positions = val_positions[seq_idx:seq_idx+1]
    initial_velocities = val_velocities[seq_idx:seq_idx+1]
    initial_species = val_species[seq_idx:seq_idx+1]
    ground_truth_pos = val_positions[seq_idx]
    
    
    logger.info(f"Selected validation sequence {seq_idx} for rollout visualization")
    
    # Use correct context length for rollout
    if not args.visualize_only:
        context_len = args.seq_len  # Use current args during training
    else:
        context_len = seq_len  # Use saved seq_len during visualization
    
    # Store context_len for visualization (gt_frames is now 0)
    gt_frames_used = 0
    
    rollout_positions, rollout_velocities, edge_probs = generate_rollout(
        model, rel_rec, rel_send,
        initial_positions, initial_velocities, initial_species,
        rollout_steps=args.rollout_steps,
        context_len=context_len,
        device=device
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Parse visualization limits
    xlim = (args.vis_limits[0], args.vis_limits[1])
    ylim = (args.vis_limits[2], args.vis_limits[3])
    
    # Static plot - show context + predictions vs corresponding ground truth
    # Include context frames for both GT and predictions to see the full trajectory
    
    # Ground truth: same period as rollout (context_len + rollout_steps frames)
    gt_full_period = ground_truth_pos[:, :rollout_positions.shape[1]]
    
    # NRI: full rollout (already includes context + predictions)  
    nri_full_trajectory = rollout_positions
    
    static_path = Path(args.output_dir) / 'nri_trajectories.png'
    plot_trajectories_and_interactions(
        ground_truth_pos=gt_full_period,
        predicted_pos=nri_full_trajectory,
        edge_probs=edge_probs,
        save_path=static_path,
        xlim=xlim,
        ylim=ylim,
        skip_frames=0
    )
    
    # Animation - also show context + predictions
    animation_path = Path(args.output_dir) / 'nri_rollout.mp4'
    create_animation(
        ground_truth_pos=gt_full_period,
        predicted_pos=nri_full_trajectory,
        save_path=animation_path,
        xlim=xlim,
        ylim=ylim
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("NRI Training and Visualization Complete")
    logger.info("="*60)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training sequences: {args.num_sequences}")
    logger.info(f"Rollout steps: {args.rollout_steps}")
    logger.info(f"\nOutputs:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Static plot: {static_path}")
    logger.info(f"  Animation: {animation_path}")
    
    # Edge statistics
    if edge_probs is not None and edge_probs.shape[1] > 1:
        interaction_prob = edge_probs[:, 1].mean().item()
        no_interaction_prob = edge_probs[:, 0].mean().item()
        logger.info(f"\nEdge Statistics:")
        logger.info(f"  Interaction probability: {interaction_prob:.3f}")
        logger.info(f"  No interaction probability: {no_interaction_prob:.3f}")
    
    logger.info("="*60)


if __name__ == '__main__':
    main()