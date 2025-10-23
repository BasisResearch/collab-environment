#!/usr/bin/env python
"""
Main script for training InteractionParticle model on boids data.

Usage:
    python -m collab_env.gnn.interaction_particles.run_training [options]

Example:
    python -m collab_env.gnn.interaction_particles.run_training \
        --dataset collab_env/data/boids/boid_single_species_basic.pt \
        --epochs 100 \
        --batch-size 32 \
        --save-dir trained_models/interaction_particle
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import yaml
from loguru import logger

from .train import train_interaction_particle, evaluate_model, evaluate_rollout
from .plotting import (
    plot_interaction_functions,
    compare_with_true_boids,
    plot_training_history,
    create_rollout_report
)
from collab_env.data.file_utils import expand_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train InteractionParticle model on boids data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='simulated_data/boid_single_species_basic.pt',
        help='Path to 2D boids dataset .pt file (AnimalTrajectoryDataset)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to 2D boids config file (.pt). If not provided, auto-detects or uses defaults.'
    )

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--visual-range', type=float, default=0.3, help='Visual range for edge construction (normalized)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for MLP')
    parser.add_argument('--embedding-dim', type=int, default=16, help='Particle embedding dimension')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of MLP layers')
    parser.add_argument('--n-particles', type=int, default=20, help='Number of particles')

    # Output arguments
    parser.add_argument(
        '--save-dir',
        type=str,
        default='trained_models/interaction_particle',
        help='Directory to save model and plots'
    )
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda/cuda:0)')

    # Evaluation
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate a saved model')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model for evaluation')
    parser.add_argument('--evaluate-rollout', action='store_true', help='Evaluate model with multi-step rollout')
    parser.add_argument('--n-rollout-steps', type=int, default=50, help='Number of steps for rollout evaluation')

    return parser.parse_args()


def load_boids_config(config_path, dataset_path=None):
    """
    Load 2D boids config from .pt file.

    The config file contains species_configs dictionary with parameters like:
    - visual_range: 50.0
    - min_distance: 15.0
    - avoid_factor: 0.05
    - matching_factor: 0.5
    - centering_factor: 0.005
    """
    # Default 2D boid parameters
    default_config = {
        'visual_range': 50.0,
        'min_distance': 15.0,
        'avoid_factor': 0.05,
        'matching_factor': 0.5,
        'centering_factor': 0.005,
    }

    if config_path is None:
        # Try to infer config from dataset path
        if dataset_path:
            config_path_guess = dataset_path.replace('.pt', '_config.pt')
            if os.path.exists(expand_path(config_path_guess)):
                config_path = config_path_guess
                logger.info(f"Auto-detected config: {config_path}")
            else:
                logger.info("No config file found, using defaults")
                return default_config
        else:
            logger.info("Using default 2D boid parameters")
            return default_config

    try:
        config_path = expand_path(config_path)
        # Load species_configs from .pt file
        species_configs = torch.load(config_path, weights_only=False)
        # Extract first species config (usually 'A')
        first_species = list(species_configs.keys())[0]
        config = species_configs[first_species]
        logger.info(f"Loaded config for species '{first_species}'")
        return config

    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        logger.warning("Using default parameters")
        return default_config


def main():
    """Main function."""
    args = parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Expand paths
    dataset_path = expand_path(args.dataset)
    save_dir = expand_path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Add file logging
    log_path = os.path.join(save_dir, 'training.log')
    logger.add(log_path, level="DEBUG")

    logger.info("=" * 60)
    logger.info("InteractionParticle Training on 2D Boids Data")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Device: {args.device or 'auto'}")

    # Load boids config
    boids_config = load_boids_config(args.config, dataset_path)
    if boids_config:
        logger.info("Boids config parameters:")
        for key, value in boids_config.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info("No boids config loaded")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Evaluation mode
    if args.eval_only:
        if args.model_path is None:
            logger.error("Must specify --model-path for evaluation mode")
            sys.exit(1)

        logger.info("Loading model for evaluation...")
        checkpoint = torch.load(args.model_path, map_location=device)

        # Create model
        from .model import InteractionParticle
        model = InteractionParticle(checkpoint['config'], device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        logger.info("Evaluating model...")
        metrics = evaluate_model(model, dataset_path, visual_range=args.visual_range, device=device)
        logger.info(f"Evaluation metrics: {metrics}")

        # Plot
        if not args.no_plot:
            logger.info("Generating plots...")
            plot_path = os.path.join(save_dir, 'interaction_functions_eval.png')
            plot_interaction_functions(model, save_path=plot_path)

            compare_path = os.path.join(save_dir, 'comparison_with_boids_eval.png')
            compare_with_true_boids(model, save_path=compare_path, config=boids_config)

        return

    # Create model config
    model_config = {
        'n_particles': args.n_particles,
        'n_particle_types': 1,
        'max_radius': 1.0,
        'hidden_dim': args.hidden_dim,
        'embedding_dim': args.embedding_dim,
        'n_mp_layers': args.n_layers,
        'input_size': 7,  # delta_pos(2) + r(1) + vel_i(2) + vel_j(2)
        'output_size': 2,
    }

    logger.info("Model configuration:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")

    # Train model
    logger.info("\nStarting training...")
    model, history = train_interaction_particle(
        dataset_path=dataset_path,
        config=model_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        visual_range=args.visual_range,
        device=device,
        save_dir=save_dir,
        seed=args.seed
    )

    logger.info("\nTraining completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")

    # Save model info
    info_path = os.path.join(save_dir, 'model_info.yaml')
    with open(info_path, 'w') as f:
        yaml.dump({
            'args': vars(args),
            'config': model_config,
            'boids_config': boids_config,
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
        }, f, default_flow_style=False)
    logger.info(f"Saved model info to {info_path}")

    # Generate plots
    if not args.no_plot:
        logger.info("\nGenerating plots...")

        # Plot training history
        history_path = os.path.join(save_dir, 'training_history.png')
        plot_training_history(history, save_path=history_path)

        # Plot interaction functions
        plot_path = os.path.join(save_dir, 'interaction_functions.png')
        plot_interaction_functions(model, save_path=plot_path)

        # Compare with true boids
        compare_path = os.path.join(save_dir, 'comparison_with_boids.png')
        compare_with_true_boids(model, save_path=compare_path, config=boids_config)

        logger.info(f"Plots saved to {save_dir}")

    # Rollout evaluation
    if args.evaluate_rollout:
        logger.info("\nEvaluating model with multi-step rollout...")
        rollout_results = evaluate_rollout(
            model,
            dataset_path,
            visual_range=args.visual_range,
            n_rollout_steps=args.n_rollout_steps,
            device=device
        )

        # Create rollout report
        rollout_dir = os.path.join(save_dir, 'rollout_evaluation')
        os.makedirs(rollout_dir, exist_ok=True)

        logger.info(f"Creating rollout visualizations...")
        create_rollout_report(rollout_results, save_dir=rollout_dir)

        logger.info(f"Rollout evaluation saved to {rollout_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {save_dir}")
    logger.info(f"  - Model: best_model.pt, final_model.pt")
    logger.info(f"  - Plots: *.png")
    logger.info(f"  - Log: training.log")
    logger.info(f"  - Info: model_info.yaml")
    if args.evaluate_rollout:
        logger.info(f"  - Rollout evaluation: rollout_evaluation/*.png")


if __name__ == '__main__':
    main()
