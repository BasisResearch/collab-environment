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

from .train import train_interaction_particle, evaluate_model
from .plotting import plot_interaction_functions, compare_with_true_boids, plot_training_history
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
        help='Path to dataset .pt file (2D or 3D boids)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to boids config file (.pt or .yaml). If not provided, uses defaults.'
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

    return parser.parse_args()


def load_boids_config(config_path, dataset_path=None):
    """
    Load boids config from .pt or .yaml file.

    For 2D boids (from boids_gnn_temp), config is a .pt file with species_configs.
    For 3D boids (from sim/boids), config is a .yaml file.
    """
    if config_path is None:
        # Try to infer config from dataset path
        if dataset_path:
            config_path_guess = dataset_path.replace('.pt', '_config.pt')
            if os.path.exists(expand_path(config_path_guess)):
                config_path = config_path_guess
                logger.info(f"Inferred config path: {config_path}")
            else:
                logger.info("No config file found, using 2D boids defaults")
                return {
                    'visual_range': 50.0,
                    'min_distance': 15.0,
                    'avoid_factor': 0.05,
                    'matching_factor': 0.5,
                    'centering_factor': 0.005,
                }
        else:
            return None

    try:
        config_path = expand_path(config_path)

        if config_path.endswith('.pt'):
            # 2D boids config
            species_configs = torch.load(config_path, weights_only=False)
            # Extract first species config (usually 'A')
            first_species = list(species_configs.keys())[0]
            config = species_configs[first_species]
            logger.info(f"Loaded 2D boids config for species '{first_species}'")
            return config

        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            # 3D boids config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Loaded 3D boids config from YAML")
            return config['agent']

        else:
            logger.warning(f"Unknown config file format: {config_path}")
            return None

    except Exception as e:
        logger.warning(f"Could not load boids config from {config_path}: {e}")
        logger.warning("Using default 2D boid parameters")
        return {
            'visual_range': 50.0,
            'min_distance': 15.0,
            'avoid_factor': 0.05,
            'matching_factor': 0.5,
            'centering_factor': 0.005,
        }


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
    logger.info("InteractionParticle Training on Boids Data")
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

    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {save_dir}")
    logger.info(f"  - Model: best_model.pt, final_model.pt")
    logger.info(f"  - Plots: *.png")
    logger.info(f"  - Log: training.log")
    logger.info(f"  - Info: model_info.yaml")


if __name__ == '__main__':
    main()
