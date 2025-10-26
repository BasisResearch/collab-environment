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

from .train import train_interaction_particle, evaluate_model, evaluate_rollout, compute_velocity_statistics
from .plotting import (
    plot_interaction_functions,
    compare_with_true_boids,
    plot_training_history,
    create_rollout_report,
    evaluate_forces_on_grid,
    evaluate_velocity_forces_on_grid,
    plot_force_decomposition,
    plot_symmetric_force_decomposition,
    evaluate_true_boid_forces,
    evaluate_true_boid_velocity_forces,
    plot_rollout_comparison
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
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--visual-range', type=float, default=0.3, help='Visual range for edge construction (normalized)')
    parser.add_argument('--scene-size', type=float, default=480.0, help='Scene size in pixels (for normalizing config parameters)')
    parser.add_argument('--rollout-steps', type=int, default=1, help='Number of rollout steps for training (1=single-step, >1=multi-step)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for MLP')
    parser.add_argument('--embedding-dim', type=int, default=16, help='Particle embedding dimension')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of MLP layers')
    parser.add_argument('--n-particles', type=int, default=20, help='Number of particles (auto-detected from config if available)')

    # Output arguments
    parser.add_argument(
        '--save-dir',
        type=str,
        default='trained_models/interaction_particle',
        help='Directory to save model and plots'
    )
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda/cuda:0/mps/auto)')
    parser.add_argument('--plot-every', type=int, default=0, help='Plot force decomposition and rollout every N epochs (0=only at end)')

    # Evaluation
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate a saved model')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model for evaluation')
    parser.add_argument('--evaluate-rollout', action='store_true', help='Evaluate model with multi-step rollout')
    parser.add_argument('--n-rollout-steps', type=int, default=50, help='Number of steps for rollout evaluation')

    return parser.parse_args()


def load_boids_config(config_path, dataset_path : Path=None):
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
            config_path_guess = dataset_path.with_stem(dataset_path.stem + '_config')
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

    # Compute velocity statistics from dataset for visualization
    logger.info("Computing velocity statistics from dataset...")
    vel_stats = compute_velocity_statistics(dataset_path)

    # Use for speed magnitude in position-dependent force plots
    speed_magnitude = vel_stats['mean_speed']
    logger.info(f"Using mean speed = {speed_magnitude:.4f} for position-dependent force visualization")

    # Use for velocity grid bounds in velocity-dependent force plots
    max_rel_vel = vel_stats['max_relative_velocity']
    logger.info(f"Using max relative velocity = {max_rel_vel:.4f} for velocity-dependent force visualization")

    # Set device
    # Device selection with MPS support
    if args.device and args.device != 'auto':
        device = torch.device(args.device)
    else:
        # Auto-detect: cuda > mps > cpu
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
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

    # Determine n_particles from boids config or command line
    if boids_config and 'counts' in boids_config:
        n_particles = boids_config['counts']
        logger.info(f"Using n_particles={n_particles} from boids config")
    else:
        n_particles = args.n_particles
        logger.info(f"Using n_particles={n_particles} from command line argument")

    # Create model config
    model_config = {
        'n_particles': n_particles,
        'n_particle_types': 1,
        'max_radius': 1.0,
        'hidden_dim': args.hidden_dim,
        'embedding_dim': args.embedding_dim,
        'n_mp_layers': args.n_layers,
        'input_size': 11,  # delta_pos(2) + r(1) + delta_vel(2) + pos_i(2) + vel_i(2) + boundary_complement(2)
        'output_size': 2,
    }

    logger.info("Model configuration:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")

    # Define epoch callback for plotting during training
    def epoch_plotting_callback(model, epoch, train_loss, val_loss, save_dir):
        """Generate plots after each validation epoch if plot_every is set."""
        if args.plot_every <= 0 or epoch % args.plot_every != 0:
            return

        logger.info(f"\nGenerating epoch {epoch} plots...")
        epoch_dir = os.path.join(save_dir, f'epoch_{epoch:04d}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Plot force decomposition comparison
        try:
            # Use the training visual_range parameter for BOTH learned and true forces
            # This ensures fair comparison (same plot range for both)
            plot_max_dist = args.visual_range * 1.5  # 50% cushion

            # Evaluate learned forces (use same speed magnitude as true forces)
            learned_forces = evaluate_forces_on_grid(
                model,
                grid_size=50,
                max_dist=plot_max_dist,
                particle_idx=0,
                speed_magnitude=speed_magnitude
            )

            # Evaluate ground truth forces (use SAME range as learned for comparison)
            true_forces = evaluate_true_boid_forces(
                boids_config,
                grid_size=50,
                max_dist=plot_max_dist,
                scene_size=args.scene_size
            )

            # Plot learned forces
            learned_plot_path = os.path.join(epoch_dir, 'learned_force_decomposition.png')
            plot_force_decomposition(
                learned_forces,
                save_path=learned_plot_path,
                title_prefix=f"Learned (Epoch {epoch})",
                visual_range=args.visual_range
            )

            # Plot ground truth forces
            true_plot_path = os.path.join(epoch_dir, 'true_boid_force_decomposition.png')
            # Extract min_distance from boids_config (normalized to [0,1] space)
            min_distance_normalized = None
            if boids_config and 'min_distance' in boids_config:
                min_distance_normalized = boids_config['min_distance'] / args.scene_size

            plot_force_decomposition(
                true_forces,
                save_path=true_plot_path,
                title_prefix=f"True Boid (Epoch {epoch})",
                visual_range=args.visual_range,
                min_distance=min_distance_normalized
            )

            logger.info(f"Saved force decomposition plots to {epoch_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate force decomposition plots: {e}")

        # Plot symmetric decomposition
        try:
            logger.info(f"Generating symmetric force decomposition for epoch {epoch}...")

            # Compute velocity statistics for max_rel_vel
            max_rel_vel = 0.05  # Use same default as after-training plots

            # Evaluate learned forces on velocity grid (delta_pos = 0)
            learned_vel_forces = evaluate_velocity_forces_on_grid(
                model,
                grid_size=50,
                max_vel=max_rel_vel,
                particle_idx=0
            )

            # Evaluate true boid forces on velocity grid (delta_pos = 0)
            true_vel_forces = evaluate_true_boid_velocity_forces(
                boids_config,
                grid_size=50,
                max_vel=max_rel_vel,
                scene_size=args.scene_size
            )

            # Plot learned symmetric decomposition
            learned_symmetric_path = os.path.join(epoch_dir, 'learned_symmetric_decomposition.png')
            plot_symmetric_force_decomposition(
                learned_forces,  # position-dependent (already computed above)
                learned_vel_forces,  # velocity-dependent
                save_path=learned_symmetric_path,
                title_prefix=f"Learned (Epoch {epoch})",
                visual_range=args.visual_range
            )

            # Plot true boid symmetric decomposition
            true_symmetric_path = os.path.join(epoch_dir, 'true_boid_symmetric_decomposition.png')
            plot_symmetric_force_decomposition(
                true_forces,  # position-dependent (already computed above)
                true_vel_forces,  # velocity-dependent
                save_path=true_symmetric_path,
                title_prefix=f"True Boid (Epoch {epoch})",
                visual_range=args.visual_range
            )

            logger.info(f"Saved symmetric decomposition plots to {epoch_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate symmetric decomposition plots: {e}")

        # Plot rollout comparison
        if args.evaluate_rollout:
            try:
                rollout_results = evaluate_rollout(
                    model,
                    dataset_path,
                    visual_range=args.visual_range,
                    n_rollout_steps=args.n_rollout_steps,
                    device=device
                )

                # Save rollout comparison (use first trajectory)
                if len(rollout_results['ground_truth_positions']) > 0:
                    rollout_plot_path = os.path.join(epoch_dir, 'rollout_comparison.png')
                    plot_rollout_comparison(
                        rollout_results['ground_truth_positions'][0],
                        rollout_results['predicted_positions'][0],
                        trajectory_idx=0,
                        save_path=rollout_plot_path
                    )
                else:
                    logger.warning(f"No rollout trajectories to plot")

                logger.info(f"Saved rollout comparison to {epoch_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate rollout comparison: {e}")

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
        rollout_steps=args.rollout_steps,
        device=device,
        save_dir=save_dir,
        seed=args.seed,
        epoch_callback=epoch_plotting_callback if args.plot_every > 0 else None
    )

    logger.info("\nTraining completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.2g}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.2g}")

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

        # Generate 2D force decomposition plots (2x3 layout)
        logger.info("Generating 2D force decomposition plots...")

        # Use the training visual_range parameter for BOTH learned and true forces
        # This ensures fair comparison (same plot range for both)
        plot_max_dist = args.visual_range * 1.5  # 50% cushion

        # Evaluate learned forces on grid (use same speed magnitude as true forces)
        learned_forces = evaluate_forces_on_grid(
            model,
            grid_size=50,
            max_dist=plot_max_dist,
            particle_idx=0,
            speed_magnitude=speed_magnitude
        )

        # Evaluate ground truth boid forces (use SAME range as learned for comparison)
        true_forces = evaluate_true_boid_forces(
            boids_config,
            grid_size=50,
            max_dist=plot_max_dist,
            scene_size=args.scene_size
        )

        # Plot learned forces
        learned_plot_path = os.path.join(save_dir, 'learned_force_decomposition.png')
        plot_force_decomposition(
            learned_forces,
            save_path=learned_plot_path,
            title_prefix="Learned",
            visual_range=args.visual_range
        )

        # Plot ground truth forces
        true_plot_path = os.path.join(save_dir, 'true_boid_force_decomposition.png')
        # Extract min_distance from boids_config (normalized to [0,1] space)
        min_distance_normalized = None
        if boids_config and 'min_distance' in boids_config:
            min_distance_normalized = boids_config['min_distance'] / args.scene_size

        plot_force_decomposition(
            true_forces,
            save_path=true_plot_path,
            title_prefix="True Boid",
            visual_range=args.visual_range,
            min_distance=min_distance_normalized
        )

        # Generate SYMMETRIC force decomposition plots (NEW!)
        logger.info("Generating symmetric force decomposition plots...")

        # Position-dependent forces (with delta_vel = 0)
        pos_forces_learned = evaluate_forces_on_grid(
            model,
            grid_size=50,
            max_dist=plot_max_dist,
            particle_idx=0,
            speed_magnitude=0.0  # Zero velocity for position-only term
        )

        # Velocity-dependent forces (with delta_pos = 0)
        vel_forces_learned = evaluate_velocity_forces_on_grid(
            model,
            grid_size=50,
            max_vel=max_rel_vel,
            particle_idx=0
        )

        # Plot symmetric decomposition for learned forces
        symmetric_learned_path = os.path.join(save_dir, 'learned_symmetric_decomposition.png')
        plot_symmetric_force_decomposition(
            pos_forces_learned,
            vel_forces_learned,
            save_path=symmetric_learned_path,
            title_prefix="Learned",
            visual_range=args.visual_range
        )

        # Generate TRUE BOID symmetric decomposition for comparison
        logger.info("Generating true boid symmetric force decomposition...")

        # Position-dependent forces (true boids, delta_vel = 0)
        # Reuse the position grid from true_forces (already computed with vel=0)
        pos_forces_true = true_forces  # This has away_pos which is computed with vel=0

        # Velocity-dependent forces (true boids, delta_pos = 0)
        vel_forces_true = evaluate_true_boid_velocity_forces(
            boids_config,
            grid_size=50,
            max_vel=max_rel_vel,
            scene_size=args.scene_size
        )

        # Plot symmetric decomposition for true boid forces
        symmetric_true_path = os.path.join(save_dir, 'true_boid_symmetric_decomposition.png')
        plot_symmetric_force_decomposition(
            pos_forces_true,
            vel_forces_true,
            save_path=symmetric_true_path,
            title_prefix="True Boid",
            visual_range=args.visual_range
        )

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
