#!/usr/bin/env python
"""
GraphGym experiment runner for GNN architecture search on boids trajectories.

This script allows you to:
1. Run single GNN experiments with specified architecture
2. Run grid search over multiple architectures
3. Compare different GNN types (GCN, GATv2, GIN, SAGE, etc.)

Examples:
    # Single experiment with GATv2
    python collab_env/gnn/run_graphgym_experiments.py --config configs/graphgym/base/boids_trajectory.yaml

    # Grid search
    python collab_env/gnn/run_graphgym_experiments.py \\
        --config configs/graphgym/base/boids_trajectory.yaml \\
        --grid configs/graphgym/grids/architecture_search.txt \\
        --max-workers 4

    # Quick test
    python collab_env/gnn/run_graphgym_experiments.py --test
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import torch
import numpy as np
from datetime import datetime
import json
import concurrent.futures
from typing import List, Dict

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collab_env.gnn.graphgym.config import (
    load_config,
    create_config,
    generate_grid_configs,
    save_config,
)
from collab_env.gnn.graphgym.dataset import create_boids_datasets
from collab_env.gnn.graphgym.trainer import train_model
from collab_env.gnn.graphgym.models import create_model_from_config


def run_single_experiment(config, worker_id: int = 0):
    """
    Run a single experiment with the given configuration.

    Args:
        config: ExperimentConfig object
        worker_id: Worker ID for logging

    Returns:
        Dictionary with results
    """
    try:
        logger.info(f"[Worker {worker_id}] Starting experiment")
        logger.info(f"[Worker {worker_id}] Config: layer={config.gnn.layer_type}, "
                   f"dim={config.gnn.dim_inner}, heads={config.gnn.heads}, "
                   f"depth={config.gnn.layers_mp}")

        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Create datasets
        train_dataset, val_dataset, test_dataset = create_boids_datasets(
            dataset_name=config.dataset.name,
            visual_range=config.dataset.visual_range,
            train_ratio=config.dataset.split[0],
            val_ratio=config.dataset.split[1],
            seed=config.seed,
            start_frame=config.dataset.start_frame,
            input_differentiation=config.dataset.input_differentiation,
        )

        logger.info(f"[Worker {worker_id}] Datasets created: "
                   f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        # Update output directory with experiment details
        exp_name = (f"{config.gnn.layer_type}_dim{config.gnn.dim_inner}_"
                   f"h{config.gnn.heads}_l{config.gnn.layers_mp}_"
                   f"s{config.seed}")
        config.out_dir = str(Path(config.out_dir) / exp_name)

        # Train model
        model, history = train_model(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )

        # Get final metrics
        best_val_loss = min(history['val_losses']) if history['val_losses'] else float('inf')
        final_train_loss = history['train_losses'][-1] if history['train_losses'] else float('inf')

        result = {
            'worker_id': worker_id,
            'layer_type': config.gnn.layer_type,
            'dim_inner': config.gnn.dim_inner,
            'heads': config.gnn.heads,
            'layers_mp': config.gnn.layers_mp,
            'stage_type': config.gnn.stage_type,
            'agg': config.gnn.agg,
            'dropout': config.gnn.dropout,
            'batchnorm': config.gnn.batchnorm,
            'visual_range': config.dataset.visual_range,
            'seed': config.seed,
            'best_val_loss': best_val_loss,
            'final_train_loss': final_train_loss,
            'status': 'success',
            'out_dir': config.out_dir,
        }

        logger.success(f"[Worker {worker_id}] Experiment complete! "
                      f"Best val loss: {best_val_loss:.6f}")

        return result

    except Exception as e:
        logger.error(f"[Worker {worker_id}] Experiment failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            'worker_id': worker_id,
            'status': 'failed',
            'error': str(e),
            'layer_type': config.gnn.layer_type if hasattr(config, 'gnn') else 'unknown',
        }


def run_grid_search(
    base_config,
    grid_file: str,
    max_workers: int = 1,
    results_dir: str = "results/graphgym/grid_search"
):
    """
    Run grid search over multiple configurations.

    Args:
        base_config: Base configuration
        grid_file: Path to grid search specification
        max_workers: Maximum number of parallel workers
        results_dir: Directory to save results

    Returns:
        List of results
    """
    # Generate all configurations
    logger.info("Generating configurations from grid search...")
    configs = generate_grid_configs(base_config, grid_file)

    logger.info(f"Generated {len(configs)} configurations")
    logger.info(f"Running with {max_workers} workers")

    # Create results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save all configs
    configs_dir = results_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    for i, cfg in enumerate(configs):
        save_config(cfg, str(configs_dir / f"config_{i:04d}.yaml"))

    # Run experiments in parallel
    results = []
    start_time = datetime.now()

    if max_workers > 1:
        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_experiment, cfg, i): i
                for i, cfg in enumerate(configs)
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

                completed = len(results)
                elapsed = (datetime.now() - start_time).total_seconds() / 60

                logger.info(f"Progress: {completed}/{len(configs)} | Elapsed: {elapsed:.1f} min")
    else:
        # Sequential execution
        for i, cfg in enumerate(configs):
            result = run_single_experiment(cfg, i)
            results.append(result)

            completed = i + 1
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            logger.info(f"Progress: {completed}/{len(configs)} | Elapsed: {elapsed:.1f} min")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.success(f"Grid search complete! Results saved to {results_file}")

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        best = min(successful, key=lambda x: x['best_val_loss'])
        logger.success(f"Best configuration: {best['layer_type']} "
                      f"(dim={best['dim_inner']}, heads={best['heads']}, "
                      f"layers={best['layers_mp']}, val_loss={best['best_val_loss']:.6f})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="GraphGym experiments for GNN architecture search on boids trajectories"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )

    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Path to grid search file (optional, for architecture search)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for grid search"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with minimal config"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="boid_single_species_basic",
        help="Dataset name"
    )

    parser.add_argument(
        "--layer-type",
        type=str,
        default="gatv2conv",
        choices=["gcnconv", "gatv2conv", "ginconv", "sageconv", "generalconv"],
        help="GNN layer type"
    )

    parser.add_argument(
        "--visual-range",
        type=float,
        default=0.1,
        help="Visual range for edge construction"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    logger.info("=" * 80)
    logger.info("GraphGym Architecture Search for Boids Trajectory Prediction")
    logger.info("=" * 80)

    # Load or create config
    if args.test:
        logger.info("Running in TEST mode with minimal config")
        config = create_config(
            dataset_name=args.dataset,
            visual_range=args.visual_range,
            layer_type=args.layer_type,
            dim_inner=64,
            heads=1,
            seed=args.seed,
        )
        config.optim.max_epoch = 5
        config.train.early_stopping_patience = 2

    elif args.config:
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)

        # Override with command line args if provided
        config.dataset.name = args.dataset
        config.dataset.visual_range = args.visual_range
        config.gnn.layer_type = args.layer_type
        config.seed = args.seed

    else:
        logger.info("No config file provided, using default")
        config = create_config(
            dataset_name=args.dataset,
            visual_range=args.visual_range,
            layer_type=args.layer_type,
            seed=args.seed,
        )

    # Run experiment(s)
    if args.grid:
        logger.info("Running GRID SEARCH")
        results = run_grid_search(
            base_config=config,
            grid_file=args.grid,
            max_workers=args.max_workers,
        )
    else:
        logger.info("Running SINGLE EXPERIMENT")
        result = run_single_experiment(config, worker_id=0)
        logger.info(f"Result: {result}")

    logger.success("All experiments complete!")


if __name__ == "__main__":
    main()
