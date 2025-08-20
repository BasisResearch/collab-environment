#!/usr/bin/env python
"""
Unified CLI for GNN training with multi-GPU support and Optuna optimization
"""

import os
import sys
import argparse
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess

import torch
import numpy as np
from itertools import product

# Add project paths
import pathlib
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from collab_env.gnn.train_gnn_optuna import (
    run_hyperparameter_search,
    train_with_best_params,
    load_dataset
)
from collab_env.gnn.train_gnn_simple import (
    load_all_datasets,
    make_specs,
    initialize_models,
    train
)


def detect_gpus() -> int:
    """Detect number of available GPUs"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_config(config: Dict[str, Any]):
    """Print configuration details"""
    for key, value in config.items():
        print(f"  {key}: {value}")


def setup_directories():
    """Create necessary directories"""
    dirs = ["logs", "models", "results"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(results_dir: str):
    """Save and organize results"""
    timestamp = get_timestamp()
    final_dir = Path(f"results/{results_dir}_{timestamp}")
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy result files
    for pattern in ["optuna_results.json", "training_summary.json", "training_results_*.json"]:
        for file in Path(".").glob(pattern):
            if file.exists():
                shutil.copy2(file, final_dir)
    
    # Copy Optuna database files
    for db_file in Path(".").glob("optuna_*.db"):
        shutil.copy2(db_file, final_dir)
    
    # If on RunPod, sync to persistent storage
    if Path("/workspace").exists():
        print("\nSyncing to RunPod persistent storage...")
        subprocess.run(["rsync", "-av", "models/", "/workspace/models/"], check=False)
        subprocess.run(["rsync", "-av", "results/", "/workspace/results/"], check=False)
    
    return final_dir


def run_test_mode():
    """Run quick test with minimal parameters"""
    print_header("QUICK TEST MODE")
    
    data_names = ['boid_single_species_basic']  # Just one dataset
    model_names = ["vpluspplus_a"]  # Just one model
    noise_levels = [0]  # Just one noise level
    heads = [1]  # Just one head config
    visual_ranges = [0.1]  # Just one visual range
    
    print("Test configuration:")
    print(f"  Dataset: {data_names[0]}")
    print(f"  Model: {model_names[0]}")
    print(f"  Epochs: 2")
    print(f"  Batch size: 4")
    
    # Load dataset
    data = load_all_datasets(data_names, batch_size=4)
    
    # Train one configuration
    model_spec, train_spec = make_specs(
        data_names[0], model_names[0], 4, noise_levels[0], heads[0], visual_ranges[0], 2
    )
    
    result = train(
        data[data_names[0]], model_spec, train_spec, 0, "test", None
    )
    
    final_loss = np.mean(result["train_losses"][-1]) if len(result["train_losses"]) > 0 else float('inf')
    print(f"\nTest completed! Final loss: {final_loss:.6f}")


def run_simple_mode(epochs: int, seed_num: int, batch_size: int):
    """Run all combinations without Optuna"""
    print_header("SIMPLE TRAINING MODE")
    
    # Configuration
    data_names = [
        'boid_single_species_basic',
        'boid_single_species_independent',
        'boid_food_basic_alignment',
        'boid_food_basic_independent',
        'boid_food_strong'
    ]
    
    model_names = ["vpluspplus_a", "lazy"]
    noise_levels = [0, 0.005]
    heads = [1, 2, 3]
    visual_ranges = [0.1, 0.5]
    
    # Generate all combinations
    all_combinations = list(product(data_names, model_names, noise_levels, heads, visual_ranges))
    total_configs = len(all_combinations) * seed_num
    
    print(f"Total configurations: {len(all_combinations)}")
    print(f"Seeds per configuration: {seed_num}")
    print(f"Total training runs: {total_configs}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    # Load all datasets
    print("\nLoading datasets...")
    all_data = load_all_datasets(data_names, batch_size=batch_size)
    
    results = []
    start_time = time.time()
    
    for idx, (data_name, model_name, noise, head, visual_range) in enumerate(all_combinations):
        for seed in range(seed_num):
            current = idx * seed_num + seed + 1
            print(f"\n[{current}/{total_configs}] Training {data_name}_{model_name}_n{noise}_h{head}_vr{visual_range}_s{seed}")
            
            model_spec, train_spec = make_specs(
                data_name, model_name, batch_size, noise, head, visual_range, epochs
            )
            
            save_name_postfix = f"noise{noise}_head{head}_vr{visual_range}"
            
            try:
                result = train(
                    all_data[data_name], model_spec, train_spec, seed, save_name_postfix, None
                )
                
                final_loss = np.mean(result["train_losses"][-1]) if len(result["train_losses"]) > 0 else float('inf')
                
                results.append({
                    "data_name": data_name,
                    "model_name": model_name,
                    "noise": noise,
                    "heads": head,
                    "visual_range": visual_range,
                    "seed": seed,
                    "final_loss": float(final_loss),
                    "file_name": result["file_name"]
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "data_name": data_name,
                    "model_name": model_name,
                    "noise": noise,
                    "heads": head,
                    "visual_range": visual_range,
                    "seed": seed,
                    "final_loss": float('inf'),
                    "error": str(e)
                })
    
    # Save results
    elapsed = time.time() - start_time
    timestamp = get_timestamp()
    summary_file = f"training_results_{timestamp}.json"
    
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print_header("TRAINING COMPLETED")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Results saved to: {summary_file}")
    
    # Print best configurations
    print("\nBest configurations per dataset:")
    for data_name in data_names:
        data_results = [r for r in results if r["data_name"] == data_name and "error" not in r]
        if data_results:
            best = min(data_results, key=lambda x: x["final_loss"])
            print(f"\n{data_name}:")
            print(f"  Best loss: {best['final_loss']:.6f}")
            print(f"  Config: {best['model_name']}, noise={best['noise']}, heads={best['heads']}, vr={best['visual_range']}")


def run_optuna_mode(mode: str, n_trials: int, epochs: int, seed_num: int, world_size: int):
    """Run with Optuna optimization"""
    print_header(f"OPTUNA MODE: {mode.upper()}")
    
    data_names = [
        'boid_single_species_basic',
        'boid_single_species_independent',
        'boid_food_basic_alignment',
        'boid_food_basic_independent',
        'boid_food_strong'
    ]
    
    model_names = ["vpluspplus_a", "lazy"]
    
    print(f"Mode: {mode}")
    print(f"Trials: {n_trials}")
    print(f"Epochs: {epochs}")
    print(f"Seeds: {seed_num}")
    print(f"GPUs: {world_size}")
    
    studies = {}
    
    if mode in ["search", "both"]:
        print("\nPhase 1: Hyperparameter search")
        studies = run_hyperparameter_search(
            data_names,
            model_names,
            n_trials=n_trials,
            epochs=epochs,
            world_size=world_size
        )
        
        # Save results
        study_results = {}
        for key, study in studies.items():
            study_results[key] = {
                "best_value": study.best_value,
                "best_params": study.best_params
            }
        
        with open("optuna_results.json", "w") as f:
            json.dump(study_results, f, indent=2)
        
        print("\nBest parameters found:")
        for key, result in study_results.items():
            print(f"\n{key}:")
            print(f"  Loss: {result['best_value']:.6f}")
            print(f"  Params: {result['best_params']}")
    
    if mode in ["train", "both"]:
        print("\nPhase 2: Training with best parameters")
        
        # Load studies if training only
        if mode == "train" and os.path.exists("optuna_results.json"):
            with open("optuna_results.json", "r") as f:
                study_results = json.load(f)
            
            # Create mock studies
            for key, result in study_results.items():
                study = type('MockStudy', (), {})()
                study.best_params = result["best_params"]
                study.best_value = result["best_value"]
                studies[key] = study
        
        if studies:
            results = train_with_best_params(
                data_names,
                model_names,
                studies,
                seed_num=seed_num,
                epochs=epochs * 2 if mode == "both" else epochs,
                world_size=world_size
            )
            
            # Save summary
            summary = {}
            for name, result in results.items():
                summary[name] = {
                    "final_loss": result["loss"],
                    "model_spec": result["model_spec"],
                    "train_spec": result["train_spec"]
                }
            
            with open("training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nTrained {len(results)} configurations")
        else:
            print("No studies found. Run search mode first.")


def main():
    parser = argparse.ArgumentParser(
        description="GNN Training CLI - Multi-GPU training with Optuna optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (2 epochs, 1 seed, 1 dataset)
  python train_cli.py test
  
  # Simple training without Optuna (all combinations)
  python train_cli.py simple --epochs 20 --seeds 5
  
  # Optuna hyperparameter search
  python train_cli.py search --trials 50 --epochs 20
  
  # Train with best parameters (after search)
  python train_cli.py train --epochs 50 --seeds 5
  
  # Complete pipeline (search + train)
  python train_cli.py both --trials 50 --epochs 20 --seeds 5
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["test", "simple", "search", "train", "both"],
        help="Training mode"
    )
    
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=50,
        help="Number of Optuna trials (for search/both modes)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=5,
        help="Number of random seeds for final training"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)"
    )
    
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        world_size = 0
    else:
        world_size = args.gpus if args.gpus else detect_gpus()
    
    # Print system info
    print_header("GNN TRAINING PIPELINE")
    print(f"Mode: {args.mode}")
    print(f"GPUs available: {world_size}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Setup directories
    setup_directories()
    
    # Record start time
    start_time = time.time()
    
    # Run appropriate mode
    try:
        if args.mode == "test":
            run_test_mode()
        elif args.mode == "simple":
            run_simple_mode(args.epochs, args.seeds, args.batch_size)
        else:  # search, train, or both
            run_optuna_mode(args.mode, args.trials, args.epochs, args.seeds, world_size)
        
        # Save and organize results
        results_dir = save_results(args.mode)
        
        # Print summary
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print_header("COMPLETED")
        print(f"Runtime: {hours}h {minutes}m {seconds}s")
        print(f"Results saved to: {results_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Enable TF32 for better performance on A100/H100 GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    main()