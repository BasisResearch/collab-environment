#!/usr/bin/env python
"""
Multi-GPU GNN training with Optuna hyperparameter optimization
"""

import os
import sys
import argparse
import pickle
from itertools import product
from typing import Dict, Any, Optional, Tuple, List
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib

# Add project paths
import pathlib
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from collab_env.gnn.gnn import train_rules_gnn, save_model, load_model
from collab_env.gnn.gnn_definition import GNN, Lazy
from collab_env.gnn.utility import dataset2testloader
from collab_env.data.file_utils import expand_path, get_project_root


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def load_dataset(data_name: str, batch_size: int = 4) -> Dict[str, Any]:
    """Load dataset and create data loaders"""
    file_name = f'{data_name}.pt'
    config_name = f'{data_name}_config.pt'
    
    dataset = torch.load(
        expand_path(f"simulated_data/{file_name}", get_project_root()),
        weights_only=False
    )
    species_configs = torch.load(
        expand_path(f"simulated_data/{config_name}", get_project_root()),
        weights_only=False
    )
    
    test_loader, train_loader = dataset2testloader(
        dataset, batch_size=batch_size, return_train=True
    )
    
    return {
        "data_name": data_name,
        "dataset": dataset,
        "species_configs": species_configs,
        "test_loader": test_loader,
        "train_loader": train_loader,
        "species_dim": len(species_configs.keys())
    }


def create_model_spec(
    data_name: str,
    model_name: str,
    noise_level: float,
    head: int,
    visual_range: float
) -> Dict[str, Any]:
    """Create model specification"""
    
    # Determine input dimension based on dataset
    in_node_dim = 20 if "food" in data_name else 19
    
    if "lazy" in model_name:
        return {
            "model_name": "lazy",
            "prediction_integration": "Euler",
            "input_differentiation": "finite",
            "in_node_dim": 3,
            "start_frame": 3,
            "heads": 1
        }
    
    return {
        "model_name": model_name,
        "node_feature_function": "vel_plus_pos_plus",
        "node_prediction": "acc",
        "prediction_integration": "Euler",
        "input_differentiation": "finite",
        "in_node_dim": in_node_dim,
        "start_frame": 3,
        "heads": head
    }


def create_train_spec(
    model_name: str,
    noise_level: float,
    visual_range: float,
    epochs: int,
    lr: float = 1e-4
) -> Dict[str, Any]:
    """Create training specification"""
    
    if "lazy" in model_name:
        return {
            "lr": None,
            "visual_range": visual_range,
            "sigma": noise_level,
            "epochs": 1,
            "training": False
        }
    
    return {
        "lr": lr,
        "visual_range": visual_range,
        "epochs": epochs,
        "sigma": noise_level,
        "training": True
    }


def initialize_model(
    model_name: str,
    model_spec: Dict[str, Any]
) -> nn.Module:
    """Initialize GNN model"""
    
    if "lazy" in model_name:
        return Lazy(**model_spec)
    
    return GNN(**model_spec)


def train_single_configuration(
    rank: int,
    world_size: int,
    data: Dict[str, Any],
    model_spec: Dict[str, Any],
    train_spec: Dict[str, Any],
    seed: int,
    save_name: str,
    distributed: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """Train a single model configuration"""
    
    if distributed and world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model
    model = initialize_model(model_spec["model_name"], model_spec)
    
    # Move model to device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Wrap model for distributed training
    if distributed and world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Create distributed data loader if needed
    train_loader = data['train_loader']
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            data['dataset'], 
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
        train_loader = DataLoader(
            data['dataset'],
            batch_size=train_spec.get('batch_size', 4),
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
    
    # Train model
    train_losses, trained_model, debug_result = train_rules_gnn(
        model,
        train_loader,
        species_dim=data['species_dim'],
        **train_spec
    )
    
    # Calculate final loss
    final_loss = np.mean(train_losses[-1]) if len(train_losses) > 0 else float('inf')
    
    # Save model (only on rank 0 for distributed training)
    if not distributed or rank == 0:
        save_model(trained_model, model_spec, train_spec, save_name)
    
    if distributed and world_size > 1:
        cleanup_distributed()
    
    return final_loss, {
        "train_losses": train_losses,
        "debug_result": debug_result
    }


def objective(
    trial: Trial,
    data_name: str,
    model_name: str,
    epochs: int,
    world_size: int
) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # Sample hyperparameters from your specific values
    noise_level = trial.suggest_categorical("noise_level", [0, 0.005])
    heads = trial.suggest_categorical("heads", [1, 2, 3])
    visual_range = trial.suggest_categorical("visual_range", [0.1, 0.5])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # Keep LR search for optimization
    
    # Load data
    data = load_dataset(data_name, batch_size=4)
    
    # Create specifications
    model_spec = create_model_spec(
        data_name, model_name, noise_level, heads, visual_range
    )
    train_spec = create_train_spec(
        model_name, noise_level, visual_range, epochs, lr
    )
    
    # Train model with multiple seeds
    losses = []
    for seed in range(3):  # Use 3 seeds for robustness
        save_name = f"{data_name}_{model_name}_trial{trial.number}_seed{seed}"
        
        if world_size > 1:
            # Use multiprocessing for distributed training
            mp.spawn(
                train_single_configuration,
                args=(world_size, data, model_spec, train_spec, seed, save_name, True),
                nprocs=world_size,
                join=True
            )
            # Load the saved results
            loss = float('inf')  # Would need to implement result loading
        else:
            loss, _ = train_single_configuration(
                0, 1, data, model_spec, train_spec, seed, save_name, False
            )
        
        losses.append(loss)
        
        # Report intermediate value for pruning
        trial.report(loss, seed)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return mean loss across seeds
    return np.mean(losses)


def run_hyperparameter_search(
    data_names: List[str],
    model_names: List[str],
    n_trials: int = 100,
    epochs: int = 20,
    world_size: int = 1,
    study_name: Optional[str] = None
) -> Dict[str, optuna.Study]:
    """Run Optuna hyperparameter optimization"""
    
    studies = {}
    
    for data_name in data_names:
        for model_name in model_names:
            if "lazy" in model_name:
                continue  # Skip lazy model optimization
            
            # Create study name
            current_study_name = study_name or f"{data_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create Optuna study
            study = optuna.create_study(
                study_name=current_study_name,
                direction="minimize",
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
                storage=f"sqlite:///optuna_{current_study_name}.db",
                load_if_exists=True
            )
            
            # Run optimization
            study.optimize(
                lambda trial: objective(trial, data_name, model_name, epochs, world_size),
                n_trials=n_trials,
                n_jobs=1,  # Use 1 job since we handle parallelism internally
                show_progress_bar=True
            )
            
            studies[f"{data_name}_{model_name}"] = study
            
            # Print best parameters
            print(f"\nBest parameters for {data_name} - {model_name}:")
            print(f"  Best value: {study.best_value}")
            print(f"  Best params: {study.best_params}")
    
    return studies


def train_with_best_params(
    data_names: List[str],
    model_names: List[str],
    studies: Dict[str, optuna.Study],
    seed_num: int = 5,
    epochs: int = 20,
    world_size: int = 1
):
    """Train models with best hyperparameters from Optuna study"""
    
    results = {}
    
    for data_name in data_names:
        # Load data once per dataset
        data = load_dataset(data_name, batch_size=4)
        
        for model_name in model_names:
            study_key = f"{data_name}_{model_name}"
            
            if "lazy" in model_name:
                # Use default parameters for lazy model
                best_params = {
                    "noise_level": 0.0,
                    "heads": 1,
                    "visual_range": 0.5,
                    "lr": None
                }
            else:
                # Get best parameters from study
                if study_key not in studies:
                    print(f"No study found for {study_key}, skipping...")
                    continue
                
                best_params = studies[study_key].best_params
            
            # Create specifications with best parameters
            model_spec = create_model_spec(
                data_name,
                model_name,
                best_params.get("noise_level", 0.0),
                best_params.get("heads", 1),
                best_params.get("visual_range", 0.5)
            )
            
            train_spec = create_train_spec(
                model_name,
                best_params.get("noise_level", 0.0),
                best_params.get("visual_range", 0.5),
                epochs,
                best_params.get("lr", 1e-4)
            )
            
            # Train with multiple seeds
            for seed in range(seed_num):
                save_name = f"{data_name}_{model_name}_best_seed{seed}"
                
                print(f"\nTraining {save_name}...")
                
                if world_size > 1:
                    mp.spawn(
                        train_single_configuration,
                        args=(world_size, data, model_spec, train_spec, seed, save_name, True),
                        nprocs=world_size,
                        join=True
                    )
                else:
                    loss, result = train_single_configuration(
                        0, 1, data, model_spec, train_spec, seed, save_name, False
                    )
                    
                    results[save_name] = {
                        "loss": loss,
                        "model_spec": model_spec,
                        "train_spec": train_spec,
                        "result": result
                    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU GNN training with Optuna")
    parser.add_argument("--mode", choices=["search", "train", "both"], default="both",
                        help="Mode: search for hyperparameters, train with best params, or both")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--seed_num", type=int, default=5,
                        help="Number of seeds for final training")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(),
                        help="Number of GPUs to use")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Name for Optuna study")
    parser.add_argument("--resume_study", action="store_true",
                        help="Resume existing Optuna study")
    
    args = parser.parse_args()
    
    # Data and model configurations
    data_names = [
        'boid_single_species_basic',
        'boid_single_species_independent',
        'boid_food_basic_alignment',
        'boid_food_basic_independent',
        'boid_food_strong'
    ]
    
    model_names = ["vpluspplus_a", "lazy"]
    
    print(f"Using {args.world_size} GPUs for training")
    
    studies = {}
    
    # Hyperparameter search
    if args.mode in ["search", "both"]:
        print("Starting hyperparameter search with Optuna...")
        studies = run_hyperparameter_search(
            data_names,
            model_names,
            n_trials=args.n_trials,
            epochs=args.epochs,
            world_size=args.world_size,
            study_name=args.study_name
        )
        
        # Save study results
        study_results = {}
        for key, study in studies.items():
            study_results[key] = {
                "best_value": study.best_value,
                "best_params": study.best_params
            }
        
        with open("optuna_results.json", "w") as f:
            json.dump(study_results, f, indent=2)
        
        print("\nOptuna results saved to optuna_results.json")
    
    # Train with best parameters
    if args.mode in ["train", "both"]:
        print("\nTraining models with best hyperparameters...")
        
        # Load studies if training only
        if args.mode == "train" and os.path.exists("optuna_results.json"):
            with open("optuna_results.json", "r") as f:
                study_results = json.load(f)
            
            # Create mock studies with best params
            for key, result in study_results.items():
                study = type('MockStudy', (), {})()
                study.best_params = result["best_params"]
                study.best_value = result["best_value"]
                studies[key] = study
        
        results = train_with_best_params(
            data_names,
            model_names,
            studies,
            seed_num=args.seed_num,
            epochs=args.epochs,
            world_size=args.world_size
        )
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Trained {len(results)} model configurations")
        
        # Save final results summary
        summary = {}
        for name, result in results.items():
            summary[name] = {
                "final_loss": result["loss"],
                "model_spec": result["model_spec"],
                "train_spec": result["train_spec"]
            }
        
        with open("training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("Training summary saved to training_summary.json")


if __name__ == "__main__":
    # Set environment variables for better GPU utilization
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Enable TF32 for better performance on A100/H100 GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    main()