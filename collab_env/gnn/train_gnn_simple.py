#!/usr/bin/env python
"""
Simple script to train all GNN configurations without Optuna
This runs the original training loop with all parameter combinations
"""

import os
import sys
import torch
import numpy as np
from itertools import product
from typing import Dict, Any
import json
from datetime import datetime
import argparse
from tqdm import tqdm

# Add project paths
import pathlib
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from collab_env.gnn.gnn import train_rules_gnn, save_model
from collab_env.gnn.gnn_definition import GNN, Lazy
from collab_env.gnn.utility import dataset2testloader
from collab_env.data.file_utils import expand_path, get_project_root


def load_all_datasets(data_names, batch_size=4):
    """Load all datasets"""
    data = {}
    
    for data_name in data_names:
        print(f"Loading dataset: {data_name}")
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
        
        data[data_name] = {
            "data_name": data_name,
            "file_name": file_name,
            "config_name": config_name,
            "dataset": dataset,
            "species_configs": species_configs,
            "test_loader": test_loader,
            "train_loader": train_loader
        }
    
    return data


def make_specs(data_name, model_name, batch_size, noise_level, head, visual_range, epoch):
    """Create model and training specifications"""
    
    # Determine input dimension
    in_node_dim = 20 if "food" in data_name else 19
    
    if "lazy" in model_name:
        model_spec = {
            "model_name": "lazy",
            "prediction_integration": "Euler",
            "input_differentiation": "finite",
            "in_node_dim": 3,
            "start_frame": 3,
            "heads": 1
        }
        
        train_spec = {
            "lr": None,
            "visual_range": visual_range,
            "sigma": noise_level,
            "epochs": 1,
            "training": False
        }
    else:
        model_spec = {
            "model_name": model_name,
            "node_feature_function": "vel_plus_pos_plus",
            "node_prediction": "acc",
            "prediction_integration": "Euler",
            "input_differentiation": "finite",
            "in_node_dim": in_node_dim,
            "start_frame": 3,
            "heads": head
        }
        
        train_spec = {
            "lr": 1e-4,
            "visual_range": visual_range,
            "epochs": epoch,
            "sigma": noise_level,
            "training": True
        }
    
    return model_spec, train_spec


def initialize_models(model_name, model_spec):
    """Initialize GNN model"""
    if "lazy" in model_name:
        return Lazy(**model_spec)
    return GNN(**model_spec)


def train(data, model_spec, train_spec, seed, save_name_postfix, device=None):
    """Train a single model configuration"""
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model
    gnn_model = initialize_models(model_spec["model_name"], model_spec)
    
    # Move to GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = gnn_model.to(device)
    
    # Get data
    data_name = data['data_name']
    train_loader = data['train_loader']
    species_dim = len(data['species_configs'].keys())
    
    # Train model
    train_losses, trained_model, debug_result = train_rules_gnn(
        gnn_model,
        train_loader,
        species_dim=species_dim,
        device=device,
        **train_spec
    )
    
    # Save model
    model_name = model_spec["model_name"]
    file_name = f"{data_name}_{model_name}_{save_name_postfix}_seed{seed}"
    save_model(trained_model, model_spec, train_spec, file_name)
    
    # Return results
    return {
        "train_losses": train_losses,
        "model": trained_model,
        "debug_result": debug_result,
        "file_name": file_name
    }


def main():
    parser = argparse.ArgumentParser(description="Train all GNN configurations")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--seed_num", type=int, default=5, help="Number of seeds")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda:0, cuda:1, etc)")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Load all datasets once
    print("Loading datasets...")
    all_data = load_all_datasets(data_names, batch_size=args.batch_size)
    
    # Generate all combinations
    all_combinations = list(product(data_names, model_names, noise_levels, heads, visual_ranges))
    total_configs = len(all_combinations) * args.seed_num
    
    print(f"\nTotal configurations to train: {len(all_combinations)}")
    print(f"Seeds per configuration: {args.seed_num}")
    print(f"Total training runs: {total_configs}")
    print(f"Device: {args.device or 'auto'}\n")
    
    # Track results
    results = []
    
    # Progress bar for all configurations
    with tqdm(total=total_configs, desc="Training progress") as pbar:
        for (data_name, model_name, noise, head, visual_range) in all_combinations:
            # Create specifications
            model_spec, train_spec = make_specs(
                data_name, model_name,
                args.batch_size, noise, head, visual_range, args.epochs
            )
            
            save_name_postfix = f"noise{noise}_head{head}_vr{visual_range}"
            
            # Train with multiple seeds
            for seed in range(args.seed_num):
                config_desc = f"{data_name}_{model_name}_{save_name_postfix}_seed{seed}"
                pbar.set_description(f"Training: {config_desc[:50]}...")
                
                try:
                    result = train(
                        all_data[data_name],
                        model_spec,
                        train_spec,
                        seed,
                        save_name_postfix,
                        device=args.device
                    )
                    
                    # Calculate final loss
                    final_loss = np.mean(result["train_losses"][-1]) if len(result["train_losses"]) > 0 else float('inf')
                    
                    # Store result
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
                    print(f"\nError training {config_desc}: {e}")
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
                
                pbar.update(1)
    
    # Save results summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"training_results_{timestamp}.json"
    
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Total runs: {len(results)}")
    print(f"Results saved to: {summary_file}")
    
    # Print best configurations for each dataset
    print(f"\n{'='*50}")
    print("Best configurations per dataset:")
    print(f"{'='*50}")
    
    for data_name in data_names:
        data_results = [r for r in results if r["data_name"] == data_name and "error" not in r]
        if data_results:
            best = min(data_results, key=lambda x: x["final_loss"])
            print(f"\n{data_name}:")
            print(f"  Model: {best['model_name']}")
            print(f"  Noise: {best['noise']}")
            print(f"  Heads: {best['heads']}")
            print(f"  Visual Range: {best['visual_range']}")
            print(f"  Best Loss: {best['final_loss']:.6f}")


if __name__ == "__main__":
    # Set environment variables for better GPU utilization
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Enable TF32 for better performance on A100/H100 GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    main()