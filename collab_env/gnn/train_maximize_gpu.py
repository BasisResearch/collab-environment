#!/usr/bin/env python
"""
Maximize GPU utilization by running multiple training jobs per GPU
Simple approach: launch N jobs per GPU concurrently
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from itertools import product
import concurrent.futures
from functools import partial
import json
from datetime import datetime
import argparse

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from collab_env.gnn.gnn import train_rules_gnn, save_model
from collab_env.gnn.gnn_definition import GNN, Lazy
from collab_env.gnn.utility import dataset2testloader
from collab_env.data.file_utils import expand_path, get_project_root


def train_single_config(params):
    """Train a single configuration - runs on any available GPU"""
    data_name, model_name, noise, heads, visual_range, seed, gpu_count, worker_id = params
    
    # Distribute workers across GPUs
    if gpu_count > 0:
        # Assign worker to GPU in round-robin fashion
        gpu_id = worker_id % gpu_count
        device = torch.device(f'cuda:{gpu_id}')
        prefix = f"[GPU{gpu_id}/W{worker_id:02d}]"
    else:
        device = torch.device('cpu')
        prefix = f"[CPU/W{worker_id:02d}]"
    
    try:
        # Load dataset
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
            dataset, batch_size=5, return_train=True
        )
        
        # Create model
        in_node_dim = 20 if "food" in data_name else 19
        
        if "lazy" in model_name:
            model = Lazy(
                model_name="lazy",
                prediction_integration="Euler",
                input_differentiation="finite",
                in_node_dim=3,
                start_frame=3,
                heads=1
            )
            epochs = 1
            lr = None
            training = False
        else:
            model = GNN(
                model_name="vpluspplus_a",
                node_feature_function="vel_plus_pos_plus",
                node_prediction="acc",
                prediction_integration="Euler",
                input_differentiation="finite",
                in_node_dim=in_node_dim,
                start_frame=3,
                heads=heads
            )
            epochs = 1  # Fixed for now, make it a parameter if needed
            lr = 1e-4
            training = True
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train
        train_losses, trained_model, debug_result = train_rules_gnn(
            model,
            train_loader,
            species_dim=len(species_configs.keys()),
            visual_range=visual_range,
            epochs=epochs,
            lr=lr,
            training=training,
            sigma=noise,
            device=device
        )
        
        # Save model
        model_spec = {"model_name": model_name, "heads": heads}
        train_spec = {"visual_range": visual_range, "sigma": noise, "epochs": epochs}
        save_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}"
        save_model(trained_model, model_spec, train_spec, save_name)
        
        # Return result
        final_loss = np.mean(train_losses[-1]) if len(train_losses) > 0 else float('inf')
        
        print(f"{prefix} ✓ Completed: {save_name} | Loss: {final_loss:.6f}")
        
        return {
            "data_name": data_name,
            "model_name": model_name,
            "noise": noise,
            "heads": heads,
            "visual_range": visual_range,
            "seed": seed,
            "final_loss": float(final_loss),
            "status": "success",
            "gpu_id": gpu_id if gpu_count > 0 else -1,
            "worker_id": worker_id
        }
        
    except Exception as e:
        print(f"{prefix} ✗ Failed: {data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed} | Error: {e}")
        return {
            "data_name": data_name,
            "model_name": model_name,
            "noise": noise,
            "heads": heads,
            "visual_range": visual_range,
            "seed": seed,
            "final_loss": float('inf'),
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id if gpu_count > 0 else -1,
            "worker_id": worker_id
        }


def main():
    # Set multiprocessing start method to 'spawn' for CUDA
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description="Maximize GPU utilization with concurrent training")
    parser.add_argument("--dataset", type=str, default="boid_single_species_basic",
                       help="Dataset to use (default: boid_single_species_basic)")
    parser.add_argument("--workers-per-gpu", type=int, default=3,
                       help="Number of concurrent jobs per GPU (default: 3)")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of seeds (default: 5)")
    parser.add_argument("--test", action="store_true",
                       help="Quick test with minimal parameters")
    
    args = parser.parse_args()
    
    # Check GPUs
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print("="*60)
    print("GPU UTILIZATION MAXIMIZER")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"GPUs available: {gpu_count}")
    
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        max_workers = gpu_count * args.workers_per_gpu
        print(f"Workers per GPU: {args.workers_per_gpu}")
        print(f"Total concurrent jobs: {max_workers}")
    else:
        max_workers = 1
        print("Running on CPU (single worker)")
    
    # Define hyperparameter grid
    if args.test:
        model_names = ["vpluspplus_a"]
        noise_levels = [0]
        heads = [1]
        visual_ranges = [0.1]
        seeds = range(1)
    else:
        model_names = ["vpluspplus_a", "lazy"]
        noise_levels = [0, 0.005]
        heads = [1, 2, 3]
        visual_ranges = [0.1, 0.5]
        seeds = range(args.seeds)
    
    # Generate all combinations with worker IDs
    all_params = []
    worker_id = 0
    for model in model_names:
        for noise in noise_levels:
            for head in heads:
                for vr in visual_ranges:
                    for seed in seeds:
                        all_params.append(
                            (args.dataset, model, noise, head, vr, seed, gpu_count, worker_id)
                        )
                        worker_id += 1
    
    print(f"\nTotal configurations: {len(all_params)}")
    print("="*60)
    
    # Show GPU assignment plan
    print("\nGPU Assignment Plan:")
    for i in range(min(8, len(all_params))):  # Show first 8 assignments
        _, model, noise, head, vr, seed, _, wid = all_params[i]
        assigned_gpu = wid % gpu_count if gpu_count > 0 else -1
        print(f"  Worker {wid:02d} -> GPU {assigned_gpu}: {model}_n{noise}_h{head}_vr{vr}_s{seed}")
    if len(all_params) > 8:
        print(f"  ... and {len(all_params) - 8} more configurations")
    print("="*60)
    
    # Run with concurrent futures - distributes across all GPUs
    results = []
    start_time = datetime.now()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {
            executor.submit(train_single_config, params): params 
            for params in all_params
        }
        
        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_params):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Progress update with GPU distribution
            if completed % 10 == 0 or completed == len(all_params):
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                gpu_dist = {}
                for r in results:
                    if 'gpu_id' in r and r['gpu_id'] >= 0:
                        gpu_dist[r['gpu_id']] = gpu_dist.get(r['gpu_id'], 0) + 1
                gpu_info = " | ".join([f"GPU{k}:{v}" for k, v in sorted(gpu_dist.items())])
                print(f"Progress: {completed}/{len(all_params)} | Elapsed: {elapsed:.1f} min | Distribution: {gpu_info}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{args.dataset}_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    successful = len([r for r in results if r["status"] == "success"])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Time: {elapsed:.2f} minutes")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Results: {results_file}")
    
    # Find best configuration
    valid_results = [r for r in results if r["status"] == "success"]
    if valid_results:
        best = min(valid_results, key=lambda x: x["final_loss"])
        print(f"\nBest configuration:")
        print(f"  Model: {best['model_name']}")
        print(f"  Noise: {best['noise']}")
        print(f"  Heads: {best['heads']}")
        print(f"  Visual Range: {best['visual_range']}")
        print(f"  Loss: {best['final_loss']:.6f}")


if __name__ == "__main__":
    main()