#!/usr/bin/env python
"""
Maximize GPU utilization by running multiple training jobs per GPU
Simple approach: launch N jobs per GPU concurrently
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import concurrent.futures
import json
from datetime import datetime
import argparse
from loguru import logger

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from collab_env.gnn.gnn import train_rules_gnn, save_model
from collab_env.gnn.gnn_definition import GNN, Lazy
from collab_env.gnn.utility import dataset2testloader
from collab_env.data.file_utils import expand_path, get_project_root


def worker_wrapper(params):
    """Wrapper to set environment before calling actual training function"""
    import os
    data_name, model_name, noise, heads, visual_range, seed, gpu_count, worker_id = params
    
    # Set CUDA_VISIBLE_DEVICES in this process before CUDA is initialized
    if gpu_count > 0:
        gpu_id = worker_id % gpu_count
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.debug(f"Worker {worker_id} assigned to GPU {gpu_id}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    else:
        logger.debug(f"Worker {worker_id} assigned to CPU")
        
    # Now call the actual training function
    return train_single_config(params)


def train_single_config(params):
    """Train a single configuration - runs on any available GPU"""
    data_name, model_name, noise, heads, visual_range, seed, gpu_count, worker_id = params
    
    # Determine device and worker label
    if gpu_count > 0:
        gpu_id = worker_id % gpu_count
        device = torch.device(f'cuda:{gpu_id}')
        worker_label = f"GPU{gpu_id}/W{worker_id:02d}"
    else:
        device = torch.device('cpu')
        worker_label = f"CPU/W{worker_id:02d}"
    
    
    # Configure logger for this worker with thread-safe serialization
    logger.remove()  # Remove default handler
    
    def format_with_worker(record):
        if "worker" in record["extra"]:
            return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[worker]}</cyan> | <level>{message}</level>\n"
        else:
            return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"
    
    # Only show INFO and above for worker processes to reduce noise
    # Use enqueue=True for thread safety without JSON serialization
    logger.add(sys.stderr, format=format_with_worker, level="DEBUG", enqueue=True)
    
    # Bind worker label to logger
    worker_logger = logger.bind(worker=worker_label)
    
    worker_logger.debug(f"Device {device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Debug: Print actual GPU being used
    if torch.cuda.is_available():
        torch.set_default_device(device)
        current_device = torch.cuda.current_device()
        worker_logger.debug(f"GPU count {gpu_count},  current_device={current_device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    else:
        worker_logger.debug(f"CUDA NOT AVAILABLE! gpu count {gpu_count}, using CPU, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        # Load dataset
        file_name = f'{data_name}.pt'
        config_name = f'{data_name}_config.pt'
        
        worker_logger.debug(f"Loading dataset {file_name}")
        dataset = torch.load(
            expand_path(f"simulated_data/{file_name}", get_project_root()),
            weights_only=False
        )
        worker_logger.debug(f"Loading species configs {config_name}")
        species_configs = torch.load(
            expand_path(f"simulated_data/{config_name}", get_project_root()),
            weights_only=False
        )
        
        worker_logger.debug(f"Creating test and train loaders")
        test_loader, train_loader = dataset2testloader(
            dataset, batch_size=5, return_train=True
        )
        
        # Create model
        in_node_dim = 20 if "food" in data_name else 19
        
        worker_logger.debug(f"Creating model {model_name}")
        
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
        
        worker_logger.debug(f"Setting seed {seed}")
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Set the device context BEFORE any CUDA operations
        if device.type == 'cuda':
            worker_logger.debug(f"Setting CUDA device {device} seed {seed}")
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(seed)
            # Ensure deterministic behavior for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        worker_logger.debug(f"Moving model to device {device}")
        # Move model to correct device
        model = model.to(device)
        
        worker_logger.debug(f"Training model")
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
        
        worker_logger.debug(f"Saving model")
        # Save model
        model_spec = {"model_name": model_name, "heads": heads}
        train_spec = {"visual_range": visual_range, "sigma": noise, "epochs": epochs}
        save_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}"
        save_model(trained_model, model_spec, train_spec, save_name)
        
        # Return result
        final_loss = np.mean(train_losses[-1]) if len(train_losses) > 0 else float('inf')
        
        worker_logger.success(f"Completed: {save_name} | Loss: {final_loss:.6f}")
        
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
        worker_logger.error(f"Failed: {data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed} | Error: {e}")
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
    
    # Configure main logger with thread-safe output
    logger.remove()
    logger.add(sys.stderr, 
              format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n",
              level="DEBUG",
              enqueue=True)  # Thread-safe logging without JSON
    
    # Check GPUs
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    logger.info("="*60)
    logger.info("GPU UTILIZATION MAXIMIZER")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"GPUs available: {gpu_count}")
    
    if gpu_count > 0:
        for i in range(gpu_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        max_workers = gpu_count * args.workers_per_gpu
        logger.info(f"Workers per GPU: {args.workers_per_gpu}")
        logger.info(f"Total concurrent jobs: {max_workers}")
    else:
        max_workers = 1
        logger.warning("Running on CPU (single worker)")
    
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
    
    logger.info(f"Total configurations: {len(all_params)}")
    logger.info("="*60)
    
    # Show GPU assignment plan
    logger.info("GPU Assignment Plan:")
    for i in range(min(8, len(all_params))):  # Show first 8 assignments
        _, model, noise, head, vr, seed, _, wid = all_params[i]
        assigned_gpu = wid % gpu_count if gpu_count > 0 else -1
        logger.debug(f"  Worker {wid:02d} -> GPU {assigned_gpu}: {model}_n{noise}_h{head}_vr{vr}_s{seed}")
    if len(all_params) > 8:
        logger.debug(f"  ... and {len(all_params) - 8} more configurations")
    logger.info("="*60)
    
    # Run with concurrent futures - distributes across all GPUs
    results = []
    start_time = datetime.now()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs using the wrapper that sets environment variables
        future_to_params = {
            executor.submit(worker_wrapper, params): params 
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
                logger.info(f"Progress: {completed}/{len(all_params)} | Elapsed: {elapsed:.1f} min | Distribution: {gpu_info}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{args.dataset}_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    successful = len([r for r in results if r["status"] == "success"])
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Time: {elapsed:.2f} minutes")
    logger.info(f"Successful: {successful}/{len(results)}")
    logger.info(f"Results: {results_file}")
    
    # Find best configuration
    valid_results = [r for r in results if r["status"] == "success"]
    if valid_results:
        best = min(valid_results, key=lambda x: x["final_loss"])
        logger.success(f"Best configuration:")
        logger.success(f"  Model: {best['model_name']}")
        logger.success(f"  Noise: {best['noise']}")
        logger.success(f"  Heads: {best['heads']}")
        logger.success(f"  Visual Range: {best['visual_range']}")
        logger.success(f"  Loss: {best['final_loss']:.6f}")


if __name__ == "__main__":
    main()