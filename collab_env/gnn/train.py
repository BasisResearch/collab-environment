#!/usr/bin/env python
"""
Maximize GPU utilization by running multiple training jobs per GPU
Simple approach: launch N jobs per GPU concurrently
"""

import sys
import os
import torch
import numpy as np
import concurrent.futures
import json
from datetime import datetime
import argparse

from loguru import logger
import pickle

# Add project paths
# current_dir = Path(__file__).parent
# project_root = current_dir.parent.parent
# sys.path.insert(0, str(project_root))

from collab_env.gnn.gnn import train_rules_gnn, save_model
from collab_env.gnn.gnn_definition import GNN, Lazy
from collab_env.gnn.utility import dataset2testloader
from collab_env.data.file_utils import expand_path, get_project_root


def worker_wrapper(params):
    """Wrapper to set environment before calling actual training function"""
    import os
    import torch

    try:
        (
            data_name,
            model_name,
            noise,
            heads,
            visual_range,
            seed,
            gpu_count,
            worker_id,
            batch_size,
            epochs,
            compile_model,
            memory_fraction,
            no_validation,
            early_stopping_patience,
            min_delta,
            train_size,
            rollout,
            total_rollout,
            ablate,
            self_loops,
            relative_positions,
        ) = params

        # Set CUDA_VISIBLE_DEVICES in this process before CUDA is initialized
        if gpu_count > 0:
            gpu_id = worker_id % gpu_count
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.debug(
                f"Worker {worker_id} assigned to GPU {gpu_id}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        else:
            logger.debug(f"Worker {worker_id} assigned to CPU")

        # Call the actual training function
        result = train_single_config(params)

        return result

    except Exception as e:
        logger.error(f"Worker {worker_id} failed with error: {e}")
        return {
            "data_name": params[0] if len(params) > 0 else "unknown",
            "model_name": params[1] if len(params) > 1 else "unknown",
            "final_loss": float("inf"),
            "status": "failed",
            "error": str(e),
            "worker_id": worker_id if "worker_id" in locals() else -1,
        }
    finally:
        # Clean up CUDA context to prevent hanging
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Force garbage collection
        import gc

        gc.collect()


def train_single_config(params):
    """Train a single configuration - runs on any available GPU"""
    (
        data_name,
        model_name,
        noise,
        heads,
        visual_range,
        seed,
        gpu_count,
        worker_id,
        batch_size,
        epochs,
        compile_model,
        memory_fraction,
        no_validation,
        early_stopping_patience,
        min_delta,
        train_size,
        rollout,
        total_rollout,
        ablate,
        self_loops,
        use_relative_positions,
    ) = params

    # Determine device and worker label
    if gpu_count > 0:
        gpu_id = worker_id % gpu_count
        device = torch.device(f"cuda:{gpu_id}")
        worker_label = f"GPU{gpu_id}/W{worker_id:02d}"
    else:
        device = torch.device("cpu")
        worker_label = f"CPU/W{worker_id:02d}"

    # Configure worker logger - loguru handles multiprocessing with enqueue=True
    # Import a fresh logger instance for this worker process
    from loguru import logger as worker_base_logger

    # Remove existing handlers and add our own for this worker process
    worker_base_logger.remove()

    def format_with_worker(record):
        if "worker" in record["extra"]:
            return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[worker]}</cyan> | <level>{message}</level>\n"
        else:
            return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"

    # Add stderr handler with worker-specific formatting
    worker_base_logger.add(
        sys.stderr, format=format_with_worker, level="DEBUG", enqueue=True
    )

    # Create worker logger with binding
    worker_logger = worker_base_logger.bind(worker=worker_label)

    # Add file logging for this specific worker using project structure
    worker_log_file = expand_path(
        f"logs/worker_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker_label.replace('/', '_')}.log",
        get_project_root(),
    )
    # Ensure logs directory exists
    worker_log_file.parent.mkdir(exist_ok=True)

    worker_logger.add(
        str(worker_log_file), format=format_with_worker, level="DEBUG", enqueue=True
    )

    worker_logger.debug(
        f"Device {device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    # Debug: Print actual device being used
    if torch.cuda.is_available() and device.type == "cuda":
        # Only set default device for CUDA devices
        torch.set_default_device(device)
        current_device = torch.cuda.current_device()

        # Set GPU memory fraction for better multi-worker performance
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
        worker_logger.debug(f"Set GPU memory fraction to {memory_fraction}")

        # Enable cudnn autotuner for better performance
        torch.backends.cudnn.benchmark = True

        worker_logger.debug(
            f"GPU count {gpu_count}, current_device={current_device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    else:
        worker_logger.debug(
            f"Using CPU device, gpu_count={gpu_count}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

    worker_logger.debug(f"Setting seed {seed}")
    # Set seed for reproducibility
    torch_generator = torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        # Load dataset
        file_name = f"runpod/{data_name}.pt"
        config_name = f"runpod/{data_name}_config.pt"

        worker_logger.debug(f"Loading dataset {file_name}")
        dataset = torch.load(
            expand_path(f"simulated_data/{file_name}", get_project_root()),
            weights_only=False,
        )
        worker_logger.debug(
            f"Dataset loaded, length: {len(dataset)}, batch size: {batch_size}"
        )
        worker_logger.debug(f"Loading species configs {config_name}")
        species_configs = torch.load(
            expand_path(f"simulated_data/{config_name}", get_project_root()),
            weights_only=False,
        )

        worker_logger.debug("Creating test and train loaders")
        test_loader, train_loader = dataset2testloader(
            dataset,
            generator=torch_generator,
            batch_size=batch_size,
            return_train=True,
            device=device,
            train_size=train_size,
        )
        worker_logger.debug(
            f"Test loader length: {len(test_loader)}, train loader length: {len(train_loader)}"
        )
        worker_logger.debug(f"Rolling out or not: {rollout}")

        # Create model
        """
               TOC -- 101325 10:00AM
               Fixed this to deal with more than 2D physical space. Also not sure 
               it originally dealt with multiple species correctly. 
               """
        # in_node_dim = 20 if "food" in data_name else 19
        num_species = len(species_configs.keys())
        num_time_steps = 3  # this should be configurable
        numbers_in_feature = (
            3  # also needs to be configurable or computed from something else
        )
        position_dim = dataset.sequences[0][0].shape[2]
        in_node_dim = position_dim * num_time_steps * numbers_in_feature + num_species
        worker_logger.debug(f"Creating model {model_name}")

        if "lazy" in model_name:
            model_spec = {
                "model_name": "lazy",
                "prediction_integration": "Euler",
                "input_differentiation": "finite",
                "in_node_dim": 3,
                "start_frame": 3,
                "heads": 1,
                "self_loops": self_loops,
                "edge_dim": position_dim if use_relative_positions else 1,
                "output_dim": position_dim,
            }
            model = Lazy(**model_spec)
            lr = None
            training = False
        else:
            model_spec = {
                "model_name": "vpluspplus_a",
                "node_feature_function": "vel_plus_pos_plus",
                "node_prediction": "acc",
                "prediction_integration": "Euler",
                "input_differentiation": "finite",
                "in_node_dim": in_node_dim,
                "start_frame": 3,
                "heads": heads,
                "self_loops": self_loops,
                "edge_dim": position_dim if use_relative_positions else 1,
                "output_dim": position_dim,
                # "use_relative_positions": use_relative_positions,
            }
            model = GNN(**model_spec)
            lr = 1e-4
            training = True

        # need to overwrite for rolling out operation.
        if rollout > 0:
            training = False
            lr = None
            epochs = 1
            loader = test_loader
            no_validation = True
            file_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}{'_selfloops' if self_loops else ''}{'_rp' if use_relative_positions else ''}"
            folder = f"trained_models/runpod/{data_name}/trained_models/"
            model_path = expand_path(
                f"{folder}/{file_name}.pt",
                get_project_root(),
            )
            """
            TOC -- 111225 1023AM
            This needs to be cpu if running cpu only 
            """
            # model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
            model.load_state_dict(torch.load(model_path, map_location=device))
            collect_debug = True  # Disable debug to avoid CPU transfers
            if ablate == 1:
                print("ablating attention layer")
                model.ablate_attention()
            elif ablate == 2:
                print("ablating attention layer by permuting")
                model.permute_attention()
            elif ablate == 3:
                print("ablating attention layer by zeroing")
                model.uni_attention()

        else:
            loader = train_loader
            collect_debug = False

        # Set the device context BEFORE any CUDA operations
        if device.type == "cuda":
            worker_logger.debug(f"Setting CUDA device {device} seed {seed}")
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(seed)
            # Ensure deterministic behavior for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        worker_logger.debug(f"Moving model to device {device}")
        # Move model to correct device
        model = model.to(device)

        # Compile model for faster execution if requested
        if compile_model and device.type == "cuda":
            try:
                worker_logger.debug("Configuring torch for compilation optimizations")
                # Set float32 matmul precision for better performance
                torch.set_float32_matmul_precision("high")
                # Skip dynamic CUDA graphs to avoid overhead from many distinct shapes
                torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
                worker_logger.debug("Compiling model with torch.compile()")
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                worker_logger.warning(
                    f"Model compilation failed: {e}, continuing without compilation"
                )

        worker_logger.debug("Training model with validation and early stopping")
        # Train with validation and early stopping
        train_losses, trained_model, debug_result = train_rules_gnn(
            model,
            loader,
            species_dim=len(species_configs.keys()),
            visual_range=visual_range,
            epochs=epochs,
            lr=lr,
            training=training,
            sigma=noise,
            device=device,
            rollout=rollout,
            total_rollout=total_rollout,
            train_logger=worker_logger,
            collect_debug=collect_debug,  # Disable debug to avoid CPU transfers
            val_dataloader=test_loader
            if not no_validation
            else None,  # Use test_loader for validation unless disabled
            early_stopping_patience=early_stopping_patience,  # Early stopping patience from args
            min_delta=min_delta,  # Minimum improvement threshold from args
            use_relative_positions=use_relative_positions,
        )

        # Save model
        # model_spec = {"model_name": model_name, "heads": heads}
        train_spec = {"visual_range": visual_range, "sigma": noise, "epochs": epochs}
        if rollout > 0:
            if ablate == 0:
                save_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}_rollout{rollout}{'_selfloops' if self_loops else ''}{'_rp' if use_relative_positions else ''}"
            elif ablate == 1:
                save_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}_ablate_rollout{rollout}{'_selfloops' if self_loops else ''}{'_rp' if use_relative_positions else ''}"
            elif ablate == 2:
                save_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}_perm_rollout{rollout}{'_selfloops' if self_loops else ''}{'_rp' if use_relative_positions else ''}"
            elif ablate == 3:
                save_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}_zero_rollout{rollout}{'_selfloops' if self_loops else ''}{'_rp' if use_relative_positions else ''}"
        else:
            save_name = f"{data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}{'_selfloops' if self_loops else ''}{'_rp' if use_relative_positions else ''}"
        
        if rollout > 0:
            folder = f"runpod/{data_name}/rollouts/"
            rollout_path = expand_path(
                f"trained_models/{folder}/{save_name}.pkl",
                get_project_root(),
            )

            """
            TOC -- 111425 9:14AM 
            Create the parent directories if they don't exist. 
            """
            logger.debug(f"rollout path {rollout_path}")
            rollout_path.parent.mkdir(exist_ok=True, parents=True)

            # rollout_path = expand_path(
            #    f"trained_models/{save_name}.pkl",
            #    get_project_root()
            # )

            with open(rollout_path, "wb") as f:  # 'wb' for write binary
                pickle.dump(debug_result, f)
                
            save_name = f"{folder}/{save_name}"
        else: # save model and training spec
            folder = f"runpod/{data_name}/trained_models/"
            expand_path(f"trained_models/{folder}", get_project_root()).mkdir(exist_ok=True, parents=True)
            save_name = f"{folder}/{save_name}"

        worker_logger.debug(f"Saving model {save_name}...")
        model_output, model_spec_path, train_spec_path = save_model(
            trained_model, model_spec, train_spec, save_name
        )
        worker_logger.debug(f"Model saved to {model_output}.")
        worker_logger.debug(f"Model spec saved to {model_spec_path}.")
        worker_logger.debug(f"Train spec saved to {train_spec_path}.")

        # Return result
        final_loss = (
            np.mean(train_losses[-1]) if len(train_losses) > 0 else float("inf")
        )

        worker_logger.success(f"Completed: {save_name} | Loss: {final_loss:.6f}")

        # Clean up to prevent hanging processes
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return {
            "data_name": data_name,
            "model_output": str(model_output),
            "model_spec_path": str(model_spec_path),
            "train_spec_path": str(train_spec_path),
            "train_losses": str(train_losses),
            "model_name": model_name,
            "noise": noise,
            "heads": heads,
            "visual_range": visual_range,
            "self_loops": self_loops,
            "use_relative_positions": use_relative_positions,
            "seed": seed,
            "final_loss": float(final_loss),
            "status": "success",
            "gpu_id": gpu_id if gpu_count > 0 else -1,
            "worker_id": worker_id,
        }

    except Exception as e:
        worker_logger.error(
            f"Failed: {data_name}_{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed} | Error: {e}"
        )
        return {
            "data_name": data_name,
            "model_name": model_name,
            "noise": noise,
            "heads": heads,
            "visual_range": visual_range,
            "self_loops": self_loops,
            "use_relative_positions": use_relative_positions,
            "seed": seed,
            "final_loss": float("inf"),
            "status": "failed",
            "error": str(e),
            "gpu_id": gpu_id if gpu_count > 0 else -1,
            "worker_id": worker_id,
        }


def main():
    # Set multiprocessing start method - 'spawn' is safer for CUDA, but 'fork' can be faster for CPU
    import multiprocessing

    try:
        # Use 'spawn' for better CUDA compatibility, fallback to default for CPU-only
        preferred_method = "spawn" if torch.cuda.is_available() else None
        if preferred_method:
            multiprocessing.set_start_method(preferred_method, force=True)
    except RuntimeError:
        pass  # Already set or not supported on this platform

    parser = argparse.ArgumentParser(
        description="Maximize GPU utilization with concurrent training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="boid_single_species_basic",
        help="Dataset to use (default: boid_single_species_basic)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=4,
        help="Number of concurrent jobs per GPU (default: 4 - optimized for RTX 4090)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override total number of workers (ignores workers-per-gpu)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16 - optimized for RTX 4090)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training (default: 1)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Early stopping patience - stop if no improvement for N epochs (default: 2)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
        help="Minimum improvement threshold for early stopping (default: 1e-6)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation and early stopping",
    )
    parser.add_argument(
        "--test", action="store_true", help="Quick test with minimal parameters"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only training, ignore GPU even if available",
    )
    parser.add_argument(
        "--cpu-optimize",
        action="store_true",
        help="Apply CPU optimizations for CUDA-enabled PyTorch builds",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable PyTorch 2.0 compilation for faster execution",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.9,
        help="GPU memory fraction to allocate (default: 0.9)",
    )
    parser.add_argument(
        "--train-size", type=float, default=0.7, help="Train size (default: 0.7)"
    )
    parser.add_argument(
        "--rollout", type=int, default=-1, help="Do rollout starting at which frame"
    )
    parser.add_argument(
        "--total_rollout", type=int, default=100, help="Total number of rollout"
    )
    parser.add_argument("--ablate", type=int, default=0, help="ablate attention layer")
    parser.add_argument(
        "--self_loops", action="store_true", help="add self loops to both layers"
    )
    parser.add_argument(
        "--relative_positions",
        action="store_true",
        help="use relative positions as edge features on first GNN layer",
    )

    args = parser.parse_args()

    # Configure main logger with thread-safe output
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n",
        level="DEBUG",
        enqueue=True,
    )  # Thread-safe logging without JSON

    # Add file logging for main process using project structure
    main_log_file = expand_path(
        f"logs/training_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        get_project_root(),
    )
    # Ensure logs directory exists
    main_log_file.parent.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = expand_path(
        f"results/results_{args.dataset[1:]}_{timestamp}.json", get_project_root()
    )
    # Ensure results directory exists
    results_file.parent.mkdir(exist_ok=True)

    logger.add(
        str(main_log_file),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n",
        level="DEBUG",
        enqueue=True,
    )

    # Apply CPU optimizations if requested
    if args.cpu_optimize or args.cpu_only:
        logger.info("Applying CPU optimizations...")
        # Disable CUDA memory allocator for CPU tensors
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        # Use CPU-optimized allocator
        os.environ["MALLOC_CONF"] = "background_thread:true,metadata_thp:auto"
        # Optimize threading
        torch.set_num_threads(min(32, torch.get_num_threads()))
        logger.info(f"Set PyTorch threads: {torch.get_num_threads()}")

    # Check GPUs
    if args.cpu_only:
        gpu_count = 0
        logger.info("GPU usage disabled by --cpu-only flag")
    else:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    logger.info("=" * 60)
    if args.cpu_only:
        logger.info("CPU-ONLY TRAINING MODE")
    else:
        logger.info("GPU UTILIZATION MAXIMIZER")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"GPUs available: {gpu_count}")

    # Calculate max_workers
    if args.max_workers is not None:
        max_workers = args.max_workers
        logger.info(f"Using manual worker override: {max_workers}")
    elif gpu_count > 0:
        max_workers = gpu_count * args.workers_per_gpu
        logger.info(f"Workers per GPU: {args.workers_per_gpu}")
    else:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        if args.cpu_only:
            # Default to 4 CPU workers for multiprocessing, each using all cores via PyTorch
            max_workers = 4
            logger.info(f"CPU cores available: {cpu_count}")
            logger.info(
                f"CPU workers: {max_workers} (each using all cores via PyTorch)"
            )
        else:
            max_workers = 1
            logger.warning("Running on CPU (single worker)")

    if gpu_count > 0:
        for i in range(gpu_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info(f"Total concurrent jobs: {max_workers}")

    # Define hyperparameter grid
    if args.test:
        model_names = ["vpluspplus_a"]
        noise_levels = [0]
        heads = [1]
        visual_ranges = [0.1]
        seeds = range(2 * max_workers)
    else:
        # model_names = ["vpluspplus_a", "lazy"]
        model_names = ["vpluspplus_a"]
        noise_levels = [0, 0.005]  # [0, 0.005]
        """
        TOC 111225 7:20AM 
        Switch to using only 1 head
        """
        # heads = [1, 2, 3]
        heads = [1]
        visual_ranges = [0.5]
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
                            (
                                args.dataset,
                                model,
                                noise,
                                head,
                                vr,
                                seed,
                                gpu_count,
                                worker_id,
                                args.batch_size,
                                args.epochs,
                                args.compile,
                                args.memory_fraction,
                                args.no_validation,
                                args.early_stopping_patience,
                                args.min_delta,
                                args.train_size,
                                args.rollout,
                                args.total_rollout,
                                args.ablate,
                                args.self_loops,
                                args.relative_positions,
                            )
                        )
                        worker_id += 1

    logger.info(f"Total configurations: {len(all_params)}")
    logger.info("=" * 60)

    # Show worker assignment plan
    if gpu_count > 0:
        logger.info("GPU Assignment Plan:")
        for i in range(min(8, len(all_params))):  # Show first 8 assignments
            (
                _,
                model,
                noise,
                head,
                vr,
                seed,
                _,
                wid,
                batch_size,
                epochs,
                _,
                _,
                _,
                _,
                _,
                _,
                rollout,
                total_rollout,
                ablate,
            ) = all_params[i]
            assigned_gpu = wid % gpu_count
            logger.info(
                f"Worker {wid:02d} -> GPU {assigned_gpu}: {model}_n{noise}_h{head}_vr{vr}_s{seed} (bs={batch_size}, ep={epochs}), rollout = {rollout}  "
            )
        if len(all_params) > 8:
            logger.info(f"  ... and {len(all_params) - 8} more configurations")
    else:
        logger.info("CPU Training Plan:")
        for i in range(min(8, len(all_params))):  # Show first 8 assignments
            (
                _,
                model,
                noise,
                head,
                vr,
                seed,
                _,
                wid,
                batch_size,
                epochs,
                _,
                _,
                _,
                _,
                _,
                _,
                rollout,
                total_rollout,
                ablate,
                self_loops,
                relative_positions,
            ) = all_params[i]
            logger.info(
                f"  Config {wid:02d} -> CPU: {model}_n{noise}_h{head}_vr{vr}_s{seed} (bs={batch_size}, ep={epochs}), rollout = {rollout}  "
            )
        if len(all_params) > 8:
            logger.info(f"  ... and {len(all_params) - 8} more configurations")
    logger.info("=" * 60)

    # Run with concurrent futures - distributes across all GPUs
    results = []
    start_time = datetime.now()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs using the wrapper that sets environment variables
        future_to_params = {
            executor.submit(worker_wrapper, params): params for params in all_params
        }

        # Collect results as they complete (no global timeout to avoid killing good jobs)
        completed = 0
        for future in concurrent.futures.as_completed(future_to_params):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress update with worker distribution
            if completed % 10 == 0 or completed == len(all_params):
                elapsed = (datetime.now() - start_time).total_seconds() / 60

                # Count GPU vs CPU workers
                gpu_workers = {}
                cpu_workers = 0
                for r in results:
                    if "gpu_id" in r:
                        if r["gpu_id"] >= 0:
                            gpu_workers[r["gpu_id"]] = (
                                gpu_workers.get(r["gpu_id"], 0) + 1
                            )
                        else:
                            cpu_workers += 1

                # Build distribution info
                dist_parts = []
                if gpu_workers:
                    gpu_info = " | ".join(
                        [f"GPU{k}:{v}" for k, v in sorted(gpu_workers.items())]
                    )
                    dist_parts.append(gpu_info)
                if cpu_workers > 0:
                    dist_parts.append(f"CPU:{cpu_workers}")

                distribution = " | ".join(dist_parts) if dist_parts else "None"
                logger.info(
                    f"Progress: {completed}/{len(all_params)} | Elapsed: {elapsed:.1f} min | Distribution: {distribution}"
                )

    # Save results using project structure
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    successful = len([r for r in results if r["status"] == "success"])

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time: {elapsed:.2f} minutes")
    logger.info(f"Successful: {successful}/{len(results)}")
    logger.info(f"Results: {results_file}")

    # Find best configuration
    valid_results = [r for r in results if r["status"] == "success"]
    if valid_results:
        best = min(valid_results, key=lambda x: x["final_loss"])
        logger.success("Best configuration:")
        logger.success(f"  Model: {best['model_name']}")
        logger.success(f"  Self Loops: {best['self_loops']}")
        logger.success(f"  Noise: {best['noise']}")
        logger.success(f"  Heads: {best['heads']}")
        logger.success(f"  Visual Range: {best['visual_range']}")
        logger.success(f"  Seed: {best['seed']}")
        logger.success(f"  Loss: {best['final_loss']:.6f}")


if __name__ == "__main__":
    main()
