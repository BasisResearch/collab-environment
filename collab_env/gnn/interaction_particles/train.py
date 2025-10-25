"""
Training script for InteractionParticle model on 2D boids data.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from loguru import logger
import os

from .model import InteractionParticle
from collab_env.gnn.utility import handle_discrete_data
from collab_env.data.file_utils import expand_path


def build_graph_from_data(positions, velocities, accelerations, visual_range=0.5, max_radius=1.0):
    """
    Build PyG graph from position, velocity, and acceleration data.

    Parameters
    ----------
    positions : torch.Tensor
        Positions [N, 2]
    velocities : torch.Tensor
        Velocities [N, 2]
    accelerations : torch.Tensor
        Accelerations [N, 2] (targets)
    visual_range : float
        Maximum distance for edges (normalized)
    max_radius : float
        Maximum radius for normalization

    Returns
    -------
    data : torch_geometric.data.Data
        Graph data
    """
    N = positions.shape[0]

    # Build edges based on visual range
    dist = torch.cdist(positions, positions, p=2)
    adj = (dist < visual_range).to(torch.bool)
    adj.fill_diagonal_(False)
    edge_index = adj.nonzero(as_tuple=False).T

    # Create node features: [particle_id, pos_x, pos_y, vel_x, vel_y]
    particle_ids = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    node_features = torch.cat([particle_ids, positions, velocities], dim=1)

    # Create graph
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=accelerations,
        pos=positions
    )

    return data


def compute_velocity_statistics(dataset_path):
    """
    Compute velocity statistics from dataset for visualization purposes.

    Parameters
    ----------
    dataset_path : str
        Path to AnimalTrajectoryDataset .pt file

    Returns
    -------
    dict with keys:
        - 'max_speed': Maximum speed (velocity magnitude) in dataset
        - 'mean_speed': Mean speed
        - 'max_relative_velocity': Maximum relative velocity magnitude between particles
        - 'mean_relative_velocity': Mean relative velocity magnitude
    """
    logger.info(f"Computing velocity statistics from {dataset_path}")

    # Load dataset
    dataset = torch.load(dataset_path, weights_only=False)

    # Extract all positions
    all_positions = []
    for i in range(len(dataset)):
        pos, species = dataset[i]
        all_positions.append(pos.unsqueeze(0))

    position = torch.cat(all_positions, dim=0)  # [B, T, N, 2]

    # Compute velocities
    velocities = torch.diff(position, dim=1)  # [B, T-1, N, 2]

    # Compute speeds
    speeds = torch.sqrt(torch.sum(velocities ** 2, dim=-1))  # [B, T-1, N]
    max_speed = speeds.max().item()
    mean_speed = speeds.mean().item()

    # Compute pairwise relative velocities
    B, T, N, D = velocities.shape
    max_rel_vel = 0.0
    sum_rel_vel = 0.0
    count = 0

    for b in range(min(B, 10)):  # Sample first 10 trajectories for efficiency
        for t in range(T):
            vel_t = velocities[b, t]  # [N, 2]
            # Compute all pairwise relative velocities
            rel_vel = vel_t.unsqueeze(0) - vel_t.unsqueeze(1)  # [N, N, 2]
            rel_vel_mag = torch.sqrt(torch.sum(rel_vel ** 2, dim=-1))  # [N, N]
            # Exclude self (diagonal)
            mask = ~torch.eye(N, dtype=torch.bool)
            rel_vel_mag_filtered = rel_vel_mag[mask]

            max_rel_vel = max(max_rel_vel, rel_vel_mag_filtered.max().item())
            sum_rel_vel += rel_vel_mag_filtered.sum().item()
            count += len(rel_vel_mag_filtered)

    mean_rel_vel = sum_rel_vel / count if count > 0 else 0.0

    stats = {
        'max_speed': max_speed,
        'mean_speed': mean_speed,
        'max_relative_velocity': max_rel_vel,
        'mean_relative_velocity': mean_rel_vel
    }

    logger.info(f"Velocity statistics:")
    logger.info(f"  Max speed: {max_speed:.4f}")
    logger.info(f"  Mean speed: {mean_speed:.4f}")
    logger.info(f"  Max relative velocity: {max_rel_vel:.4f}")
    logger.info(f"  Mean relative velocity: {mean_rel_vel:.4f}")

    return stats


def prepare_dataset(dataset_path, visual_range=0.5, max_radius=1.0, input_differentiation='finite_diff'):
    """
    Load and prepare 2D boids dataset for training.

    Parameters
    ----------
    dataset_path : str
        Path to AnimalTrajectoryDataset .pt file
    visual_range : float
        Maximum distance for edges (in normalized coordinates)
    max_radius : float
        Maximum radius for normalization
    input_differentiation : str
        Method for computing velocities/accelerations ('finite_diff' or 'spline')

    Returns
    -------
    graphs : list
        List of PyG Data objects
    p_range : float
        Position range for normalization

    Notes
    -----
    The simulation has a natural timestep of 1.0 between frames.
    Velocities and accelerations are computed as-is without rescaling.
    """
    logger.info(f"Loading 2D boids dataset from {dataset_path}")

    # Load AnimalTrajectoryDataset
    dataset = torch.load(dataset_path, weights_only=False)
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    # Extract all positions from the dataset
    # Each sample is (positions, species) where positions: [steps, N, 2]
    all_positions = []
    for i in range(len(dataset)):
        pos, species = dataset[i]  # [steps, N, 2], [N]
        # Add batch dimension: [1, steps, N, 2]
        all_positions.append(pos.unsqueeze(0))

    # Stack all samples: [B, steps, N, 2]
    position = torch.cat(all_positions, dim=0)
    logger.info(f"Extracted positions with shape: {position.shape}")

    # Note: 2D boids data is already normalized by scene size [width, height]
    # Positions are in [0, 1] range
    # We just need to compute velocities and accelerations

    # Compute velocities and accelerations using finite differences
    # Natural timestep is 1.0 between frames
    # v[t] = p[t+1] - p[t] (displacement per timestep)
    # a[t] = v[t+1] - v[t] (velocity change per timestep)
    p_smooth, v_smooth, a_smooth, v_function = handle_discrete_data(
        position, input_differentiation
    )

    # Use as-is (no rescaling)
    p_normalized = p_smooth.float()
    v_normalized = v_smooth.float()
    a_normalized = a_smooth.float()

    # For 2D boids, data is already normalized
    # p_range is 1.0 (scene is normalized to [0,1] x [0,1])
    p_range = 1.0

    # Build graphs
    B, F, N, D = p_normalized.shape
    logger.info(f"Building graphs from {B} trajectories, {F} frames, {N} particles")

    graphs = []
    for b in range(B):
        for f in range(1, F-1):  # Skip first and last frame (no acceleration)
            if torch.isnan(a_normalized[b, f]).any() or torch.isinf(a_normalized[b, f]).any():
                continue

            graph = build_graph_from_data(
                p_normalized[b, f],
                v_normalized[b, f],
                a_normalized[b, f],
                visual_range=visual_range,
                max_radius=max_radius
            )
            graphs.append(graph)

    logger.info(f"Created {len(graphs)} graph samples")
    return graphs, p_range


def create_multistep_dataset(dataset_path, rollout_steps, visual_range=0.5, max_radius=1.0,
                             input_differentiation='finite_diff'):
    """
    Create dataset of multi-step trajectory chunks for rollout training.

    Parameters
    ----------
    dataset_path : str
        Path to AnimalTrajectoryDataset .pt file
    rollout_steps : int
        Length of trajectory chunks (M)
    visual_range : float
        Maximum distance for edges (in normalized coordinates)
    max_radius : float
        Maximum radius for normalization
    input_differentiation : str
        Method for computing velocities/accelerations ('finite_diff' or 'spline')

    Returns
    -------
    sequences : list of dict
        Each dict contains:
        - 'initial_pos': torch.Tensor [N, 2]
        - 'initial_vel': torch.Tensor [N, 2]
        - 'gt_positions': torch.Tensor [rollout_steps, N, 2]
        - 'gt_velocities': torch.Tensor [rollout_steps, N, 2]
    p_range : float
        Position range for normalization

    Notes
    -----
    The simulation has a natural timestep of 1.0 between frames.
    Velocities are computed as-is without rescaling.
    """
    logger.info(f"Creating multi-step dataset with rollout_steps={rollout_steps}")
    logger.info(f"Loading 2D boids dataset from {dataset_path}")

    # Load AnimalTrajectoryDataset
    dataset = torch.load(dataset_path, weights_only=False)
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    # Extract all positions from the dataset
    all_positions = []
    for i in range(len(dataset)):
        pos, species = dataset[i]  # [steps, N, 2], [N]
        all_positions.append(pos.unsqueeze(0))

    # Stack all samples: [B, steps, N, 2]
    position = torch.cat(all_positions, dim=0)
    logger.info(f"Extracted positions with shape: {position.shape}")

    # Compute velocities (natural timestep = 1.0)
    p_smooth, v_smooth, a_smooth, v_function = handle_discrete_data(
        position, input_differentiation
    )

    # Use as-is (no rescaling)
    v_normalized = v_smooth.float()
    p_normalized = p_smooth.float()

    p_range = 1.0

    # Build trajectory sequences
    B, F, N, D = p_normalized.shape
    logger.info(f"Building trajectory sequences from {B} trajectories, {F} frames, {N} particles")
    logger.info(f"Rollout steps: {rollout_steps}, need at least {rollout_steps} frames per trajectory")

    if F < rollout_steps:
        raise ValueError(
            f"Dataset trajectories have only {F} frames, but rollout_steps={rollout_steps}. "
            f"Either reduce --rollout-steps to <{F}, or use a dataset with longer trajectories."
        )

    sequences = []
    nan_count = 0
    for b in range(B):
        # Use NON-OVERLAPPING windows to avoid data leakage
        # Step by rollout_steps to get independent chunks
        for start_frame in range(0, F - rollout_steps + 1, rollout_steps):
            # Extract chunk of rollout_steps frames
            pos_chunk = p_normalized[b, start_frame:start_frame + rollout_steps]  # [M, N, 2]
            vel_chunk = v_normalized[b, start_frame:start_frame + rollout_steps]  # [M, N, 2]

            # Check for NaN/Inf
            if torch.isnan(pos_chunk).any() or torch.isinf(pos_chunk).any():
                nan_count += 1
                continue
            if torch.isnan(vel_chunk).any() or torch.isinf(vel_chunk).any():
                nan_count += 1
                continue

            sequences.append({
                'initial_pos': pos_chunk[0].clone(),  # [N, 2]
                'initial_vel': vel_chunk[0].clone(),  # [N, 2]
                'gt_positions': pos_chunk.clone(),    # [M, N, 2]
                'gt_velocities': vel_chunk.clone(),   # [M, N, 2]
            })

    logger.info(f"Created {len(sequences)} non-overlapping trajectory sequences (skipped {nan_count} due to NaN/Inf)")
    return sequences, p_range


def train_interaction_particle(
    dataset_path,
    config=None,
    epochs=100,
    batch_size=32,
    learning_rate=1e-5,
    train_split=0.8,
    visual_range=0.5,
    rollout_steps=1,
    device=None,
    save_dir=None,
    seed=42,
    epoch_callback=None
):
    """
    Train InteractionParticle model on 2D boids data.

    Supports both single-step (rollout_steps=1) and multi-step rollout training.

    Parameters
    ----------
    dataset_path : str
        Path to dataset .pt file
    config : dict, optional
        Model configuration
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    train_split : float
        Fraction of data for training
    visual_range : float
        Maximum distance for edges
    rollout_steps : int
        Number of rollout steps for training (default: 1 for single-step)
        If > 1, uses multi-step rollout training with position-based loss
    device : str or torch.device, optional
        Device to use
    save_dir : str, optional
        Directory to save model checkpoints
    seed : int
        Random seed
    epoch_callback : callable, optional
        Callback function called after each validation epoch with signature:
        callback(model, epoch, train_loss, val_loss, save_dir)

    Returns
    -------
    model : InteractionParticle
        Trained model
    history : dict
        Training history

    Notes
    -----
    The simulation uses a natural timestep of 1.0 between frames.
    For integration during rollout, dt=1.0 is used implicitly.
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Log training mode
    if rollout_steps > 1:
        logger.info(f"Using multi-step rollout training with {rollout_steps} steps")
        logger.info(f"Loss will be computed on POSITIONS (not accelerations)")
    else:
        logger.info("Using single-step training (acceleration prediction)")

    logger.info("Using natural timestep dt=1.0 from simulation")

    # Default config
    if config is None:
        config = {
            'n_particles': 20,
            'n_particle_types': 1,
            'max_radius': 1.0,
            'hidden_dim': 128,
            'embedding_dim': 16,
            'n_mp_layers': 3,
            'input_size': 7,  # delta_pos(2) + r(1) + delta_vel(2) + pos_i(2)
            'output_size': 2,
        }

    # Prepare dataset (different for single-step vs multi-step)
    if rollout_steps == 1:
        # Single-step: use individual frames
        graphs, p_range = prepare_dataset(
            dataset_path,
            visual_range=visual_range,
            max_radius=config['max_radius']
        )

        # Split into train and validation
        n_train = int(len(graphs) * train_split)
        train_data = graphs[:n_train]
        val_data = graphs[n_train:]

        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    else:
        # Multi-step: use trajectory sequences
        sequences, p_range = create_multistep_dataset(
            dataset_path,
            rollout_steps=rollout_steps,
            visual_range=visual_range,
            max_radius=config['max_radius']
        )

        # Split into train and validation
        n_train = int(len(sequences) * train_split)
        train_data = sequences[:n_train]
        val_data = sequences[n_train:]

        logger.info(f"Train sequences: {len(train_data)}, Val sequences: {len(val_data)}")

        # Check if we have enough data
        if len(train_data) == 0:
            raise ValueError(
                f"No training sequences created! Dataset may be too short for rollout_steps={rollout_steps}. "
                f"Try reducing --rollout-steps or using a longer dataset."
            )

        # For multi-step, we'll manually batch in the training loop
        # (can't use DataLoader directly with dict format)
        train_loader = train_data
        val_loader = val_data

    # Create model
    model = InteractionParticle(config, device=device)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'config': config,
        'p_range': p_range,
        'rollout_steps': rollout_steps
    }

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        if rollout_steps == 1:
            # Single-step training: standard batch loop
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                batch = batch.to(device)

                optimizer.zero_grad()

                # Forward pass
                pred = model(batch)

                # Compute loss (acceleration MSE)
                loss = F.mse_loss(pred, batch.y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1
        else:
            # Multi-step rollout training
            # Manually batch the sequences
            import random
            random.shuffle(train_loader)  # train_loader is just a list of sequences

            for i in tqdm(range(0, len(train_loader), batch_size), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                batch_sequences = train_loader[i:i+batch_size]

                optimizer.zero_grad()
                batch_loss = 0.0

                # Process each sequence in the batch
                for seq in batch_sequences:
                    initial_pos = seq['initial_pos'].to(device)
                    initial_vel = seq['initial_vel'].to(device)
                    gt_positions = seq['gt_positions'].to(device)  # [M, N, 2]

                    # Perform differentiable rollout (dt=1.0, natural timestep)
                    pred_positions, _ = differentiable_rollout(
                        model, initial_pos, initial_vel, rollout_steps,
                        visual_range=visual_range, delta_t=1.0, device=device,
                        requires_grad=True, return_intermediates=True
                    )

                    # Compute position MSE loss
                    loss = F.mse_loss(pred_positions, gt_positions)
                    batch_loss += loss

                # Average loss over batch
                batch_loss = batch_loss / len(batch_sequences)

                # Backward pass
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                train_batches += 1

        # Average training loss
        if train_batches > 0:
            train_loss /= train_batches
        else:
            logger.warning("No training batches processed in this epoch!")
            train_loss = float('inf')

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        if rollout_steps == 1:
            # Single-step validation
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    batch = batch.to(device)
                    pred = model(batch)
                    loss = F.mse_loss(pred, batch.y)
                    val_loss += loss.item()
                    val_batches += 1
        else:
            # Multi-step rollout validation
            with torch.no_grad():
                for i in tqdm(range(0, len(val_loader), batch_size), desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    batch_sequences = val_loader[i:i+batch_size]

                    batch_loss = 0.0
                    for seq in batch_sequences:
                        initial_pos = seq['initial_pos'].to(device)
                        initial_vel = seq['initial_vel'].to(device)
                        gt_positions = seq['gt_positions'].to(device)

                        # Perform rollout (dt=1.0, natural timestep)
                        pred_positions, _ = differentiable_rollout(
                            model, initial_pos, initial_vel, rollout_steps,
                            visual_range=visual_range, delta_t=1.0, device=device,
                            requires_grad=False, return_intermediates=True
                        )

                        # Compute position MSE loss
                        loss = F.mse_loss(pred_positions, gt_positions)
                        batch_loss += loss

                    batch_loss = batch_loss / len(batch_sequences)
                    val_loss += batch_loss.item()
                    val_batches += 1

        # Average validation loss
        if val_batches > 0:
            val_loss /= val_batches
        else:
            logger.warning("No validation batches processed in this epoch!")
            val_loss = float('inf')

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss = {train_loss:.2g}, Val Loss = {val_loss:.2g}"
        )

        # Save best model
        if save_dir and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'p_range': p_range,
                'rollout_steps': rollout_steps
            }, save_path)
            logger.info(f"Saved best model to {save_path}")

        # Call epoch callback if provided
        if epoch_callback is not None:
            epoch_callback(model, epoch + 1, train_loss, val_loss, save_dir)

    # Save final model
    if save_dir:
        final_path = os.path.join(save_dir, 'final_model.pt')
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'history': history
        }, final_path)
        logger.info(f"Saved final model to {final_path}")

    return model, history


def evaluate_model(model, dataset_path, visual_range=0.5, device=None):
    """
    Evaluate model on dataset.

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    dataset_path : str
        Path to dataset
    visual_range : float
        Maximum distance for edges
    device : str or torch.device, optional
        Device to use

    Returns
    -------
    metrics : dict
        Evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device

    # Prepare dataset
    graphs, _ = prepare_dataset(dataset_path, visual_range=visual_range)
    loader = DataLoader(graphs, batch_size=32, shuffle=False)

    # Evaluate
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y, reduction='sum')
            total_loss += loss.item()
            total_samples += batch.y.shape[0]

    mse = total_loss / total_samples
    rmse = np.sqrt(mse)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'n_samples': total_samples
    }

    logger.info(f"Evaluation: MSE = {mse:.6f}, RMSE = {rmse:.6f}")

    return metrics


def build_graph_from_current_state(positions, velocities, visual_range, device):
    """
    Build PyG graph from current positions and velocities (for rollout).

    Parameters
    ----------
    positions : torch.Tensor
        Positions [N, 2]
    velocities : torch.Tensor
        Velocities [N, 2]
    visual_range : float
        Maximum distance for edges
    device : torch.device
        Device to use

    Returns
    -------
    data : torch_geometric.data.Data
        Graph data
    """
    N = positions.shape[0]

    # Build edges based on visual range
    dist = torch.cdist(positions, positions, p=2)
    adj = (dist < visual_range).to(torch.bool)
    adj.fill_diagonal_(False)
    edge_index = adj.nonzero(as_tuple=False).T

    # Create node features: [particle_id, pos_x, pos_y, vel_x, vel_y]
    particle_ids = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)
    node_features = torch.cat([particle_ids, positions, velocities], dim=1)

    # Create dummy accelerations (not used for prediction)
    dummy_acc = torch.zeros_like(positions)

    # Create graph
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=dummy_acc,
        pos=positions
    )

    return data


def differentiable_rollout(model, initial_positions, initial_velocities, n_steps,
                          visual_range=0.5, delta_t=0.1, device=None,
                          requires_grad=False, return_intermediates=False):
    """
    Perform differentiable multi-step rollout.

    This function can be used for both training (with gradients) and evaluation (without).
    It performs Euler integration of the predicted accelerations.

    Parameters
    ----------
    model : InteractionParticle
        Model to use for prediction
    initial_positions : torch.Tensor
        Initial positions [N, 2]
    initial_velocities : torch.Tensor
        Initial velocities [N, 2]
    n_steps : int
        Number of steps to predict (including initial state)
    visual_range : float
        Maximum distance for edges
    delta_t : float
        Time step for integration (default: 0.1)
    device : torch.device, optional
        Device to use
    requires_grad : bool
        If True, enables gradient computation (for training)
        If False, uses torch.no_grad() (for evaluation)
    return_intermediates : bool
        If True, returns all intermediate states on device
        If False, returns final states moved to CPU

    Returns
    -------
    predicted_positions : torch.Tensor
        Predicted positions [n_steps, N, 2]
    predicted_velocities : torch.Tensor
        Predicted velocities [n_steps, N, 2]
    """
    if device is None:
        device = next(model.parameters()).device

    # Initialize
    pos = initial_positions.clone().to(device)
    vel = initial_velocities.clone().to(device)

    predicted_positions = []
    predicted_velocities = []

    # Store initial state
    if return_intermediates:
        predicted_positions.append(pos)
        predicted_velocities.append(vel)
    else:
        predicted_positions.append(pos.detach().cpu())
        predicted_velocities.append(vel.detach().cpu())

    # Choose gradient context
    def rollout_step():
        nonlocal pos, vel

        for step in range(n_steps - 1):
            # Build graph from current state
            graph = build_graph_from_current_state(pos, vel, visual_range, device)

            # Predict acceleration
            pred_acc = model(graph)

            # Euler integration
            vel = vel + pred_acc * delta_t
            pos = pos + vel * delta_t

            # Store predictions
            if return_intermediates:
                predicted_positions.append(pos)
                predicted_velocities.append(vel)
            else:
                predicted_positions.append(pos.detach().cpu())
                predicted_velocities.append(vel.detach().cpu())

    if requires_grad:
        # Training mode: keep gradients
        rollout_step()
    else:
        # Evaluation mode: no gradients
        with torch.no_grad():
            rollout_step()

    predicted_positions = torch.stack(predicted_positions, dim=0)
    predicted_velocities = torch.stack(predicted_velocities, dim=0)

    return predicted_positions, predicted_velocities


def generate_rollout(model, initial_positions, initial_velocities, n_steps,
                    visual_range=0.5, delta_t=1.0, device=None):
    """
    Generate autoregressive rollout predictions (evaluation mode).

    This is a convenience wrapper around differentiable_rollout for backward compatibility.

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    initial_positions : torch.Tensor
        Initial positions [N, 2]
    initial_velocities : torch.Tensor
        Initial velocities [N, 2]
    n_steps : int
        Number of steps to predict
    visual_range : float
        Maximum distance for edges
    delta_t : float
        Time step for integration (default: 1.0, natural simulation timestep)
    device : str or torch.device, optional
        Device to use

    Returns
    -------
    predicted_positions : torch.Tensor
        Predicted positions [n_steps, N, 2]
    predicted_velocities : torch.Tensor
        Predicted velocities [n_steps, N, 2]
    """
    model.eval()
    return differentiable_rollout(
        model, initial_positions, initial_velocities, n_steps,
        visual_range=visual_range, delta_t=delta_t, device=device,
        requires_grad=False, return_intermediates=False
    )


def evaluate_rollout(model, dataset_path, visual_range=0.5, n_rollout_steps=50, delta_t=1.0, device=None):
    """
    Evaluate model with multi-step rollout on validation data.

    Parameters
    ----------
    model : InteractionParticle
        Trained model
    dataset_path : str
        Path to dataset
    visual_range : float
        Maximum distance for edges
    n_rollout_steps : int
        Number of steps for rollout
    delta_t : float
        Time step for integration (default: 1.0, natural simulation timestep)
    device : str or torch.device, optional
        Device to use

    Returns
    -------
    results : dict
        Dictionary containing:
        - ground_truth_positions: list of [T, N, 2]
        - predicted_positions: list of [T, N, 2]
        - ground_truth_velocities: list of [T, N, 2]
        - predicted_velocities: list of [T, N, 2]
        - metrics: dict of aggregate metrics
    """
    if device is None:
        device = next(model.parameters()).device

    logger.info(f"Loading dataset for rollout evaluation from {dataset_path}")

    # Load dataset
    dataset = torch.load(dataset_path, weights_only=False)

    # Use validation split (last 20%)
    n_train = int(len(dataset) * 0.8)
    val_dataset = [dataset[i] for i in range(n_train, len(dataset))]

    logger.info(f"Evaluating rollout on {len(val_dataset)} validation samples")

    all_gt_positions = []
    all_pred_positions = []
    all_gt_velocities = []
    all_pred_velocities = []

    position_errors = []
    velocity_errors = []

    model.eval()

    for idx, (positions, species) in enumerate(tqdm(val_dataset, desc="Generating rollouts")):
        # positions: [T, N, 2]
        T, N, D = positions.shape

        # Use available timesteps (need at least 2 for initial velocity)
        if T < 2:
            continue

        # Limit rollout steps to available data
        # generate_rollout returns n_steps timesteps (including initial)
        actual_rollout_steps = min(n_rollout_steps, T)

        # Extract ground truth trajectory (same number of timesteps as rollout will produce)
        gt_positions = positions[:actual_rollout_steps].numpy()

        # Compute ground truth velocities
        gt_velocities = np.diff(gt_positions, axis=0)
        gt_velocities = np.concatenate([gt_velocities, gt_velocities[-1:]], axis=0)  # Repeat last

        # Get initial conditions
        initial_pos = positions[0]  # [N, 2]
        initial_vel = positions[1] - positions[0]  # [N, 2]

        # Generate rollout
        pred_pos, pred_vel = generate_rollout(
            model, initial_pos, initial_vel,
            n_steps=actual_rollout_steps,
            visual_range=visual_range,
            delta_t=delta_t,
            device=device
        )

        # Convert to numpy
        pred_pos = pred_pos.numpy()
        pred_vel = pred_vel.numpy()

        # Store results
        all_gt_positions.append(gt_positions)
        all_pred_positions.append(pred_pos)
        all_gt_velocities.append(gt_velocities)
        all_pred_velocities.append(pred_vel)

        # Compute errors for this trajectory
        pos_error = np.mean(np.linalg.norm(gt_positions - pred_pos, axis=-1))  # Mean over time and particles
        vel_error = np.mean(np.linalg.norm(gt_velocities - pred_vel, axis=-1))

        position_errors.append(pos_error)
        velocity_errors.append(vel_error)

    # Aggregate metrics
    metrics = {
        'mean_position_error': np.mean(position_errors),
        'std_position_error': np.std(position_errors),
        'mean_velocity_error': np.mean(velocity_errors),
        'std_velocity_error': np.std(velocity_errors),
        'n_trajectories': len(position_errors)
    }

    logger.info(f"Rollout evaluation complete:")
    logger.info(f"  Mean position error: {metrics['mean_position_error']:.6f} ± {metrics['std_position_error']:.6f}")
    logger.info(f"  Mean velocity error: {metrics['mean_velocity_error']:.6f} ± {metrics['std_velocity_error']:.6f}")

    results = {
        'ground_truth_positions': all_gt_positions,
        'predicted_positions': all_pred_positions,
        'ground_truth_velocities': all_gt_velocities,
        'predicted_velocities': all_pred_velocities,
        'metrics': metrics
    }

    return results
