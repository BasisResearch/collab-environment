"""
Training script for InteractionParticle model on boids data.
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
        all_positions.append(pos.unsqueeze(0).numpy())

    # Stack all samples: [B, steps, N, 2]
    position = np.concatenate(all_positions, axis=0)
    logger.info(f"Extracted positions with shape: {position.shape}")

    # Note: 2D boids data is already normalized by scene size [width, height]
    # Positions are in [0, 1] range
    # We just need to compute velocities and accelerations

    # Compute velocities and accelerations using finite differences
    p_smooth, v_smooth, a_smooth, v_function = handle_discrete_data(
        position, input_differentiation
    )

    # Convert to torch
    p_normalized = torch.tensor(p_smooth, dtype=torch.float32)
    v_normalized = torch.tensor(v_smooth, dtype=torch.float32)
    a_normalized = torch.tensor(a_smooth, dtype=torch.float32)

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
    return graphs, p_range.item()


def train_interaction_particle(
    dataset_path,
    config=None,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    train_split=0.8,
    visual_range=0.5,
    device=None,
    save_dir=None,
    seed=42
):
    """
    Train InteractionParticle model on boids data.

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
    device : str or torch.device, optional
        Device to use
    save_dir : str, optional
        Directory to save model checkpoints
    seed : int
        Random seed

    Returns
    -------
    model : InteractionParticle
        Trained model
    history : dict
        Training history
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Default config
    if config is None:
        config = {
            'n_particles': 20,
            'n_particle_types': 1,
            'max_radius': 1.0,
            'hidden_dim': 128,
            'embedding_dim': 16,
            'n_mp_layers': 3,
            'input_size': 7,  # delta_pos(2) + r(1) + vel_i(2) + vel_j(2)
            'output_size': 2,
        }

    # Prepare dataset
    graphs, p_range = prepare_dataset(
        dataset_path,
        visual_range=visual_range,
        max_radius=config['max_radius']
    )

    # Split into train and validation
    n_train = int(len(graphs) * train_split)
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:]

    logger.info(f"Train samples: {len(train_graphs)}, Val samples: {len(val_graphs)}")

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

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
        'p_range': p_range
    }

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred = model(batch)

            # Compute loss
            loss = F.mse_loss(pred, batch.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                batch = batch.to(device)
                pred = model(batch)
                loss = F.mse_loss(pred, batch.y)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
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
                'p_range': p_range
            }, save_path)
            logger.info(f"Saved best model to {save_path}")

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
