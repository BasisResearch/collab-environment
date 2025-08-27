"""Training utilities for NRI model."""

import torch
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from collab_env.data.file_utils import expand_path


def load_boids_dataset(data_path, num_sequences=None, device='cpu'):
    """
    Load and prepare boids dataset for NRI training.
    
    Args:
        data_path: Path to dataset file (.pt or .pkl)
        num_sequences: Number of sequences to load (None = all)
        device: Device to load data to
        
    Returns:
        positions: [batch, agents, timesteps, 2]
        velocities: [batch, agents, timesteps, 2]
        species: [batch, agents]
    """
    expanded_path = expand_path(data_path)
    
    # Load dataset
    if data_path.endswith('.pt'):
        dataset = torch.load(expanded_path, map_location=device)
    elif data_path.endswith('.pkl'):
        with open(expanded_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Handle custom dataset format
    if hasattr(dataset, 'sequences'):
        all_positions = []
        all_species = []
        
        sequences_to_load = dataset.sequences[:num_sequences] if num_sequences else dataset.sequences
        
        for pos_array, species_array in sequences_to_load:
            # Convert positions [timesteps, agents, 2] -> [agents, timesteps, 2]
            if isinstance(pos_array, np.ndarray):
                pos_tensor = torch.from_numpy(pos_array).float().transpose(0, 1)
            else:
                pos_tensor = torch.tensor(pos_array).float().transpose(0, 1)
            
            # Handle species
            if isinstance(species_array, list):
                species_tensor = torch.tensor(species_array).long()
            elif isinstance(species_array, np.ndarray):
                if species_array.ndim == 1:
                    species_tensor = torch.from_numpy(species_array).long()
                else:
                    species_tensor = torch.from_numpy(species_array[0]).long()
            else:
                species_tensor = torch.zeros(pos_tensor.shape[0], dtype=torch.long)
            
            all_positions.append(pos_tensor)
            all_species.append(species_tensor)
        
        positions = torch.stack(all_positions, dim=0).to(device)
        species = torch.stack(all_species, dim=0).to(device)
        
        # Compute velocities
        velocities = torch.zeros_like(positions)
        velocities[:, :, 1:] = positions[:, :, 1:] - positions[:, :, :-1]
        
        # Check raw data range
        logger.info(f"Position range: [{positions.min().item():.3f}, {positions.max().item():.3f}]")
        
        # Data appears to already be normalized to [0,1], convert to [-1,1] for better training
        if positions.min() >= 0 and positions.max() <= 1:
            logger.info("Data appears normalized to [0,1], converting to [-1,1]")
            positions = positions * 2.0 - 1.0
            velocities = velocities * 2.0  # Scale velocities accordingly
        elif hasattr(dataset, 'width') and positions.max() > 10:
            # Data in pixel coordinates, normalize to [-1,1]
            logger.info(f"Converting pixel coordinates (width: {dataset.width}) to [-1,1]")
            positions = (positions - dataset.width/2) / (dataset.width/2)
            velocities = velocities / (dataset.width/2)
        
        logger.info(f"Final position range: [{positions.min().item():.3f}, {positions.max().item():.3f}]")
        logger.info(f"Final velocity range: [{velocities.min().item():.3f}, {velocities.max().item():.3f}]")
        
        # Check for NaN or infinite values
        if torch.isnan(positions).any() or torch.isinf(positions).any():
            logger.error("NaN or infinite values detected in positions after loading!")
        if torch.isnan(velocities).any() or torch.isinf(velocities).any():
            logger.error("NaN or infinite values detected in velocities after loading!")
    
    elif isinstance(dataset, dict):
        # Dictionary format
        positions = dataset.get('positions')
        velocities = dataset.get('velocities')
        species = dataset.get('species', None)
        
        # Convert to tensors if needed
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        if not isinstance(velocities, torch.Tensor):
            velocities = torch.tensor(velocities, dtype=torch.float32)
        if species is not None and not isinstance(species, torch.Tensor):
            species = torch.tensor(species, dtype=torch.long)
        else:
            species = torch.zeros(positions.shape[0], positions.shape[1], dtype=torch.long)
        
        positions = positions.to(device)
        velocities = velocities.to(device)
        species = species.to(device)
    
    else:
        raise ValueError(f"Unsupported dataset format: {type(dataset)}")
    
    logger.info(f"Loaded {positions.shape[0]} sequences")
    logger.info(f"Shape: {positions.shape}")
    
    return positions, velocities, species


def prepare_nri_data(positions, velocities, species, seq_len=10, pred_len=1):
    """
    Prepare sliding window data for NRI training.
    
    Args:
        positions: [batch, agents, timesteps, 2]
        velocities: [batch, agents, timesteps, 2]
        species: [batch, agents]
        seq_len: Length of input sequence
        pred_len: Length of prediction sequence
        
    Returns:
        inputs: [num_windows, agents, seq_len, 4]
        targets: [num_windows, agents, pred_len, 4]
        species_expanded: [num_windows, agents]
    """
    batch_size, n_agents, total_timesteps, _ = positions.shape
    
    inputs = []
    targets = []
    
    for t in range(total_timesteps - seq_len - pred_len + 1):
        # Input sequence
        input_pos = positions[:, :, t:t+seq_len, :]
        input_vel = velocities[:, :, t:t+seq_len, :]
        input_data = torch.cat([input_pos, input_vel], dim=-1)
        inputs.append(input_data)
        
        # Target sequence
        target_pos = positions[:, :, t+seq_len:t+seq_len+pred_len, :]
        target_vel = velocities[:, :, t+seq_len:t+seq_len+pred_len, :]
        target_data = torch.cat([target_pos, target_vel], dim=-1)
        targets.append(target_data)
    
    # Stack and reshape
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    
    num_windows = inputs.shape[0]
    inputs = inputs.view(-1, n_agents, seq_len, inputs.shape[-1])
    targets = targets.view(-1, n_agents, pred_len, targets.shape[-1])
    
    # Expand species
    species_expanded = species.unsqueeze(0).expand(num_windows, -1, -1)
    species_expanded = species_expanded.contiguous().view(-1, n_agents)
    
    return inputs, targets, species_expanded


def prepare_data_loaders(positions, velocities, species, seq_len=10, pred_len=1,
                         batch_size=32, train_split=0.7):
    """Create training and validation data loaders with sequence-based splitting."""
    # Split sequences first (like existing GNN), then prepare sliding windows
    batch_size_seqs, n_agents, total_timesteps, _ = positions.shape
    
    # Use fixed seed for reproducible splits like existing GNN
    train_size = int(train_split * batch_size_seqs)
    val_size = batch_size_seqs - train_size
    
    generator = torch.Generator().manual_seed(2025)
    indices = torch.randperm(batch_size_seqs, generator=generator)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Split sequences
    train_positions = positions[train_indices]
    train_velocities = velocities[train_indices]
    train_species = species[train_indices]
    
    val_positions = positions[val_indices]
    val_velocities = velocities[val_indices]
    val_species = species[val_indices]
    
    # Prepare sliding window data for each split
    train_inputs, train_targets, train_species_expanded = prepare_nri_data(
        train_positions, train_velocities, train_species, seq_len, pred_len
    )
    val_inputs, val_targets, val_species_expanded = prepare_nri_data(
        val_positions, val_velocities, val_species, seq_len, pred_len
    )
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets, train_species_expanded)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets, val_species_expanded)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Train sequences: {len(train_indices)}, Val sequences: {len(val_indices)}")
    
    return train_loader, val_loader, (val_positions, val_velocities, val_species)


def train_epoch(model, dataloader, optimizer, rel_rec, rel_send, device, beta=1.0, alpha=0.1):
    """Train NRI model for one epoch."""
    model.train()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets, species) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        species = species.to(device)
        
        # Prepare NRI data
        nri_data = model.prepare_boids_data(
            inputs[..., :2],  # positions
            inputs[..., 2:4], # velocities
            species
        )
        
        # Debug: Check for NaN in training data
        if torch.isnan(nri_data).any():
            print(f"NaN detected in training nri_data!")
            print(f"inputs shape: {inputs.shape}, targets shape: {targets.shape}")
            print(f"inputs stats: min={inputs.min()}, max={inputs.max()}")
            print(f"nri_data stats: min={nri_data.min()}, max={nri_data.max()}")
            raise RuntimeError("NaN detected in training data! This indicates data loading/preprocessing issues.")
        
        # Forward pass
        optimizer.zero_grad()
        output, prob, logits = model(nri_data, rel_rec, rel_send)
        
        # Compute loss
        loss, nll_loss, kl_loss, sparse_loss = model.compute_loss(
            output, targets, prob, logits, beta=beta, alpha=alpha
        )
        
        # Backward pass - no gradient clipping, let's find the real issue
        loss.backward()
        
        # Apply very aggressive gradient clipping to prevent exploding gradients
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        logger.debug(f"Batch {batch_idx} - Gradient norm: {total_norm:.6f}")
        
        # Check if gradients are NaN and reset if needed
        if torch.isnan(total_norm):
            logger.warning(f"NaN gradient detected at batch {batch_idx}! Resetting gradients.")
            optimizer.zero_grad()
            return total_loss, total_nll, total_kl  # Exit early
            
        optimizer.step()
        
        total_loss += loss.item()
        total_nll += nll_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
        
        logger.debug(f"  Batch {batch_idx}, Loss: {loss.item():.4f}, NLL: {nll_loss.item():.4f}, KL: {kl_loss.item():.4f}")
    
    return total_loss / num_batches, total_nll / num_batches, total_kl / num_batches


def validate_rollout(model, val_positions, val_velocities, val_species, rel_rec, rel_send, 
                     device, rollout_start=5, rollout_steps=50, context_len=10):
    """Validate using rollouts from initial frames (like existing GNN)."""
    model.eval()
    total_mse = 0
    num_sequences = 0
    
    with torch.no_grad():
        for seq_idx in range(val_positions.shape[0]):
            pos = val_positions[seq_idx:seq_idx+1]  # [1, agents, time, 2]
            vel = val_velocities[seq_idx:seq_idx+1]  # [1, agents, time, 2]  
            species = val_species[seq_idx:seq_idx+1]  # [1, agents]
            
            # Use sufficient frames for initial condition (need context_len frames)
            if pos.shape[2] <= rollout_start + rollout_steps:
                continue  # Skip sequences that are too short
                
            # Use rollout_start + context_len frames as initial condition
            initial_pos = pos[:, :, :rollout_start + context_len]
            initial_vel = vel[:, :, :rollout_start + context_len]
            ground_truth_pos = pos[:, :, rollout_start + context_len:rollout_start + context_len + rollout_steps]
            
            # Generate rollout starting from rollout_start + context_len frame
            from collab_env.gnn.nri_visualization import generate_rollout
            try:
                rollout_pos, _, _ = generate_rollout(
                    model, rel_rec, rel_send,
                    initial_pos, initial_vel, species,
                    rollout_steps=rollout_steps,
                    context_len=context_len,
                    device=device
                )
                
                # Compare rollout to ground truth
                # rollout_pos: [agents, rollout_steps, 2], ground_truth_pos: [1, agents, timesteps, 2]
                if rollout_pos.shape[1] >= ground_truth_pos.shape[2]:
                    pred_steps = ground_truth_pos.shape[2]
                    rollout_slice = rollout_pos[:, :pred_steps, :]  # [agents, steps, 2]
                    ground_truth_slice = ground_truth_pos[0]  # [agents, steps, 2]
                    mse = torch.mean((rollout_slice - ground_truth_slice) ** 2)
                    total_mse += mse.item()
                    num_sequences += 1
                    
            except Exception as e:
                logger.warning(f"Rollout failed for sequence {seq_idx}: {e}")
                continue
    
    if num_sequences == 0:
        return float('inf')
        
    return total_mse / num_sequences


def validate(model, dataloader, rel_rec, rel_send, device, beta=1.0, alpha=0.1):
    """Validate NRI model with standard next-step prediction."""
    model.eval()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, species in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            species = species.to(device)
            
            # Prepare NRI data
            nri_data = model.prepare_boids_data(
                inputs[..., :2], inputs[..., 2:4], species
            )
            
            # Forward pass
            output, prob, logits = model(nri_data, rel_rec, rel_send)
            
            # Compute loss
            loss, nll_loss, kl_loss, _ = model.compute_loss(
                output, targets, prob, logits, beta=beta, alpha=alpha
            )
            
            total_loss += loss.item()
            total_nll += nll_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
    
    return total_loss / num_batches, total_nll / num_batches, total_kl / num_batches


def train_model(model, train_loader, val_loader, val_data, rel_rec, rel_send,
                epochs=10, lr=1e-3, beta=1.0, alpha=0.1, device='cpu', 
                use_rollout_validation=False, rollout_start=5, rollout_steps=50):
    """Train NRI model with rollout-based validation like existing GNN."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_metric = float('inf')
    best_model_state = None
    
    val_positions, val_velocities, val_species = val_data
    
    for epoch in range(epochs):
        # Train
        train_loss, train_nll, train_kl = train_epoch(
            model, train_loader, optimizer, rel_rec, rel_send, device, beta, alpha
        )
        
        if use_rollout_validation and epoch % 5 == 0 and epoch > 0:  # Rollout validation every 5 epochs, skip first
            # Rollout-based validation (like existing GNN)
            val_mse = validate_rollout(
                model, val_positions, val_velocities, val_species, 
                rel_rec, rel_send, device, rollout_start, rollout_steps
            )
            val_metric = val_mse
            val_type = "MSE"
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train - Loss: {train_loss:.4f}, NLL: {train_nll:.4f}, KL: {train_kl:.4f}")
            logger.info(f"  Val   - Rollout MSE: {val_mse:.6f}")
        else:
            # Standard next-step validation
            val_loss, val_nll, val_kl = validate(
                model, val_loader, rel_rec, rel_send, device, beta, alpha
            )
            val_metric = val_loss
            val_type = "Loss"
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train - Loss: {train_loss:.4f}, NLL: {train_nll:.4f}, KL: {train_kl:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, NLL: {val_nll:.4f}, KL: {val_kl:.4f}")
        
        # Save best model
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_model_state = model.state_dict()
            logger.info(f"  New best model (val {val_type.lower()}: {val_metric:.6f})")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def save_model(model, rel_rec, rel_send, path, args=None):
    """Save NRI model and matrices."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'rel_rec': rel_rec,
        'rel_send': rel_send,
        'args': args
    }, path)


def load_model(path, device='cpu'):
    """Load NRI model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    # Extract dimensions from relation matrices
    rel_rec = checkpoint['rel_rec']
    rel_send = checkpoint['rel_send']
    n_agents = rel_rec.shape[1]
    
    # Create model (you may need to adjust parameters based on saved args)
    from collab_env.gnn.nri_model import create_nri_model_for_boids
    
    args = checkpoint.get('args', {})
    model, _, _ = create_nri_model_for_boids(
        num_boids=n_agents,
        n_species=1,  # Default, adjust if needed
        n_edge_types=getattr(args, 'n_edge_types', 2),
        hidden_dim=getattr(args, 'hidden_dim', 128),
        dropout=getattr(args, 'dropout', 0.1),
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, rel_rec.to(device), rel_send.to(device)