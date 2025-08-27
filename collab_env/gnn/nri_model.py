"""Complete NRI model implementation for boids data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .nri_modules import (
    NRIEncoder, NRIDecoder, create_relational_matrices, gumbel_softmax
)


class NRIModel(nn.Module):
    """Neural Relational Inference model for boids data."""
    
    def __init__(
        self,
        n_in,
        n_hid=256,
        n_out=2,  # x, y acceleration
        n_edge_types=2,  # no interaction, interaction
        do_prob=0.0,
        skip_first=True,
        temp=0.5,
        hard_sample=False,
    ):
        super(NRIModel, self).__init__()
        
        self.n_edge_types = n_edge_types
        self.skip_first = skip_first
        self.temp = temp
        self.hard_sample = hard_sample
        
        # Encoder: infer edge types
        self.encoder = NRIEncoder(n_in, n_hid, n_edge_types, do_prob)
        
        # Decoder: predict dynamics given edge types  
        self.decoder = NRIDecoder(n_in, n_edge_types, n_hid, do_prob)
        
        # For integrating predictions
        self.n_in = n_in
        self.n_out = n_out
        
    def forward(self, data, rel_rec, rel_send, pred_steps=1):
        """
        Forward pass of NRI model.
        
        Args:
            data: [batch_size, n_atoms, timesteps, n_dims] 
            rel_rec: [n_edges, n_atoms] relation matrix for receivers
            rel_send: [n_edges, n_atoms] relation matrix for senders
            pred_steps: number of prediction steps
            
        Returns:
            output: predicted trajectories
            prob: edge type probabilities 
            logits: raw edge type logits
        """
        # Encode edge types from input data
        logits = self.encoder(data, rel_rec, rel_send)
        
        # Debug: Check for NaN in logits
        if torch.isnan(logits).any():
            print(f"NaN detected in encoder logits! logits shape: {logits.shape}")
            print(f"logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
            print(f"input data stats: min={data.min()}, max={data.max()}, mean={data.mean()}")
            raise RuntimeError("NaN detected in encoder logits! This indicates exploding gradients or weight corruption.")
        
        # Sample edge types using Gumbel-Softmax
        edges = gumbel_softmax(
            logits, tau=self.temp, hard=self.hard_sample, dim=-1
        )
        prob = F.softmax(logits, dim=-1)
        
        # Debug: Check for NaN in prob
        if torch.isnan(prob).any():
            print(f"NaN detected in prob after softmax! prob shape: {prob.shape}")
            print(f"prob stats: min={prob.min()}, max={prob.max()}, mean={prob.mean()}")
            raise RuntimeError("NaN detected in prob after softmax! This indicates issues with logits.")
        
        # Decode dynamics using sampled edge types
        if pred_steps == 1:
            output = self.decoder(data, edges, rel_rec, rel_send)
            # Add timestep dimension: [batch, nodes, features] -> [batch, nodes, 1, features]
            output = output.unsqueeze(2)
        else:
            # Multi-step prediction
            output = self._predict_multi_step(
                data, edges, rel_rec, rel_send, pred_steps
            )
            
        return output, prob, logits
    
    def _predict_multi_step(self, data, edges, rel_rec, rel_send, pred_steps):
        """Predict multiple steps ahead."""
        outputs = []
        current_data = data
        
        for _ in range(pred_steps):
            # Predict next step
            pred = self.decoder(current_data, edges, rel_rec, rel_send)
            outputs.append(pred)
            
            # Update current data for next prediction
            # Assume pred is acceleration, integrate to get next positions/velocities
            current_data = self._integrate_dynamics(current_data, pred)
            
        return torch.stack(outputs, dim=2)  # [batch, atoms, pred_steps, dims]
    
    def _integrate_dynamics(self, data, acceleration, dt=1.0):
        """Integrate acceleration to update positions and velocities."""
        # Assume data format: [batch, atoms, timesteps, dims]
        # where dims = [pos_x, pos_y, vel_x, vel_y, ...]
        
        # Extract last timestep
        last_state = data[:, :, -1, :]  # [batch, atoms, dims]
        
        if self.n_in >= 4:  # Has position and velocity
            # Split into position and velocity
            pos = last_state[:, :, :2]  # [batch, atoms, 2]
            vel = last_state[:, :, 2:4]  # [batch, atoms, 2] 
            
            # Euler integration
            new_vel = vel + acceleration * dt
            new_pos = pos + new_vel * dt
            
            # Concatenate with other features if they exist
            if self.n_in > 4:
                other_features = last_state[:, :, 4:]
                new_state = torch.cat([new_pos, new_vel, other_features], dim=-1)
            else:
                new_state = torch.cat([new_pos, new_vel], dim=-1)
        else:
            # Simple case: just add acceleration
            new_state = last_state + acceleration * dt
            
        # Add new timestep to data
        new_data = torch.cat([
            data,
            new_state.unsqueeze(2)
        ], dim=2)
        
        return new_data
    
    def compute_loss(self, output, target, prob, logits, beta=1.0, alpha=1.0):
        """
        Compute NRI loss combining reconstruction and KL divergence.
        
        Args:
            output: predicted dynamics [batch, atoms, timesteps, dims]
            target: ground truth dynamics [batch, atoms, timesteps, dims] 
            prob: edge type probabilities [batch, edges, n_edge_types]
            logits: raw edge type logits [batch, edges, n_edge_types]
            beta: weight for KL divergence term
            alpha: weight for sparsity regularization
            
        Returns:
            total_loss: combined loss
            nll_loss: reconstruction loss
            kl_loss: KL divergence loss 
            sparse_loss: sparsity regularization loss
        """
        # Reconstruction loss (negative log likelihood)
        nll_loss = F.mse_loss(output, target)
        
        # KL divergence with uniform prior - original NRI implementation
        # KL(q||p) = sum(q * log(q/p)) where q=prob, p=uniform
        # For numerical stability, use log_softmax
        log_prob = F.log_softmax(logits, dim=-1)
        uniform_prob = 1.0 / self.n_edge_types
        
        # KL divergence: sum over edge types, mean over edges and batch
        kl_loss = prob * (log_prob - torch.log(torch.tensor(uniform_prob, device=logits.device)))
        kl_loss = kl_loss.sum(dim=-1).mean()
        
        # Add L2 regularization to prevent exploding weights
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.sum(param * param)
        
        # Original NRI: no additional sparsity loss, KL handles sparsity
        total_loss = nll_loss + beta * kl_loss + 1e-5 * l2_reg
        sparse_loss = torch.tensor(0.0, device=output.device)  # For compatibility
        
        return total_loss, nll_loss, kl_loss, sparse_loss
        
    def encode_edges(self, data, rel_rec, rel_send):
        """Encode edge types without sampling."""
        logits = self.encoder(data, rel_rec, rel_send)
        prob = F.softmax(logits, dim=-1)
        return prob, logits
    
    def decode_dynamics(self, data, edges, rel_rec, rel_send):
        """Decode dynamics given edge types."""
        return self.decoder(data, edges, rel_rec, rel_send)


class NRIBoidsModel(NRIModel):
    """Specialized NRI model for boids data."""
    
    def __init__(
        self,
        n_species=1,
        use_species_features=True,
        **kwargs
    ):
        # Calculate input dimensions based on boids features
        # Note: input will be flattened across timesteps
        base_features = 4  # pos_x, pos_y, vel_x, vel_y per timestep
        if use_species_features:
            base_features += n_species
        
        # Assume 5 timesteps by default (will be set dynamically)
        n_in = base_features  # Will be multiplied by timesteps in encoder
            
        super().__init__(n_in=n_in, **kwargs)
        
        self.n_species = n_species
        self.use_species_features = use_species_features
        
    def prepare_boids_data(self, positions, velocities, species=None):
        """
        Prepare boids data for NRI model.
        
        Args:
            positions: [batch, atoms, timesteps, 2] positions
            velocities: [batch, atoms, timesteps, 2] velocities  
            species: [batch, atoms] species labels (optional)
            
        Returns:
            data: [batch, atoms, timesteps, n_in] formatted data
        """
        # Concatenate positions and velocities
        data = torch.cat([positions, velocities], dim=-1)
        
        if self.use_species_features and species is not None:
            # Add species as one-hot encoding
            species_onehot = F.one_hot(species, num_classes=self.n_species).float()
            # Expand to match timesteps
            species_expanded = species_onehot.unsqueeze(2).expand(
                -1, -1, data.size(2), -1
            )
            data = torch.cat([data, species_expanded], dim=-1)
            
        return data
        
    def extract_predictions(self, output):
        """Extract position and velocity predictions from output."""
        if output.size(-1) >= 4:
            positions = output[..., :2]
            velocities = output[..., 2:4] 
            return positions, velocities
        else:
            # Output is just accelerations
            return None, output


def create_nri_model_for_boids(
    num_boids,
    n_species=1,
    use_species_features=True,
    n_edge_types=2,
    hidden_dim=256,
    dropout=0.0,
    device='cuda'
):
    """
    Create an NRI model configured for boids data.
    
    Args:
        num_boids: number of boids in the system
        n_species: number of species types
        use_species_features: whether to include species features
        n_edge_types: number of edge types (interaction types)
        hidden_dim: hidden layer dimension
        dropout: dropout probability
        device: device to put model on
        
    Returns:
        model: configured NRI model
        rel_rec: receiver relation matrix
        rel_send: sender relation matrix
    """
    model = NRIBoidsModel(
        n_species=n_species,
        use_species_features=use_species_features,
        n_hid=hidden_dim,
        n_edge_types=n_edge_types,
        do_prob=dropout,
    ).to(device)
    
    rel_rec, rel_send = create_relational_matrices(num_boids)
    rel_rec = rel_rec.to(device)
    rel_send = rel_send.to(device)
    
    return model, rel_rec, rel_send