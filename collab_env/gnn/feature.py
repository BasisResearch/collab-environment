import numpy as np
import torch
import math
import torch.nn.functional as functional

def return_position_encoding(position, d_model, max_T):
    """
    position: torch tensor, of shape node number (N) by dim of positions
    d_model: int, dimension of the model
    max_T: max period

    returns: pe, of shape N x d_model x dim
    """
    pe = torch.zeros(position.shape[0], d_model, position.shape[1])
    

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(max_T) / d_model)
    )
    
    ind_sin = torch.arange(0,d_model,2)
    ind_cos = torch.arange(1,d_model,2)

    position = position.unsqueeze(1)
    div_term = div_term.unsqueeze(0)
    div_term = div_term.unsqueeze(2)

    # make sure everything is on GPU
    div_term = div_term.to(position.device)
    pe = pe.to(position.device)

    pe[:, ind_sin, :] = torch.sin(position * div_term)
    pe[:, ind_cos, :] = torch.cos(position * div_term)[:,:len(ind_cos)]

    return pe

def node_feature_vel(past_p, past_v, past_a, species_idx, species_dim):
    """
    Only has the agent velocities and position as feature.
    """
    init_p = past_p[:, -1, :, :]
    init_v = past_v[:, -1, :, :]
    S, N, dim = init_p.shape

    # Flatten
    x_vel = init_v.contiguous().view(S * N, dim)
    x_pos = init_p.contiguous().view(S * N, dim)
    x_pos_boundary = torch.maximum(
        1 - x_pos, torch.ones_like(x_pos) * 0.5
    )  # as in Allen et al., CoRL 2022
    x_species = species_idx.contiguous().view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()
    x_input = torch.cat(
        [x_vel, x_pos_boundary, species_onehot], dim=-1
    )  # [S*N, in_node_dim]

    return x_input

def node_feature_vel_pos(past_p, past_v, past_a, species_idx, species_dim):
    """
    Only has the agent velocities and position as feature.
    """
    init_p = past_p[:, -1, :, :]
    init_v = past_v[:, -1, :, :]
    S, N, dim = init_p.shape

    # Flatten
    x_vel = init_v.contiguous().view(S * N, dim)
    x_pos = init_p.contiguous().view(S * N, dim)
    x_pos_boundary = torch.maximum(
        1 - x_pos, torch.ones_like(x_pos) * 0.5
    )  # as in Allen et al., CoRL 2022
    x_species = species_idx.contiguous().view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()
    x_input = torch.cat(
        [x_vel, x_pos, x_pos_boundary, species_onehot], dim=-1
    )  # [S*N, in_node_dim]

    return x_input

def node_feature_vel_pos_plus(
    past_p, past_v, past_a, species_idx, species_dim, past_time=3
):
    """
    Only has the agent velocities and position as feature.
    """
    init_v = past_v[:, -1, :, :]
    S, F, N, dim = past_p.shape
    past_p = past_p[:, -past_time:, :, :]
    past_p = torch.permute(past_p, [0, 2, 1, 3])

    # Flatten
    x_vel = init_v.contiguous().view(S * N, dim)
    x_pos = past_p.contiguous().view((S * N, past_time * dim))
    x_pos_boundary = torch.maximum(
        1 - x_pos, torch.ones_like(x_pos) * 0.5
    )  # as in Allen et al., CoRL 2022
    x_species = species_idx.contiguous().view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()
    x_input = torch.cat(
        [x_vel, x_pos, x_pos_boundary, species_onehot], dim=-1
    )  # [S*N, in_node_dim]

    # del x_vel
    # del x_pos
    # del x_pos_boundary
    # del species_onehot

    return x_input

def node_feature_vel_plus_pos_plus(
    past_p, past_v, past_a, species_idx, species_dim, past_time=3
):
    """
    Only has the agent velocities and position as feature.
    """

    S, F, N, dim = past_p.shape
    past_p = past_p[:, -past_time:, :, :]
    past_p = torch.permute(past_p, [0, 2, 1, 3])

    S, F, N, _ = past_v.shape
    past_v = past_v[:, -past_time:, :, :]
    past_v = torch.permute(past_v, [0, 2, 1, 3])

    # Flatten
    x_vel = past_v.contiguous().view((S * N, past_time * dim))
    x_pos = past_p.contiguous().view((S * N, past_time * dim))
    x_pos_boundary = torch.maximum(
        1 - x_pos, torch.ones_like(x_pos) * 0.5
    )  # as in Allen et al., CoRL 2022

    x_species = species_idx.contiguous().view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()

    x_input = torch.cat(
        [x_vel, x_pos, x_pos_boundary, species_onehot], dim=-1
    )  # [S*N, in_node_dim]

    # del x_vel
    # del x_pos
    # del x_pos_boundary
    # del species_onehot

    return x_input

def node_feature_vel_pos_sinusoidal(
    past_p, past_v, past_a, species_idx, species_dim, past_time=2
):
    """
    Only has the agent velocities and position as feature.
    """

    # [x_vel, x_pos, x_pos_boundary, species_onehot]
    x_input = node_feature_vel_plus_pos_plus(past_p, past_v, past_a,
                                             species_idx, species_dim, past_time=past_time)
    # apply sinusidal operation on position
    S, F, N, dim = past_p.shape
    position_index = torch.arange(past_time * dim, 2 * past_time * dim)
    pos = x_input[:, position_index]
    velocity_index = torch.arange(0, past_time * dim)
    vel = x_input[:, velocity_index]

    d_model = 10
    max_T = 0.25/(2*math.pi)
    pe = return_position_encoding(pos, d_model, max_T)
    pe_reshape = torch.cat([pe[:,:,d] for d in range(pe.shape[2])], dim = 1)

    ve = return_position_encoding(vel, d_model, max_T)
    ve_reshape = torch.cat([ve[:,:,d] for d in range(ve.shape[2])], dim = 1)

    x_input = torch.cat([x_input, ve_reshape, pe_reshape], dim = 1)

    return x_input.contiguous()

