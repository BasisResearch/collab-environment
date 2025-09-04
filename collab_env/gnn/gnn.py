import torch
import torch.nn.functional as functional
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collab_env.gnn.utility import handle_discrete_data, v_function_2_vminushalf

# import gc  # Commented out - not used when gc.collect() is disabled
from collab_env.gnn.gnn_definition import GNN, Lazy
from collab_env.data.file_utils import expand_path, get_project_root
from loguru import logger
from torch_geometric.data import Data, Batch
from itertools import groupby


def build_single_graph_edges(positions, visual_range):
    """
    Build edge index for a single graph (not batched).
    positions: [N, 2] torch tensor (normalized position)
    visual_range: scalar (in normalized units, e.g. 0.1-- higher is more connected)
    Returns: edge_index [2, E] for single graph
    """
    # N, _ = positions.shape
    # device = positions.device

    # Compute pairwise distances
    dist = torch.cdist(positions, positions, p=2)  # [N, N]

    # Apply visual range filter and remove self-loops
    threshold = visual_range
    adj = (dist < threshold).to(torch.bool)
    adj.fill_diagonal_(False)  # remove self-loops
    edge_index = adj.nonzero(as_tuple=False).T  # [2, E]

    return edge_index


def build_pyg_batch(
    past_p,
    past_v,
    past_a,
    species_idx,
    species_dim,
    visual_range,
    node_feature_function,
):
    """
    Build PyTorch Geometric batch from individual graphs.
    past_p, past_v, past_a: [B, F, N, D] tensors
    species_idx: [B, N] tensor
    Returns: PyG Batch object
    """
    B, F, N, D = past_p.shape
    # device = past_p.device

    data_list = []

    for b in range(B):
        # Build edge index for this single graph
        positions = past_p[b, -1]  # [N, D] - use latest frame
        edge_index = build_single_graph_edges(positions, visual_range)

        # Get node features for this graph
        node_features = node_feature_function(
            past_p[b : b + 1],
            past_v[b : b + 1],
            past_a[b : b + 1],
            species_idx[b : b + 1],
            species_dim,
        )  # [1*N, feature_dim]

        # Create edge attributes (weights) - uniform weights for all edges
        num_edges = edge_index.shape[1]
        edge_attr = torch.ones(num_edges, 1, device=edge_index.device)

        # Create Data object
        data = Data(
            x=node_features.squeeze(0) if node_features.dim() > 2 else node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,  # Add edge attributes
            batch_idx=b,  # Store original batch index
            init_pos=past_p[b, -1],  # Store initial positions [N, D]
            init_vel=past_v[b, -1],  # Store initial velocities [N, D]
        )
        data_list.append(data)

    # Create PyG batch - this handles edge index offsetting automatically
    batch = Batch.from_data_list(data_list)
    return batch


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


def clip_acc(a, clip=1):
    # Due to the unstable nature of Euler integrator. We do this to safe guard.
    a_clip = torch.ones_like(a) * clip
    return torch.minimum(a, a_clip)


def bound_location(x, B=2):
    if np.linalg.norm(x.cpu()) > B:
        return -x
    else:
        return 0


def adjacency2edge(A):
    """
    Given an adjacency matrix, return edge_index,edge_weight, that is used for torch.nn.GNN
    A is a pytorch tensor.
    """
    edge_index = A.nonzero().t().contiguous()
    edge_weight = torch.tensor(
        [A[edge_index[0, i], edge_index[1, i]] for i in range(edge_index.shape[1])]
    )

    return edge_index, edge_weight


def return_adjacency_matrix(N, edge_index, edge_weight, batch_num=1):
    # move data to cpu if they are on GPU
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    if isinstance(edge_weight, torch.Tensor):
        edge_weight = edge_weight.detach().cpu().numpy()

    A_output = np.zeros((N * batch_num, N * batch_num)).astype("float32")

    for edge_ind in range(edge_index.shape[1]):
        (source, target) = edge_index[:, edge_ind]
        w = edge_weight[edge_ind]
        A_output[source][target] = np.min(w)

    A_output[np.isnan(A_output)] = 0
    return A_output


def normalize_by_col(N, batch_num, edge_index, return_matrix=False):
    # takes uniform adjacency matrix and makes it row sum to 1
    A = return_adjacency_matrix(N, edge_index, np.ones(edge_index.shape[1]), batch_num)

    col_sum = np.sum(A, axis=0)
    A_output = A / col_sum[np.newaxis, :]
    A_output[np.isnan(A_output)] = 0

    if return_matrix:
        return A_output

    # pick entries out - vectorized version
    weight = A_output[edge_index[0], edge_index[1]]

    return weight


def identify_frames(pos, vel):
    """
    TODO: more robust filtering
    """
    # construct lazy model prediction error
    # actual = pos[:,1:]
    # predicted = pos[:,:-1] + vel[:,:-1]
    # loss = [functional.mse_loss(actual[:,f],predicted[:,f]) for f in range(actual.shape[1])]

    # polarity
    polarity = torch.norm(torch.mean(vel, axis=2), dim=2)  # across birds
    polarity = polarity[:, :-4]  # skip last 4 frames
    polarity_diff = torch.abs(torch.diff(polarity, axis=1))
    threshold = torch.mean(polarity_diff, axis=1) + torch.std(polarity_diff, axis=1)
    threshold = threshold.reshape((-1, 1))
    diff_ind = (
        torch.argwhere(torch.sum(polarity_diff >= threshold, axis=0))
        .ravel()
        .cpu()
        .numpy()
    )
    candidate_frames = np.unique(
        np.concatenate([(d - 2, d - 1, d, d + 1, d + 2, d + 3) for d in diff_ind])
    )
    candidate_frames = candidate_frames[candidate_frames >= 0]

    return candidate_frames


def add_noise(x_input, sigma=0.001):
    noise = torch.normal(0, torch.ones_like(x_input) * sigma)
    return x_input + noise


def remove_boid_interaction(edge_index, species_idx):
    boid_index = torch.argwhere(
        species_idx.ravel() == 0
    ).ravel()  # assume boids are of class 0
    nonboid_edge_index = torch.isin(edge_index, boid_index).sum(axis=0) < 2

    edge_index_out = edge_index[:, nonboid_edge_index]
    return edge_index_out


def run_gnn_frame_pyg(model, pyg_batch, v_minushalf, delta_t, device=None):
    """
    Run GNN forward pass using PyTorch Geometric batch.

    model: GNN model
    pyg_batch: PyG Batch object with x, edge_index, batch
    v_minushalf: v_{t-1/2}, used only for Leapfrog integration
    delta_t: discrete time
    device: compute device

    Returns: pred_pos, pred_vel, pred_acc, pred_vplushalf, W
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pyg_batch = pyg_batch.to(device)

    node_prediction = model.node_prediction
    prediction_integration = model.prediction_integration

    # Forward pass - PyG handles batching automatically!
    pred, W = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr.squeeze(-1))

    pred = pred.to(device)

    # Reconstruct batch dimensions from PyG batch
    B = pyg_batch.batch.max().item() + 1  # Number of graphs in batch
    N = pyg_batch.batch.bincount()[0].item()  # Nodes per graph (assuming same size)

    pred_pos, pred_vel, pred_vplushalf, pred_acc = None, None, None, None

    if node_prediction == "position":
        pred_pos = pred.view(B, N, 2)

    elif node_prediction == "acc":
        pred_acc = pred.view(B, N, 2)
        pred_acc = clip_acc(pred_acc, clip=1)

        # Extract initial positions and velocities from PyG batch
        init_p = pyg_batch.init_pos.view(B, N, 2)
        init_v = pyg_batch.init_vel.view(B, N, 2)

        if v_minushalf is not None:
            v_minushalf = torch.as_tensor(v_minushalf, device=device)
        else:
            v_minushalf = init_v

        if "leap" in prediction_integration:
            pred_vplushalf = v_minushalf + pred_acc * delta_t
            pred_pos = init_p + pred_vplushalf * delta_t
            pred_vel = pred_vplushalf
        else:
            pred_vel = init_v + pred_acc * delta_t
            pred_pos = init_p + pred_vel * delta_t

    elif node_prediction == "velocity":
        pred_vel = pred.view(B, N, 2)
        init_p = pyg_batch.init_pos.view(B, N, 2)
        pred_pos = init_p + pred_vel

    return pred_pos, pred_vel, pred_acc, pred_vplushalf, W


def run_gnn(
    model,
    position,
    species_idx,
    species_dim,
    visual_range=0.5,
    sigma=0.001,
    device=None,
    training=True,
    lr=1e-3,
    debug_result=None,
    full_frames=False,
    rollout=-1,
    total_rollout=60,
    rollout_everyother=-1,
    ablate_boid_interaction=False,
    collect_debug=True,
):
    """
    This functions calls run_gnn_frame()
    training: set to True if training, set to false to test on heldout dataset.
    rollout: the frame number to start predicting for mult-frame rollout
    rollout_everyother: every x frame number do mult-frame rollout
    """
    if lr is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if debug_result is None:
        debug_result = {}
        debug_result["actual"] = []  # where we keep the actual frame t + 1
        debug_result[
            "predicted"
        ] = []  # where we keep the predicted result of frame t + 1 from information 1:t
        debug_result[
            "actual_acc"
        ] = []  # where we keep the actual acceleration at frame t.
        debug_result[
            "predicted_acc"
        ] = []  # where we keep the predicted acceleration at frame t.
        debug_result[
            "loss"
        ] = []  # where we keep loss for prediction of acceleration at frame t.
        debug_result["W"] = []  # where we keep updated adjcency matrix frame t

    start_frame = model.start_frame

    batch_loss = 0
    delta_t = 1

    S, Frame, N, _ = position.shape

    pos, vel, acc, v_function = handle_discrete_data(
        position, model.input_differentiation
    )

    pos = torch.as_tensor(pos)
    vel = torch.as_tensor(vel)
    acc = torch.as_tensor(acc)

    pos = pos.to(device)
    vel = vel.to(device)
    acc = acc.to(device)

    # frame roll out
    roll_out_flag = np.zeros(Frame)
    if rollout > 0:
        roll_out_flag[rollout:] = 1
    if rollout_everyother > 0:
        roll_out_flag = np.ones(Frame)
        roll_out_flag[::rollout_everyother] = 0

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine node feature function
    node_feature_function_name = model.node_feature_function
    if node_feature_function_name == "vel":
        node_feature = node_feature_vel
    elif node_feature_function_name == "vel_pos":
        node_feature = node_feature_vel_pos
    elif node_feature_function_name == "vel_pos_plus":
        node_feature = node_feature_vel_pos_plus
    elif node_feature_function_name == "vel_plus_pos_plus":
        node_feature = node_feature_vel_plus_pos_plus
    else:
        raise ValueError(f"Unknown node feature function: {node_feature_function_name}")

    frames = identify_frames(pos, vel)
    if full_frames:
        frames = np.arange(Frame - 1)

    if total_rollout > 0:
        frames = frames[:total_rollout]

    for frame_ind in range(
        len(frames) - 1
    ):  # range(Frame-1): #since we are predicting frame + 1, we have to stop 1 frame earlier
        frame = frames[frame_ind]
        if frame_ind == 0:
            frame_prev = None
        else:
            frame_prev = frames[frame_ind - 1]
        frame_next = frames[frame_ind + 1]

        # dummy initialization
        a = acc[:, 0]
        v = vel[:, 0]
        p = pos[:, 0]
        # have to be this convoluted instead of a simple :(frame+1) because we will want to replace p,v,a with rollout predicted value.
        if frame == 0 or frame_prev != frame - 1:  # not continuous!
            past_p = pos[:, frame, :, :]  # include the current frame
            past_v = vel[:, frame, :, :]  # include the current frame
            past_a = acc[:, frame, :, :]  # include the current frame

            past_p = past_p[:, None, :, :]
            past_v = past_v[:, None, :, :]
            past_a = past_a[:, None, :, :]
        else:
            past_p = torch.cat([past_p, p[:, None, :, :]], axis=1)
            past_v = torch.cat([past_v, v[:, None, :, :]], axis=1)
            past_a = torch.cat([past_a, a[:, None, :, :]], axis=1)

        if past_p.shape[1] < start_frame:
            v = vel[:, frame_next]  # 1 step ahead
            p = pos[:, frame_next]  # 1 step ahead
            a = acc[:, frame_next]  # 1 step ahead
            if v_function is not None:
                vminushalf = v_function_2_vminushalf(v_function, frame)  # v_t-1/2
                vminushalf = torch.as_tensor(vminushalf, device=device)
            else:
                vminushalf = v  # Use current velocity as fallback
            if training:
                p, v, a = add_noise(p, sigma), add_noise(v, sigma), add_noise(a, sigma)
                vminushalf = add_noise(vminushalf, sigma)
            continue

        target_pos = pos[:, frame_next]  # 1 step ahead/next frame
        target_acc = acc[:, frame_next]
        species_idx = species_idx.to(device)  # [S, N]

        # build PyG batch - much simpler!
        pyg_batch = build_pyg_batch(
            past_p, past_v, past_a, species_idx, species_dim, visual_range, node_feature
        )

        if ablate_boid_interaction:
            # TODO: Implement for PyG batch if needed
            pass

        (pred_pos, pred_vel, pred_acc, pred_vplushalf, W) = run_gnn_frame_pyg(
            model, pyg_batch, vminushalf, delta_t, device
        )
        # print("pred_pos.shape", pred_pos.shape)
        # print("target_pos.shape", target_pos.shape)
        # print("pred_pos.shape", pred_pos.shape)

        # Loss
        edge_index, edge_weight = W
        pred_pos_ = pred_pos.detach().clone().to(device)
        if pred_vel is not None:
            pred_vel_ = pred_vel.detach().clone().to(device)
        if pred_acc is not None:
            pred_acc_ = pred_acc.detach().clone().to(device)
        loss = functional.mse_loss(
            pred_acc, target_acc
        )  # + 0.1 * torch.sum(edge_weight)

        if lr is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_loss += loss.item()

        # update frame information
        if roll_out_flag[frame_next]:
            v = pred_vel_
            p = pred_pos_
            a = pred_acc_
            vminushalf = pred_vel_
            if training:  # protect transient training dynamics
                delat_p = bound_location(p, B=2)  # prevent blowing up
                p += delat_p
        else:
            v = vel[:, frame_next]  # 1 step ahead
            p = pos[:, frame_next]  # 1 step ahead
            a = acc[:, frame_next]  # 1 step ahead
            if v_function is not None:
                vminushalf = v_function_2_vminushalf(v_function, frame)
                vminushalf = torch.as_tensor(vminushalf, device=device)  # v_t-1/2
            else:
                vminushalf = v  # Use current velocity as fallback

        if training:
            p, v, a = add_noise(p, sigma), add_noise(v, sigma), add_noise(a, sigma)
            vminushalf = add_noise(vminushalf, sigma)

        # # Debug: show sample boid before/after
        if collect_debug:
            debug_result["predicted"].append(pred_pos_.detach().cpu().numpy())
            debug_result["actual"].append(target_pos.detach().cpu().numpy())
            # if training:
            debug_result["loss"].append(loss.detach().cpu().numpy())
            debug_result["W"].append(W)
            debug_result["predicted_acc"].append(pred_acc_.detach().cpu().numpy())
            debug_result["actual_acc"].append(target_acc.detach().cpu().numpy())

        # del pred_pos_
        # del pred_vel_
        # del pred_acc_

    # del pos
    # del vel
    # del acc
    # del vminushalf
    # del v_function
    # gc.collect()

    return batch_loss, debug_result, model


def train_rules_gnn(
    model,
    dataloader,
    visual_range=0.1,
    epochs=300,
    lr=1e-3,
    training=True,
    full_frames=True,
    species_dim=1,
    sigma=0.001,
    device=None,
    aux_data=None,
    rollout=-1,
    rollout_everyother=-1,
    total_rollout=100,
    ablate_boid_interaction=False,
    train_logger=None,
    collect_debug=False,
    val_dataloader=None,
    early_stopping_patience=10,
    min_delta=1e-6,
):
    if train_logger is None:
        train_logger = logger

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    debug_result_all = {}

    train_losses = []  # train loss by epoch
    val_losses = []  # validation loss by epoch
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    if not training:  # testing mode
        epochs = 1
        model.training = False
        model.eval()
    else:
        model.training = True
        model.train()

    if rollout > 0:
        train_logger.debug("rolling out...")
        full_frames = True

    for ep in range(epochs):
        torch.cuda.empty_cache()

        # Only log epoch start for longer training
        train_logger.debug(f"Starting epoch {ep + 1}/{epochs}")

        debug_result_all[ep] = {}

        train_losses_by_batch = []  # train loss this epoch

        for batch_idx, (position, species_idx) in enumerate(dataloader):
            # Only log every 10th batch to reduce noise
            train_logger.debug(
                f"Epoch {ep + 1}/{epochs} | Processing batch {batch_idx + 1}/{len(dataloader)}"
            )

            S, Frame, N, _ = position.shape

            (loss, debug_result_all[ep][batch_idx], model) = run_gnn(
                model,
                position,
                species_idx,
                species_dim,
                visual_range=visual_range,
                sigma=sigma,
                device=device,
                training=training,
                lr=lr,
                full_frames=full_frames,
                rollout=rollout,
                total_rollout=total_rollout,
                rollout_everyother=rollout_everyother,
                ablate_boid_interaction=ablate_boid_interaction,
                collect_debug=collect_debug,
            )

            train_losses_by_batch.append(loss)

        train_losses.append(train_losses_by_batch)
        epoch_train_loss = np.mean(train_losses[-1])
        train_logger.debug(f"Epoch {ep:03d} | Train loss: {epoch_train_loss:.4g}")

        # Validation evaluation
        epoch_val_loss = None
        if val_dataloader is not None and training:
            train_logger.debug("Validating...")
            model.eval()  # Set to eval mode for validation
            with torch.no_grad():
                val_losses_by_batch = []
                for batch_idx, (position, species_idx) in enumerate(val_dataloader):
                    S, Frame, N, _ = position.shape
                    (val_loss, _, _) = run_gnn(
                        model,
                        position,
                        species_idx,
                        species_dim,
                        visual_range=visual_range,
                        sigma=0,  # No noise during validation
                        device=device,
                        training=False,  # No gradient updates
                        lr=None,
                        full_frames=full_frames,
                        rollout=rollout,
                        total_rollout=total_rollout,
                        rollout_everyother=rollout_everyother,
                        ablate_boid_interaction=ablate_boid_interaction,
                        collect_debug=False,
                    )
                    val_losses_by_batch.append(val_loss)

                epoch_val_loss = np.mean(val_losses_by_batch)
                val_losses.append(val_losses_by_batch)

                # Early stopping logic
                if epoch_val_loss < best_val_loss - min_delta:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                train_logger.debug(
                    f"Epoch {ep:03d} | Train: {epoch_train_loss:.4g} | Val: {epoch_val_loss:.4g} | Patience: {patience_counter}/{early_stopping_patience}"
                )

                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    train_logger.info(
                        f"Early stopping at epoch {ep + 1}. Best val loss: {best_val_loss:.4g}"
                    )
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

            model.train()  # Set back to train mode

    # If we have a best model state and we did early stopping, use it
    if best_model_state is not None and training and val_dataloader is not None:
        model.load_state_dict(best_model_state)
        train_logger.info(
            f"Restored best model with validation loss: {best_val_loss:.4f}"
        )

    return np.array(train_losses), model, debug_result_all


def debatch_edge_index_weight(edge_index_output, edge_weight_output, INODE_NUM, files):
    """
    INODE_NUM: node dimension number for each file
    Takes batched sparse adjacency matrix
        and return seperate adjacency matrix for each file.
    """
    file_ind = torch.floor(edge_index_output / INODE_NUM)  # file in this batch

    assert torch.all(file_ind[0] == file_ind[1])
    node_ind = edge_index_output % INODE_NUM  # node in this batch

    W_by_file = []
    file_IDs = []
    for file in np.unique(file_ind):
        file_IDs.append(file)

        col_ind = file_ind[0] == file
        edge_index_output_file = node_ind[:, col_ind]
        edge_weight_output_file = edge_weight_output[col_ind]
        W = return_adjacency_matrix(
            INODE_NUM, edge_index_output_file, edge_weight_output_file
        )
        W_by_file.append(W)

    # for all the files without adjacency matrix this frame

    for file in np.setdiff1d(files, np.unique(file_ind)):
        W_by_file.append(np.zeros((INODE_NUM, INODE_NUM)))
        file_IDs.append(file)

    return W_by_file, file_IDs


def get_input_adjcency_from_debug_batch(
    input_data_loader, starting_frame, ending_frame, visual_range, epoch_list
):
    "organization is epoch -> file -> frame"

    data_list = list(iter(input_data_loader))
    B = len(data_list)

    W_out_epoch = {}
    for epoch in epoch_list:
        W_out = {}
        file_num = 0

        for batch_ind in range(B):
            position, species_idx = data_list[batch_ind]
            S, F, N, dim = position.shape

            for file in range(S):
                W_out[file_num + file] = []
                for f in range(starting_frame, ending_frame):
                    edge_index_input = build_single_graph_edges(
                        position[file, f, :, :], visual_range=visual_range
                    )
                    W_input_frame = normalize_by_col(
                        N, 1, edge_index_input, return_matrix=True
                    )
                    W_out[file_num + file].append(W_input_frame)
            file_num += S

        W_out_epoch[epoch] = W_out

    return W_out_epoch


def get_output_adjcency_from_debug_batch(
    debug_result,
    starting_frame,
    ending_frame,
    epoch_list=None,
    visual_range=0.5,
    INODE_NUM=20,
):
    "organization is epoch -> file -> frame"

    if epoch_list is None:
        epochs = list(debug_result.keys())
        epoch_list = [epochs[-1]]

    epoch_num = 0
    FRAME_NUM = 0
    # batch_num = 0
    frame = 0

    W_out_epoch = {}
    W_in_epoch = {}
    for epoch in epoch_list:
        W_out = {}
        W_in = {}

        file_num = 0
        B = len(debug_result[epoch_num])
        # F = len(debug_result[epoch_num][batch_num]['actual'])

        for batch_ind in range(B):
            file_num_batch = len(
                debug_result[epoch_num][batch_ind]["actual"][FRAME_NUM]
            )  # first frame

            # initialize all files in this batch
            for file_id in range(file_num, file_num + file_num_batch):
                W_out[file_id] = []
                W_in[file_id] = []

            for frame in range(starting_frame, ending_frame):
                """output W"""
                edge_index_output, edge_weight_output = debug_result[epoch_num][
                    batch_ind
                ]["W"][frame]

                W_by_file, file_ID = debatch_edge_index_weight(
                    edge_index_output,
                    edge_weight_output,
                    INODE_NUM,
                    np.arange(file_num_batch),
                )
                for _ in range(len(W_by_file)):
                    file_id = file_ID[_] + file_num
                    W_out[file_id].append(W_by_file[_])

                """input W"""
                pos = debug_result[epoch_num][batch_ind]["actual"][frame]
                pos = torch.tensor(pos)
                for file_ind in range(pos.shape[0]):
                    edge_index_input = build_single_graph_edges(
                        pos[file_ind], visual_range=visual_range
                    )
                    W_input_frame = normalize_by_col(
                        INODE_NUM, 1, edge_index_input, return_matrix=True
                    )
                    W_in[file_ind + file_num].append(W_input_frame)

            file_num = file_num + file_num_batch

        W_out_epoch[epoch] = W_out
        W_in_epoch[epoch] = W_in

    return W_out_epoch, W_in_epoch


def get_adjcency_from_debug_batch(
    debug_result,
    input_data_loader,
    visual_range,
    starting_frame,
    ending_frame,
    model_starting_frame,
    inode,
):
    """
    handle batching
    """
    starting_frame_ = starting_frame - model_starting_frame
    W_out_epoch, W_input_epoch = get_output_adjcency_from_debug_batch(
        debug_result,
        starting_frame_,
        starting_frame_ + (ending_frame - starting_frame),
        epoch_list=None,
        visual_range=visual_range,
        INODE_NUM=inode,
    )
    # W_input_epoch = get_input_adjcency_from_debug_batch(input_data_loader,
    #    starting_frame, ending_frame, visual_range, epoch_list = list(W_out_epoch.keys()))
    return W_input_epoch, W_out_epoch


## plotting


def plot_log_loss(
    train_losses,
    labels=None,
    alpha=0.05,
    log_base=10,
    title="Training Loss",
    ylabel="MSE Loss (log scale)",
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for i in range(len(train_losses)):
        # plot the mean line
        train_loss = train_losses[i]
        label = labels[i]

        #
        x = np.arange(train_loss.shape[0])
        y_mean = np.mean(train_loss, axis=1)  # across datasets
        # y_std = np.std(train_loss, axis = 1) #across datasets
        ax.plot(x, y_mean, label=label, color="C" + str(i))

        # y_lower = y_mean - 2 * y_std
        # y_upper = y_mean + 2 * y_std
        y_lower = np.quantile(train_loss, 0.5 * alpha, axis=1)  # across datasets
        y_upper = np.quantile(train_loss, 1 - 0.5 * alpha, axis=1)  # across datasets

        # Shade the confidence interval
        if alpha is not None:
            ax.fill_between(x, y_lower, y_upper, alpha=0.4, color="C" + str(i))

    plt.yscale("log", base=log_base)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    # plt.show()

    return ax


def find_frame_sets(lst):
    out = []
    for _, g in groupby(enumerate(lst), lambda k: k[0] - k[1]):
        start = next(g)[1]
        end = list(v for _, v in g) or [start]
        out.append(np.arange(start, end[-1] + 1))

    return out


def load_model(name, folder, file_name):
    model_path = expand_path(
        f"{folder}/{file_name}.pt",
        get_project_root(),
    )
    model_spec_path = expand_path(
        f"{folder}/{file_name}_model_spec.pkl",
        get_project_root(),
    )
    train_spec_path = expand_path(
        f"{folder}/{file_name}_train_spec.pkl",
        get_project_root(),
    )

    # save model spec
    with open(model_spec_path, "rb") as f:  # 'wb' for write binary
        model_spec = pickle.load(f)
    logger.debug("Loaded model spec.")

    # save training spec
    with open(train_spec_path, "rb") as f:  # 'wb' for write binary
        train_spec = pickle.load(f)
    logger.debug("Loaded training spec.")

    # Create an instance of the model (with the same architecture)
    model_spec["model_name"] = name
    model_spec["node_feature_function"] = "vel_plus_pos_plus"
    model_spec["node_prediction"] = "acc"
    model_spec["prediction_integration"] = "Euler"
    model_spec["input_differentiation"] = "finite"
    model_spec["in_node_dim"] = 19
    model_spec["start_frame"] = 3

    if "lazy" in name:
        gnn_model = Lazy(**model_spec)
        logger.debug("Loaded lazy model.")
        return gnn_model, model_spec, train_spec

    gnn_model = GNN(**model_spec)

    # load model
    # Load the state dictionary
    gnn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    logger.debug("Loaded model.")

    # Set the model to evaluation mode (important for inference)
    gnn_model.eval()

    return gnn_model, model_spec, train_spec


def save_model(model, model_spec, train_spec, file_name):
    # model weights
    model_output = expand_path(
        f"trained_models/{file_name}.pt",
        get_project_root(),
    )
    torch.save(model.state_dict(), model_output)

    # model specs
    model_spec_path = expand_path(
        f"trained_models/{file_name}_model_spec.pkl",
        get_project_root(),
    )
    with open(model_spec_path, "wb") as f:  # 'wb' for write binary
        pickle.dump(model_spec, f)

    # training specs
    train_spec_path = expand_path(
        f"trained_models/{file_name}_train_spec.pkl",
        get_project_root(),
    )
    with open(train_spec_path, "wb") as f:  # 'wb' for write binary
        pickle.dump(train_spec, f)

    return model_output, model_spec_path, train_spec_path


def debug_result2prediction(rollout_debug_result, file_id, epoch_num=0):
    # infer batch size
    predicted = rollout_debug_result[epoch_num][0]["predicted"][0]
    batch_size = predicted.shape[0]

    # infer batch number and the index within that batch
    batch_ind = int(np.floor(file_id / batch_size))
    file_ind = file_id % batch_size

    # rollout result fro this batch
    rollout_batch = rollout_debug_result[epoch_num][batch_ind]

    # concatenate across frames
    actual = np.array(
        [
            rollout_batch["actual"][f][file_ind]
            for f in range(len(rollout_batch["actual"]))
        ]
    )

    predicted = np.array(
        [
            rollout_batch["predicted"][f][file_ind]
            for f in range(len(rollout_batch["predicted"]))
        ]
    )

    # loss = np.array(
    #     [rollout_batch["loss"][f]
    #         for f in range(len(rollout_batch["loss"]))
    #         ]
    #     )

    pos_gnn, vel_gnn, acc_gnn, _ = handle_discrete_data(
        torch.tensor(predicted[np.newaxis, :]), "Euler"
    )
    pos, vel, acc, _ = handle_discrete_data(
        torch.tensor(actual[np.newaxis, :]), "Euler"
    )

    frames = identify_frames(pos, vel)
    frame_sets = find_frame_sets(frames)

    return pos, vel, acc, pos_gnn, vel_gnn, acc_gnn, frame_sets
