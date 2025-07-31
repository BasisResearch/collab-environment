import torch
import torch.nn.functional as functional
import numpy as np
import matplotlib.pyplot as plt
from utility import fit_spline_to_data, upgrade_data, upgrade_data_finite_diff, v_function_2_vminushalf
import gc

def build_edge_index(positions, visual_range):
    """
    positions: [B, N, 2] torch tensor (normalized position)
    visual_range: scalar (in normalized units, e.g. 0.1-- higher is more connected)
    Returns: edge_index [2, E] for full batch
    """
    B, N, _ = positions.shape
    device = positions.device

    # Flatten across batch: [B*N, 2]
    flat_pos = positions.reshape((B * N, 2))

    # Compute pairwise distances
    dist = torch.cdist(flat_pos, flat_pos, p=2)  # [B*N, B*N]

    # Only allow intra-batch connections
    mask = torch.ones_like(dist, dtype=torch.bool, device=device)
    for b in range(B):
        start = b * N
        end = (b + 1) * N
        mask[start:end, start:end] = False  # mask self-group
    dist[mask] = float('inf')

    # Apply visual range filter
    threshold = visual_range
    adj = (dist < threshold).to(torch.bool)
    adj.fill_diagonal_(False)  # remove self-loops
    edge_index = adj.nonzero(as_tuple=False).T  # [2, E]

    return edge_index  # optional: return mask or edge_attr

def extract_adjacency_difference(N, edge_index_input, edge_weight_input, edge_index_output, edge_weight_output):
    """input"""
    A_input = normalize_by_col(N, edge_index_input, return_matrix = True)

    """output"""
    A_output = return_adjacency_matrix(N, edge_index_output, edge_weight_output)
    
    delta = np.linalg.norm(A_output) - np.linalg.norm(A_input)

    return A_input, A_output, delta
    
def node_feature_vel(past_p,past_v,past_a,species_idx,species_dim):
    """
    Only has the agent velocities and position as feature.
    """
    init_p = past_p[:,-1,:,:]
    init_v = past_v[:,-1,:,:]
    S, N, _ = init_p.shape

    # Flatten
    x_vel = init_v.view(S * N, 2)
    x_pos = init_p.view(S * N, 2)
    x_pos_boundary = np.maximum( 1 - x_pos, np.ones_like(x_pos) * 0.5 )
    x_species = species_idx.view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()
    x_input = torch.cat([x_vel, x_pos_boundary, species_onehot], dim=-1)  # [S*N, in_node_dim]

    return x_input

def node_feature_vel_pos(past_p,past_v,past_a,species_idx,species_dim):
    """
    Only has the agent velocities and position as feature.
    """
    init_p = past_p[:,-1,:,:]
    init_v = past_v[:,-1,:,:]
    S, N, _ = init_p.shape

    # Flatten
    x_vel = init_v.view(S * N, 2)
    x_pos = init_p.view(S * N, 2)
    x_pos_boundary = np.maximum( 1 - x_pos, np.ones_like(x_pos) * 0.5 )
    x_species = species_idx.view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()
    x_input = torch.cat([x_vel, x_pos, x_pos_boundary, species_onehot], dim=-1)  # [S*N, in_node_dim]

    return x_input

def node_feature_vel_pos_plus(past_p,past_v,past_a,species_idx,species_dim,past_time = 3):
    """
    Only has the agent velocities and position as feature.
    """
    init_v = past_v[:,-1,:,:]
    S, F, N, _ = past_p.shape
    past_p = past_p[:,-past_time:,:,:]
    past_p = torch.permute(past_p, [0, 2, 1, 3])
    
    # Flatten
    x_vel = init_v.view(S * N, 2)
    x_pos = past_p.reshape((S * N, past_time * 2))
    x_pos_boundary = np.maximum( 1 - x_pos, np.ones_like(x_pos) * 0.5 )
    x_species = species_idx.view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()
    x_input = torch.cat([x_vel, x_pos, x_pos_boundary, species_onehot], dim=-1)  # [S*N, in_node_dim]

    del x_vel
    del x_pos
    del x_pos_boundary
    del species_onehot

    return x_input

def node_feature_vel_plus_pos_plus(past_p,past_v,past_a,
            species_idx,species_dim,past_time = 3,adj = None):
    """
    Only has the agent velocities and position as feature.
    """

    S, F, N, _ = past_p.shape
    past_p = past_p[:,-past_time:,:,:]
    past_p = torch.permute(past_p, [0, 2, 1, 3])

    S, F, N, _ = past_v.shape
    past_v = past_v[:,-past_time:,:,:]
    past_v = torch.permute(past_v, [0, 2, 1, 3])

    # Flatten
    x_vel = past_v.reshape((S * N, past_time * 2))
    x_pos = past_p.reshape((S * N, past_time * 2))
    #x_pos_boundary = np.maximum( 1 - x_pos, np.ones_like(x_pos) * 0.5 )
    x_pos_boundary = 1 - x_pos
    pairwise_distances = functional.pairwise_distance(x_pos, x_pos, p=2) # Euclidean distance
    pairwise_diff_x = x_pos[:,0].reshape((-1,1)) - x_pos[:,0].reshape((1,-1))
    pairwise_diff_y = x_pos[:,1].reshape((-1,1)) - x_pos[:,1].reshape((1,-1))
    nonlinear_feature0 = pairwise_distances
    nonlinear_feature1x = pairwise_diff_x / pairwise_distances ** 2
    nonlinear_feature1y = pairwise_diff_y / pairwise_distances ** 2
    nonlinear_feature2x = pairwise_diff_x / pairwise_distances ** 4
    nonlinear_feature2y = pairwise_diff_y / pairwise_distances ** 4

    nonlinear_feature0 = adj @ nonlinear_feature0.reshape((-1,1))
    nonlinear_feature1x = torch.diag(adj @ nonlinear_feature1x)
    nonlinear_feature1y = torch.diag(adj @ nonlinear_feature1y)
    nonlinear_feature2x = torch.diag(adj @ nonlinear_feature2x)
    nonlinear_feature2y = torch.diag(adj @ nonlinear_feature2y)
    #assert 1 == 0

    x_species = species_idx.view(S * N)
    species_onehot = functional.one_hot(x_species, num_classes=species_dim).float()

    x_input = torch.cat([x_vel, x_pos, x_pos_boundary,
                         nonlinear_feature0,
                         nonlinear_feature1x.reshape((-1,1)),
                         nonlinear_feature1y.reshape((-1,1)),
                         nonlinear_feature2x.reshape((-1,1)),
                         nonlinear_feature2y.reshape((-1,1)),
                         species_onehot], dim=-1)  # [S*N, in_node_dim]

    del x_vel
    del x_pos
    del x_pos_boundary
    del species_onehot
    del nonlinear_feature0
    del nonlinear_feature1x
    del nonlinear_feature1y
    del nonlinear_feature2x
    del nonlinear_feature2y

    assert 1 == 0

    return x_input

def clip_acc(a,clip = 1):
    # Due to the unstable nature of Euler integrator. We do this to safe guard.
    a_clip = torch.ones_like(a) * clip
    return torch.minimum(a, a_clip)

def bound_location(x, B = 2):
    if np.linalg.norm(x.cpu()) > B:
        return -x
    else:
        return 0

def adjacency2edge(A):
    """
    Given an adjacency matrix, return edge_index,edge_weight, that is used for torch.nn.GNN
    A is a tensor
    """
    edge_index = A.nonzero().t().contiguous()
    edge_weight = torch.tensor([A[edge_index[0,i],edge_index[1,i]]for i in range(edge_index.shape[1])])

    return edge_index, edge_weight

def return_adjacency_matrix(N,edge_index,edge_weight):
    # move data to cpu if they are on GPU
    if edge_index.device != "cpu":
        edge_index = edge_index.detach().cpu().numpy()
    if edge_weight.device != "cpu":
        edge_weight = edge_weight.detach().cpu().numpy()

    A_output = np.zeros((N,N)).astype("float32")
    
    for edge_ind in range(edge_index.shape[1]):
        (source, target) = edge_index[:,edge_ind]
        w = edge_weight[edge_ind]
        A_output[source][target] = np.min(w)
    
    A_output[np.isnan(A_output)] = 0
    return A_output

def normalize_by_col(N, edge_index, return_matrix = False):
    # takes uniform adjacency matrix and makes it row sum to 1
    A = return_adjacency_matrix(N,edge_index,np.ones(edge_index.shape[1]))
    
    col_sum = np.sum(A, axis = 0)
    A_output = A / col_sum[np.newaxis, :]
    A_output[np.isnan(A_output)] = 0

    if return_matrix:
        return A_output

    # pick entries out
    weight = [A_output[edge_index[0,i],edge_index[1,i]]for i in range(edge_index.shape[1])]

    return np.array(weight)

def run_gnn_frame(model, edge_index, edge_weight,
                    past_p, past_v, past_a, v_minushalf, delta_t,
                    species_idx, species_dim):
    """
    given model, initial position and velocity of all animals,
    predict the next frame position of all animals.
    
    node_feature_function: vel, vel_pos, vel_pos_plus, vel_plus_pos_plus
    node_prediction: position, velocity, or acceleration
    
    assumes x_input, edge_index, and edge_weight have all been sent to device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    S, F, N, _ = past_p.shape
    node_feature_function = model.node_feature_function
    node_prediction = model.node_prediction

    adj = None
    if node_feature_function == "vel":
        node_feature = node_feature_vel
    elif node_feature_function == "vel_pos":
        node_feature = node_feature_vel_pos
    elif node_feature_function == "vel_pos_plus":
        node_feature = node_feature_vel_pos_plus
    elif node_feature_function == "vel_plus_pos_plus":
        node_feature = node_feature_vel_plus_pos_plus
        adj = return_adjacency_matrix(N, edge_index, edge_weight)
        adj = torch.tensor(adj).to(device)

    x_input = node_feature(past_p, past_v, past_a, species_idx, 1, adj = adj.T)

    x_input = x_input.to(device)

    # Forward
    pred, W = model(x_input, edge_index, edge_weight)
    pred = pred.to(device)
    init_p = past_p[:,-1,:,:]
    init_v = past_v[:,-1,:,:]
    init_a = past_a[:,-1,:,:]

    pred_pos, pred_vel, pred_vplushalf, pred_acc = None, None, None, None

    if node_prediction == "position":
        pred_pos = pred.view(S, N, 2)

    elif node_prediction == "acc":
        pred_acc = pred.view(S, N, 2)
        pred_acc = clip_acc(pred_acc,clip = 1)

        # leap frog integration
        """
        # v_{i+1/2} = v_{i-1/2} + a_i * delta_t
        # x_{i+1} = x_{i} + v_{i+1/2} * delta_t
        
        
        pred_vplushalf = v_minushalf + pred_acc * delta_t
        pred_pos = init_p + pred_vplushalf * delta_t
        pred_vel = pred_vplushalf #init_v + pred_acc * delta_t
        """

        # Euler
        pred_vel = init_v + pred_acc 
        pred_pos = init_p + pred_vel 
        
    elif node_prediction == "velocity":
        pred_vel = pred.view(S, N, 2)
        pred_pos = init_p + pred_vel #TO DO: use other integration

    return pred_pos, pred_vel, pred_vplushalf, pred_acc, W
    

def run_gnn(model,
            pos, vel, acc, v_function, species_idx, species_dim,
            visual_range = 0.5,
            device = None,
            training = True, lr = 1e-3,
            debug_result = None,
            ep = None, batch_idx = None,
            rollout = -1, #rollout: the frame number to start predicting for mult-frame rollout
            rollout_everyother = -1): #rollout_everyother: every x frame number do mult-frame rollout
    """
    training: set to True if training, set to false to test on heldout dataset.
    """
    if lr is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if debug_result is None:
        debug_result = {}
        debug_result["actual"] = []
        debug_result["predicted"] = []
        debug_result["epoch_batch"] = []
        debug_result["W"] = []
        debug_result["loss"] = []
        debug_result["actual_acc"] = []
        debug_result["predicted_acc"] = []
    
    start_frame = model.start_frame

    batch_loss = 0
    delta_t = 1

    pos = pos.to(device)
    vel = vel.to(device)
    acc = acc.to(device)

    S, Frame, N, _ = pos.shape

    # frame roll out
    roll_out_flag = np.zeros(Frame)
    if rollout > 0:
        roll_out_flag[rollout:] = 1
    if rollout_everyother > 0:
        roll_out_flag = np.ones(Frame)
        roll_out_flag[::rollout_everyother] = 0
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build graph:
    # we can use input to initialize
    #if aux_data is not None:
    #    edge_index, edge_weight = aux_data[batch_idx]["edge_index"], aux_data[batch_idx]["edge_weight"]
    #else: 
        # or we initialize through the first frame
    #    init_p = p_smooth[:,0,:,:]
    #    edge_index = build_edge_index(init_p, visual_range=visual_range)
    #    edge_weight = normalize_by_col(N,edge_index)

    #    edge_index = edge_index.to(device)
    #    edge_weight = torch.tensor(edge_weight).to(device)
    #    #edge_index = edge_index.to(torch.float64)
    

    for frame in range(Frame-1):
        if frame == 0:
            past_p = pos[:,0,:,:] #include the current frame
            past_v = vel[:,0,:,:] #include the current frame
            past_a = acc[:,0,:,:] #include the current frame

            past_p = past_p[:, torch.newaxis, :, :]
            past_v = past_v[:, torch.newaxis, :, :]
            past_a = past_a[:, torch.newaxis, :, :]
        else:    
            past_p = torch.cat([past_p, p[:, torch.newaxis, :, :]], axis = 1)             
            past_v = torch.cat([past_v, v[:, torch.newaxis, :, :]], axis = 1)
            past_a = torch.cat([past_a, a[:, torch.newaxis, :, :]], axis = 1)
        
        #past_p = past_p.to(device)      # [S, F, N, 2]
        #past_v = past_v.to(device)      # [S, F, N, 2]
        #past_a = past_a.to(device)      # [S, F, N, 2]

        if frame < start_frame:
            v = vel[:, frame + 1]  # 1 step ahead
            p = pos[:, frame + 1]  # 1 step ahead
            a = acc[:, frame + 1]  # 1 step ahead
            vminushalf = v_function_2_vminushalf(v_function, frame + 1)
            vminushalf = torch.tensor(vminushalf).to(device) #v_t-1/2
            continue

        target_pos = pos[:, frame + 1]  # 1 step ahead/next frame
        target_acc = acc[:, frame + 1]
        species_idx = species_idx.to(device)    # [S, N]

        # build graph
        init_p = past_p[:, -1, :, :]
        edge_index = build_edge_index(init_p, visual_range=visual_range)
        edge_weight = normalize_by_col(N,edge_index)

        edge_index = torch.tensor(edge_index).to(device)
        edge_weight = torch.tensor(edge_weight).to(device)       

        (pred_pos, pred_vel,
            pred_vplushalf, pred_acc, W) = run_gnn_frame(model, edge_index, edge_weight,
                        past_p, past_v, past_a, vminushalf, delta_t,
                        species_idx, species_dim)
        #print("pred_pos.shape", pred_pos.shape)
        #print("target_pos.shape", target_pos.shape)
        #print("pred_pos.shape", pred_pos.shape)

        # Loss
        edge_index, edge_weight = W
        A = torch.tensor(return_adjacency_matrix(N,edge_index,edge_weight))

        pred_pos_ = pred_pos.detach().clone().to(device)
        if pred_vel is not None:
            pred_vel_ = pred_vel.detach().clone().to(device)
        if pred_acc is not None:
            pred_acc_ = pred_acc.detach().clone().to(device)
        loss = functional.mse_loss(pred_acc, target_acc) #+ 0.1 * torch.sum(edge_weight)

        if training:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        batch_loss += loss.item()

        # update frame information
        if roll_out_flag[frame + 1]:
            v = pred_vel_
            p = pred_pos_
            a = pred_acc_
            vminushalf = pred_vplushalf
            if training: #protect transient training dynamics
                delat_p = bound_location(p, B = 2) #prevent blowing up
                p += delat_p
        else:
            v = vel[:, frame + 1]  # 1 step ahead
            p = pos[:, frame + 1]  # 1 step ahead
            a = acc[:, frame + 1]  # 1 step ahead
            vminushalf = v_function_2_vminushalf(v_function, frame + 1)
            vminushalf = torch.tensor(vminushalf).to(device) #v_t-1/2
            
        # # Debug: show sample boid before/after
        debug_result["actual"].append(target_pos.detach().cpu().numpy())
        debug_result["predicted"].append(pred_pos_.detach().cpu().numpy())
        #if training:
        debug_result["loss"].append(loss.detach().cpu().numpy())
        debug_result["epoch_batch"].append((ep,batch_idx))
        debug_result["W"].append((ep,W))
        debug_result["predicted_acc"].append(pred_acc_.detach().cpu().numpy())
        debug_result["actual_acc"].append(target_acc.detach().cpu().numpy())

        del pred_pos_
        del pred_vel_
        del pred_acc_
    
    del pos
    del vel
    del acc
    del vminushalf
    del v_function
    gc.collect()

    return batch_loss, debug_result, model

def train_rules_gnn(model, dataloader,
                    visual_range = 0.1,
                    epochs = 300,
                    lr=1e-3,
                    training = True,
                    species_dim = 1,
                    device = None,
                    aux_data = None,
                    rollout = -1,
                    rollout_everyother = -1):
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    debug_result_all = {}

    train_losses = [] # train loss by epoch
    for ep in range(epochs):
        torch.cuda.empty_cache()

        print("epoch", ep)
        print("\n")

        debug_result_all[ep] = {}

        model.train()

        train_losses_by_batch = [] # train loss this epoch

        for batch_idx, (position, species_idx) in enumerate(dataloader):
            print("batch", batch_idx)
            print("\n")

            S, Frame, N, _ = position.shape

            if training:
                p_smooth, v_smooth, a_smooth, v_function = upgrade_data(position)
            else:
                p_smooth, v_smooth, a_smooth, v_function = upgrade_data_finite_diff(position)
            p_smooth = torch.tensor(p_smooth)
            v_smooth = torch.tensor(v_smooth)
            a_smooth = torch.tensor(a_smooth)
            
            (loss, debug_result_all[ep][batch_idx], model) = run_gnn(
                                        model, p_smooth, v_smooth, a_smooth, v_function[0], species_idx, species_dim,
                                        visual_range = visual_range,
                                        device = device,
                                        training = training,
                                        lr = lr,
                                        ep = ep, batch_idx = batch_idx,
                                        rollout = rollout,
                                        rollout_everyother = rollout_everyother)

            train_losses_by_batch.append(loss)

        train_losses.append(train_losses_by_batch)
        if ep % 50 == 0 or ep == epochs:
            
            print(f"Epoch {ep:03d} | Train: {np.mean(train_losses[-1]):.4f}")
                
    return np.array(train_losses), model, debug_result_all

def parse_debug(debug_result, input_data_loader, epoch_num, visual_range):
    # additional quantifications using the output from train_rules_gnn

    W_input_epoch, W_output_epoch, W_difference_epoch = {}, {}, {}
    epochs = list(debug_result.keys())
    for epoch in [epochs[-1]]:
    
        W_across_batch = debug_result[epoch] #W_across_batch has all the batches

        (W_input_batch, W_output_batch, W_difference_batch) = ([], [], [])
        
        for batch_ind in W_across_batch.keys():

            # the input for each video
            position, species_idx = list(iter(input_data_loader))[batch_ind]
            N = position.shape[2]
            edge_index_input = build_edge_index(position[:,-1,:,:], visual_range = visual_range)

            # the output varies over frames
            F = len(W_across_batch[batch_ind]['W'])
            W_difference_batch_frame = []
            W_output_frame = []

            for f in range(F): #over frames
                (_, (edge_index_output, edge_weight_output)) = W_across_batch[batch_ind]['W'][f]
    

                W_input, W_output, W_difference = extract_adjacency_difference(
                    N, edge_index_input, None, edge_index_output, edge_weight_output)
                W_output_frame.append(W_output)
                W_difference_batch_frame.append(W_difference)
                
            W_input_batch.append(W_input)
            W_output_batch.append(W_output_frame)
            W_difference_batch.append(W_difference_batch_frame)
    
        print("finished one epoch")
        W_input_epoch[epoch] = W_input_batch
        W_output_epoch[epoch] = W_output_batch
        W_difference_epoch[epoch] = W_difference_batch

    return W_input_epoch, W_output_epoch, W_difference_epoch

## plotting 

def plot_log_loss(train_losses, labels = None,
                  alpha = 0.05, log_base=10, title = "Training Loss", ylabel = "MSE Loss (log scale)"):

    fig, ax = plt.subplots(1,1,figsize = (8,5))

    for i in range(len(train_losses)):

        # plot the mean line
        train_loss = train_losses[i]
        label = labels[i]

        # 
        x = np.arange(train_loss.shape[0])
        y_mean = np.mean(train_loss, axis = 1) #across datasets
        #y_std = np.std(train_loss, axis = 1) #across datasets
        ax.plot(x, y_mean, label = label, color='C'+str(i))

        #y_lower = y_mean - 2 * y_std
        #y_upper = y_mean + 2 * y_std
        y_lower = np.quantile(train_loss, 0.5 * alpha, axis = 1) #across datasets
        y_upper = np.quantile(train_loss, 1 - 0.5 * alpha, axis = 1) #across datasets

        # Shade the confidence interval
        if alpha is not None:
            ax.fill_between(x, y_lower, y_upper, alpha=0.4, color='C'+str(i))

    plt.yscale('log', base=log_base)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    #plt.show()

    return ax
