import torch
import torch.nn.functional as functional
import numpy as np

from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.gnn import debug_result2prediction
from collab_env.gnn.utility import handle_discrete_data

bird_num = 20

def rollout_to_pos_vel_acc(rollout_debug_result, starting_frame = 0, ending_frame = 50, subsample = 10):
    """given rollout result of positions, produce discrete velocity, acceleration"""
    pos_all_files = []
    vel_all_files = []
    acc_all_files = []
    
    for file_id in rollout_debug_result[0]:
        pos, vel, acc, pos_gnn, vel_gnn, acc_gnn, frame_sets = debug_result2prediction(
                            rollout_debug_result,
                            file_id = file_id, epoch_num = 0)
        
        pos_all_files.append(pos[0,starting_frame:ending_frame:subsample,:bird_num])
        vel_all_files.append(vel[0,starting_frame:ending_frame:subsample,:bird_num])
        acc_all_files.append(acc[0,starting_frame:ending_frame:subsample,:bird_num])
        
    pos_concatenated = torch.concatenate(pos_all_files)
    vel_concatenated = torch.concatenate(vel_all_files)
    acc_concatenated = torch.concatenate(acc_all_files)

    return pos_concatenated, vel_concatenated, acc_concatenated

def data_to_pos_vel_acc(loader, starting_frame = 0, ending_frame = 50, subsample = 10):
    """given a loader of Pytorch dataset, produce discrete velocity, acceleration"""

    pos_all_files = []
    vel_all_files = []
    acc_all_files = []
    
    for (position_gt, _) in loader:
        
        pos, vel, acc, v_function = handle_discrete_data(position_gt, "Euler")
        
        pos_all_files.append(pos[0,starting_frame:ending_frame:subsample,:bird_num])
        vel_all_files.append(vel[0,starting_frame:ending_frame:subsample,:bird_num])
        acc_all_files.append(acc[0,starting_frame:ending_frame:subsample,:bird_num])
    
    pos_concatenated = torch.concatenate(pos_all_files)
    vel_concatenated = torch.concatenate(vel_all_files)
    acc_concatenated = torch.concatenate(acc_all_files)

    return pos_concatenated, vel_concatenated, acc_concatenated

def return_deltapos_vnext(pos, vel, acc, threshold = 0.2):
    """Panel C helper"""
    pos = pos.squeeze()
    vel = vel.squeeze()
    acc = acc.squeeze()
    F, N, dim = pos.shape

    v_next = []
    pos_diff = []
    for f in range(F-1):
        for bi in range(N):
            # find the average position of flock
            setdiff = torch.from_numpy(np.setdiff1d(np.arange(N),bi))
            diff = torch.mean(pos[f,setdiff], axis = 0) - pos[f,bi] #across birds
            if torch.norm(diff) < threshold:
                pos_diff.append(diff)
                v_next.append(vel[f, bi])

    return pos_diff, v_next

def figure_data_C(test_loader, rollout_debug_result, model = False, starting_frame = 5, ending_frame = 50):
    """Panel C """
    if model:
        (pos_concatenated, vel_concatenated, acc_concatenated) = rollout_to_pos_vel_acc(
            rollout_debug_result, starting_frame, ending_frame)
    else:
        (pos_concatenated, vel_concatenated, acc_concatenated) = data_to_pos_vel_acc(test_loader,
                                                                                    starting_frame,
                                                                                    ending_frame)

    pos_diff, v_next = return_deltapos_vnext(pos_concatenated,
                                             vel_concatenated,
                                             acc_concatenated)
    return np.array(pos_diff), np.array(v_next)

def mean_traces_C(pos_diff, v_next):
    """Panel C helper"""
    
    bins_x = np.linspace(0,0.2,21) # Bin edge
    bins_y = np.linspace(0,0.2,21) # Bin edge
    
    indices_x = np.digitize(pos_diff[:,0], bins_x)
    indices_y = np.digitize(pos_diff[:,1], bins_y)

    indices_set_x = np.unique(indices_x)[:-1] # -1 due to the bins
    indices_set_y = np.unique(indices_y)[:-1] # -1 due to the bins
    
    data = []
    location = []
    sd = []

    for ind_x in indices_set_x:
        for ind_y in indices_set_y:
            ind_data = np.logical_and(indices_x==ind_x, indices_y==ind_y)
            if np.sum(ind_data) > 100:
                location.append( [bins_x[ind_x], bins_x[ind_y]] )
                data.append( [np.mean(v_next[ind_data,0]),np.mean(v_next[ind_data,1])]  )
                sd.append([np.std(v_next[ind_data,0]),np.std(v_next[ind_data,1])])
                
            
    location = np.array(location)
    data = np.array(data)
    sd = np.array(sd)
    return location, data, sd

def return_deltav_acc_flock(pos, vel, acc, threshold = 0.2):
    """
    Sister function of return_deltav_acc_singleton(),
    instead of comparing pairs of boids, this function 
        compares a boid and the mean of all the boids:
            for a boid of a boid and all the rest of boids:
            find del_v between the boid and the mean of the rest of the boids.
            and find acc of the boid.

    Have not optimized for vectorization yet.
    """
    pos = pos.squeeze()
    vel = vel.squeeze()
    acc = acc.squeeze()
    F, N, dim = pos.shape

    del_v = []
    acc_ = []
    for f in range(F):
        for bi in range(N):
            setdiff = torch.from_numpy(np.setdiff1d(np.arange(N),bi))
            diff_pos = torch.norm(torch.mean(pos[f,setdiff], axis = 0) - pos[f,bi]) #across birds
            diff_vel = torch.norm(torch.mean(vel[f,setdiff], axis = 0) - vel[f,bi]) #across birds

            if diff_pos < threshold:

                del_v.append(diff_vel)
                acc_.append(torch.norm(acc[f,bi]))
    return del_v, acc_

def return_deltav_acc_singleton(pos, vel, acc, threshold = 0.2):
    """
    Sister function of return_deltav_acc_singleton(),
    instead of comparing between a boid and all the rest of the boids, this function 
        compares a boid and another boid:
            for a boid of a boid and another boid:
            find del_v between them.
            and find acc of the boid.

    Have not optimized for vectorization yet.
    """
    pos = pos.squeeze()
    vel = vel.squeeze()
    acc = acc.squeeze()
    F, N, dim = pos.shape

    del_v = []
    acc_ = []
    for f in range(F):
        for bi in range(N):
            for bj in range(bi,N):
                diff_pos = torch.norm(pos[f,bj] - pos[f,bi]) #across birds
                diff_vel = torch.norm(vel[f,bj] - vel[f,bi]) #across birds

                if diff_pos < threshold:

                    del_v.append(diff_vel)
                    acc_.append(torch.norm(acc[f,bi]))
    return del_v, acc_

def figure_data_B(test_loader, rollout_debug_result, model = False, starting_frame = 5, ending_frame = 50,
        threshold = 0.1, version = "singleton"):

    if model:
        (pos_concatenated, vel_concatenated, acc_concatenated) = rollout_to_pos_vel_acc(
            rollout_debug_result, starting_frame, ending_frame)
    else:
        (pos_concatenated, vel_concatenated, acc_concatenated) = data_to_pos_vel_acc(test_loader,
                                                                                    starting_frame,
                                                                                    ending_frame)
    if "singleton" in version:
        del_v, acc_ = return_deltav_acc_singleton(pos_concatenated,
                                                vel_concatenated,
                                                acc_concatenated, threshold = threshold)
    else:
        del_v, acc_ = return_deltav_acc_flock(pos_concatenated,
                                                vel_concatenated,
                                                acc_concatenated, threshold = threshold)
    return del_v, acc_

def mean_trace_B(del_v, acc_):
    bins = np.linspace(0,0.05,10) # Bin edge
    indices = np.digitize(del_v, bins)
    indices_set = np.unique(indices)
    bins = bins[indices_set-1]
    
    mean = []
    sd = []
    for s in indices_set:
        mean.append( np.nanmean(np.array(acc_)[indices == s]) )
        sd.append( np.nanstd(np.array(acc_)[indices == s]) )
    mean = np.array(mean)
    sd = np.array(sd)

    return bins, mean, sd