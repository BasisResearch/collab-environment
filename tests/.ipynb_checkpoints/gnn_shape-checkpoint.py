"""Tests if GNN objects initializes correctly"""
import os
import sys

script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
sys.path.append("/workspace/collab-environment/")
sys.path.append("/workspace/collab-environment/collab_env/gnn")
sys.path.append("/workspace/collab-environment/collab_env/data/boids")

import numpy as np
import torch

from collab_env.gnn.gnn import node_feature_vel
from collab_env.gnn.gnn_definition import GNN, Lazy

output_dim = 3
gnn_model1  = GNN(     model_name = "bogus1",
                        node_feature_function = "vel",
                        node_prediction = "acc",
                        input_differentiation = "finite",
                        prediction_integration = "Euler",
                        start_frame = 1,
                        heads = 1, in_node_dim = 7, hidden_dim = 128, output_dim = output_dim)

assert gnn_model1.out.in_features == 128
assert gnn_model1.out.out_features == 3

# run the model forward
edge_index = np.array([[1,2,3,4,5],[6,7,8,9,10]])
edge_weight = np.random.rand(5)

bird_num = 21
frame_num = 100
dim = 3
bogus_v = torch.rand(1,frame_num,bird_num,dim) #1 file, 100 frames, 21 boids, dim = 3
bogus_p = torch.rand(1,frame_num,bird_num,dim)
bogus_a = torch.rand(1,frame_num,bird_num,dim)
species_idx = torch.zeros(bird_num,dtype=torch.int64)
species_dim = 1

x_input = node_feature_vel(bogus_p,bogus_v,bogus_a,species_idx,species_dim)

pred, W = gnn_model1(x_input, torch.tensor(edge_index), torch.tensor(edge_weight))
assert pred.shape == (bird_num, dim)