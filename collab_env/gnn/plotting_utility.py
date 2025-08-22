"""
Each GNN model has 1) model spec 2) training spec.
Each dataset has data spec.

We usually load trained GNN and the test dataset, then rollout GNN prediction.
This script contains utility functions to load various datasets, and load models.
"""
import torch
import pickle
from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.utility import dataset2testloader

def load_various_data(data_names, batch_size):

    data = {}

    for data_name in data_names:
        file_name = f'{data_name}.pt'
        config_name = f'{data_name}_config.pt'
        
        dataset = torch.load(expand_path(
                "simulated_data/" + file_name, get_project_root()), weights_only = False)
        species_configs = torch.load(expand_path(
                "simulated_data/" + config_name, get_project_root()), weights_only = False)

        test_loader = dataset2testloader(dataset, batch_size = batch_size)

        if len(data_names) == 1:
            data["file_name"] = file_name
            data["config_name"] = config_name
            data["dataset"] = dataset
            data["species_configs"] = species_configs
            data['test_loader'] = test_loader
        else:
            data[data_name] = {}
            data[data_name]["file_name"] = file_name
            data[data_name]["config_name"] = config_name
            data[data_name]["dataset"] = dataset
            data[data_name]["species_configs"] = species_configs
            data[data_name]['test_loader'] = test_loader

    return data

def load_rollout(model_name,
                data_name = None,
                noise = 0,
                head = 1,
                visual_range = 0.1, seed = 0, rollout_starting_frame = 5):
    save_name_postfix = f"noise{noise}_head{head}_visual_range{visual_range}"
    file_name = f"{data_name}_{model_name}_{save_name_postfix}_seed{seed}"

    rollout_path = expand_path(
            f"trained_models/{file_name}_rollout_{rollout_starting_frame}.pkl",
            get_project_root()
    )

    
    with open(rollout_path, "rb") as f: # 'wb' for write binary
        rollout_result = pickle.load(f)
    return rollout_result
