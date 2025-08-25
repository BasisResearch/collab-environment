"""
Each GNN model has 1) model spec 2) training spec.
Each dataset has data spec.

We usually load trained GNN and the test dataset, then rollout GNN prediction.
This script contains utility functions to load various datasets, and load models.
"""
TRAIN_SIZE = 0.7

import torch
import pickle
from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.utility import dataset2testloader

import io

class DeviceUnpickler(pickle.Unpickler):
    def __init__(self, file, device='cpu'):
        super().__init__(file)
        self.device = device
    
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
        else:
            return super().find_class(module, name)


def load_various_data(data_names, batch_size, device = 'cpu',return_dict = True):

    data = {}

    for data_name in data_names:
        file_name = f'{data_name}.pt'
        config_name = f'{data_name}_config.pt'
        
        dataset = torch.load(expand_path(
                "simulated_data/" + file_name, get_project_root()), weights_only = False)
        species_configs = torch.load(expand_path(
                "simulated_data/" + config_name, get_project_root()), weights_only = False)

        test_loader = dataset2testloader(
            dataset, batch_size=batch_size, device=device, train_size=TRAIN_SIZE
        )

        if len(data_names) == 1 and not return_dict:
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
                root_path = "trained_models",
                noise = 0,
                head = 1,
                visual_range = 0.1, seed = 0, rollout_starting_frame = 5):
    save_name_postfix = f"n{noise}_h{head}_vr{visual_range}_s{seed}"
    file_name = f"{data_name}_{model_name}_{save_name_postfix}"

    rollout_path = expand_path(
            f"{root_path}/{file_name}_rollout_{rollout_starting_frame}.pkl",
            get_project_root()
    )

    with open(rollout_path, "rb") as f: # 'wb' for write binary
        rollout_result = DeviceUnpickler(f, device='cpu').load()
    return rollout_result
