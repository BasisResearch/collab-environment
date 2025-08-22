import pandas as pd
from collab_env.gnn.utility import dataset2testloader
from collab_env.data.file_utils import expand_path, get_project_root

data_name = 'boid_single_species_basic'

def data2csv(data_name):
    file_name = f'{data_name}.pt'
    
    dataset = torch.load(expand_path(
            "simulated_data/" + file_name, get_project_root()), weights_only = False)

    test_loader = dataset2testloader(dataset)

    pos, species = list(test_loader)[0]
    pos = pos.numpy().squeeze()
    
    pos_pd_x = pd.DataFrame(pos[:,:,0].squeeze())
    save_path = expand_path(
                "simulated_data/" + file_name + '_x.csv', get_project_root())
    pos_pd_x.to_csv(save_path, index = False)
    
    
    pos_pd_y = pd.DataFrame(pos[:,:,1].squeeze())
    save_path = expand_path(
                "simulated_data/" + file_name + '_y.csv', get_project_root())
    pos_pd_y.to_csv(save_path, index = False)

data_names = ['boid_single_species_basic', 'boid_single_species_independent', # without food
              'boid_food_basic_alignment', 'boid_food_basic_independent', # with food
              'boid_food_strong'] # with food, strong influence
for data_name in data_names:
    data2csv(data_name)