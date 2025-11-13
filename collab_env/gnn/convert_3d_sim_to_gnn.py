import argparse
import shutil
import sys
import subprocess

import pickle
import yaml

import torch
import numpy as np

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.train import main as train_main
from collab_env.sim.boids.sim_mesh_dataset import SimMeshDataset
from collab_env.sim.boids.show_trajectories_all import show_trajectories


def create_species_config(folder_path, config_filename="config.yaml"):
    config_path = expand_path(config_filename, folder_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    height = depth = width = config["environment"]["box_size"]
    target_position = config["environment"]["target_position"][0]
    food_config = {
        "x": target_position[0],
        "y": target_position[1],
        "z": target_position[2],
        "counts": 1,
    }
    species_config = {
        "agent": {
            "visual_range": config["agent"]["neighborhood_dist"],
            "centering_factor": config["agent"]["cohesion_weight"],
            "min_distance": config["agent"]["min_separation"],
            "avoid_factor": config["agent"]["separation_weight"],
            "matching_factor": config["agent"]["alignment_weight"],
            "margin": config["agent"]["min_ground_separation"],
            "turn_factor": 4,
            "speed_limit": config["agent"]["max_speed"],
            "food_factor": config["agent"]["target_weight"][0],
            "food_visual_range": config["environment"]["box_size"],
            "food_eating_range": 20,
            "food_time": config["simulator"]["target_creation_time"][0],
            "hunger_threshold": -100,
            "counts": config["simulator"]["num_agents"],
            "width": width,
            "height": height,
            "depth": depth,
            "independent": config["agent"]["random_walk"],
            "food": food_config,
        },  # the food that species A can see
        "food0": food_config,
    }

    return species_config


def convert_sim_output_to_gnn_input(folder_name):
    folder_path = expand_path(
        #    f"{config['files']['trajectory_folder']}/{config['files']['trajectory_file']}",
        #    get_project_root(),
        folder_name,
        get_project_root(),
    )
    species_config = create_species_config(folder_path)
    print(species_config)

    dataset = SimMeshDataset(
        folder_path,
        species_config["agent"]["width"],
        species_config["agent"]["height"],
        species_config["agent"]["depth"],
    )

    # print(dataset.__getitem__(0))
    print(dataset.__getitem__(0)[0].shape)
    print(dataset.__getitem__(0)[1].shape)
    print(len(dataset))

    file_name = run_folder + "_food.pt"
    config_name = run_folder + "_food_config.pt"
    data_path = expand_path("simulated_data/runpod/" + file_name, get_project_root())

    torch.save(dataset, data_path)

    torch.save(
        species_config,
        expand_path("simulated_data/runpod/" + config_name, get_project_root()),
    )

    print(f"data saved to {data_path}")

    return data_path.name


def convert_gnn_rollout_to_sim_trajectory(full_path, box_size=1500):
    with open(full_path, "rb") as f:
        data = pickle.load(f)
    """
    TOC -- 11:39AM 
    The length of the first dimension is the number of episodes, which should be 1 for the rollout. 
    The length of the second dimension is the batch size, not sure why we would want this > 1 for a rollout. Though
    I guess we may want to look at multiple starting positions, so maybe this needs to be dealt with in a better way.
    The first three frames we need to take from the actual data since we start after that for predicted. The rest
    we can take from predicted. I suppose that also needs to be configurable, since we may have different time windows
    in the features given to the GNN. 
    
    Well that idea didn't work since the actual starts at frame 3 as well. Would need to change the GNN rollout code to
    give me the actual start frames that prime the model. 
    
    Not sure how I am dealing with the types. Maybe assume all are agents since I don't think it matters for 
    show_trajectory(). 
    
    Need to convert the predicted tensors into a dataframe. Are the predicted vectors grouped by time or agent? I 
    believe they are grouped by time step.      
    
    
    """
    print(f"len of data {len(data)}")
    print(f"len of data[0] {len(data[0])}")
    print("keys", data[0][0].keys())
    # print('keys', data[0][1].keys())
    print(f'len of d[0][0]["predicted"] {len(data[0][0]["predicted"])}')

    for batch_index in range(len(data[0])):
        df = pd.DataFrame()
        print("df \n", df)
        # actual = data[0][batch_index]['actual']
        print(f"batch_index {batch_index}")
        predicted = np.array(data[0][batch_index]["predicted"])
        print(f"predicted shape {predicted.shape}")

        predicted = np.squeeze(predicted, axis=1)
        predicted = np.clip(predicted * box_size, a_min=-box_size, a_max=box_size)

        T, N, L = predicted.shape
        print("shape ", predicted.shape)
        times = np.arange(T)
        ids = np.arange(1, N + 1)
        print("times ", times)
        print("ids ", ids)
        predicted_combined = predicted.reshape(T * N, L)
        print("combined shape ", predicted_combined.shape)
        time_col = np.repeat(times, N)
        id_col = np.tile(ids, T)
        print("time col ", time_col)
        print("id col ", id_col)
        df = pd.DataFrame(predicted_combined, columns=["x", "y", "z"])
        df["type"] = "agent"
        df.insert(0, "id", id_col)
        df.insert(0, "time", time_col)
        df.loc[df["id"] == N, "type"] = "env"

        print("df \n", df[df["time"] < 20])

        table = pa.Table.from_pandas(df)
        file_name = "sim-output/test-rollout/test-rollout.parquet"
        rollout_path = expand_path(file_name, get_project_root())
        pq.write_table(table, rollout_path)
        return file_name

        #
        # for i in range(len(predicted)):
        #     print('actual \n', actual[i] - predicted[i])
        #     print('predicted \n', predicted[i])
        #     print(f'{i} : {predicted[i].shape}')
        # num_start_frames = len(actual) - len(predicted)
        # print(f'num_start_frames {num_start_frames}')
        # full_trajectory = actual[:num_start_frames] + predicted
        # print(len(full_trajectory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="convert_3d_sim_to_gnn",
        description="Converts the 3D simulator output trajectories into the input file for the GNN, trains the GNN on those trajectories, performs a rollout, then converts the rollout trajectories into an input file for the 3D simulator to display the trajectories in the 3D environment",
        epilog="---",
    )
    parser.add_argument("-rf", "--run_folder")
    parser.add_argument(
        "-c",
        "--convert_from_sim",
        action="store_true",
        help="Convert the simulator parquet file to the GNN input file. Don't bother if this has already been done.",
    )
    parser.add_argument(
        "-t",
        "--train_gnn",
        action="store_true",
        help="Train the GNN. Don't bother if this completed successfully previously.",
    )
    parser.add_argument(
        "-r",
        "--rollout",
        action="store_true",
        help="Compute the rollouts from the GNN. Don't bother if this successfully previously.",
    )
    parser.add_argument(
        "-p",
        "--parquet",
        action="store_true",
        help="Create the simulator parquet files from the rollouts to be use to show trajectories in the 3D environment.",
    )
    parser.add_argument(
        "-s",
        "--show_trajectories",
        action="store_true",
        help="Show trajectories in the 3D environment.",
    )

    args, rest = parser.parse_known_args()
    """
    Need to get all of the args before calling train_main() because train_main() has a parser that I don't want to mess
    with but that messes with me. 
    """
    parquet = args.parquet
    show_tracks = args.show_trajectories
    rollout = args.rollout
    if args.run_folder:
        run_folder = args.run_folder
    else:
        run_folder = "test-rollout-multitype-started-20251015-161205"

    # filename = sys.argv[1a]
    # run_folder = 'hackathon-homogenous-2.0-200-20-0.0-0.0-sim_run-started-20251002-142914'
    # run_folder = 'test-rollout-multitype-started-20251015-161205'
    folder_name = "sim-output/" + run_folder

    input_file_name = convert_sim_output_to_gnn_input(folder_name)
    input_file_name = input_file_name.split(".")[0]
    print("input_file_name", input_file_name)
    best_model_output = None
    if args.train_gnn:
        print("training gnn")
        best_model = train_main(dataset_arg=input_file_name, remaining_args=rest)
        # result = subprocess.run([sys.executable, 'collab_env/gnn/train.py', '--dataset', input_file_name])
        print("gnn training returned ", best_model)
        best_model_output = best_model["model_output"]

    if rollout:
        print(f"rollout gnn on {input_file_name}")
        result = subprocess.run(
            [
                sys.executable,
                "collab_env/gnn/train.py",
                "--dataset",
                input_file_name,
                "--rollout",
                "1",
                "--total_rollout",
                "3000",
                "--batch-size",
                "1",
            ],
            check=True,
        )
        print("gnn rollout returned ", result)

    # full_path = "/Users/tc/ArchivedBoxSync/Research/Basis/collab-env-stuff/collab-environment/trained_models/runpod/hackathon-homogenous-2.0-200-20-0.0-0.0-sim_run-started-20251002-142914_food/rollouts/hackathon-homogenous-2.0-200-20-0.0-0.0-sim_run-started-20251002-142914_food_vpluspplus_a_n0.005_h3_vr0.5_s4_rollout1.pkl"

    if parquet:
        if best_model_output is None:
            model_name = (
                run_folder + "_food_vpluspplus_a_n0.005_h3_vr0.5_s4_rollout1.pkl"
            )
        else:
            model_name = (
                best_model_output.split("/")[-1].split(".pt")[0] + "_rollout1.pkl"
            )
        # full_path = expand_path('trained_models/runpod/' + input_file_name + "/rollouts/" + run_folder + '_food_vpluspplus_a_n0.005_h3_vr0.5_s4_rollout1.pkl', get_project_root())
        full_path = expand_path(
            "trained_models/runpod/" + run_folder + "_food/rollouts/" + model_name,
            get_project_root(),
        )
        # full_path = "/Users/tc/ArchivedBoxSync/Research/Basis/collab-env-stuff/collab-environment/trained_models/runpod/hackathon-homogenous-2.0-200-20-0.0-0.0-sim_run-started-20251002-142914_food/rollouts/hackathon-homogenous-2.0-200-20-0.0-0.0-sim_run-started-20251002-142914_food_vpluspplus_a_n0.005_h3_vr0.5_s4_rollout1.pkl"
        # full_path.parent.parent.parent.mkdir(exist_ok=True)
        # full_path.parent.parent.mkdir(exist_ok=True)
        # full_path.parent.mkdir(exist_ok=True)
        print(f"converting {full_path} to parquet")
        parquet_file_name = convert_gnn_rollout_to_sim_trajectory(full_path)

    if show_tracks:
        print(f"copying {parquet_file_name} to {folder_name}")
        shutil.copy(parquet_file_name, folder_name)
        show_trajectories(
            config_filename=folder_name + "/config.yaml",
            trajectory_file_name=parquet_file_name.split("/")[-1],
            trajectory_directory_name=folder_name,
            show_visualizer=True,
        )
