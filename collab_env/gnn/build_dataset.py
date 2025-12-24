import argparse
from pathlib import Path

# from typing import Tuple
import numpy as np
import yaml
import torch
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from collab_env.data.file_utils import expand_path, get_project_root


def convert_pandas_to_node_features(df: pd.DataFrame):
    agents_df = df.copy()
    """
    TOC -- 103025 
    Only dealing with the first target mesh right now. Need to fix this when there are more meshes -- though
    we may want to change the format of the dataframe for multiple targets, not sure. (Update the simulator to
    output as a list rather than subscripting _1, not sure why I did that.)
    """
    # print(f'target_mesh_closest_point_1 to list {len(agents_df["target_mesh_closest_point_1"].to_list())}')
    # s = agents_df ["target_mesh_closest_point_1"]
    # print(s.isnull().sum())  # how many nulls
    # print([i for i, x in enumerate(s) if x is None][:10])  # indices of None
    # print([type(x) for x in s.head(20)])  # sample types

    agents_df[
        [
            "x",
            "y",
            "z",
        ]
    ] = pd.DataFrame(agents_df["position"].to_list(), index=agents_df.index)

    # agents_df[
    #     [
    #         "v_x",
    #         "v_y",
    #         "v_z",
    #     ]
    # ] = pd.DataFrame(agents_df["velocity"].to_list(), index=agents_df.index)

    agents_df[
        [
            "target_mesh_closest_point_x",
            "target_mesh_closest_point_y",
            "target_mesh_closest_point_z",
        ]
    ] = pd.DataFrame(
        agents_df["target_mesh_closest_point_1"].to_list(), index=agents_df.index
    )
    # print('target mesh closest point 1 \n', agents_df["target_mesh_closest_point_1"].to_list())
    agents_df[
        [
            "mesh_scene_closest_point_x",
            "mesh_scene_closest_point_y",
            "mesh_scene_closest_point_z",
        ]
    ] = pd.DataFrame(
        agents_df["mesh_scene_closest_point"].to_list(), index=agents_df.index
    )

    # agents_df[
    #     [
    #         "distance_to_target_mesh_closest_point_1",
    #     ]
    # ] = pd.DataFrame(
    #     np.array(agents_df["distance_to_target_mesh_closest_points"].to_list())[:, 0],
    #     index=agents_df.index,
    # )
    # print('mesh dist:', df[['time', 'id', 'mesh_scene_distance', 'mesh_scene_closest_point_x', 'mesh_scene_closest_point_y','mesh_scene_closest_point_z']])
    groups = agents_df.groupby("time")[
        [
            "x",
            "y",
            "z",
            "v_x",
            "v_y",
            "v_z",
            "distance_to_target_mesh_closest_point_1",
            "target_mesh_closest_point_x",
            "target_mesh_closest_point_y",
            "target_mesh_closest_point_z",
            "mesh_scene_distance",
            "mesh_scene_closest_point_x",
            "mesh_scene_closest_point_y",
            "mesh_scene_closest_point_z",
        ]
    ]

    #
    # Convert everything to torch tensors
    #
    node_feature_groups = groups.apply(
        lambda g: torch.tensor(g.to_numpy(), dtype=torch.float32)
    )

    #
    # Stack everything into a tensor with shape (num frames, num agents, num input parameters)
    #
    node_features = torch.stack(node_feature_groups.to_list())

    return node_features


def compute_positions(agents_df: pd.DataFrame):
    #
    # Reshape the dataframe so the index is time, the column is id and the value
    # is the position vector.
    #

    agents_df["position"] = agents_df[["x", "y", "z"]].values.tolist()
    agents_df["position"] = agents_df[["x", "y", "z"]].values.tolist()
    pivot = agents_df.pivot(index="time", columns="id", values="position")
    # pivot = agents_df.pivot(index="time", columns="id", values=["x", "y", "z"])
    print("pivot\n", pivot)

    #
    # convert into a np array of shape (num time steps, num agents, dimension of world)
    #
    positions = torch.from_numpy(
        np.stack(pivot.values.flatten()).reshape(pivot.shape + (-1,))
    ).float()

    #
    # Compute relative positions for each time step and each agent. If we have a tensor called positions of
    # shape (num time steps, num agents, 3), we should get a tensor relative_positions of shape
    # (num time steps, num agents, num agents, 3), where relative_positions[t, i, j] is the position of
    # agent j relative to agent i at time t.
    #
    relative_positions = positions[:, None, :, :] - positions[:, :, None, :]

    return positions, relative_positions


def complete_graph_edge_attributes(relative_position, source, destination, device=None):
    """

    Args:
        relative_position (torch.Tensor): the relative positions of the agents. This should have shape
            (num agents, num agents, dimension of physical space) with relative_position[i,j] = pos_j - pos_i (i.e.,
            the position of agent j with respect to agent i).
        device (str, optional): the device to use. Defaults to None.

    Returns:
        edge_index (torch.LongTensor): the edge index for the graph.
        edge_attr (torch.FloatTensor): the attributes of the edges. Attribute for directed edge (i,j) will be relative_position[i,j].

    need comments to explain how this works
    """
    # rel_pos: torch.Tensor shape (N, N, D) for a single time step
    # returns Data with edge_index (2, E) and edge_attr (E, D)
    relative_position_on_device = relative_position.to(device=device)
    # num_agents = relative_position_on_device.shape[0]

    # idx = torch.arange(num_agents, device=device)
    # src, dst = torch.meshgrid(idx, idx, indexing='ij')   # shape (num_agents, num_agents)
    #
    # src = src.flatten()
    # dst = dst.flatten()

    # edge_index = torch.stack([src, dst], dim=0)          # (2, num_agents**2)
    edge_attr = relative_position_on_device[
        source, destination
    ]  # (E, D) via advanced indexing

    return edge_attr


class Sim3DInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """

        Args:
            root (string): full path to dataset directory -- this is used to set raw_dir, which is needed
                            in super.__init__()
            transform ():
            pre_transform ():
        """
        self.root_path = expand_path(root, get_project_root())
        root = str(self.root_path)

        super(Sim3DInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.sim_data_folder_name = root
        self.episodes = self.load_episodes()
        # print('episodes:', self.episodes)
        # print(type(self.raw_dir))

    @property
    def raw_file_names(self):
        # List of the raw files
        # there is no download necessary since we are working with local files.
        return ["test"]

    @property
    def processed_file_names(self):
        return ["sim3D_gnn_data"]

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        pass
        # path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # The zip file is removed
        # os.unlink(path)

    def get_filename_for_saved_episode(self, episode_number):
        filename = f"{self.processed_paths[0]}_episode_{episode_number}.pt"
        return filename

    def process(self):
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.

        # Get the raw data directory as a Path object
        raw_dir_path = Path(self.raw_dir)

        # Load the config file for the simulator run that generated the data
        # The name is fixed as config.yaml in the output directory for the simulator.
        config = yaml.safe_load(open(expand_path("../config.yaml", raw_dir_path)))
        num_agents = config["simulator"]["num_agents"]
        num_time_steps = config["simulator"]["num_frames"] + 1  # add 1 for time step 0
        box_size = config["environment"]["box_size"]

        # Process each episode file in the directory.
        episode_file_list = list(raw_dir_path.glob("../episode*.parquet"))
        episode_number = 0
        for episode_file in tqdm(episode_file_list):
            episode_number += 1
            trajectory_path = expand_path(episode_file, raw_dir_path)
            """
            -- 101325 4:03PM
            TODO: Change this to read only the columns we need. 
            """
            df = pq.read_pandas(trajectory_path).to_pandas()
            agents_df = df[df["type"] == "agent"]

            #
            # Get the positions and relative positions of the agents.
            #
            positions, relative_positions = compute_positions(agents_df)

            #
            # Convert data frame to nodes features for this episode
            #
            node_features = convert_pandas_to_node_features(agents_df)
            # print('node features\n', node_features)
            #
            # Scale all the coordinates and distances by the size of the box (assumes width = height = depth)
            #
            node_features = node_features / torch.tensor(
                [box_size], dtype=torch.float32
            )
            labels = positions / torch.tensor([box_size], dtype=torch.float32)
            # print('labels\n', labels)
            # assert False, 'process'

            """
            TOC -- 121925 11:21AM
            For the first pass, create a homogeneous graph where there are only agent nodes and all of the 
            node features are included in the agent. (Debatable as to whether the positions of the target and
            mesh scene should be relative.)
            """

            """
            TOC -- 121925 10:18PM
            edge index is now computed with edge attributes in function
            """
            #
            # Create edge indices for a complete graph with self loops.
            #
            from_nodes = torch.arange(num_agents).repeat_interleave(
                num_agents
            )  # sources
            to_nodes = torch.arange(num_agents).repeat(num_agents)  # targets
            edge_index = torch.stack(
                [from_nodes, to_nodes], dim=0
            )  # shape [2, num_agents**2]

            #
            # Add relative positions for each time step and each agent as the edge attributes.
            # Create the data object for this episode and add it to the data list.
            # We loop to num_time_steps - 1 because the last graph in the sequence has nothing to predict.
            #
            data_list = [
                Data(
                    x=node_features[t],
                    y=labels[t + 1],  # labels are the position at the next time step
                    edge_index=edge_index,
                    edge_attr=complete_graph_edge_attributes(
                        relative_positions[t], from_nodes, to_nodes
                    ),
                )
                for t in range(num_time_steps - 1)
            ]

            # Store the processed data
            # data, slices = self.collate(data_list)
            # print("saving to ", self.get_filename_for_saved_episode(episode_number))

            torch.save(
                # (data, slices), self.get_filename_for_saved_episode(episode_number)
                data_list,
                self.get_filename_for_saved_episode(episode_number),
            )

        # indicate that processing is complete.
        with open(self.processed_paths[0], "w") as _:
            pass  # create an empty file

    def load_episodes(self):
        episode_file_list = list(self.root_path.glob("processed/*episode*.pt"))
        num_episodes = len(episode_file_list)
        print(f"loading {num_episodes} episodes")
        episodes = [
            torch.load(
                f"{self.processed_dir}/{self.processed_file_names[0]}_episode_{episode_number}.pt",
                weights_only=False,
            )
            for episode_number in tqdm(range(1, num_episodes + 1))
        ]
        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        return self.episodes[index]


if __name__ == "__main__":
    #
    # Get the config file name if specified on the command line
    #
    parser = argparse.ArgumentParser(
        prog="build_dataset.py",
        description="Builds a graph dataset from 3D simulation data.",
        epilog="---",
    )
    parser.add_argument("-d", "--directory", type=str)
    args = parser.parse_args()
    if not args.directory:
        print("Specify directory as command-line argument -d.")
        exit("1")

    dataset = Sim3DInMemoryDataset(args.directory)
    print("dataset length = ", len(dataset))

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    print("checking data from loader")
    for episode_number, episode in enumerate(loader):
        print(f"episode {episode_number} length {len(episode)}")
        for graph in episode:
            print("first graph: \n", graph)
            break
            # print('graph.x: \n', graph.x)
            # print('-' * a10)
