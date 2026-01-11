import argparse
import json
import re
from typing import Optional

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


def convert_dataframe_to_node_features_old(
    df: pd.DataFrame, columns=None, time_window_length=1
):
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

    # agents_df[
    #     [
    #         "x",
    #         "y",
    #         "z",
    #     ]
    # ] = pd.DataFrame(agents_df["position"].to_list(), index=agents_df.index)

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
    if columns is None:
        groups = agents_df.groupby("time")[
            # columns = [
            # "x",
            # "y",
            # "z",
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

    else:
        groups = agents_df.groupby("time")[columns]

    # print('agents_df\n', agents_df[["x", "y", "z"]])
    #
    # Convert everything to torch tensors
    #

    # print('groups head  ', groups.head())
    #
    node_feature_groups = groups.apply(
        lambda g: torch.tensor(g.to_numpy(), dtype=torch.float32)
    )
    #
    # print("node feature group shape ", node_feature_groups.shape)
    # print("node feature group [0] ", node_feature_groups[0])

    #
    # Stack everything into a tensor with shape (num frames, num agents, num input parameters)
    #
    node_features = torch.stack(node_feature_groups.to_list())
    # print("node features shape ", node_features.shape)

    return node_features


def convert_dataframe_to_node_features(
    df: pd.DataFrame, columns=None, time_window_length=1
):
    """
    Args:
     df (pd.DataFrame): the dataframe containing a description of the agents in the environment
     columns (): the columns in the dataframe to include in the node features
     time_window_length (int): the time window we want to include in the node features.
    """

    """
    TOC -- 011026
    This should be changed to take a full dataframe rather than just the agent rows. We may want to create graphs that
    include information about the targets and mesh directly rather than only using those for the relative positions in
    the agent node features.   
    """
    agents_df = df.copy()
    """
    TOC -- 103025 
    Only dealing with the first target mesh right now. Need to fix this when there are more meshes -- though
    we may want to change the format of the dataframe for multiple targets, not sure. (Update the simulator to
    output as a list rather than subscripting _1, not sure why I did that.)
    
    TOC -- 011026 
    I did that because parquet can handle 2-dimensional lists in a dataframe. 
    """

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

    if columns is None:
        # groups = agents_df.groupby("time")[
        columns = [
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

    #
    # Create a pivot tabel indexed by time. The agent id will be the columns, and the values in each column will
    # be the columns for that agent id. For example, with time 0 and 1, agents 1 and 2, and columns x and y, the
    # pivot table will look something like this:
    #        agent 1             agent 2
    #       x       y           x      y
    # 0    0.4     0.7         0.5    0.6
    # 1    0.3     0.1         0.2    0.9
    #
    pivot = agents_df.pivot(index="time", columns="id", values=columns)
    num_time_steps = pivot.index.size
    num_agents = len(pivot.columns.levels[1])

    # Convert the pivot table to a numpy array with shape (num time steps, num agents, num columns). For
    # the example above, this will give us:
    #        agent 1             agent 2
    #       x       y           x      y
    # 0    0.1     0.2         0.3    0.4
    # 1    1.1     1.2         1.3    1.4
    # 2    2.1     2.2         2.3    2.4
    #
    # [ [  [0.1, 0.2], [0.3, 0.4] ], [ [1.1, 1.2], [1.3, 1.4] ] , [ [2.1, 2.2], [2.3, 2.4] ] ]
    feature_array = pivot.to_numpy().reshape(num_time_steps, num_agents, len(columns))
    # print("feature array shape: ", feature_array.shape)

    # Stack the features into time windows. The resulting shape will be:
    # (time steps - window length, agents, window length, columns).
    # For the example above with a time window of length 2, this should be:
    # [
    # t = 0: [
    #         agent 0: [
    #                   window 0: [0.1, 0.2]
    #                   window 1: [1.1, 1.2]
    #                  ],
    #         agent 1: [
    #                   window 0: [0.3, 0.4],
    #                   window 1: [1.3, 1.4],
    #                  ],
    #       ]
    # t = 1: [
    #         agent 0: [
    #                   window 0: [1.1, 1.2],
    #                   window 1: [2.1, 2.2],
    #                  ],
    #         agent 1: [
    #                   window 0: [1.3, 1.4],
    #                   window 1: [2.3, 2.4],
    #                  ],
    #       ]
    # ]
    stacked_features = np.stack(
        [  # make sure we have at least one time step in the stack.
            feature_array[t : t + 1 + num_time_steps - time_window_length, :, :]
            for t in range(time_window_length)
        ],
        axis=2,
    )
    # print(
    #     "new convert(): before reshape stacked features shape = ",
    #     stacked_features.shape,
    # )

    # Reshape the stacked features to flatten the time windows by columns into a single dimension of node features
    # for each agent in each time step. For the example above, this should be:
    # [
    # t = 0: [
    #         agent 0: [0.1, 0.2, 1.1, 1.2],
    #         agent 1: [0.3, 0.4, 1.3, 1.4],
    #        ]
    # t = 1: [
    #         agent 0: [1.1, 1.2, 2.1, 2.2],
    #         agent 1: [1.3, 1.4, 2.3, 2.4],
    #        ]
    # ]
    node_features = stacked_features.reshape(
        num_time_steps - time_window_length + 1,
        num_agents,
        time_window_length
        * len(columns),  # one column for each time step in the window
    )
    # print("new convert(): after reshape node features shape = ", node_features.shape)
    return torch.from_numpy(node_features)


def compute_positions(agents_df: pd.DataFrame):
    """
    Args:
    """
    #
    # Reshape the dataframe so the index is time, the column is id and the value
    # is the position vector.
    #
    agents_df = agents_df.copy()
    agents_df.loc[:, "position"] = agents_df[["x", "y", "z"]].values.tolist()
    agents_df.loc[:, "velocity"] = agents_df[["v_x", "v_y", "v_z"]].values.tolist()

    # print('compute_positions(): agents_df\n', agents_df)
    pivot = agents_df.pivot(index="time", columns="id", values=["position", "velocity"])
    # pivot = agents_df.pivot(index="time", columns="id", values=["x", "y", "z"])
    # print("pivot\n", pivot)

    #
    # convert into a np array of shape (num time steps, num agents, dimension of world)
    #
    positions = torch.from_numpy(
        np.stack(pivot["position"].values.flatten()).reshape(
            pivot["position"].shape + (-1,)
        )
    ).float()

    velocities = torch.from_numpy(
        np.stack(pivot["velocity"].values.flatten()).reshape(
            pivot["velocity"].shape + (-1,)
        )
    ).float()

    #
    #
    # Compute relative positions for each time step and each agent. If we have a tensor called
    # positions of shape (num time steps, num agents, 3), we should get a tensor relative_positions
    # of shape (num time steps, num agents, num agents, 3), where relative_positions[t, i, j] is
    # the position of agent j relative to agent i at time t.
    #
    relative_positions = positions[:, None, :, :] - positions[:, :, None, :]

    return positions, relative_positions, velocities


def edge_attributes_for_complete_graph(
    relative_position, source, destination, device=None
):
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


def compute_data_list_from_dataframe(
    agents_df: pd.DataFrame,
    num_time_steps: int,  # do we need this or should we compute it from DataFrame?
    num_agents: int,
    box_size: float,
    label_offset: int = 1,
    node_feature_columns=None,
    label_type: Optional[str] = None,
    time_window_length: int = 1,
):
    #
    # Get the positions and relative positions of the agents.
    #
    positions, relative_positions, velocities = compute_positions(agents_df)

    #
    # Convert data frame to nodes features for this episode
    # (must be called after compute positions because that function
    # sets up the position column -- I don't like this)
    #
    node_features = convert_dataframe_to_node_features(
        agents_df, node_feature_columns, time_window_length=time_window_length
    )
    # print(
    #     "compute data list from dataframe(): node feature shape: ", node_features.shape
    # )
    # print("compute data list from dataframe(): num_time_steps: ", num_time_steps)

    #
    # Scale all the coordinates and distances by the size of the box (assumes width = height = depth)
    #
    node_features = node_features / torch.tensor([box_size], dtype=torch.float32)

    # print("node features after division\n", node_features)
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
    from_nodes = torch.arange(num_agents).repeat_interleave(num_agents)  # sources
    to_nodes = torch.arange(num_agents).repeat(num_agents)  # targets
    edge_index = torch.stack([from_nodes, to_nodes], dim=0)  # shape [2, num_agents**2]

    # the labels will be the positions of the agent at the next time step.
    if label_type is not None and label_type.lower() == "velocities":
        labels = velocities / torch.tensor([box_size], dtype=torch.float32)
    else:
        labels = positions / torch.tensor([box_size], dtype=torch.float32)
    # print('labels\n', labels)
    # assert False, 'process'

    #
    # Add relative positions for each time step and each agent as the edge attributes.
    # Create the data object for this episode and add it to the data list.
    # We loop to num_time_steps - 1 because the last graph in the sequence has nothing to predict.
    #
    # print("compute data list(): node features shape ", node_features.shape)
    # print("compute data list(): node feature[0] shape ", node_features[0].shape)
    # print('compute data list(): num_time_steps: ', num_time_steps)
    # print('compute data list(): upper range: ', num_time_steps - label_offset - (time_window_length - 1))
    # The number of time windows will be num_time_steps - label_offset - time_window_length + 1
    # proof:
    # for notational convenience let n = num_time_steps, k = label_offset, w = time_window_length
    # Since we have to predict k steps out, the last time step that can end a window is the (n-k)th time step. That
    # window has to start w-1 time steps before that, which is the (n-k-w+1)-th time step. If you prefer pictures with
    # indices:
    # |      first window     |     |        last window            |           | label |
    # | 0 | 1 | 2 | ... | w-1 | ... | n-k-w | n-k-w+1 | ... | n-k-1 | n-k | ... | n-1   |
    # since the first window starts at 0 and the last window starts at n-k-w, there are n-k-w+1 windows.
    #
    data_list = [
        Data(
            x=node_features[t],
            y=labels[
                t + label_offset
            ],  # labels are whatever we are predicting label_offset time steps into the future
            edge_index=edge_index,
            edge_attr=edge_attributes_for_complete_graph(
                relative_positions[t], from_nodes, to_nodes
            ),
        )
        for t in range(
            #    time_window_length - 1,
            num_time_steps - label_offset - (time_window_length - 1),
        )  # don't go beyond the labels.
    ]
    # print('compute datalist from dataframw(): len data list ', len(data_list))
    return data_list


def sort_using_numbers(path):
    """
    Creates a list of substrings in the file name that separates out numeric parts so that the filenames can
    be sorted by episode number.
    Args:
        path(Path): the path of the file

    """
    parts = re.split(r"(\d+.)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class Sim3DInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        node_feature_columns=None,
        load_only=False,
        data_directory_name=None,
        labels: Optional[str] = None,
        time_window_length: int = 1,
    ):
        """

        Args:
            root (string): full path to dataset directory -- this is used to set raw_dir, which is needed
                            in super.__init__()
            transform ():
            pre_transform ():
        """
        self.root_path = expand_path(root, get_project_root())
        root = str(self.root_path)
        self.node_feature_columns = node_feature_columns

        if load_only:
            process_indicator_path = expand_path(
                "processed/" + self.processed_file_names[0], self.root_path
            )
            if not process_indicator_path.exists():
                raise FileNotFoundError(
                    f"Dataset constructor called in load only mode but path {process_indicator_path} doesn't exists."
                )

        if data_directory_name is not None:
            self.sim_data_folder_path = expand_path(
                data_directory_name, get_project_root()
            )
            if self.root_path.exists():
                raise FileExistsError(
                    f"Output directory {self.root_path.name} already exists. I don't want to overwrite it, so do something about that."
                )
            self.root_path.mkdir(parents=True, exist_ok=False)
            self.link_to_data = True
        else:
            self.sim_data_folder_path = self.root_path
            self.link_to_data = False

        self.label_type = labels
        self.time_window_length = time_window_length

        super(Sim3DInMemoryDataset, self).__init__(root, transform, pre_transform)

        self.episode_file_list = None  # this is created in load_episodes()
        self._input_node_dim = None
        self.episodes = self.load_episodes()
        # print('self.episodes', self.episodes)
        self._input_node_dim = self.episodes[0][0].x.shape[1]
        self._edge_attr_dim = self.episodes[0][0].edge_attr.shape[1]
        self._label_dim = self.episodes[0][0].y.shape[1]
        # print('episodes:', self.episodes)
        # print(type(self.raw_dir))

    @property
    def raw_file_names(self):
        # List of the raw files
        # there is no download necessary since we are working with local files.
        return ["link_to_raw_data"]

    @property
    def processed_file_names(self):
        return ["gnn3D_data"]

    def download(self):
        """
        Gets called if raw_file_names do not exist in raw directory. If the link to data flag is set in the
        constructor, we will create a symbolic link in the raw directory to point to the data directory. This
        may not work in Windows -- silly Windows.
        """
        if self.link_to_data:
            data_link = expand_path("raw/" + self.raw_file_names[0], self.root_path)
            data_link.symlink_to(self.sim_data_folder_path, target_is_directory=True)

    def get_filename_for_saved_episode(self, episode_file_list, episode_number):
        episode_file_name = episode_file_list[episode_number].name.split(".parquet")[0]
        # filename = f"{self.processed_paths[0]}_episode_{episode_number}.pt"
        filename = f"{self.processed_paths[0]}_{episode_file_name}.pt"
        return filename

    def process(self):
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.

        # Load the config file for the simulator run that generated the data
        # The name is fixed as config.yaml in the output directory for the simulator.
        config = yaml.safe_load(
            open(expand_path("config.yaml", self.sim_data_folder_path))
        )

        # Process each episode file in the directory.
        # We need these sorted by episode number to match the dataset indices to the episodes. (Maybe
        # this should be done in a more robust way, by keeping a mapping of indices to file names since
        # there is nothing that forces the episodes to have the current format.)

        episode_file_list = sorted(
            list(self.sim_data_folder_path.glob("episode*.parquet")),
            key=sort_using_numbers,
        )
        # print("len(episode_file_list)", len(episode_file_list))
        for episode_number, episode_file in enumerate(tqdm(episode_file_list)):
            # print('processing episode', episode_number, ' episode file', episode_file)
            trajectory_path = expand_path(episode_file, self.sim_data_folder_path)
            """
            -- 101325 4:03PM
            TODO: Change this to read only the columns we need. 
            """
            df = pq.read_pandas(trajectory_path).to_pandas()
            # need a copy because we mess with positions in compute_positions().
            agents_df = df[df["type"] == "agent"].copy()

            data_list = compute_data_list_from_dataframe(
                agents_df,
                num_agents=config["simulator"]["num_agents"],
                # add 1 to num_frames for time step 0
                num_time_steps=config["simulator"]["num_frames"] + 1,
                box_size=config["environment"]["box_size"],
                node_feature_columns=self.node_feature_columns,
                label_type=self.label_type,
                time_window_length=self.time_window_length,
            )

            # Store the processed data
            # print("saving to ", self.get_filename_for_saved_episode(episode_file_list, episode_number))
            torch.save(
                data_list,
                self.get_filename_for_saved_episode(episode_file_list, episode_number),
            )

        # indicate that processing is complete by creating the indicator file
        dataset_metadata = {
            "node_feature_columns": self.node_feature_columns,
            "time_window_length": self.time_window_length,
        }

        with open(self.processed_paths[0], "w", encoding="utf-8") as f:
            f.write(json.dumps(dataset_metadata, ensure_ascii=False))

        # torch.save(dataset_metadata, self.processed_paths[0])
        # print("saved to ", self.processed_paths[0])
        # with open(self.processed_paths[0], "w") as f:
        #     f.write(dataset_data)

    def load_episodes(self):
        self.episode_file_list = sorted(
            list(self.root_path.glob("processed/*episode*.pt")),
            key=sort_using_numbers,
        )
        # episode_file_list = list(self.root_path.glob("processed/*episode*.pt"))
        num_episodes = len(self.episode_file_list)
        print(f"loading {num_episodes} episodes")
        episodes = [
            torch.load(
                episode_file,
                weights_only=False,
            )
            for episode_file in tqdm(self.episode_file_list, leave=True)
        ]
        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        """
        Return the episode and the index so the training code can figure out what episode it is
        being trained on. This makes it easier to correctly compare the predicted trajectories
        and the actual trajectories when analyzing the results.
        """
        return self.episodes[index], index

    @property
    def input_node_dim(self):
        return self._input_node_dim

    @property
    def label_dim(self):
        return self._label_dim

    @property
    def edge_attr_dim(self):
        return self._edge_attr_dim


def csv_strings(arg):
    return [x for x in arg.split(",")]


if __name__ == "__main__":
    #
    # Get the config file name if specified on the command line
    #
    parser = argparse.ArgumentParser(
        prog="build_dataset.py",
        description="Builds a graph dataset from 3D simulation data.",
        epilog="---",
    )
    parser.add_argument("-d", "--directory", type=str, required=True)
    parser.add_argument("-id", "--input_data_directory", type=str, default=None)
    parser.add_argument(
        "-lv", "--use_velocities_as_labels", action="store_true", default=False
    )
    parser.add_argument(
        "-nfc", "--node_feature_columns", type=csv_strings, default=None
    )
    parser.add_argument("-twl", "--time_window_length", type=int, default=1)

    args = parser.parse_args()

    dataset = Sim3DInMemoryDataset(
        args.directory,
        data_directory_name=args.input_data_directory,
        labels="velocities" if args.use_velocities_as_labels else "positions",
        node_feature_columns=args.node_feature_columns,
        time_window_length=args.time_window_length,
    )
    print("dataset length = ", len(dataset))

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    # dataset_metadata = torch.load(dataset.processed_paths[0])
    with open(dataset.processed_paths[0], "r") as f:
        dataset_metadata = json.load(f)
    print(dataset_metadata)

    # for episode_index, episode_data in enumerate(dataset):
    #     for graph in episode_data:
    #         print("x", graph[0].x)
    #         print("y", graph[0].y)
    #         break
    #     break

    # print("checking data from loader")
    # for episode_number, episode in enumerate(loader):
    #     print(f"episode {episode_number} length {len(episode)}")
    #     for graph in episode:
    #         print("first graph: \n", graph)
    #         break
    #         # print('graph.x: \n', graph.x)
    #         # print('-' * a10)
