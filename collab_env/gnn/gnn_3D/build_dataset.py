import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

# from typing import Tuple
import numpy as np
import yaml
import torch
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from collab_env.data.file_utils import expand_path, get_project_root


def convert_dataframe_to_node_features(
    df: pd.DataFrame, columns=None, time_window_length=1
) -> torch.Tensor:
    """
    Args:
     df (pd.DataFrame): the dataframe containing a description of the agents in the environment. The dataframe is
     assumed to have the following columns:
            "x",
            "y",
            "z",
            "v_x",
            "v_y",
            "v_z",
            "distance_to_target_mesh_closest_point_1",
            "target_mesh_closest_point_1",
            "mesh_scene_distance",
            "mesh_scene_closest_point_1",
     However, if the columns parameter is specified, then the only assumed columns are "target_mesh_closest_point_1" and
     "mesh_scene_closest_point_1". These are lists of 3 floats.

     columns (List[str}): the columns in the dataframe to include in the node features.

     time_window_length (int): the time window we want to include in the node features.

    Returns:
        a torch.Tensor containing the node features for each agent corresponding to the data in the dataframe.
        The shape of the tensor will be (num time steps - window length, num agents, window length * num columns)
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
    I did the subscripting thing because parquet can't handle 2-dimensional lists in a dataframe. 
    """

    # split the lists in the dataframe into separate columns.
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

    # if columns is not specified, we take all the columns in the standard simulation dataframe.
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

    return torch.from_numpy(node_features)


def compute_positions(
    agents_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        agents_df (pd.DataFrame): a dataframe containing the agents' positions and velocities as
        columns "x", "y", "z", v_x", "v_y", "v_z"
    Returns:
        A tuple containing:
            positions (torch.FloatTensor): the positions of the agents at each time step.
            relative_positions (torch.FloatTensor): the relative positions of the agents with respect to
            each other agents at each time step.
            velocities (torch.FloatTensor): the velocities of the agents at each time step.
        The shape of positions and velocities is (num time steps, num agents, dimension of world).
        The shape of relative_positions is (num time steps, num agents, num_agents, dimension of world).
    """

    #
    # Reshape the dataframe so the index is time, the column is id and the value
    # is the position vector.
    #
    agents_df = agents_df.copy()
    agents_df.loc[:, "position"] = agents_df[["x", "y", "z"]].values.tolist()
    agents_df.loc[:, "velocity"] = agents_df[["v_x", "v_y", "v_z"]].values.tolist()

    pivot = agents_df.pivot(index="time", columns="id", values=["position", "velocity"])

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
    # Compute relative positions for each time step and each agent. If we have a tensor called
    # positions of shape (num time steps, num agents, 3), we should get a tensor relative_positions
    # of shape (num time steps, num agents, num agents, 3), where relative_positions[t, i, j] is
    # the position of agent j relative to agent i at time t.
    #
    relative_positions = positions[:, None, :, :] - positions[:, :, None, :]

    return positions, relative_positions, velocities


def edge_attributes_for_complete_graph(
    relative_position: torch.Tensor, source, destination, device=None
) -> torch.Tensor:
    """
    Args:
        relative_position (torch.Tensor): the relative positions of the agents at a single time step. This should have
        shape (num agents, num agents, dimension of physical space) with relative_position[i,j] =
        pos_j - pos_i (i.e., the position of agent j with respect to agent i).
        source (int): the index of the agent
        destination (int): the index of the agent

        device (str, optional): the device to use. Defaults to None.

    Returns:
        edge_attr (torch.FloatTensor): the attributes of the edges moved to device. For every pair of edges, the
        attribute for directed edge (i,j) will be relative_position[i,j]. The shape of the edge_attr will match
        the edge_index shape.

    """
    relative_position_on_device = relative_position.to(device=device)

    edge_attr = relative_position_on_device[source, destination]

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
) -> list[Data]:
    """
    Args:
        agents_df (pd.DataFrame): a dataframe containing the agents' observations
        num_time_steps (int): the number of time steps to output
        num_agents (int): the number of agents in the dataframe
        box_size (float): the size of the cube delineating the world (assumes width = height = depth)
        label_offset (int, optional): how many time steps into the future should we be predicting.  Defaults to 1.
        node_feature_columns (list[str], optional): the list of features to use for nodes. Defaults to None.
        label_type (str, optional): the type of the labels to use (either "velocities" or "positions". Defaults to None.
    Returns:
        A Data list that includes the sequence of graphs for the GNN. Each graph will include the node features for each
        node and the edges features for each edge, as well as the labels for each node.
    """
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

    #
    # Scale all the coordinates and distances by the size of the box (assumes width = height = depth)
    #
    node_features = node_features / torch.tensor([box_size], dtype=torch.float32)
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
    # Create edge indices for a complete graph with self loops in COO format, i.e., the edge indices for a complete 3
    # node graph would be: [ [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 3, 0, 1, 2] ]. This represents the edges
    # (0,0), (0,1), ..., (2,1), (2,2) in that order.
    #
    from_nodes = torch.arange(num_agents).repeat_interleave(num_agents)  # sources
    to_nodes = torch.arange(num_agents).repeat(num_agents)  # targets
    edge_index = torch.stack([from_nodes, to_nodes], dim=0)  # shape [2, num_agents**2]

    # the labels will be the positions of the agent at the next time step.
    if label_type is not None and label_type.lower() == "velocities":
        labels = velocities / torch.tensor([box_size], dtype=torch.float32)
    else:
        labels = positions / torch.tensor([box_size], dtype=torch.float32)

    #
    # Add relative positions for each time step and each agent as the edge attributes.
    # Create the data object for this episode and add it to the data list.
    # We loop to num_time_steps - 1 because the last graph in the sequence has nothing to predict.
    #
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
            num_time_steps - label_offset - (time_window_length - 1),
        )  # don't go beyond the labels.
    ]
    return data_list


def sort_using_numbers(path) -> list[int | str]:
    """
    Args:
        path(Path): the path of the file
    Returns:
        Creates a list of substrings in the file name that converts the numeric parts to ints so the filenames can be
        sorted by episode number.
    """
    parts = re.split(r"(\d+.)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class Sim3DInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        node_feature_columns: Optional[list[str]] = None,
        load_only: bool = False,
        data_directory_name: Optional[str] = None,
        label_type: Optional[str] = None,
        time_window_length: int = 1,
    ):
        """

        Args:
            root (string): full path to dataset directory -- this is used to set raw_dir, which is needed
                            in super.__init__()
            transform ():
            pre_transform ():
            node_feature_columns (List(str)):
            load_only (bool): True if we only want to load data that was already processed and don't want to process
                data this time around. (Used by train_3DGNN because it doesn't know the node columns.)
            data_directory_name (string): the directory of the data that contains the data to be processed and turned into
                GNN data.
            label_type (str): velocities or positions depending on whether the labels should be the agents velocities or
                positions.

        """
        self.root_path = expand_path(root, get_project_root())
        root = str(self.root_path)

        # if we are in load_only mode, make sure the data had already been processed, raise exception otherwise.
        if load_only:
            process_indicator_path = expand_path(
                "processed/" + self.processed_file_names[0], self.root_path
            )
            if not process_indicator_path.exists():
                raise FileNotFoundError(
                    f"Dataset constructor called in load only mode but path {process_indicator_path} doesn't exists."
                )

        # if we are outputting to a separate output directory, make sure that directory does not already exist so we
        # don't trash someone's data.
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

        self.node_feature_columns = node_feature_columns
        self.label_type = label_type
        self.time_window_length = time_window_length
        self._metadata = None

        #
        # Call super()
        #
        super(Sim3DInMemoryDataset, self).__init__(root, transform, pre_transform)

        self.episode_file_list: list[str] = []  # this is created in load_episodes()
        self._input_node_dim = None
        self.episodes = self.load_episodes()
        # print('self.episodes', self.episodes)
        self._input_node_dim = self.episodes[0][0].x.shape[1]
        self._edge_attr_dim = self.episodes[0][0].edge_attr.shape[1]
        self._label_dim = self.episodes[0][0].y.shape[1]

        # get the metadata from the file that process() previously dumped the metadata to.
        with open(self.processed_paths[0], "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

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
        Gets called if the files listed in self.raw_file_names do not exist in raw directory. If the link to data flag
        is set in the constructor, we will create a symbolic link in the raw directory to point to the data directory.
        This may not work in Windows -- silly Windows.
        """
        if self.link_to_data:
            data_link = expand_path("raw/" + self.raw_file_names[0], self.root_path)
            data_link.symlink_to(self.sim_data_folder_path, target_is_directory=True)

    def get_filename_for_saved_episode(
        self, episode_file_list: list[Path], episode_number: int
    ) -> str:
        """
        Args:
            episode_file_list (list(Path)): list of file paths for the episode parquet files.
        Returns:
            the filename of the torch file containing the list of graphs for the given episode
        """
        episode_file_name = episode_file_list[episode_number].name.split(".parquet")[0]
        # filename = f"{self.processed_paths[0]}_episode_{episode_number}.pt"
        filename = f"{self.processed_paths[0]}_{episode_file_name}.pt"
        return filename

    def process(self):
        """
        Reads all the episode parquet files, loading the dataframes and converting the dataframes into node features
        and edge attributes for the graphs at each time step.
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.
        """

        # Load the config file for the simulator run that generated the data.
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

            # create the list of graphs
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
            torch.save(
                data_list,
                self.get_filename_for_saved_episode(episode_file_list, episode_number),
            )

        # indicate that processing is complete by creating the indicator file
        self._metadata = {
            "node_feature_columns": self.node_feature_columns,
            "time_window_length": self.time_window_length,
            "input_node_dim": data_list[0].x.shape[1],
            "edge_attr_dim": data_list[0].edge_attr.shape[1],
            "label_dim": data_list[0].y.shape[1],
            "episode_file_list": [ep_path.name for ep_path in episode_file_list],
            "label_type": self.label_type,
        }

        with open(self.processed_paths[0], "w", encoding="utf-8") as f:
            f.write(json.dumps(self._metadata, ensure_ascii=False))

    def load_episodes(self) -> list[list[Data]]:
        """
        Loads the data that was previously processed.
        Returns:
            A list of Data lists with length = the number of episodes, where each Data list contains the graph sequence
            for a single episode.
        """
        self.episode_file_list = sorted(
            # couldn't figure out this type checking thing from lint, so ignored it due to lack of time.
            list(self.root_path.glob("processed/*episode*.pt")),  # type: ignore [arg-type]
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

    def __getitem__(self, index: int) -> Data:
        """
        Args:
            index (int): index of the episode
        Return the episode and the index so the training code can figure out what episode it is
        being trained on. This makes it easier to correctly compare the predicted trajectories
        and the actual trajectories when analyzing the results.
        """
        return self.episodes[index], index

    @property
    def metadata(self):
        return self._metadata

    @property
    def input_node_dim(self):
        return self._input_node_dim

    @property
    def label_dim(self):
        return self._label_dim

    @property
    def edge_attr_dim(self):
        return self._edge_attr_dim


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
    parser.add_argument("-nfc", "--node_feature_columns", type=str, default=None)
    parser.add_argument("-twl", "--time_window_length", type=int, default=1)

    args = parser.parse_args()

    dataset = Sim3DInMemoryDataset(
        args.directory,
        data_directory_name=args.input_data_directory,
        label_type="velocities" if args.use_velocities_as_labels else "positions",
        node_feature_columns=None
        if args.node_feature_columns is None
        else args.node_feature_columns.split(","),
        time_window_length=args.time_window_length,
    )
    print("dataset length = ", len(dataset))
