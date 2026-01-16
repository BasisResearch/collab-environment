import argparse
import shutil
from pathlib import Path
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn
import torch
import yaml
from datetime import datetime

from collab_env.gnn.gnn_3D.gnn_agent import GNN_Agents
from collab_env.sim.boids.run_simulator import create_environment, run_simulator


# matplotlib.use("TkAgg") # this was needed to do the matplotlib animation in OS X. Can't use for remote test so dropping.

from matplotlib import pyplot as plt, animation
import pyarrow as pa
import pyarrow.parquet as pq

from collab_env.data.file_utils import get_project_root, expand_path


def save_attention(attention_weights_list: list[Any], filename: str) -> None:
    """
    Saves attention weights to a parquet file.
    Args:
        attention_weights_list (list[Any]): this is a list of attention weights for each graph over the
            time steps in a single episode.
        filename (str): the name of the file off of the project root to save the attention weights to.

    """
    from_nodes = []
    to_nodes = []
    alpha_list = []
    for attention_weights in attention_weights_list:
        edge_index, alpha = attention_weights

        # alpha_list.append(alpha.view(-1).cpu().numpy())
        # need this to be 1D to store in DataFrame for parquet
        alpha_list.append(alpha.ravel())

        # note that the to node is paying attention to the from node.
        # alpha_{ij} is the weight on the edge (j, i) that is directed from j
        # toward node i in the GNN.
        from_nodes.append(edge_index[0])
        to_nodes.append(edge_index[1])

    # create the dataframe where the rows correspond to the time steps in the episode
    df = pd.DataFrame(
        {
            "time": np.arange(len(attention_weights_list)),
            "from": from_nodes,
            "to": to_nodes,
            "attention_weight": alpha_list,
        }
    )

    #
    # Dump data to output file
    #
    attention_table = pa.Table.from_pandas(df)
    file_path = expand_path(filename, get_project_root())
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(attention_table, file_path)


def save_predictions(predictions: torch.Tensor, filename: str) -> None:
    """
    Args:
        predictions (torch.Tensor): this is a list of predictions for each episode, shape (num_time_steps, num_agents, label length)
        filename (str): the name of the file off of the project root to save the predictions to.
    """
    num_time_steps, num_agents, _ = predictions.shape

    time_col = np.repeat(np.arange(0, num_time_steps), num_agents)
    agent_col = np.tile(np.arange(1, num_agents + 1), num_time_steps)
    position_col = predictions.reshape((num_time_steps) * num_agents, -1)

    df = pd.DataFrame(
        {
            "time": time_col,
            "id": agent_col,
            "x": position_col[:, 0],
            "y": position_col[:, 1],
            "z": position_col[:, 2],
            "type": "agent",
        }
    )

    #
    # Dump data to output file
    #
    prediction_table = pa.Table.from_pandas(df)
    file_path = expand_path(filename, get_project_root())
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(prediction_table, file_path)


def save_episode(
    episode_file_list: list[str],
    predictions: torch.Tensor,
    weights,
    indices: torch.IntTensor,
    directory: str,
    prefix: str,
) -> None:
    """
    Args:
        episode_file_list (list[str]): the list of file names of an episode.
        predictions (torch.Tensor): the predictions for an episode. Shape will be (num_time_steps, num_agents * batch_size, label length).
        weights (torch.Tensor): the weights for each episode.
        indices (torch.IntTensor): the indices indicating the actual episode number used from the simulation data (this
            is shuffled so the episode number and the episode index may not match).
        directory (str): the directory to save the episode.
        prefix (str): the prefix of the save file (e.g. "training", "validation").
    """
    num_batches = len(indices)  # there is one index for each episode in the batch.

    # break up the batch into predictions for each episode in the batch.
    prediction_chunks = torch.chunk(predictions, num_batches, dim=1)
    for i in range(num_batches):
        episode_file_name = episode_file_list[indices[i]]  # .split(".pt")[0]

        # save the results labeled with the corresponding episode file name
        save_predictions(
            prediction_chunks[i],
            directory + f"/{prefix}_predictions_{episode_file_name}",  # .parquet",
        )

    # attention is a little trickier because it has edge_indices in COO format and edge weights separately.
    # there is probably a cool np array indexing thingamajig that could be used here to make this faster.
    num_times_steps = len(weights)
    nodes_per_graph = predictions.shape[1] / num_batches
    node_ranges = torch.arange(0, (num_batches + 1) * nodes_per_graph, nodes_per_graph)

    split_weight_list: list[list] = [[] for _ in range(num_times_steps)]
    for time in range(num_times_steps):
        edge_index, edge_weights = weights[time]

        inner_list: list[Optional[Tuple[Any, Any]]] = [None] * num_batches
        for episode in range(num_batches):
            start_index = int(node_ranges[episode])
            end_index = int(node_ranges[episode + 1]) - 1
            source, destination = edge_index
            mask = (source >= start_index) & (source <= end_index)

            edge_index_for_graph = edge_index[:, mask] - start_index
            edge_weights_for_graph = edge_weights[mask]
            inner_list[episode] = (edge_index_for_graph, edge_weights_for_graph)

        split_weight_list[time] = inner_list

    for episode in range(num_batches):
        # need to get all time steps for this episode to pass to save_attention
        episode_attention_weights = [
            split_weight_list[time][episode] for time in range(num_times_steps)
        ]

        save_attention(
            episode_attention_weights,
            directory
            + f"/{prefix}_attention_weights_{episode_file_list[indices[episode]]}",  # .parquet",
        )


def process_training_result(training_result: dict, directory: str) -> None:
    """
    Args:
        training_result (dict): the result of the training process.
        directory (str): the directory pff of the project root to save the results.
    """
    print("processing training result")

    # use the episode file list to match result files names to input data file names
    episode_file_list = list(training_result["dataset_metadata"]["episode_file_list"])

    # save losses
    loss_df = pd.DataFrame(
        {
            "training loss": training_result["train_losses"],
            "validation loss": training_result["val_losses"],
        }
    )

    loss_table = pa.Table.from_pandas(loss_df)
    file_path = expand_path(directory + "/losses.parquet", get_project_root())
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(loss_table, file_path)

    # get the indices of the actual episode file used for each training episode
    val_indices = training_result["val_dataset_indices"]

    for episode in range(len(training_result["val_predictions"])):
        """
        TOC -- 011126 8:15PM 
        episode file list is now a list of names, not paths, so we don't have to get
        the name property anymore. 
        """
        save_episode(
            indices=val_indices[episode],
            predictions=torch.Tensor(
                np.array(training_result["val_predictions"][episode])
            ),
            weights=training_result["val_attention"][episode],
            prefix="validation",
            episode_file_list=episode_file_list,
            directory=directory,
        )

    train_indices = training_result["train_dataset_indices"]

    for episode in range(len(training_result["train_predictions"])):
        save_episode(
            indices=train_indices[episode],
            predictions=torch.Tensor(
                np.array(training_result["train_predictions"][episode])
            ),
            weights=training_result["train_attention"][episode],
            prefix="train",
            episode_file_list=episode_file_list,
            directory=directory,
        )

    # plot_attention_weights(val_attention_weights[-40:])

    # plot_losses(training_result["train_losses"], training_result["val_losses"])


def plot_losses(train_loss_list: list[Any], val_loss_list: list[Any]) -> None:
    """
    Args:
        train_loss_list (list): list of training losses
        val_loss_list (list): list of validation losses
    """
    plt.plot(train_loss_list[1:], label="train loss")
    plt.plot(val_loss_list, label="val loss")
    plt.show()


def animate_attention_weights(attention_weights_list: list[Any]) -> None:
    """
    Args:
        attention_weights_list (list): list of attention weights of shape (time, num agents, num agents)
    """
    attention_matrices = [
        convert_attention_weights_to_adj_matrix(w) for w in attention_weights_list
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        attention_matrices[0], cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto"
    )
    title = ax.set_title("Frame 0", animated=False)

    nrows = len(attention_matrices[0])
    ncols = len(attention_matrices[0][0])
    # Place ticks centered on each pixel/index
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))

    # Label ticks with integers 0..n-1
    ax.set_xticklabels(np.arange(ncols))
    ax.set_yticklabels(np.arange(nrows))

    def init():
        im.set_data(attention_matrices[0])
        title.set_text("Frame 0")
        return im, title

    def update(frame):
        im.set_data(attention_matrices[frame])
        title.set_text(f"Frame: {frame:10.0f}")
        return (im, title)

    # lint doesn't like when this is a named variable, but I get a warning in the jupyter notebook  when I don't name it,
    # though even when I do name it, the animation doesn't work in the jupyter notebook but that might be a matplotlib
    # configuration issue on OS X.
    _ = animation.FuncAnimation(
        fig,
        update,
        frames=len(attention_matrices),
        init_func=init,
        interval=100,
        blit=False,
    )
    plt.show()


def plot_attention_weights(attention_weight_list: list[Any], num_cols: int = 2) -> None:
    """
    Args:
        attention_weight_list (list[Any]): list of attention weights for each time step in COO format like the edge_index in torch-geometric
        num_cols (int, optional): number of columns in the plot. Defaults to 2.
    """
    attention_matrices = [
        convert_attention_weights_to_adj_matrix(w) for w in attention_weight_list
    ]
    # figure out how many rows will be needed for plot all time steps for the specified number of columns
    num_rows = int(np.ceil(len(attention_weight_list) / num_cols))

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 15, num_rows * 10),
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()
    for i, axis in enumerate(axes):
        seaborn.heatmap(
            attention_matrices[i],
            cmap="viridis",
            annot=True,
            ax=axis,
            vmin=0.0,
            vmax=1.0,
            xticklabels=False,
            yticklabels=False,
        )
    plt.show()


def convert_attention_weights_to_adj_matrix(
    attention_weights: list[Any],
) -> torch.Tensor:
    """
    converts the attention weights from COO format to an adjacency matrix

    Args:
        attention_weights (Tuple): attention weights in COO format, i.e., (edge_index, weights)
        where edge_index is a list of two lists [from, to]. For example, if from[0] = 8 and to[0]
        is 4, then there is a directed edge from node 8 to node 4 in the GNN and the attention weight on
        this edge is weight[0].

    Returns:
        a torch tensor adj, such that adj[i,j] is the attention that node i place
        on node j. This is alpha_{ij} in the GATv2Conv documentation.

    """
    edge_index, alpha = attention_weights

    num_nodes = edge_index.max().item() + 1
    alpha = alpha.view(-1)

    src = edge_index[0]
    dst = edge_index[1]
    adj = torch.zeros(num_nodes, num_nodes)
    adj[src, dst] = alpha
    return adj.t()


def load_attention_weights(
    directory: str, filename: str
) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        directory (str): the parent directory containing the training results data.
        filename (str): name of the parquet file containing the attention weights in a dataframe

    Returns:
        attention_weights_list (list[Tuple[torch.Tensor, torch.Tensor]): list of attention weights for each time step in COO format
    """

    path = expand_path(directory + "/" + filename, get_project_root())
    df = pd.read_parquet(path)
    attention_weights_list = [
        (torch.tensor([f, t]), torch.tensor(w))
        for f, t, w in zip(df["from"], df["to"], df["attention_weight"])
    ]
    return attention_weights_list


def rollout(
    simulator_config: dict,
    agent_config: dict,
    rollout_path: Path,
    predictions_are_velocities: bool = False,
) -> None:
    """
    Args:
        simulator_config (dict): configuration dictionary for the simulator
        agent_config (dict): configuration dictionary for the rollout agent
        rollout_path (Path): path to the rollout output folder
        predictions_are_velocities (bool): if True, the agent positions are velocities
    This function displays the predicted trajectories of the agents. It needs to be given the simulator config and the
    agent config so that the environment and agents can be constructed and the simulator can be run.
    """

    print("run folder is ", rollout_path)

    # create the environment for the simulator based on the simulator config file specified
    env = create_environment(config=simulator_config, run_folder=rollout_path)

    # create the agents to choose the actions in the simulator based on the agent config file specified
    agents = GNN_Agents(
        simulator_config=simulator_config,
        agent_config=agent_config,
        env=env,
        predictions_are_velocities=predictions_are_velocities,
    )

    # Run the simulator with the environment and agents created. The output of the simulator will go in the rollout path.
    run_simulator(
        config=simulator_config, env=env, agents=agents, run_folder=rollout_path
    )

    print(f"rollout completed at {datetime.now().strftime('%Y%m%d-%H%M%S')}")


def analyze_results(args: argparse.Namespace) -> None:
    """
    Args:
        args (argparse.Namespace): arguments parsed from the command line
    """
    if args.plot_attention:
        attention_weights_list = load_attention_weights(args.directory, args.filename)
        plot_attention_weights(
            attention_weights_list[args.start_time : args.finish_time + 1], num_cols=1
        )

    if args.animate_attention:
        attention_weights_list = load_attention_weights(args.directory, args.filename)
        animate_attention_weights(attention_weights_list)

    if args.rollout:
        simulator_config_path = expand_path(
            args.directory + "/" + args.simulator_config_file, get_project_root()
        )
        simulator_config = yaml.safe_load(open(simulator_config_path, "r"))
        agent_config_path = expand_path(
            args.directory + "/" + args.agent_config_file, get_project_root()
        )
        agent_config = yaml.safe_load(open(agent_config_path, "r"))
        rollout_path = expand_path(
            args.directory + "/" + args.rollout_subdirectory, get_project_root()
        )
        rollout_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(simulator_config_path, expand_path("sim_config.yaml", rollout_path))
        shutil.copy(agent_config_path, expand_path("agent_config.yaml", rollout_path))
        if args.show_visualizer:
            simulator_config["visuals"]["show_visualizer"] = True

        # set the number of episodes to 1, so we don't see the same thing over and over.
        simulator_config["simulator"]["num_episodes"] = 1
        rollout(
            simulator_config,
            agent_config,
            rollout_path,
            args.predictions_are_velocities,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="analyze_results.py",
        description="Analyze result of training on 3D simulation data.",
        epilog="---",
    )
    parser.add_argument("-d", "--directory", type=str, required=True)
    parser.add_argument("-pa", "--plot_attention", action="store_true")
    parser.add_argument("-aa", "--animate_attention", action="store_true")
    parser.add_argument("-f", "--filename", type=str)
    parser.add_argument("-st", "--start_time", default=0, type=int)
    parser.add_argument("-ft", "--finish_time", default=0, type=int)

    parser.add_argument("-r", "--rollout", action="store_true")
    parser.add_argument("-acf", "--agent_config_file", type=str)
    parser.add_argument("-scf", "--simulator_config_file", type=str)
    parser.add_argument("-rsd", "--rollout_subdirectory", type=str)
    parser.add_argument("-v", "--show_visualizer", action="store_true")
    parser.add_argument("-pv", "--predictions_are_velocities", action="store_true")

    args = parser.parse_args()

    analyze_results(args)

    print("analyze results completed")
