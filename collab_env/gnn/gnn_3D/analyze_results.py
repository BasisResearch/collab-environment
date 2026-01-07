import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn
import torch
import matplotlib
import yaml
from datetime import datetime

from collab_env.gnn.gnn_3D.gnn_agent import GNN_Agents
from collab_env.sim.boids.run_simulator import create_environment, run_simulator

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt, animation
import pyarrow as pa
import pyarrow.parquet as pq

from collab_env.data.file_utils import get_project_root, expand_path


def save_attention(attention_weights_list, filename):
    """
    Saves attention weights to a parquet file.
    Args:
        attention_weights_list (): this is a list of attention weights for each graph over the
                            time steps in a single episode.
        filename ():

    Returns:

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


def save_predictions(predictions, filename):
    # print("saving predictions\n", predictions)
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
    # print('df\n', df)

    #
    # Dump data to output file
    #
    prediction_table = pa.Table.from_pandas(df)
    file_path = expand_path(filename, get_project_root())
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(prediction_table, file_path)


def process_training_result(training_result, directory):
    print("processing training result")

    # use the episode file list to match result files names to input data file names
    episode_file_list = list(training_result["episode_file_list"])

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
        episode_file_name = episode_file_list[val_indices[episode]].name.split(".pt")[0]

        # save the results labeled with the corresponding episode file name
        val_predictions = np.array(training_result["val_predictions"][episode])
        save_predictions(
            val_predictions,
            directory + f"/validation_predictions_{episode_file_name}.parquet",
        )

        val_attention_weights = training_result["val_attention"][episode]
        save_attention(
            val_attention_weights,
            directory + f"/validation_attention_weights_{episode_file_name}.parquet",
        )

    train_indices = training_result["train_dataset_indices"]
    print("train_indices", train_indices)
    print("number of train_predictions", len(training_result["train_predictions"]))
    for episode in range(len(training_result["train_predictions"])):
        episode_file_name = episode_file_list[train_indices[episode]].name.split(".pt")[
            0
        ]
        # save the results labeled with the corresponding episode number
        train_predictions = np.array(training_result["train_predictions"][episode])
        save_predictions(
            train_predictions,
            directory + f"/training_predictions_{episode_file_name}.parquet",
        )

        train_attention_weights = training_result["train_attention"][episode]
        save_attention(
            train_attention_weights,
            directory + f"/training_attention_weights_{episode_file_name}.parquet",
        )

    # plot_attention_weights(val_attention_weights[-40:])

    # plot_losses(training_result["train_losses"], training_result["val_losses"])


def plot_losses(train_loss_list, val_loss_list):
    plt.plot(train_loss_list[1:], label="train loss")
    plt.plot(val_loss_list, label="val loss")
    plt.show()


def animate_attention_weights(attention_weights_list):
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

    _ = animation.FuncAnimation(
        fig,
        update,
        frames=len(attention_matrices),
        init_func=init,
        interval=100,
        blit=False,
    )
    plt.show()


def plot_attention_weights(attention_weight_list, num_cols=2):
    attention_matrices = [
        convert_attention_weights_to_adj_matrix(w) for w in attention_weight_list
    ]
    num_rows = int(np.ceil(len(attention_weight_list) / num_cols))
    print("num_rows", num_rows)
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


def convert_attention_weights_to_adj_matrix(attention_weights):
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


def load_attention_weights(directory, filename):
    """
    Args:
        directory (): the parent directory containing the training data. This function
                    assumes training_results is a subdirectory containing the parquet files
        filename (): name of the parquet file containing the attention weights in a dataframe

    Returns:

    """

    path = expand_path(directory + "/" + filename, get_project_root())
    df = pd.read_parquet(path)
    attention_weights_list = [
        (torch.tensor([f, t]), torch.tensor(w))
        for f, t, w in zip(df["from"], df["to"], df["attention_weight"])
    ]
    return attention_weights_list


def rollout(simulator_config: dict, agent_config: dict, rollout_path: Path):
    """
    Args:
        args (): arguments parsed from the command line
        config (dict): configuration dictionary

    This function displays the predicted trajectories of the agents. It needs to be given the simulator config and the
    agent config so that the environment and agents can be constructed and the simulator can be run.
    """

    print("run folder is ", rollout_path)

    env = create_environment(config=simulator_config, run_folder=rollout_path)
    agents = GNN_Agents(
        simulator_config=simulator_config, agent_config=agent_config, env=env
    )
    run_simulator(
        config=simulator_config, env=env, agents=agents, run_folder=rollout_path
    )

    print(f"rollout completed at {datetime.now().strftime('%Y%m%d-%H%M%S')}")


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

    args = parser.parse_args()

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
        rollout(simulator_config, agent_config, rollout_path)

    print("analyze results completed")
