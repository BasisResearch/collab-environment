import argparse

import numpy as np
import pandas as pd
import seaborn
import torch
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt, animation

from collab_env.data.file_utils import get_project_root, expand_path


def animate_attention_weights(attention_weights_list):
    attention_matrices = [
        convert_attention_weights_to_adj_matrix(w) for w in attention_weights_list
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        attention_matrices[0], cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto"
    )
    title = ax.set_title("Frame 0", animated=False)

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
    print("type ", type(axes))
    axes = axes.ravel()
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

    path = expand_path(directory + "/training_results/" + filename, get_project_root())
    df = pd.read_parquet(path)
    attention_weights_list = [
        (torch.tensor([f, t]), torch.tensor(w))
        for f, t, w in zip(df["from"], df["to"], df["attention_weight"])
    ]
    return attention_weights_list


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

    args = parser.parse_args()

    if args.plot_attention:
        attention_weights_list = load_attention_weights(args.directory, args.filename)
        plot_attention_weights(
            attention_weights_list[args.start_time : args.finish_time + 1], num_cols=1
        )

    if args.animate_attention:
        attention_weights_list = load_attention_weights(args.directory, args.filename)
        animate_attention_weights(attention_weights_list)
