import argparse

import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Subset
from torchinfo import summary
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq


from tqdm.auto import tqdm as auto_tgdm

from collab_env.data.file_utils import expand_path, get_project_root
# from collab_env.gnn3D.analyze_results import plot_attention_weights
# from collab_env.gnn3D.build_dataset import Sim3DInMemoryDataset
# from collab_env.gnn3D.gnn_models import GNN_Attention

from contextlib import nullcontext

from collab_env.gnn.build_dataset import Sim3DInMemoryDataset
from collab_env.gnn.gnn_models import GNN_Attention


def train_epoch(model, loader, optimizer, train=True):
    """
    Trains the given model for one epoch or evluates the  for one epoch.
    Args:
        model (): pytorch model to train
        loader (): dataset loader
        optimizer (): pytorch optimizer
        train (bool): indicates whether this is an evaluation only run or if we should train):

    Returns:
        total_loss (float): the total loss per time step for this epoch
        prediction_list (list): the predictions for every episode and every time step within the episode
        attention_weights_list (list): the attention weights for every episode and every time step within the episode

    """
    if train:
        model.train()
        context = nullcontext()
    else:
        model.eval()
        context = torch.no_grad()

    # use torch.no_grad() when just evaluating; otherwise this context is nullcontext.
    with context:
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

        episode_bar = auto_tgdm(
            loader,
            unit="episode",
            bar_format=bar_format,
            desc=f"{'train' if train else 'val'}",
            leave=True,
        )

        # create lists for results, there will be one entry for each episode
        prediction_list = []
        attention_weights_list = []

        total_loss = 0.0
        for episode in episode_bar:
            episode_loss = 0.0

            # create lists for all the predictions and attention weights in this episode.
            prediction_list.append([])
            attention_weights_list.append([])

            # there is a graph for every time step.
            stored_init_post = False
            for graph in episode:
                if not stored_init_post:
                    stored_init_post = True
                    # print('x shape: ', graph.x.shape)
                    input_position = graph.x[:, :3].detach().numpy()
                    # print('input\n ', input_position)
                    # print('input shape: ', input_position.shape)
                    prediction_list[-1].append(input_position)

                prediction, attention_weights = model(graph)

                edge_index, alpha = attention_weights

                # attention_weights = convert_attention_weights_to_adj_matrix(attention_weights)

                # store the predictions and attention weights for this time step
                prediction_list[-1].append(prediction.detach().cpu().numpy())
                attention_weights_list[-1].append(attention_weights)

                loss = F.mse_loss(prediction, graph.y)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                episode_loss += loss.item()
            # end for graph

            total_loss += episode_loss / (
                len(episode) * loader.batch_size
            )  # divide by the number of time steps in episode
            episode_bar.set_postfix(
                {
                    "(total loss per step, episode loss) ": f"({total_loss:.4f},{episode_loss:.4f})"
                }
            )
        # end for episode

    return (
        total_loss
        / len(loader),  # divide by num episodes to get the loss per time step
        prediction_list,
        attention_weights_list,
    )


def load_dataset(directory: str):
    """
    Loads training and validation datasets from specified directory.
    Args:
        directory (string): the path to the directory containing sim3d dataset

    Returns:
        train_loader (torch.utils.data.DataLoader): the training dataset loader
        val_loader (torch.utils.data.DataLoader): the validation dataset loader
    """
    dataset = Sim3DInMemoryDataset(directory)
    print("dataset length = ", len(dataset))

    seed = np.random.randint(low=0, high=2**31)
    torch_generator = torch.manual_seed(seed)
    np.random.seed(seed)

    train_size = int(len(dataset) * 0.8)
    print("train size: ", train_size)
    train_dataset: Subset
    val_dataset: Subset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch_generator
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader


def train_3DGNN(directory, num_epochs=1, evaluate_only=False):
    """
    Loads training and validation datasets from specified directory, creates the GNN model, and runs the training loop
    by calling train_epoch() for each epoch.

    Args:
        directory (string): the path to the directory containing sim3d dataset
        num_epochs (int): the number of epochs to run):
        evaluate_only (bool): indicates whether this is an evaluation only run or if we should train

    Returns:
        training result dictionary (Dict): this dictionary contains the losses for all epochs, and the predictions and
        attention weights for the last epoch.

    """
    train_loader, val_loader = load_dataset(directory)

    model = GNN_Attention(
        model_name="gnn-Attention-GConv-Linear",
        in_node_dim=14,
        edge_dim=3,
        output_dim=3,
        self_loops=True,
        fill_value=torch.zeros(3).float(),
    )
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        # print(f"epoch {epoch} ")
        # print("-"*40)
        val_loss, val_prediction_list, val_attention_weights_list = train_epoch(
            model=model, loader=val_loader, optimizer=optimizer, train=False
        )
        val_loss_list.append(val_loss)
        # print("val loss", val_loss)

        train_loss, train_prediction_list, train_attention_weights_list = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            train=not evaluate_only,
        )
        train_loss_list.append(train_loss)
        # print("training loss", train_loss)

    if not evaluate_only:
        train_loss, train_prediction_list, train_attention_weights_list = train_epoch(
            model=model, loader=val_loader, optimizer=optimizer, train=False
        )
        # print("final val loss", val_loss)

    return {
        "train_losses": train_loss_list,
        "train_predictions": train_prediction_list,
        "train_attention": train_attention_weights_list,
        "val_losses": val_loss_list,
        "val_predictions": val_prediction_list,
        "val_attention": val_attention_weights_list,
    }


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

        alpha_list.append(alpha.view(-1).cpu().numpy())

        # note that the to node is paying attention to the from node.
        # alpha_{ij} is the weight on the edge (j, i) that is directed from j
        # toward node i in the GNN.
        from_nodes.append(edge_index[0].cpu().numpy())
        to_nodes.append(edge_index[1].cpu().numpy())

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
    # only doing the last epoch for now.
    print("processing training result")
    # print(training_result["val_predictions"])
    val_predictions = np.array(training_result["val_predictions"][-1])
    # print("val predictions shape", val_predictions.shape)
    # print("val predictions", val_predictions)
    save_predictions(
        val_predictions, directory + "/training_results/validation_predictions.parquet"
    )
    # what is the -1 for? can't i save them all.
    train_predictions = np.array(training_result["train_predictions"][-1])
    save_predictions(
        train_predictions, directory + "/training_results/training_predictions.parquet"
    )

    train_attention_weights = training_result["train_attention"][-1]
    save_attention(
        train_attention_weights,
        directory + "/training_results/train_attention_weights.parquet",
    )

    val_attention_weights = training_result["val_attention"][-1]
    save_attention(
        val_attention_weights,
        directory + "/training_results/val_attention_weights.parquet",
    )

    loss_df = pd.DataFrame(
        {
            "training loss": training_result["train_losses"],
            "validation loss": training_result["val_losses"],
        }
    )

    loss_table = pa.Table.from_pandas(loss_df)
    file_path = expand_path(directory + "/losses.parquet", get_project_root())
    pq.write_table(loss_table, file_path)

    # plot_attention_weights(val_attention_weights[-40:])

    # plot_losses(training_result["train_losses"], training_result["val_losses"])


def plot_losses(train_loss_list, val_loss_list):
    plt.plot(train_loss_list[1:], label="train loss")
    plt.plot(val_loss_list, label="val loss")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="build_dataset.py",
        description="Builds a graph dataset from 3D simulation data.",
        epilog="---",
    )
    parser.add_argument("-d", "--directory", type=str, required=True)
    parser.add_argument("-e", "--evaluate-only", action="store_true")
    parser.add_argument("-l", "--load", type=str)
    parser.add_argument("-ne", "--num-epochs", default=1, type=int)

    args = parser.parse_args()

    result = train_3DGNN(
        args.directory, num_epochs=args.num_epochs, evaluate_only=args.evaluate_only
    )

    process_training_result(result, args.directory)

    print("training complete")
