import argparse


import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.nn import MLP
from torchinfo import summary
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


from tqdm.auto import tqdm as auto_tgdm

# from collab_env.gnn3D.analyze_results import plot_attention_weights
# from collab_env.gnn3D.build_dataset import Sim3DInMemoryDataset
# from collab_env.gnn3D.gnn_models import GNN_Attention

from contextlib import nullcontext

from collab_env.gnn.analyze_results import process_training_result
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
            stored_init_pos = False
            for graph in episode:
                if not stored_init_pos:
                    stored_init_pos = True
                    # print('x shape: ', graph.x.shape)
                    input_position = graph.x[:, :3].detach().numpy()
                    # print('input\n ', input_position)
                    # print('input shape: ', input_position.shape)
                    prediction_list[-1].append(input_position)

                prediction, attention_weights = model(graph)

                # edge_index, alpha = attention_weights

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


def train_3DGNN(
    directory,
    num_epochs=1,
    evaluate_only=False,
    include_second_layer=True,
    mlp_layers=None,
):
    """
    Loads training and validation datasets from specified directory, creates the GNN model, and runs the training loop
    by calling train_epoch() for each epoch.

    Args:
        directory (string): the path to the directory containing sim3d dataset
        num_epochs (int): the number of epochs to run
        evaluate_only (bool): indicates whether this is an evaluation only run or if we should train

    Returns:
        training result dictionary (Dict): this dictionary contains the losses for all epochs, and the predictions and
        attention weights for the last epoch.

    """
    train_loader, val_loader = load_dataset(directory)
    if mlp_layers is not None:
        mlp = MLP(mlp_layers)
    else:
        mlp = None

    model = GNN_Attention(
        model_name="gnn-Attention-GConv-Linear",
        in_node_dim=14,
        edge_dim=3,
        output_dim=3,
        self_loops=True,
        fill_value=torch.zeros(3).float(),
        include_convolutional_layer=include_second_layer,
        mlp=mlp,
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


def csv_ints(arg):
    return [int(x) for x in arg.split(",")]


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
    parser.add_argument("-ncl", "--no_convolutional_layer", action="store_true")
    parser.add_argument("-mlp", "--multilayer_perceptron_layers", type=csv_ints)

    args = parser.parse_args()

    result = train_3DGNN(
        args.directory,
        num_epochs=args.num_epochs,
        evaluate_only=args.evaluate_only,
        include_second_layer=not args.no_convolutional_layer,
        mlp_layers=args.multilayer_perceptron_layers,
    )

    process_training_result(result, args.directory)

    print("training complete")
