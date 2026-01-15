import argparse
from pathlib import Path
from typing import Optional, Tuple, Any, ContextManager, Sequence, Callable

import torch
import numpy as np
import torch_geometric
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

from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.gnn_3D.analyze_results import process_training_result
from collab_env.gnn.gnn_3D.build_dataset import Sim3DInMemoryDataset
from collab_env.gnn.gnn_3D.gnn_models import GNN_Attention


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train: bool = True,
) -> Tuple[float, list[Any], list[Any], list[Any]]:
    """
    Trains the given model for one epoch or evaluates the model for one epoch.
    Args:
        model (torch.nn.Module): pytorch model to train
        loader (DataLoader): dataset loader
        optimizer (torch.optim.Optimizer): pytorch optimizer
        train (bool): indicates whether this is an evaluation only run or if we should train):

    Returns:
        total_loss (float): the total loss per time step for this epoch
        prediction_list (list): the predictions for every episode and every time step within the episode
        attention_weights_list (list): the attention weights for every episode and every time step within the episode

    """
    context: ContextManager[Optional[None]]  # needed for lint
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
        prediction_list: list[Any] = []
        attention_weights_list: list[Any] = []
        episode_index_list = []
        total_loss = 0.0
        for episode, index in episode_bar:
            episode_index_list.append(index)
            episode_loss = 0.0

            # create lists for all the predictions and attention weights in this episode.
            prediction_list.append([])
            attention_weights_list.append([])

            # there is a graph for every time step.
            for graph in episode:
                prediction, attention_weights = model(graph)

                # store the predictions and attention weights for this time step
                prediction_list[-1].append(prediction.detach().cpu().numpy())

                # need to detach each part of the tuple separately because we can't just detach the tuple
                attention_weight_edge_index, attention_weight_alpha = attention_weights
                attention_weights_list[-1].append(
                    (
                        attention_weight_edge_index.detach().cpu().numpy(),
                        attention_weight_alpha.detach().cpu().numpy(),
                    )
                )

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
                    "(total loss per step, episode loss)": f"({total_loss:.9f},{episode_loss:.6f})"
                }
            )
        # end for episode

    return (
        total_loss
        / len(loader),  # divide by num episodes to get the loss per time step
        prediction_list,
        attention_weights_list,
        episode_index_list,
    )


def load_dataset(
    directory: str,
) -> Tuple[
    torch_geometric.data.DataLoader,
    Sequence[int],
    torch_geometric.data.DataLoader,
    Sequence[int],
    dict,
]:
    """
    Loads training and validation datasets from specified directory.
    Args:
        directory (string): the path to the directory containing sim3d dataset
        node_feature_columns (list): the list of columns from the dataframe to include as input features

    Returns:
        train_loader (torch.utils.data.DataLoader): the training dataset loader
        val_loader (torch.utils.data.DataLoader): the validation dataset loader
    """
    dataset = Sim3DInMemoryDataset(directory, load_only=True)

    seed = np.random.randint(low=0, high=2**31)
    torch_generator = torch.manual_seed(seed)
    np.random.seed(seed)

    train_size = int(len(dataset) * 0.8)

    train_dataset: Subset
    val_dataset: Subset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch_generator
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

    return (
        train_loader,
        train_dataset.indices,
        val_loader,
        val_dataset.indices,
        dataset.metadata,
    )


def model_factory(
    mlp_layers=None,
    load_model: Optional[Path] = None,
    include_second_layer: bool = False,
) -> Callable[[dict], Tuple[torch.nn.Module, torch.optim.Optimizer]]:
    """
    Args:
        mlp_layers (list): list of dimensions of each layer in the MLP to creates.
    Returns:
        A function that creates a GNN model and optimizer.
    """
    if mlp_layers is not None:
        mlp = MLP(mlp_layers)
    else:
        mlp = None

    def model_creator(dataset_metadata: dict):
        if load_model is not None:
            # model_state = torch.load(load_model)
            # model.load_state_dict(model_state)
            # full model was saved for now -- change this later to be more portable
            model = torch.load(load_model)
        else:
            model = GNN_Attention(
                model_name="GNN-Attention-Linear",
                in_node_dim=dataset_metadata["input_node_dim"],
                edge_dim=dataset_metadata["edge_attr_dim"],  # get this from dataset
                output_dim=dataset_metadata["label_dim"],  # get this from dataset
                self_loops=True,
                fill_value=torch.zeros(
                    dataset_metadata["edge_attr_dim"]
                ).float(),  # get dimension from dataset same as edge_dim
                include_convolutional_layer=include_second_layer,
                mlp=mlp,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        return model, optimizer

    return model_creator


def train_3DGNN(
    directory: str,
    training_result_path: Path,
    model_creation_function: Callable[
        ..., Tuple[torch.nn.Module, torch.optim.Optimizer]
    ],
    num_epochs: int = 1,
    evaluate_only: bool = False,
) -> dict:
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
    # load the dataset
    (
        train_loader,
        training_dataset_indices,
        val_loader,
        val_dataset_indices,
        dataset_metadata,
    ) = load_dataset(directory)

    # create the model and optimizer
    model, optimizer = model_creation_function(dataset_metadata=dataset_metadata)
    summary(model)

    # train or evaluate the model as specified
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        # print(f"epoch {epoch} ")
        # print("-"*40)
        val_loss, val_prediction_list, val_attention_weights_list, val_index_list = (
            train_epoch(
                model=model, loader=val_loader, optimizer=optimizer, train=False
            )
        )
        val_loss_list.append(val_loss)
        # print("val loss", val_loss)

        (
            train_loss,
            train_prediction_list,
            train_attention_weights_list,
            train_index_list,
        ) = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            train=not evaluate_only,
        )
        train_loss_list.append(train_loss)
        # print("training loss", train_loss)

        saved_model_path = expand_path(
            f"saved_models/{model.name}_epoch_{epoch}.pt",
            training_result_path,
        )
        saved_model_path.parent.mkdir(parents=True, exist_ok=True)
        # saving full model -- not portable or particularly secure but easier for now
        torch.save(model, saved_model_path)

    # if we are training, we need to run validation for the final epoch.
    if not evaluate_only:
        val_loss, val_prediction_list, val_attention_weights_list, val_index_list = (
            train_epoch(
                model=model, loader=val_loader, optimizer=optimizer, train=False
            )
        )
        # print("final val loss", val_loss)

    return {
        "train_losses": train_loss_list,
        "train_predictions": train_prediction_list,
        "train_attention": train_attention_weights_list,
        "train_dataset_indices": train_index_list,
        "val_losses": val_loss_list,
        "val_predictions": val_prediction_list,
        "val_attention": val_attention_weights_list,
        "val_dataset_indices": val_index_list,
        "trained_model": model,
        "dataset_metadata": dataset_metadata,
    }


if __name__ == "__main__":
    # need this for using lists in command line arguments
    def csv_ints(arg):
        return [int(x) for x in arg.split(",")]

    parser = argparse.ArgumentParser(
        prog="train_3DGNN.py",
        description="Trains a GNN on the graph dataset generated from 3D simulation data.",
        epilog="---",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="path to the directory containing simulation data",
    )
    parser.add_argument(
        "-e", "--evaluate-only", action="store_true", help="do not train, only evaluate"
    )
    parser.add_argument(
        "-l",
        "--load_model",
        type=str,
        default=None,
        help="specifies the path to the file containing the model we should load",
    )  # not implemented yet
    # parser.add_argument(
    #     "-fr", "--force_reload", type=str, help="force the dataset to be reprocessed"
    # )  # not implemented yet
    parser.add_argument(
        "-ne",
        "--num-epochs",
        default=1,
        type=int,
        help="the number of epochs for training",
    )
    parser.add_argument(
        "-cl",
        "--convolutional_layer",
        action="store_true",
        help="include a convolutional layer after the attention layer",
    )
    parser.add_argument(
        "-mlp",
        "--multilayer_perceptron_layers",
        type=csv_ints,
        help="include a multilayer perceptron with dimensions specified as a list of integers",
    )
    parser.add_argument(
        "-trd",
        "--training_result_subdirectory",
        type=str,
        default="training_results",
        help="subdirectory with the directory to store the results",
    )

    args = parser.parse_args()

    # create the training result folder right away so we don't waste time training
    # and then blow up trying to save the results.
    training_result_path = expand_path(
        args.directory + "/" + args.training_result_subdirectory, get_project_root()
    )

    # if the training result directory exists and is not empty, raise an exception.
    if (
        training_result_path.exists()
        and next(training_result_path.iterdir(), None) is not None
    ):
        raise FileExistsError(
            f"Training result directory already exists. Please move it so I don't overwrite it, which could be sad for you. \n Full path is {training_result_path}."
        )

    training_result_path.mkdir(parents=True, exist_ok=False)

    result = train_3DGNN(
        directory=args.directory,
        training_result_path=training_result_path,
        num_epochs=args.num_epochs,
        evaluate_only=args.evaluate_only,
        model_creation_function=model_factory(
            include_second_layer=args.convolutional_layer,
            mlp_layers=args.multilayer_perceptron_layers,
            load_model=args.load_model,
        ),
    )

    print("calling process training result")
    process_training_result(
        result, args.directory + "/" + args.training_result_subdirectory
    )

    print("training complete")
