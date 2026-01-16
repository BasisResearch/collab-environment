"""
CollabSQLDataset: PyTorch Geometric dataset that fetches data from a SQL database.

This module provides a dataset class that:
1. Downloads episode data from a PostgreSQL database
2. Caches it as parquet files
3. Processes it into PyTorch Geometric Data objects for GNN training
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sqlalchemy import create_engine, text
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.gnn_3D.schema_config import SchemaConfig


def sort_using_numbers(path: Path):
    """
    Creates a list of substrings separating numeric parts for natural sorting.
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def get_valid_window_starts(
    df: pd.DataFrame,
    time_col: str,
    agent_col: str,
    window_length: int,
    label_offset: int,
) -> List[Tuple[int, Set[int]]]:
    """
    Find starting timesteps where all frames in the window have identical agent sets.

    A valid window spans frames [t, t + window_length - 1] for input features,
    plus frame [t + window_length - 1 + label_offset] for the label target.

    Args:
        df: DataFrame with observations
        time_col: Name of the time/frame column
        agent_col: Name of the agent ID column
        window_length: Number of frames in input window
        label_offset: Number of frames ahead for prediction target

    Returns:
        List of tuples (start_time, agent_set) for valid windows
    """
    # Get unique timesteps sorted
    timesteps = sorted(df[time_col].unique())
    if len(timesteps) == 0:
        return []

    # Build a dict of timestep -> set of agent IDs
    agents_by_time = {
        t: set(df[df[time_col] == t][agent_col].unique()) for t in timesteps
    }

    # Total span needed: window_length frames for input + label_offset for target
    # The label frame is at t + window_length - 1 + label_offset
    total_span = window_length + label_offset

    valid_windows = []
    for i, start_t in enumerate(timesteps):
        # Check if we have enough frames ahead
        if i + total_span > len(timesteps):
            break

        # Get all frames in this window (including label frame)
        window_times = timesteps[i : i + total_span]

        # Check that these are consecutive integers
        if window_times != list(range(start_t, start_t + total_span)):
            # Gap in timesteps, skip this window
            continue

        # Get agent sets for all frames in window
        agent_sets = [agents_by_time[t] for t in window_times]

        # Check if all sets are identical and non-empty
        if len(agent_sets[0]) > 0 and all(s == agent_sets[0] for s in agent_sets):
            valid_windows.append((start_t, agent_sets[0]))

    return valid_windows


def compute_positions_generic(
    df: pd.DataFrame,
    schema: SchemaConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute positions and relative positions from DataFrame.

    Args:
        df: DataFrame with position data for a single timestep (or pivoted)
        schema: Schema configuration

    Returns:
        positions: Tensor of shape (num_agents, num_dims)
        relative_positions: Tensor of shape (num_agents, num_agents, num_dims)
            where relative_positions[i, j] = position[j] - position[i]
    """
    pos_cols = schema.position_columns

    # Get positions as numpy array, sorted by agent_id
    df_sorted = df.sort_values(schema.agent_id_column)
    positions = torch.from_numpy(df_sorted[pos_cols].to_numpy()).float()

    # Compute relative positions: rel[i,j] = pos[j] - pos[i]
    # Shape: (num_agents, num_agents, num_dims)
    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

    return positions, relative_positions


def build_complete_graph_edges(
    num_nodes: int,
    relative_positions: torch.Tensor,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    Build edge index and attributes for a complete graph.

    Args:
        num_nodes: Number of nodes (agents)
        relative_positions: Tensor of shape (num_nodes, num_nodes, num_dims)

    Returns:
        edge_index: Shape (2, num_nodes^2)
        edge_attr: Shape (num_nodes^2, num_dims) - relative positions as edge features
    """
    # Create complete graph edge index (including self-loops)
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    edge_index = torch.stack([src, dst], dim=0)

    # Edge attributes are relative positions
    edge_attr = relative_positions[src, dst]

    return edge_index, edge_attr


def build_node_features(
    df: pd.DataFrame,
    schema: SchemaConfig,
    agent_ids: List[int],
    timesteps: List[int],
) -> torch.Tensor:
    """
    Build node feature tensor from DataFrame for given agents and timesteps.

    Args:
        df: DataFrame with observation data
        schema: Schema configuration
        agent_ids: Ordered list of agent IDs (determines node ordering)
        timesteps: List of timesteps to include in the window

    Returns:
        Tensor of shape (num_agents, num_features * window_length)
    """
    feature_cols = schema.node_feature_columns
    time_col = schema.time_column
    agent_col = schema.agent_id_column

    # Create agent_id to index mapping for consistent ordering
    agent_to_idx = {aid: idx for idx, aid in enumerate(sorted(agent_ids))}
    num_agents = len(agent_ids)
    num_features = len(feature_cols)
    window_length = len(timesteps)

    # Initialize feature tensor
    features = torch.zeros(num_agents, num_features * window_length)

    for t_idx, t in enumerate(timesteps):
        df_t = df[df[time_col] == t]
        for _, row in df_t.iterrows():
            agent_idx = agent_to_idx[row[agent_col]]
            for f_idx, col in enumerate(feature_cols):
                feat_idx = t_idx * num_features + f_idx
                features[agent_idx, feat_idx] = row[col]

    return features


def build_labels(
    df: pd.DataFrame,
    schema: SchemaConfig,
    agent_ids: List[int],
    timestep: int,
) -> torch.Tensor:
    """
    Build label tensor from DataFrame for given agents at a specific timestep.

    Args:
        df: DataFrame with observation data
        schema: Schema configuration
        agent_ids: Ordered list of agent IDs
        timestep: Timestep to get labels from

    Returns:
        Tensor of shape (num_agents, num_label_dims)
    """
    label_cols = schema.label_columns
    time_col = schema.time_column
    agent_col = schema.agent_id_column

    agent_to_idx = {aid: idx for idx, aid in enumerate(sorted(agent_ids))}
    num_agents = len(agent_ids)

    labels = torch.zeros(num_agents, len(label_cols))

    df_t = df[df[time_col] == timestep]
    for _, row in df_t.iterrows():
        agent_idx = agent_to_idx[row[agent_col]]
        for l_idx, col in enumerate(label_cols):
            labels[agent_idx, l_idx] = row[col]

    return labels


class CollabSQLDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset that fetches episode data from a SQL database.

    This dataset:
    1. Queries episodes for a given session from the database
    2. Downloads and caches observation data as parquet files
    3. Processes the data into PyG Data objects for GNN training
    4. Handles variable agent counts by only including valid windows

    Args:
        root: Directory for caching downloaded/processed data
        connection_string: PostgreSQL connection string
        session_id: Session identifier to fetch episodes for
        schema_config: Configuration mapping SQL columns to node features
        time_window_length: Number of consecutive frames per input window
        label_offset: How many frames ahead to predict (default: 1)
        transform: PyG transform to apply to each Data object
        pre_transform: PyG pre-transform to apply during processing
        load_only: If True, only load existing processed data (don't download/process)
    """

    def __init__(
        self,
        root: str,
        connection_string: str,
        session_id: str,
        schema_config: SchemaConfig,
        time_window_length: int = 1,
        label_offset: int = 1,
        transform=None,
        pre_transform=None,
        load_only: bool = False,
    ):
        self.connection_string = connection_string
        self.session_id = session_id
        self.schema_config = schema_config
        self.time_window_length = time_window_length
        self.label_offset = label_offset
        self._load_only = load_only

        # Will be populated during download()
        self._episode_info: List[dict] = []
        self._raw_file_names: List[str] = []

        # Expand root path
        self.root_path = expand_path(root, get_project_root())
        root = str(self.root_path)

        # Check if we're in load-only mode and data doesn't exist
        if load_only:
            processed_path = self.root_path / "processed" / "metadata.json"
            if not processed_path.exists():
                raise FileNotFoundError(
                    f"Dataset in load-only mode but {processed_path} doesn't exist."
                )
            # Load episode info from metadata
            with open(processed_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self._episode_info = metadata.get("episode_info", [])
            self._raw_file_names = [
                f"episode_{ep['episode_id']}.parquet" for ep in self._episode_info
            ]

        # Call parent init (triggers download and process if needed)
        super().__init__(root, transform, pre_transform)

        # Load processed episodes
        self._metadata = None
        self.episodes = self._load_episodes()

        # Set dimension info from first episode
        if len(self.episodes) > 0 and len(self.episodes[0]) > 0:
            first_data = self.episodes[0][0]
            self._input_node_dim = first_data.x.shape[1]
            self._edge_attr_dim = first_data.edge_attr.shape[1]
            self._label_dim = first_data.y.shape[1]
        else:
            self._input_node_dim = 0
            self._edge_attr_dim = 0
            self._label_dim = 0

        # Load metadata
        with open(self.processed_paths[0], "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    @property
    def raw_file_names(self) -> List[str]:
        """List of raw parquet files (one per episode)."""
        if self._raw_file_names:
            return self._raw_file_names

        # If not populated yet, query the database
        if not self._load_only:
            self._fetch_episode_info()

        return self._raw_file_names

    @property
    def processed_file_names(self) -> List[str]:
        """Indicator file for processed data."""
        return ["metadata.json"]

    def _fetch_episode_info(self):
        """Query database for episode information."""
        engine = create_engine(self.connection_string)
        query = text("""
            SELECT episode_id, episode_number, num_frames, num_agents
            FROM episodes
            WHERE session_id = :session_id
            ORDER BY episode_number
        """)

        with engine.connect() as conn:
            result = conn.execute(query, {"session_id": self.session_id})
            self._episode_info = [
                {
                    "episode_id": row[0],
                    "episode_number": row[1],
                    "num_frames": row[2],
                    "num_agents": row[3],
                }
                for row in result
            ]

        self._raw_file_names = [
            f"episode_{ep['episode_id']}.parquet" for ep in self._episode_info
        ]
        engine.dispose()

    def _get_required_columns(self) -> List[str]:
        """Get list of columns needed from observations table based on schema."""
        schema = self.schema_config
        columns = set()

        # Always need time and agent ID
        columns.add(schema.time_column)
        columns.add(schema.agent_id_column)

        # Add columns from schema config
        columns.update(schema.node_feature_columns)
        columns.update(schema.position_columns)
        columns.update(schema.label_columns)

        # Add agent_type_id if we need to filter
        if schema.agent_type_filter is not None:
            columns.add("agent_type_id")

        return sorted(columns)

    def download(self):
        """Download episode data from SQL database and save as parquet files."""
        if self._load_only:
            return

        # Ensure episode info is fetched
        if not self._episode_info:
            self._fetch_episode_info()

        engine = create_engine(self.connection_string)

        # Build dynamic query based on required columns
        required_cols = self._get_required_columns()
        cols_str = ", ".join(required_cols)
        obs_query = text(f"""
            SELECT {cols_str}
            FROM observations
            WHERE episode_id = :episode_id
            ORDER BY {self.schema_config.time_column}, {self.schema_config.agent_id_column}
        """)

        raw_dir = self.root_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {len(self._episode_info)} episodes from database...")
        for ep_info in tqdm(self._episode_info, desc="Downloading episodes"):
            episode_id = ep_info["episode_id"]

            with engine.connect() as conn:
                df = pd.read_sql(obs_query, conn, params={"episode_id": episode_id})

            # Save as parquet
            output_path = raw_dir / f"episode_{episode_id}.parquet"
            df.to_parquet(output_path, index=False)

        engine.dispose()

    def process(self):
        """Process raw parquet files into PyG Data objects."""
        schema = self.schema_config
        time_col = schema.time_column
        agent_col = schema.agent_id_column

        # Ensure episode info is available
        if not self._episode_info:
            # Load from downloaded files if needed
            raw_files = sorted(
                list((self.root_path / "raw").glob("episode_*.parquet")),
                key=sort_using_numbers,
            )
            self._episode_info = [
                {"episode_id": f.stem.replace("episode_", ""), "episode_number": i}
                for i, f in enumerate(raw_files)
            ]
            self._raw_file_names = [f.name for f in raw_files]

        processed_dir = self.root_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        all_episode_metadata = []

        print(f"Processing {len(self._episode_info)} episodes...")
        for ep_idx, ep_info in enumerate(tqdm(self._episode_info, desc="Processing")):
            episode_id = ep_info["episode_id"]
            raw_path = self.root_path / "raw" / f"episode_{episode_id}.parquet"

            # Load parquet
            df = pd.read_parquet(raw_path)

            # Apply agent type filter if specified
            if schema.agent_type_filter is not None:
                df = df[df["agent_type_id"] == schema.agent_type_filter].copy()

            # Find valid windows
            valid_windows = get_valid_window_starts(
                df,
                time_col,
                agent_col,
                self.time_window_length,
                self.label_offset,
            )

            if len(valid_windows) == 0:
                print(f"Warning: Episode {episode_id} has no valid windows, skipping.")
                continue

            # Build Data objects for each valid window
            data_list = []
            for start_t, agent_set in valid_windows:
                agent_ids = sorted(agent_set)
                num_agents = len(agent_ids)

                # Window timesteps for input features
                input_timesteps = list(
                    range(start_t, start_t + self.time_window_length)
                )

                # Label timestep
                label_t = start_t + self.time_window_length - 1 + self.label_offset

                # Filter DataFrame to relevant agents
                df_window = df[df[agent_col].isin(agent_ids)]

                # Build node features
                node_features = build_node_features(
                    df_window, schema, agent_ids, input_timesteps
                )

                # Build labels
                labels = build_labels(df_window, schema, agent_ids, label_t)

                # Get positions at last input timestep for edge attributes
                df_last_input = df_window[
                    df_window[time_col] == input_timesteps[-1]
                ]
                positions, relative_positions = compute_positions_generic(
                    df_last_input, schema
                )

                # Build graph edges
                edge_index, edge_attr = build_complete_graph_edges(
                    num_agents, relative_positions
                )

                # Apply normalization if scale factor is set
                if schema.scale_factor is not None:
                    scale = schema.scale_factor
                    node_features = node_features / scale
                    labels = labels / scale
                    edge_attr = edge_attr / scale

                # Create Data object
                data = Data(
                    x=node_features,
                    y=labels,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )

                data_list.append(data)

            # Save processed episode
            output_path = processed_dir / f"episode_{episode_id}.pt"
            torch.save(data_list, output_path)

            all_episode_metadata.append(
                {
                    "episode_id": episode_id,
                    "episode_number": ep_info.get("episode_number", ep_idx),
                    "num_graphs": len(data_list),
                    "num_valid_windows": len(valid_windows),
                }
            )

        # Save metadata
        metadata = {
            "session_id": self.session_id,
            "schema_config": schema.to_dict(),
            "time_window_length": self.time_window_length,
            "label_offset": self.label_offset,
            "episode_info": self._episode_info,
            "episode_metadata": all_episode_metadata,
            "input_node_dim": (
                data_list[0].x.shape[1] if data_list else schema.num_node_features * self.time_window_length
            ),
            "edge_attr_dim": schema.num_position_dims,
            "label_dim": schema.num_label_dims,
        }

        with open(self.processed_paths[0], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _load_episodes(self) -> List[List[Data]]:
        """Load all processed episode files."""
        episode_files = sorted(
            list((self.root_path / "processed").glob("episode_*.pt")),
            key=sort_using_numbers,
        )

        print(f"Loading {len(episode_files)} processed episodes...")
        episodes = []
        for ep_file in tqdm(episode_files, desc="Loading episodes"):
            data_list = torch.load(ep_file, weights_only=False)
            episodes.append(data_list)

        return episodes

    def __len__(self) -> int:
        """Number of episodes in the dataset."""
        return len(self.episodes)

    def __getitem__(self, index: int) -> Tuple[List[Data], int]:
        """
        Get episode and its index.

        Returns:
            Tuple of (list of Data objects for this episode, episode index)
        """
        return self.episodes[index], index

    @property
    def metadata(self) -> dict:
        """Dataset metadata."""
        return self._metadata

    @property
    def input_node_dim(self) -> int:
        """Dimensionality of node input features."""
        return self._input_node_dim

    @property
    def edge_attr_dim(self) -> int:
        """Dimensionality of edge attributes."""
        return self._edge_attr_dim

    @property
    def label_dim(self) -> int:
        """Dimensionality of labels."""
        return self._label_dim


class FlatGraphDataset(torch.utils.data.Dataset):
    """
    Wraps an episode-based dataset to expose individual graphs.

    This enables graph-level train/val splitting instead of episode-level splitting,
    which is useful when you have few episodes but many graphs.

    Each graph is wrapped as a single-element list so it's compatible with
    training loops that expect episodes (lists of graphs).

    Args:
        episode_dataset: Dataset where each item is (list of graphs, episode_index)
    """

    def __init__(self, episode_dataset):
        self.graphs = []
        self.source_info = []  # Track which episode each graph came from

        for episode_idx in range(len(episode_dataset)):
            episode_graphs, _ = episode_dataset[episode_idx]
            for graph_idx, graph in enumerate(episode_graphs):
                self.graphs.append(graph)
                self.source_info.append(
                    {"episode_idx": episode_idx, "graph_idx": graph_idx}
                )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Tuple[List[Data], int]:
        """
        Returns graph wrapped as single-element list for compatibility with
        episode-based training loops.

        Returns:
            Tuple of ([single graph], index)
        """
        return [self.graphs[index]], index


def split_flat_dataset(
    flat_dataset: FlatGraphDataset,
    train_fraction: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[torch.utils.data.Subset, List[int], torch.utils.data.Subset, List[int]]:
    """
    Split a FlatGraphDataset into train and validation subsets.

    Args:
        flat_dataset: FlatGraphDataset to split
        train_fraction: Fraction of data for training (default: 0.8)
        seed: Random seed for reproducibility (default: random)

    Returns:
        Tuple of (train_subset, train_indices, val_subset, val_indices)
    """
    if seed is None:
        seed = np.random.randint(low=0, high=2**31)

    torch_generator = torch.manual_seed(seed)
    np.random.seed(seed)

    train_size = int(len(flat_dataset) * train_fraction)
    val_size = len(flat_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        flat_dataset, [train_size, val_size], generator=torch_generator
    )

    return train_subset, list(train_subset.indices), val_subset, list(val_subset.indices)


def train_sql_dataset(
    dataset: CollabSQLDataset,
    output_dir: str,
    num_epochs: int = 100,
    train_fraction: float = 0.8,
    learning_rate: float = 1e-3,
    seed: Optional[int] = None,
    include_second_layer: bool = False,
    mlp_layers: Optional[List[int]] = None,
) -> dict:
    """
    Train a GNN on a CollabSQLDataset using graph-level splitting.

    This function:
    1. Flattens all episode graphs into a single pool
    2. Splits at graph level (not episode level)
    3. Trains using the train_epoch function from train_3DGNN.py

    Args:
        dataset: CollabSQLDataset to train on
        output_dir: Directory to save trained models
        num_epochs: Number of training epochs
        train_fraction: Fraction of graphs for training
        learning_rate: Learning rate for optimizer
        seed: Random seed for reproducibility
        include_second_layer: Add convolutional layer after attention
        mlp_layers: Optional MLP layer dimensions

    Returns:
        Dictionary with training results (losses, final predictions, model)
    """
    # Import here to avoid circular imports
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MLP
    from torchinfo import summary

    from collab_env.gnn.gnn_3D.train_3DGNN import train_epoch
    from collab_env.gnn.gnn_3D.gnn_models import GNN_Attention

    # Flatten dataset for graph-level splitting
    flat_dataset = FlatGraphDataset(dataset)
    print(f"Total graphs: {len(flat_dataset)}")

    # Split into train/val
    train_subset, train_indices, val_subset, val_indices = split_flat_dataset(
        flat_dataset, train_fraction=train_fraction, seed=seed
    )
    print(f"Train graphs: {len(train_subset)}, Val graphs: {len(val_subset)}")

    # Create data loaders
    train_loader = DataLoader(dataset=train_subset, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=1, shuffle=False)

    # Create model
    if mlp_layers is not None:
        mlp = MLP(mlp_layers)
    else:
        mlp = None

    model = GNN_Attention(
        model_name="GNN-Attention-SQL",
        in_node_dim=dataset.input_node_dim,
        edge_dim=dataset.edge_attr_dim,
        output_dim=dataset.label_dim,
        self_loops=True,
        fill_value=torch.zeros(dataset.edge_attr_dim).float(),
        include_convolutional_layer=include_second_layer,
        mlp=mlp,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    summary(model)

    # Create output directory
    output_path = expand_path(output_dir, get_project_root())
    output_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Validation first
        val_loss, val_predictions, val_attention, val_idx_list = train_epoch(
            model=model, loader=val_loader, optimizer=optimizer, train=False
        )
        val_loss_list.append(val_loss)
        print(f"Val loss: {val_loss:.6f}")

        # Training
        train_loss, train_predictions, train_attention, train_idx_list = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, train=True
        )
        train_loss_list.append(train_loss)
        print(f"Train loss: {train_loss:.6f}")

        # Save model checkpoint
        model_path = output_path / f"model_epoch_{epoch}.pt"
        torch.save(model, model_path)

    # Final validation
    val_loss, val_predictions, val_attention, val_idx_list = train_epoch(
        model=model, loader=val_loader, optimizer=optimizer, train=False
    )
    print(f"\nFinal val loss: {val_loss:.6f}")

    # Save final model
    final_model_path = output_path / "model_final.pt"
    torch.save(model, final_model_path)

    # Save training metadata
    training_metadata = {
        "num_epochs": num_epochs,
        "train_fraction": train_fraction,
        "learning_rate": learning_rate,
        "seed": seed,
        "num_train_graphs": len(train_subset),
        "num_val_graphs": len(val_subset),
        "train_losses": train_loss_list,
        "val_losses": val_loss_list,
        "dataset_metadata": dataset.metadata,
    }

    metadata_path = output_path / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)

    return {
        "train_losses": train_loss_list,
        "val_losses": val_loss_list,
        "train_predictions": train_predictions,
        "train_attention": train_attention,
        "train_indices": train_idx_list,
        "val_predictions": val_predictions,
        "val_attention": val_attention,
        "val_indices": val_idx_list,
        "trained_model": model,
        "dataset_metadata": dataset.metadata,
        "output_path": output_path,
    }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Build GNN dataset from SQL database."
    )
    parser.add_argument(
        "-c", "--connection_string", type=str, required=True, help="Database connection string"
    )
    parser.add_argument(
        "-s", "--session_id", type=str, required=True, help="Session ID to fetch"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Output directory for cached data"
    )
    parser.add_argument(
        "-twl", "--time_window_length", type=int, default=1, help="Time window length"
    )
    parser.add_argument(
        "-lo", "--label_offset", type=int, default=1, help="Label offset"
    )
    parser.add_argument(
        "--scale_factor", type=float, default=None, help="Normalization scale factor"
    )

    args = parser.parse_args()

    schema = SchemaConfig(
        node_feature_columns=["v_x", "v_y", "v_z"],
        position_columns=["x", "y", "z"],
        label_columns=["x", "y", "z"],
        scale_factor=args.scale_factor,
    )

    dataset = CollabSQLDataset(
        root=args.output_dir,
        connection_string=args.connection_string,
        session_id=args.session_id,
        schema_config=schema,
        time_window_length=args.time_window_length,
        label_offset=args.label_offset,
    )

    print(f"Dataset created with {len(dataset)} episodes")
    print(f"Input node dim: {dataset.input_node_dim}")
    print(f"Edge attr dim: {dataset.edge_attr_dim}")
    print(f"Label dim: {dataset.label_dim}")
