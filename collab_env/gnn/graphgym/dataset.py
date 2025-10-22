"""
Data adapter to convert boids trajectory data to GraphGym-compatible format.

This module converts the boids tracking data (both simulated and real) into
temporal graphs suitable for GraphGym's architecture search framework.
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from loguru import logger

from collab_env.data.file_utils import expand_path, get_project_root
from collab_env.gnn.utility import handle_discrete_data


class BoidsGraphGymDataset(Dataset):
    """
    Convert boids trajectory data to temporal graphs for GraphGym.

    Each sample in the dataset represents a trajectory snippet where we predict
    agent accelerations at time t given positions/velocities at t-k:t.

    Graph construction:
    - Nodes: Agents at a given timestep
    - Edges: Based on visual range (spatial proximity)
    - Node features: Position, velocity, species
    - Node labels: Acceleration (regression target)
    """

    def __init__(
        self,
        dataset_name: str = "boid_single_species_basic",
        visual_range: float = 0.1,
        start_frame: int = 3,
        input_differentiation: str = "finite",
        split: str = "train",  # 'train', 'val', or 'test'
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        root: Optional[str] = None,
        seed: int = 0,
    ):
        """
        Args:
            dataset_name: Name of the dataset (e.g., 'boid_single_species_basic')
            visual_range: Distance threshold for edge creation
            start_frame: First frame to start predictions (need history for velocity/acc)
            input_differentiation: Method to compute velocities ('finite' or 'spline')
            split: Which split to load ('train', 'val', 'test')
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            root: Root directory for data (defaults to project root)
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.visual_range = visual_range
        self.start_frame = start_frame
        self.input_differentiation = input_differentiation
        self.split = split
        self.seed = seed

        # Set root directory
        if root is None:
            self.root = get_project_root()
        else:
            self.root = Path(root)

        # Load dataset
        logger.info(f"Loading dataset: {dataset_name} (split: {split})")
        self._load_dataset()

        # Split data
        self._split_data(train_ratio, val_ratio)

        logger.info(f"Dataset loaded: {len(self)} samples in {split} split")

    def _load_dataset(self):
        """Load the raw trajectory dataset."""
        file_name = f"runpod/{self.dataset_name}.pt"
        config_name = f"runpod/{self.dataset_name}_config.pt"

        dataset_path = expand_path(f"simulated_data/{file_name}", self.root)
        config_path = expand_path(f"simulated_data/{config_name}", self.root)

        logger.debug(f"Loading from: {dataset_path}")

        # Load dataset - it's an AnimalTrajectoryDataset
        raw_dataset = torch.load(dataset_path, weights_only=False)
        self.species_configs = torch.load(config_path, weights_only=False)
        self.species_dim = len(self.species_configs.keys())

        # Extract all trajectories
        self.trajectories = []
        self.species_labels = []

        for idx in range(len(raw_dataset)):
            position, species = raw_dataset[idx]  # position: [T, N, 2], species: [N]
            self.trajectories.append(position)
            self.species_labels.append(species)

        logger.debug(f"Loaded {len(self.trajectories)} trajectories")

    def _split_data(self, train_ratio: float, val_ratio: float):
        """Split data into train/val/test."""
        total = len(self.trajectories)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size

        # Set seed for reproducible splits
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Create indices
        indices = list(range(total))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Select split
        if self.split == "train":
            self.indices = train_indices
        elif self.split == "val":
            self.indices = val_indices
        elif self.split == "test":
            self.indices = test_indices
        else:
            raise ValueError(f"Unknown split: {self.split}")

        logger.debug(f"Split sizes - train: {len(train_indices)}, "
                    f"val: {len(val_indices)}, test: {len(test_indices)}")

    def len(self):
        """Return number of samples in this split."""
        return len(self.indices)

    def __len__(self):
        """Return number of samples in this split."""
        return len(self.indices)

    def get(self, idx):
        """Get a single graph sample."""
        # Map to actual trajectory index
        traj_idx = self.indices[idx]

        position = self.trajectories[traj_idx]  # [T, N, 2]
        species = self.species_labels[traj_idx]  # [N]

        # Convert to numpy for differentiation
        position_np = position.numpy() if isinstance(position, torch.Tensor) else position
        position_np = np.expand_dims(position_np, axis=0)  # [1, T, N, 2]

        # Compute velocities and accelerations
        p_smooth, v_smooth, a_smooth, _ = handle_discrete_data(
            position_np, self.input_differentiation
        )

        # Convert back to tensors
        p = torch.from_numpy(p_smooth).float().squeeze(0)  # [T, N, 2]
        v = torch.from_numpy(v_smooth).float().squeeze(0)  # [T, N, 2]
        a = torch.from_numpy(a_smooth).float().squeeze(0)  # [T, N, 2]

        # Create list of graphs (one per timestep, starting from start_frame)
        graphs = []
        for t in range(self.start_frame, p.shape[0]):
            graph = self._create_graph_at_timestep(p, v, a, species, t)
            if graph is not None:
                graphs.append(graph)

        # For now, return a random timestep's graph
        # In future, could return sequence of graphs for temporal modeling
        if len(graphs) == 0:
            # Fallback: return a dummy graph
            return self._create_dummy_graph()

        # Return a random graph from this trajectory
        graph_idx = np.random.randint(0, len(graphs))
        return graphs[graph_idx]

    def _create_graph_at_timestep(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        species: torch.Tensor,
        t: int
    ) -> Optional[Data]:
        """
        Create a graph at a specific timestep.

        Args:
            p: Positions [T, N, 2]
            v: Velocities [T, N, 2]
            a: Accelerations [T, N, 2]
            species: Species labels [N]
            t: Timestep index

        Returns:
            PyG Data object
        """
        num_agents = p.shape[1]

        # Node features: [pos_x, pos_y, vel_x, vel_y, species_one_hot...]
        pos_t = p[t]  # [N, 2]
        vel_t = v[t]  # [N, 2]

        # One-hot encode species
        species_onehot = torch.zeros(num_agents, self.species_dim)
        species_onehot.scatter_(1, species.unsqueeze(1), 1)

        # Concatenate features
        node_features = torch.cat([pos_t, vel_t, species_onehot], dim=1)  # [N, 4+species_dim]

        # Node labels: acceleration at time t
        node_labels = a[t]  # [N, 2]

        # Build edges based on visual range
        edge_index = self._build_edges(pos_t, self.visual_range)

        # Edge attributes: distance
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            edge_distances = torch.norm(pos_t[src] - pos_t[dst], dim=1, keepdim=True)
            edge_attr = edge_distances
        else:
            edge_attr = torch.zeros(0, 1)

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_labels,
            num_nodes=num_agents,
            # Metadata
            timestep=t,
            trajectory_id=0,  # Could track this if needed
        )

        return data

    def _build_edges(self, positions: torch.Tensor, visual_range: float) -> torch.Tensor:
        """
        Build edge index based on visual range.

        Args:
            positions: Node positions [N, 2]
            visual_range: Distance threshold

        Returns:
            edge_index [2, E]
        """
        N = positions.shape[0]

        # Compute pairwise distances
        dist = torch.cdist(positions, positions, p=2)  # [N, N]

        # Apply visual range filter and remove self-loops
        adj = (dist < visual_range).to(torch.bool)
        adj.fill_diagonal_(False)

        # Convert to edge index
        edge_index = adj.nonzero(as_tuple=False).T  # [2, E]

        return edge_index

    def _create_dummy_graph(self) -> Data:
        """Create a dummy graph for error cases."""
        return Data(
            x=torch.zeros(1, 4 + self.species_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 1),
            y=torch.zeros(1, 2),
            num_nodes=1,
        )


def create_boids_datasets(
    dataset_name: str = "boid_single_species_basic",
    visual_range: float = 0.1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
    **kwargs
) -> Tuple[BoidsGraphGymDataset, BoidsGraphGymDataset, BoidsGraphGymDataset]:
    """
    Create train, validation, and test datasets for boids trajectories.

    Args:
        dataset_name: Name of the dataset
        visual_range: Distance threshold for edges
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        seed: Random seed
        **kwargs: Additional arguments for BoidsGraphGymDataset

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = BoidsGraphGymDataset(
        dataset_name=dataset_name,
        visual_range=visual_range,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        **kwargs
    )

    val_dataset = BoidsGraphGymDataset(
        dataset_name=dataset_name,
        visual_range=visual_range,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        **kwargs
    )

    test_dataset = BoidsGraphGymDataset(
        dataset_name=dataset_name,
        visual_range=visual_range,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        **kwargs
    )

    return train_dataset, val_dataset, test_dataset
