"""GraphGym integration for GNN architecture search on boids trajectory data."""

from .dataset import BoidsGraphGymDataset
from .trainer import GraphGymTrainer
from .config import create_config, load_config

__all__ = [
    'BoidsGraphGymDataset',
    'GraphGymTrainer',
    'create_config',
    'load_config',
]
