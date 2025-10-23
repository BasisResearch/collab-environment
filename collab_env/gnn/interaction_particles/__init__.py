"""
InteractionParticle model training for 2D boids data.
Adapted from https://github.com/saalfeldlab/decomp-gnn

Quick start:
    python -m collab_env.gnn.interaction_particles.train_2d_boids
"""

from .model import InteractionParticle
from .train import train_interaction_particle, evaluate_model
from .plotting import plot_interaction_functions, compare_with_true_boids

__all__ = [
    'InteractionParticle',
    'train_interaction_particle',
    'evaluate_model',
    'plot_interaction_functions',
    'compare_with_true_boids'
]
