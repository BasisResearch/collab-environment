"""Collab Environment - A collaborative environment for multi-agent simulations and tracking.

This package provides tools for:
- Multi-agent simulations (boids, swarm behavior)
- Graph Neural Networks for agent modeling
- Tracking and alignment of agents
- Data processing and visualization utilities
"""

# Import all subpackages to make them available
from . import alignment
from . import dashboard
from . import data
from . import gnn
from . import sim
from . import tracking
from . import utils

__version__ = "0.1.0"
__all__ = ["alignment", "dashboard", "data", "gnn", "sim", "tracking", "utils"]
