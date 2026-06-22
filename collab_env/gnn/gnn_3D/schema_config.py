"""
Schema configuration for GNN datasets.

This module defines the SchemaConfig dataclass that specifies how to map
tabular data columns to GNN node features, edge attributes, and labels.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class SchemaConfig:
    """
    Configuration for mapping tabular data to GNN graph structures.

    Attributes:
        node_feature_columns: Columns to use as node features (e.g., ["v_x", "v_y", "v_z"])
        position_columns: Columns for spatial positions, used to compute edge attributes
            as relative positions (e.g., ["x", "y", "z"])
        label_columns: Columns to use as prediction targets (e.g., ["x", "y", "z"])
        scale_factor: Optional normalization scale factor. If provided, all features,
            positions, and labels are divided by this value.
        agent_type_filter: Optional filter to select only specific agent types.
            If None, all agents are included.
        time_column: Name of the column containing time/frame indices (default: "time_index")
        agent_id_column: Name of the column containing agent identifiers (default: "agent_id")
    """

    node_feature_columns: List[str] = field(default_factory=list)
    position_columns: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    label_columns: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    scale_factor: Optional[float] = None
    agent_type_filter: Optional[str] = None
    time_column: str = "time_index"
    agent_id_column: str = "agent_id"

    @classmethod
    def from_yaml(cls, path: str) -> "SchemaConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("schema", data))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaConfig":
        """Create configuration from a dictionary."""
        return cls(
            node_feature_columns=data.get("node_feature_columns", []),
            position_columns=data.get("position_columns", ["x", "y", "z"]),
            label_columns=data.get("label_columns", ["x", "y", "z"]),
            scale_factor=data.get("scale_factor"),
            agent_type_filter=data.get("agent_type_filter"),
            time_column=data.get("time_column", "time_index"),
            agent_id_column=data.get("agent_id_column", "agent_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "node_feature_columns": self.node_feature_columns,
            "position_columns": self.position_columns,
            "label_columns": self.label_columns,
            "scale_factor": self.scale_factor,
            "agent_type_filter": self.agent_type_filter,
            "time_column": self.time_column,
            "agent_id_column": self.agent_id_column,
        }

    @property
    def num_node_features(self) -> int:
        """Number of node features per timestep."""
        return len(self.node_feature_columns)

    @property
    def num_position_dims(self) -> int:
        """Dimensionality of position space (for edge attributes)."""
        return len(self.position_columns)

    @property
    def num_label_dims(self) -> int:
        """Dimensionality of labels."""
        return len(self.label_columns)
