"""
Configuration system for GraphGym experiments on boids trajectories.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    format: str = "BoidsTrajectory"
    name: str = "boid_single_species_basic"
    task: str = "node"
    task_type: str = "regression"
    transductive: bool = False
    split: List[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    visual_range: float = 0.1
    start_frame: int = 3
    input_differentiation: str = "finite"


@dataclass
class TrainConfig:
    """Training configuration."""
    mode: str = "custom"
    batch_size: int = 16
    eval_period: int = 1
    ckpt_period: int = 50
    enable_ckpt: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-6


@dataclass
class ModelConfig:
    """Model configuration."""
    type: str = "gnn"
    loss_fun: str = "mse"
    edge_decoding: str = "none"
    graph_pooling: str = "none"


@dataclass
class GNNConfig:
    """GNN architecture configuration."""
    # Layer structure
    layers_pre_mp: int = 1
    layers_mp: int = 2
    layers_post_mp: int = 1

    # Architecture
    dim_inner: int = 128
    layer_type: str = "gatv2conv"
    stage_type: str = "stack"
    agg: str = "add"

    # Attention-specific
    heads: int = 1
    head_mode: str = "concat"

    # Regularization
    batchnorm: bool = True
    act: str = "prelu"
    dropout: float = 0.0
    normalize_adj: bool = False

    # Output
    out_dim: int = 2


@dataclass
class OptimConfig:
    """Optimizer configuration."""
    optimizer: str = "adam"
    base_lr: float = 0.0001
    weight_decay: float = 0.0
    max_epoch: int = 50
    scheduler: str = "reduce_on_plateau"
    reduce_factor: float = 0.5
    schedule_patience: int = 5
    min_lr: float = 1e-6


@dataclass
class CustomConfig:
    """Custom configuration for trajectory prediction."""
    rollout: bool = True
    rollout_start_frame: int = 5
    rollout_total_frames: int = 100
    compute_mae: bool = True
    compute_r2: bool = True
    compute_per_timestep_acc: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    out_dir: str = "results/graphgym"
    metric_best: str = "loss"

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    custom: CustomConfig = field(default_factory=CustomConfig)

    seed: int = 0
    num_threads: int = 4
    device: str = "auto"

    print_freq: int = 10
    tensorboard_each_run: bool = False
    tensorboard_agg: bool = True


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ExperimentConfig object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert nested dicts to dataclass instances
    config = ExperimentConfig()

    if 'dataset' in config_dict:
        config.dataset = DatasetConfig(**config_dict['dataset'])
    if 'train' in config_dict:
        config.train = TrainConfig(**config_dict['train'])
    if 'model' in config_dict:
        config.model = ModelConfig(**config_dict['model'])
    if 'gnn' in config_dict:
        config.gnn = GNNConfig(**config_dict['gnn'])
    if 'optim' in config_dict:
        config.optim = OptimConfig(**config_dict['optim'])
    if 'custom' in config_dict:
        config.custom = CustomConfig(**config_dict['custom'])

    # Top-level fields
    for key in ['out_dir', 'metric_best', 'seed', 'num_threads', 'device',
                'print_freq', 'tensorboard_each_run', 'tensorboard_agg']:
        if key in config_dict:
            setattr(config, key, config_dict[key])

    logger.info(f"Config loaded successfully")
    return config


def create_config(
    dataset_name: str = "boid_single_species_basic",
    visual_range: float = 0.1,
    layer_type: str = "gatv2conv",
    dim_inner: int = 128,
    heads: int = 1,
    **kwargs
) -> ExperimentConfig:
    """
    Create a configuration programmatically.

    Args:
        dataset_name: Name of the dataset
        visual_range: Visual range for edge construction
        layer_type: GNN layer type
        dim_inner: Hidden dimension
        heads: Number of attention heads
        **kwargs: Additional config overrides

    Returns:
        ExperimentConfig object
    """
    config = ExperimentConfig()

    # Set dataset config
    config.dataset.name = dataset_name
    config.dataset.visual_range = visual_range

    # Set GNN config
    config.gnn.layer_type = layer_type
    config.gnn.dim_inner = dim_inner
    config.gnn.heads = heads

    # Apply any additional overrides
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested keys like 'gnn.dropout'
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Top-level key
            setattr(config, key, value)

    return config


def save_config(config: ExperimentConfig, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: ExperimentConfig object
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = asdict(config)

    logger.info(f"Saving config to: {save_path}")

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def parse_grid_search(grid_file: str) -> List[Dict[str, Any]]:
    """
    Parse grid search configuration from text file.

    Format of grid file:
        config.path alias [val1,val2,val3]

    Example:
        gnn.dim_inner dim [64,128,256]
        gnn.layer_type layer ['gcnconv','gatv2conv']

    Args:
        grid_file: Path to grid search file

    Returns:
        List of parameter grids
    """
    grid_file = Path(grid_file)

    if not grid_file.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_file}")

    logger.info(f"Parsing grid search from: {grid_file}")

    grids = []

    with open(grid_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse line: config_path alias values
            parts = line.split()
            if len(parts) < 3:
                logger.warning(f"Skipping invalid line: {line}")
                continue

            config_path = parts[0]
            alias = parts[1]
            values_str = ' '.join(parts[2:])

            # Parse values list
            try:
                values = eval(values_str)  # Safely parse Python list
                if not isinstance(values, list):
                    values = [values]
            except Exception as e:
                logger.error(f"Failed to parse values '{values_str}': {e}")
                continue

            grids.append({
                'config_path': config_path,
                'alias': alias,
                'values': values,
            })

    logger.info(f"Parsed {len(grids)} parameter grids")
    return grids


def generate_grid_configs(
    base_config: ExperimentConfig,
    grid_file: str,
    out_dir: Optional[str] = None
) -> List[ExperimentConfig]:
    """
    Generate all configurations for a grid search.

    Args:
        base_config: Base configuration
        grid_file: Path to grid search specification
        out_dir: Output directory for configs (optional)

    Returns:
        List of ExperimentConfig objects
    """
    grids = parse_grid_search(grid_file)

    # Generate all combinations using recursive approach
    def generate_combinations(grids, idx=0, current_config=None):
        if current_config is None:
            current_config = base_config

        if idx >= len(grids):
            return [current_config]

        grid = grids[idx]
        config_path = grid['config_path']
        values = grid['values']

        configs = []
        for value in values:
            # Create a copy of the config
            import copy
            new_config = copy.deepcopy(current_config)

            # Set the value
            parts = config_path.split('.')
            obj = new_config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

            # Recursively generate for remaining grids
            configs.extend(generate_combinations(grids, idx + 1, new_config))

        return configs

    all_configs = generate_combinations(grids)

    logger.info(f"Generated {len(all_configs)} configurations from grid search")

    # Optionally save configs
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, config in enumerate(all_configs):
            save_path = out_dir / f"config_{i:04d}.yaml"
            save_config(config, str(save_path))

    return all_configs
