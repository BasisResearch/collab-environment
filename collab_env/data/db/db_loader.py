"""
Data loader for tracking analytics database.

Loads data from various sources into PostgreSQL or DuckDB:
- 3D Boids: Parquet files from collab_env.sim.boids
- 2D Boids: PyTorch .pt files from collab_env.sim.boids_gnn_temp
- Tracking CSV: Real-world tracking data from collab_env.tracking
- GNN Rollout: Model evaluation rollout pickle files with predictions
"""

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from collab_env.data.db.config import DBConfig, get_db_config

# Configure loguru logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level="INFO")

# Add file output to logs directory
log_dir = Path(__file__).parent.parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "db_loader_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # Rotate at midnight
    retention="30 days",  # Keep logs for 30 days
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy/PyTorch types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, numpy type, torch type, etc.)

    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def get_food_location_from_config(config: Dict[str, Any], scene_size: float) -> Optional[Tuple[float, float]]:
    """
    Extract food location from 2D boids species config.

    The config structure is:
    {
        'A': {'width': 480, 'height': 480, ...},
        'food0': {'x': 160.0, 'y': 0, 'counts': 1}
    }

    Food coordinates in config are in pixel coordinates. We normalize them
    and then scale by scene_size to match the coordinate system used for
    agent positions.

    Args:
        config: Species configuration dictionary
        scene_size: Scene size for coordinate scaling (default 480.0)

    Returns:
        (food_x, food_y) in scaled scene coordinates, or None if no food
    """
    if not config or 'food0' not in config:
        return None

    food_config = config['food0']

    # Get scene dimensions (default to scene_size if not in config)
    if 'A' in config:
        width = config['A'].get('width', scene_size)
        height = config['A'].get('height', scene_size)
    else:
        width = height = scene_size

    # Food config stores pixel coordinates, normalize first
    food_x_normalized = food_config['x'] / width
    food_y_normalized = food_config['y'] / height

    # Scale to scene coordinates (same as agent positions)
    food_x = food_x_normalized * scene_size
    food_y = food_y_normalized * scene_size

    logger.debug(f"Food location: pixel=({food_config['x']}, {food_config['y']}), "
                 f"normalized=({food_x_normalized:.4f}, {food_y_normalized:.4f}), "
                 f"scaled=({food_x:.2f}, {food_y:.2f})")

    return (food_x, food_y)


@dataclass
class SessionMetadata:
    """Metadata for a simulation/tracking session."""
    session_id: str
    session_name: str
    category_id: str  # 'boids_3d', 'boids_2d', 'tracking_csv' (references categories table)
    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""
    episode_id: str
    session_id: str
    episode_number: int
    num_frames: int
    num_agents: int
    frame_rate: float
    file_path: str


class DatabaseConnection:
    """Unified database connection using SQLAlchemy."""

    def __init__(self, config: DBConfig):
        self.config = config
        self.engine: Optional[Engine] = None

    def connect(self):
        """Establish database connection using SQLAlchemy."""
        url = self.config.sqlalchemy_url()
        self.engine = create_engine(url, echo=False)

        # Test connection
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        if self.config.backend == 'postgres':
            logger.info(f"Connected to PostgreSQL: {self.config.postgres.dbname}")
        else:
            logger.info(f"Connected to DuckDB: {self.config.duckdb.dbpath}")

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def transaction(self):
        """
        Context manager for transaction handling.

        Usage:
            with db.transaction() as conn:
                # All operations here are in a single transaction
                conn.execute(text("INSERT ..."))
                conn.execute(text("INSERT ..."))
                # Auto-commits on exit, rolls back on exception

        This is much faster for bulk operations as it commits only once.
        """
        return self.engine.begin()

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None, conn=None):
        """Execute a query with named parameters.

        Args:
            query: SQL query string
            params: Named parameters for query
            conn: Optional connection (for transactional usage)
        """
        if conn is not None:
            # Use provided connection (transactional mode - no auto-commit)
            conn.execute(text(query), params or {})
        else:
            # Create connection and auto-commit (non-transactional mode)
            with self.engine.connect() as conn:
                conn.execute(text(query), params or {})
                conn.commit()

    def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[tuple]:
        """Fetch one result."""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchone()

    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """Fetch all results."""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append', conn=None):
        """Insert DataFrame using pandas to_sql (much faster for bulk inserts).

        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: What to do if table exists ('append', 'replace', 'fail')
            conn: Optional connection (for transactional usage)
        """
        if conn is not None:
            # Use provided connection (transactional mode)
            df.to_sql(table_name, conn, if_exists=if_exists, index=False, method='multi')
        else:
            # Use engine (non-transactional mode - auto-commits)
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False, method='multi')


class BaseDataLoader:
    """Base class for data loaders."""

    def __init__(self, db_conn: DatabaseConnection, max_episodes: int = np.inf):
        self.db = db_conn
        self.max_episodes = max_episodes

    def load_session(self, metadata: SessionMetadata, conn=None):
        """Load session metadata into database.

        Args:
            metadata: Session metadata
            conn: Optional connection (for transactional usage)
        """
        # Convert config and metadata to JSON strings (convert numpy/torch types first)
        config_json = json.dumps(convert_to_json_serializable(metadata.config))
        metadata_json = json.dumps(convert_to_json_serializable(metadata.metadata)) if metadata.metadata else None

        query = """
        INSERT INTO sessions (session_id, session_name, category_id, config, metadata)
        VALUES (:session_id, :session_name, :category_id, :config, :metadata)
        """

        self.db.execute(query, {
            'session_id': metadata.session_id,
            'session_name': metadata.session_name,
            'category_id': metadata.category_id,
            'config': config_json,
            'metadata': metadata_json
        }, conn=conn)
        logger.info(f"Loaded session: {metadata.session_id}")

    def load_episode(self, metadata: EpisodeMetadata, conn=None):
        """Load episode metadata into database.

        Args:
            metadata: Episode metadata
            conn: Optional connection (for transactional usage)
        """
        query = """
        INSERT INTO episodes (episode_id, session_id, episode_number, num_frames, num_agents, frame_rate, file_path)
        VALUES (:episode_id, :session_id, :episode_number, :num_frames, :num_agents, :frame_rate, :file_path)
        """

        self.db.execute(query, {
            'episode_id': metadata.episode_id,
            'session_id': metadata.session_id,
            'episode_number': metadata.episode_number,
            'num_frames': metadata.num_frames,
            'num_agents': metadata.num_agents,
            'frame_rate': metadata.frame_rate,
            'file_path': metadata.file_path
        }, conn=conn)
        logger.info(f"Loaded episode: {metadata.episode_id}")

    def load_observations_batch(self, observations: pd.DataFrame, episode_id: str, conn=None):
        """Load observations in batch using pandas to_sql.

        Args:
            observations: DataFrame with observation data
            episode_id: Episode ID
            conn: Optional connection (for transactional usage)
        """
        # Ensure required columns exist
        required_cols = ['time_index', 'agent_id', 'x', 'y']
        for col in required_cols:
            if col not in observations.columns:
                raise ValueError(f"Missing required column: {col}")

        # Prepare DataFrame for insertion (optimized - work with copy to avoid side effects)
        df = observations.copy()

        # Add episode_id column (use assign for efficiency)
        df['episode_id'] = episode_id

        # Set default agent_type_id if missing
        if 'agent_type_id' not in df.columns:
            df['agent_type_id'] = 'agent'

        # Select only the columns we need in the correct order
        # This avoids type conversions for columns that are already correct
        col_order = ['episode_id', 'time_index', 'agent_id', 'agent_type_id', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z']
        existing_cols = [c for c in col_order if c in df.columns]
        df = df[existing_cols]

        # Use pandas to_sql for fast bulk insert
        self.db.insert_dataframe(df, 'observations', if_exists='append', conn=conn)
        logger.info(f"Loaded {len(df)} observations for episode {episode_id}")

    def load_extended_properties_batch(
        self,
        episode_id: str,
        property_data: Dict[str, pd.Series],
        conn=None
    ):
        """
        Load extended properties in batch.

        Args:
            episode_id: Episode identifier
            property_data: Dict mapping property_id to Series with values
                          Series index should match observations (time_index, agent_id)
            conn: Optional connection (for transactional usage)
        """
        # Get observation IDs for this episode
        # Optimize: try simple 2-tuple mapping first (fastest path for 90% of episodes)
        # IMPORTANT: Use the same connection to see uncommitted inserts in the transaction
        query = """
        SELECT observation_id, time_index, agent_id
        FROM observations
        WHERE episode_id = :episode_id
        ORDER BY time_index, agent_id
        """

        if conn is not None:
            # Use transactional connection
            result = conn.execute(text(query), {'episode_id': episode_id})
            obs_rows = result.fetchall()
        else:
            # Use non-transactional connection
            obs_rows = self.db.fetch_all(query, {'episode_id': episode_id})

        # Build simple 2-tuple mapping - works for most cases
        obs_id_map = {}
        has_collision = False
        for row in obs_rows:
            key = (row[1], row[2])  # (time_index, agent_id)
            if key in obs_id_map:
                # Collision detected - need to use 3-tuple mapping
                has_collision = True
                break
            obs_id_map[key] = row[0]

        # If collision detected, rebuild with 3-tuple mapping
        if has_collision:
            logger.info(f"Multiple agent types detected for episode {episode_id}, using 3-tuple mapping")
            query_3tuple = """
            SELECT observation_id, time_index, agent_id, agent_type_id
            FROM observations
            WHERE episode_id = :episode_id
            ORDER BY time_index, agent_id, agent_type_id
            """
            if conn is not None:
                result = conn.execute(text(query_3tuple), {'episode_id': episode_id})
                obs_rows = result.fetchall()
            else:
                obs_rows = self.db.fetch_all(query_3tuple, {'episode_id': episode_id})

            obs_id_map = {(row[1], row[2], row[3]): row[0] for row in obs_rows}
            # Also create 2-tuple fallback for 'agent' type (property data uses 2-tuples)
            obs_id_map_2tuple = {(row[1], row[2]): row[0] for row in obs_rows if row[3] == 'agent'}
        else:
            obs_id_map_2tuple = None

        # Prepare data for batch insert
        records = []
        total_property_values = sum(len(v) for v in property_data.values())
        logger.info(f"Processing {len(property_data)} properties with {total_property_values} total values")

        for property_id, values in property_data.items():
            property_records = 0
            for idx, value in values.items():
                # Determine observation_id based on index structure
                if isinstance(idx, tuple):
                    if len(idx) == 2:
                        # Try 2-tuple mapping (most common)
                        obs_id = obs_id_map_2tuple.get(idx) if obs_id_map_2tuple is not None else obs_id_map.get(idx)
                    elif len(idx) == 3 and has_collision:
                        # Try 3-tuple mapping
                        obs_id = obs_id_map.get(idx)
                    else:
                        obs_id = None
                else:
                    obs_id = None

                if obs_id is None:
                    logger.warning(f"No observation found for index {idx}")
                    continue

                if pd.notna(value):
                    records.append({
                        'observation_id': obs_id,
                        'property_id': property_id,
                        'value_float': float(value) if isinstance(value, (int, float)) else None,
                        'value_text': str(value) if not isinstance(value, (int, float)) else None
                    })
                    property_records += 1

            logger.info(f"Property '{property_id}': created {property_records} records from {len(values)} values")

        if not records:
            logger.info(f"No extended properties to load for episode {episode_id}")
            return

        # Use pandas to_sql for fast bulk insert
        df = pd.DataFrame(records)
        self.db.insert_dataframe(df, 'extended_properties', if_exists='append', conn=conn)
        logger.info(f"Loaded {len(records)} extended property values for episode {episode_id}")


class Boids3DLoader(BaseDataLoader):
    """Loader for 3D boids simulation data from parquet files."""

    def load_simulations_bulk(self, parent_dir: Path):
        """
        Load multiple simulations from a parent directory.

        Args:
            parent_dir: Directory containing multiple simulation subdirectories
        """
        # Discover all simulation directories (those with config.yaml)
        simulation_dirs = []
        for subdir in sorted(parent_dir.iterdir()):
            if subdir.is_dir() and (subdir / "config.yaml").exists():
                simulation_dirs.append(subdir)

        if not simulation_dirs:
            raise ValueError(f"No simulation directories found in {parent_dir}")

        logger.info(f"Found {len(simulation_dirs)} simulation directories in {parent_dir}")
        logger.info(f"Loading all simulations in single transaction...")

        # Load all simulations in one transaction for maximum performance
        with self.db.transaction() as conn:
            for idx, sim_dir in enumerate(simulation_dirs, 1):
                logger.info(f"[{idx}/{len(simulation_dirs)}] Loading {sim_dir.name}...")
                self._load_simulation_no_transaction(sim_dir, conn=conn)

        logger.info(f"Completed loading {len(simulation_dirs)} simulations")

    def _load_simulation_no_transaction(self, simulation_dir: Path, conn=None):
        """Internal method to load a simulation without starting a transaction.

        Args:
            simulation_dir: Path to simulation directory
            conn: Optional connection (for transactional usage)
        """
        logger.info(f"Loading 3D boids simulation from: {simulation_dir}")

        # Load config
        config_path = simulation_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create session metadata
        session_id = f"session-{simulation_dir.name}"
        session_metadata = SessionMetadata(
            session_id=session_id,
            session_name=simulation_dir.name,
            category_id='boids_3d',
            config=config,
            metadata={
                'simulation_dir': str(simulation_dir),
                'loader': 'Boids3DLoader'
            }
        )

        self.load_session(session_metadata, conn=conn)

        # Load each episode
        episode_files = sorted(simulation_dir.glob("episode-*.parquet"))
        logger.info(f"Loading up to {min(len(episode_files), self.max_episodes)} episodes...")

        episodes_loaded = 0
        for episode_num, episode_file in enumerate(episode_files):
            if episode_num >= self.max_episodes:
                logger.info(f"Reached maximum number of episodes ({self.max_episodes}) for session {session_id}, stopping...")
                break
            logger.info(f"[{episode_num + 1}/{len(episode_files)}] Loading episode {episode_num} from: {episode_file}")
            self.load_episode_file(session_id, episode_num, episode_file, config, conn=conn)
            episodes_loaded += 1

        logger.info(f"Completed loading simulation: {session_id} ({episodes_loaded} out of {len(episode_files)} episodes)")

    def load_simulation(self, simulation_dir: Path):
        """
        Load a complete 3D boids simulation.

        Args:
            simulation_dir: Directory containing config.yaml and episode-*.parquet files
        """
        logger.info(f"Loading 3D boids simulation from: {simulation_dir}")

        # Load config
        config_path = simulation_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create session metadata
        session_id = f"session-{simulation_dir.name}"
        session_metadata = SessionMetadata(
            session_id=session_id,
            session_name=simulation_dir.name,
            category_id='boids_3d',
            config=config,
            metadata={
                'simulation_dir': str(simulation_dir),
                'loader': 'Boids3DLoader'
            }
        )

        # Load all data in a single transaction for performance
        episode_files = sorted(simulation_dir.glob("episode-*.parquet"))
        logger.info(f"Loading up to {min(len(episode_files), self.max_episodes)} episodes in single transaction...")

        with self.db.transaction() as conn:
            self.load_session(session_metadata, conn=conn)

            # Load each episode
            episodes_loaded = 0
            for episode_num, episode_file in enumerate(episode_files):
                if episode_num >= self.max_episodes:
                    logger.info(f"Reached maximum number of episodes ({self.max_episodes}) for session {session_id}, stopping...")
                    break
                logger.info(f"[{episode_num + 1}/{len(episode_files)}] Loading episode {episode_num} from: {episode_file}")
                self.load_episode_file(session_id, episode_num, episode_file, config, conn=conn)
                episodes_loaded += 1

        logger.info(f"Completed loading simulation: {session_id} ({episodes_loaded} out of {len(episode_files)} episodes)")

    def load_episode_file(
        self,
        session_id: str,
        episode_number: int,
        file_path: Path,
        config: Dict[str, Any],
        conn=None
    ):
        """Load a single episode parquet file.

        Args:
            session_id: Session ID
            episode_number: Episode number
            file_path: Path to parquet file
            config: Configuration dictionary
            conn: Optional connection (for transactional usage)
        """
        logger.info(f"Loading episode {episode_number} from: {file_path}")

        # Read parquet file
        df = pd.read_parquet(file_path)

        # Log entity types (includes both agents and environment entities)
        if 'type' in df.columns:
            logger.info(f"Row count: {len(df)}, Types: {df['type'].value_counts().to_dict()}")

        # Extract metadata (convert numpy types to native Python types)
        num_frames = int(df['time'].max() + 1)
        num_agents = int(df['id'].nunique())
        frame_rate = float(config.get('frame_rate', 30.0))

        episode_id = f"episode-{episode_number}-{file_path.stem}"

        episode_metadata = EpisodeMetadata(
            episode_id=episode_id,
            session_id=session_id,
            episode_number=episode_number,
            num_frames=num_frames,
            num_agents=num_agents,
            frame_rate=frame_rate,
            file_path=str(file_path)
        )

        self.load_episode(episode_metadata, conn=conn)

        # Prepare observations DataFrame
        observations = pd.DataFrame({
            'time_index': df['time'],
            'agent_id': df['id'],
            'agent_type_id': df['type'],
            'x': df['x'],
            'y': df['y'],
            'z': df['z'],
            'v_x': df['v_x'],
            'v_y': df['v_y'],
            'v_z': df['v_z']
        })

        self.load_observations_batch(observations, episode_id, conn=conn)

        # Load extended properties if they exist
        # IMPORTANT: Only load extended properties for 'agent' type observations
        # Environment entities ('env' type) don't have extended properties and create
        # index collisions when they share agent_ids with actual agents
        import numpy as np
        extended_props = {}

        # Filter DataFrame to only include 'agent' type rows for extended properties
        agent_df = df[df['type'] == 'agent'].copy()

        # Map actual parquet column names to property IDs
        # Distance to target center (may have suffix like _1, and may or may not have '_to_')
        target_center_cols = [c for c in agent_df.columns if c.startswith('distance_target_center') or c.startswith('distance_to_target_center')]
        if target_center_cols:
            extended_props['distance_to_target_center'] = agent_df.set_index(['time', 'id'])[target_center_cols[0]]

        # Distance to target mesh (may have suffix like _1)
        target_mesh_cols = [c for c in agent_df.columns if 'distance_to_target_mesh' in c]
        if target_mesh_cols:
            extended_props['distance_to_target_mesh'] = agent_df.set_index(['time', 'id'])[target_mesh_cols[0]]

        # Distance to scene mesh
        if 'mesh_scene_distance' in agent_df.columns:
            extended_props['distance_to_scene_mesh'] = agent_df.set_index(['time', 'id'])['mesh_scene_distance']

        # Handle array-type closest point columns
        # Target mesh closest point (stored as array [x, y, z])
        # Match columns like 'target_mesh_closest_point_1' but NOT 'distance_to_target_mesh_closest_point_1'
        target_closest_cols = [c for c in agent_df.columns if 'target_mesh_closest_point' in c and not c.startswith('distance')]
        if target_closest_cols:
            # Extract array column and filter out None values
            arr_col = agent_df[target_closest_cols[0]]
            # Create mask for non-None values
            mask = arr_col.notna()
            filtered_df = agent_df[mask]
            filtered_arr_col = arr_col[mask]

            if len(filtered_arr_col) > 0:
                # Stack arrays into 2D numpy array
                logger.info(f"Before stack: filtered_arr_col length={len(filtered_arr_col)}, first element type={type(filtered_arr_col.iloc[0])}")
                coords_array = np.stack(filtered_arr_col.to_numpy())
                logger.info(f"After stack: coords_array shape={coords_array.shape}, dtype={coords_array.dtype}")

                # Create Series for each coordinate with (time, id) multi-index (only for non-None values)
                idx = pd.MultiIndex.from_arrays([filtered_df['time'], filtered_df['id']])
                for i, suffix in enumerate(['x', 'y', 'z']):
                    prop_id = f'target_mesh_closest_{suffix}'
                    extended_props[prop_id] = pd.Series(coords_array[:, i], index=idx)

        # Scene mesh closest point (stored as array [x, y, z])
        if 'mesh_scene_closest_point' in agent_df.columns:
            # Extract array column and filter out None values
            arr_col = agent_df['mesh_scene_closest_point']
            # Create mask for non-None values
            mask = arr_col.notna()
            filtered_df = agent_df[mask]
            filtered_arr_col = arr_col[mask]

            if len(filtered_arr_col) > 0:
                # Stack arrays into 2D numpy array
                coords_array = np.stack(filtered_arr_col.to_numpy())

                # Create Series for each coordinate with (time, id) multi-index (only for non-None values)
                idx = pd.MultiIndex.from_arrays([filtered_df['time'], filtered_df['id']])
                for i, suffix in enumerate(['x', 'y', 'z']):
                    prop_id = f'scene_mesh_closest_{suffix}'
                    extended_props[prop_id] = pd.Series(coords_array[:, i], index=idx)

        if extended_props:
            logger.info(f"Loading {len(extended_props)} extended properties for episode {episode_id}")
            self.load_extended_properties_batch(episode_id, extended_props, conn=conn)


class Boids2DLoader(BaseDataLoader):
    """Loader for 2D boids simulation data from PyTorch .pt files."""

    def load_datasets_bulk(self, parent_dir: Path):
        """
        Load multiple 2D boids datasets from a parent directory.

        Args:
            parent_dir: Directory containing multiple .pt dataset files
        """
        # Discover all .pt files (excluding *_config.pt)
        dataset_files = []
        for pt_file in sorted(parent_dir.glob("*.pt")):
            if not pt_file.name.endswith("_config.pt"):
                dataset_files.append(pt_file)

        if not dataset_files:
            raise ValueError(f"No .pt dataset files found in {parent_dir}")

        logger.info(f"Found {len(dataset_files)} dataset files in {parent_dir}")
        logger.info(f"Loading all datasets in single transaction...")

        # Load all datasets in one transaction for maximum performance
        with self.db.transaction() as conn:
            for idx, data_file in enumerate(dataset_files, 1):
                logger.info(f"[{idx}/{len(dataset_files)}] Loading {data_file.name}...")
                self._load_dataset_no_transaction(data_file, conn=conn)

        logger.info(f"Completed loading {len(dataset_files)} datasets")

    def _load_dataset_no_transaction(self, data_path: Path, conn=None):
        """Internal method to load a dataset without starting a transaction.

        Args:
            data_path: Path to .pt file
            conn: Optional connection (for transactional usage)
        """
        logger.info(f"Loading 2D boids dataset from: {data_path}")

        # Validate file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Construct config file path
        basename = data_path.stem
        config_path = data_path.parent / f"{basename}_config.pt"

        # Load config (may not exist for all datasets)
        config = {}
        if config_path.exists():
            logger.info(f"Loading config from: {config_path}")
            config = torch.load(config_path, weights_only=False)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")

        # Load dataset
        logger.info(f"Loading PyTorch dataset from: {data_path}")
        dataset = torch.load(data_path, weights_only=False)

        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # Extract scene size from config
        scene_size = config.get('scene_size', 480.0)
        if not isinstance(scene_size, (int, float)):
            scene_size = 480.0

        num_samples = np.minimum(len(dataset), self.max_episodes)
        # Create session metadata
        session_id = f"session-2d-{basename}"
        session_metadata = SessionMetadata(
            session_id=session_id,
            session_name=basename,
            category_id='boids_2d',
            config=config,
            metadata={
                'data_file': str(data_path),
                'config_file': str(config_path) if config_path.exists() else None,
                'num_samples': num_samples,
                'scene_size': scene_size,
                'loader': 'Boids2DLoader'
            }
        )

        self.load_session(session_metadata, conn=conn)

        # Load each sample as an episode
        logger.info(f"Loading up to {min(len(dataset), self.max_episodes)} episodes...")
        episodes_loaded = 0
        for sample_idx in range(len(dataset)):
            if sample_idx >= self.max_episodes:
                logger.info(f"Reached maximum number of episodes ({self.max_episodes}) for session {session_id}, stopping...")
                break
            logger.info(f"[{sample_idx + 1}/{len(dataset)}] Loading sample {sample_idx}")
            self.load_sample(session_id, sample_idx, dataset[sample_idx], config, scene_size, conn=conn)
            episodes_loaded += 1

        logger.info(f"Completed loading dataset: {session_id} ({episodes_loaded} out of {len(dataset)} episodes)")

    def load_dataset(self, data_path: Path):
        """
        Load a complete 2D boids dataset.

        Args:
            data_path: Path to .pt file (e.g., boid_single_species_basic.pt)
        """
        logger.info(f"Loading 2D boids dataset from: {data_path}")

        # Validate file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Construct config file path
        # Config files follow pattern: {basename}_config.pt
        basename = data_path.stem  # e.g., "boid_single_species_basic"
        config_path = data_path.parent / f"{basename}_config.pt"

        # Load config (may not exist for all datasets)
        config = {}
        if config_path.exists():
            logger.info(f"Loading config from: {config_path}")
            config = torch.load(config_path, weights_only=False)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")

        # Load dataset
        logger.info(f"Loading PyTorch dataset from: {data_path}")
        dataset = torch.load(data_path, weights_only=False)

        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # Extract scene size from config
        scene_size = config.get('scene_size', 480.0)
        if not isinstance(scene_size, (int, float)):
            # If scene_size is not in top level, try to infer from species config
            scene_size = 480.0  # Default value

        num_samples = np.minimum(len(dataset), self.max_episodes)

        # Create session metadata
        session_id = f"session-2d-{basename}"
        session_metadata = SessionMetadata(
            session_id=session_id,
            session_name=basename,
            category_id='boids_2d',
            config=config,
            metadata={
                'data_file': str(data_path),
                'config_file': str(config_path) if config_path.exists() else None,
                'num_samples': num_samples,
                'scene_size': scene_size,
                'loader': 'Boids2DLoader'
            }
        )

        # Load all data in a single transaction for performance
        logger.info(f"Loading up to {num_samples} out of {len(dataset)} episodes in single transaction...")

        with self.db.transaction() as conn:
            self.load_session(session_metadata, conn=conn)

            # Load each sample as an episode
            episodes_loaded = 0
            for sample_idx in range(len(dataset)):
                if sample_idx >= self.max_episodes:
                    logger.info(f"Reached maximum number of episodes ({self.max_episodes}) for session {session_id}, stopping...")
                    break
                logger.info(f"[{sample_idx + 1}/{len(dataset)}] Loading sample {sample_idx}")
                self.load_sample(session_id, sample_idx, dataset[sample_idx], config, scene_size, conn=conn)
                episodes_loaded += 1

        logger.info(f"Completed loading dataset: {session_id} ({episodes_loaded} out of {len(dataset)} episodes)")

    def load_sample(
        self,
        session_id: str,
        sample_idx: int,
        sample: tuple,
        config: Dict[str, Any],
        scene_size: float,
        conn=None
    ):
        """
        Load a single sample (trajectory) from the dataset.

        Args:
            session_id: Session identifier
            sample_idx: Sample index in dataset
            sample: Tuple of (positions, species)
            config: Configuration dictionary
            scene_size: Scene size in pixels
            conn: Optional connection (for transactional usage)
        """
        logger.info(f"Loading sample {sample_idx}")

        positions, species = sample

        # positions: [timesteps, num_agents, 2]
        # species: [num_agents]

        # Convert to numpy for easier manipulation
        import numpy as np
        positions_np = positions.numpy()
        species_np = species.numpy()

        num_timesteps, num_agents, num_dims = positions_np.shape
        assert num_dims == 2, f"Expected 2D positions, got {num_dims}D"

        # Compute velocities: v[t] = p[t+1] - p[t]
        # Shape: [timesteps-1, num_agents, 2]
        velocities_np = np.diff(positions_np, axis=0)

        # For last timestep, use velocity from previous timestep
        # This maintains consistent timestep count
        last_velocity = velocities_np[-1:, :, :]
        velocities_np = np.vstack([velocities_np, last_velocity])

        assert velocities_np.shape[0] == num_timesteps, \
            f"Velocity timesteps mismatch: {velocities_np.shape[0]} != {num_timesteps}"

        # Assume frame rate of 1.0 for 2D boids (discrete timesteps)
        # This can be overridden if specified in config
        frame_rate = config.get('frame_rate', 1.0)

        # Create episode ID
        episode_id = f"episode-{sample_idx:04d}-{session_id}"

        episode_metadata = EpisodeMetadata(
            episode_id=episode_id,
            session_id=session_id,
            episode_number=sample_idx,
            num_frames=num_timesteps,
            num_agents=num_agents,
            frame_rate=frame_rate,
            file_path=f"sample-{sample_idx}"
        )

        self.load_episode(episode_metadata, conn=conn)

        # Prepare observations DataFrame using vectorized numpy operations (MUCH faster)
        # Total rows = num_timesteps * num_agents
        total_rows = num_timesteps * num_agents

        # Create index arrays using numpy broadcasting
        # time_index: [0,0,0,...,1,1,1,...,2,2,2,...] (repeat each timestep num_agents times)
        time_indices = np.repeat(np.arange(num_timesteps), num_agents)

        # agent_id: [0,1,2,...,0,1,2,...,0,1,2,...] (tile agent IDs num_timesteps times)
        agent_ids = np.tile(np.arange(num_agents), num_timesteps)

        # Scale and flatten position/velocity arrays
        # positions_np: [timesteps, agents, 2] -> reshape to [timesteps*agents, 2]
        positions_flat = positions_np.reshape(-1, 2) * scene_size
        velocities_flat = velocities_np.reshape(-1, 2) * scene_size

        # Create DataFrame directly from numpy arrays (100x faster than list of dicts)
        observations = pd.DataFrame({
            'time_index': time_indices,
            'agent_id': agent_ids,
            'agent_type_id': 'agent',  # Same for all rows
            'x': positions_flat[:, 0],
            'y': positions_flat[:, 1],
            'z': None,  # No z-coordinate for 2D boids
            'v_x': velocities_flat[:, 0],
            'v_y': velocities_flat[:, 1],
            'v_z': None  # No z-velocity for 2D boids
        })

        self.load_observations_batch(observations, episode_id, conn=conn)

        # Compute extended properties: distance to food (if food is present)
        extended_props = {}
        food_location = get_food_location_from_config(config, scene_size)

        if food_location is not None:
            food_x, food_y = food_location

            # Compute distance to food for each boid at each timestep
            # Distance = sqrt((boid_x - food_x)^2 + (boid_y - food_y)^2)
            #
            # Note: Food agent itself (last agent if species[-1] == 1) gets distance 0
            # but we compute for all agents for simplicity

            dx = positions_flat[:, 0] - food_x
            dy = positions_flat[:, 1] - food_y
            distances = np.sqrt(dx**2 + dy**2)

            # Create multi-index for extended properties
            idx = pd.MultiIndex.from_arrays([time_indices, agent_ids])
            extended_props['distance_to_food'] = pd.Series(distances, index=idx)

            logger.info(f"Computed distance_to_food for {len(distances)} observations")

        if extended_props:
            self.load_extended_properties_batch(episode_id, extended_props, conn=conn)

        logger.info(f"Loaded sample {sample_idx}: {num_timesteps} frames, {num_agents} agents")


class TrackingCSVLoader(BaseDataLoader):
    """Loader for real-world tracking CSV data from processed_tracks."""

    def load_sessions_bulk(self, parent_dir: Path):
        """
        Load multiple tracking sessions from a parent directory.

        Args:
            parent_dir: Directory containing multiple session subdirectories
                       (e.g., data/processed_tracks/)
        """
        # Discover all session directories (those with aligned_frames/)
        session_dirs = []
        for subdir in sorted(parent_dir.iterdir()):
            if subdir.is_dir() and (subdir / "aligned_frames").exists():
                session_dirs.append(subdir)

        if not session_dirs:
            raise ValueError(f"No session directories found in {parent_dir}")

        logger.info(f"Found {len(session_dirs)} session directories in {parent_dir}")
        logger.info(f"Loading all sessions in single transaction...")

        # Load all sessions in one transaction for maximum performance
        with self.db.transaction() as conn:
            for idx, session_dir in enumerate(session_dirs, 1):
                logger.info(f"[{idx}/{len(session_dirs)}] Loading {session_dir.name}...")
                self._load_session_no_transaction(session_dir, conn)

        logger.info(f"Completed loading {len(session_dirs)} sessions")

    def _load_session_no_transaction(self, session_dir: Path, conn=None):
        """Internal method to load a session without starting a transaction.

        Args:
            session_dir: Directory containing aligned_frames/ subdirectory
            conn: Optional connection (for transactional usage)
        """
        logger.info(f"Loading tracking session from: {session_dir}")

        # Load optional Metadata.yaml if exists
        metadata_path = session_dir / "Metadata.yaml"
        config = {}
        if metadata_path.exists():
            logger.info(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, 'r') as f:
                config = yaml.safe_load(f)

        # Create session metadata
        session_id = session_dir.name  # e.g., "2024_06_01-session_0003"
        session_metadata = SessionMetadata(
            session_id=session_id,
            session_name=session_dir.name,
            category_id='tracking_csv',
            config=config,
            metadata={
                'session_dir': str(session_dir),
                'loader': 'TrackingCSVLoader'
            }
        )

        self.load_session(session_metadata, conn=conn)

        # Discover all camera episodes in aligned_frames/
        aligned_frames = session_dir / "aligned_frames"
        camera_dirs = sorted([d for d in aligned_frames.iterdir()
                              if d.is_dir() and not d.name.startswith('.')])

        logger.info(f"Loading up to {min(len(camera_dirs), self.max_episodes)} camera episodes...")

        episodes_loaded = 0
        for episode_num, camera_dir in enumerate(camera_dirs):
            if episode_num >= self.max_episodes:
                logger.info(f"Reached maximum number of episodes ({self.max_episodes}) for session {session_id}, stopping...")
                break

            # Find CSV file (e.g., thermal_2_tracks.csv)
            csv_files = list(camera_dir.glob("*_tracks.csv"))
            if not csv_files:
                logger.warning(f"No CSV file found in {camera_dir}, skipping")
                continue

            csv_path = csv_files[0]
            episode_name = camera_dir.name  # e.g., "thermal_2"

            self.load_episode_csv(session_id, episode_num, episode_name, csv_path, conn=conn)
            episodes_loaded += 1

        logger.info(f"Completed loading session: {session_id} ({episodes_loaded} out of {len(camera_dirs)} episodes)")

    def load_tracking_session(self, session_dir: Path):
        """
        Load a single tracking session.

        Args:
            session_dir: Directory containing aligned_frames/ subdirectory
        """
        logger.info(f"Loading tracking session from: {session_dir}")

        # Validate structure
        aligned_frames = session_dir / "aligned_frames"
        if not aligned_frames.exists():
            raise FileNotFoundError(f"aligned_frames not found in {session_dir}")

        # Load in single transaction
        with self.db.transaction() as conn:
            self._load_session_no_transaction(session_dir, conn)

        logger.info(f"Completed loading session: {session_dir.name}")

    def load_episode_csv(
        self,
        session_id: str,
        episode_number: int,
        episode_name: str,
        csv_path: Path,
        conn=None
    ):
        """Load a single camera's tracking CSV file.

        Args:
            session_id: Session identifier
            episode_number: Episode number within session
            episode_name: Episode name (camera name)
            csv_path: Path to CSV file
            conn: Optional connection (for transactional usage)
        """
        logger.info(f"Loading episode {episode_name} from: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = ['track_id', 'frame', 'x', 'y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")

        # Compute velocities per track with proper time scaling (vectorized approach)
        import numpy as np
        frame_rate = 30.0  # Default for video tracking

        # Sort by track_id and frame for proper diff() calculation
        df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)

        # Compute position and frame differences using groupby (vectorized, much faster than loop)
        df['dx'] = df.groupby('track_id')['x'].diff()
        df['dy'] = df.groupby('track_id')['y'].diff()
        df['frame_diff'] = df.groupby('track_id')['frame'].diff()

        # Compute time delta: dt = frame_diff / frame_rate (seconds)
        df['dt'] = df['frame_diff'] / frame_rate

        # Compute velocities: v = dx / dt (pixels/second)
        df['v_x'] = df['dx'] / df['dt']
        df['v_y'] = df['dy'] / df['dt']

        # Set velocity to NaN when there are frame gaps (frame_diff > 1 means missing frames)
        df.loc[df['frame_diff'] > 1, ['v_x', 'v_y']] = np.nan

        # For first frame of each track, set velocity to 0.0 (instead of NaN)
        first_frame_mask = df['frame_diff'].isna()
        df.loc[first_frame_mask, ['v_x', 'v_y']] = 0.0

        # Drop temporary columns
        df = df.drop(columns=['dx', 'dy', 'frame_diff', 'dt'])

        # Extract metadata
        num_frames = int(df['frame'].max() + 1)
        num_agents = int(df['track_id'].nunique())

        episode_id = f"episode-{episode_name}-{session_id}"

        episode_metadata = EpisodeMetadata(
            episode_id=episode_id,
            session_id=session_id,
            episode_number=episode_number,
            num_frames=num_frames,
            num_agents=num_agents,
            frame_rate=frame_rate,
            file_path=str(csv_path)
        )

        self.load_episode(episode_metadata, conn=conn)

        # Prepare observations DataFrame
        observations = pd.DataFrame({
            'time_index': df['frame'],
            'agent_id': df['track_id'],
            'agent_type_id': 'agent',  # Default type
            'x': df['x'],
            'y': df['y'],
            'z': None,  # 2D data
            'v_x': df['v_x'],
            'v_y': df['v_y'],
            'v_z': None
        })

        self.load_observations_batch(observations, episode_id, conn=conn)

        logger.info(f"Loaded episode {episode_name}: {num_frames} frames, {num_agents} tracks, {len(df)} observations")


class GNNRolloutLoader(BaseDataLoader):
    """Loader for GNN model evaluation rollout data from pickle files."""

    def _detect_food_agent(self, dataset_name: str, num_agents: int) -> int:
        """
        Detect food agent by filename heuristic.

        Args:
            dataset_name: Dataset name from rollout filename
            num_agents: Number of agents in trajectory

        Returns:
            Agent index of food agent, or -1 if none found
        """
        # If 'food' is in the dataset name, food agent is the last agent
        if 'food' in dataset_name.lower():
            return num_agents - 1

        return -1  # No food agent

    def _create_agent_type_map(
        self,
        food_agent_idx: int,
        num_frames: int,
        num_agents: int
    ) -> Dict[Tuple[int, int], str]:
        """
        Create mapping from (time_index, agent_id) to agent_type.

        Args:
            food_agent_idx: Index of food agent (-1 if none)
            num_frames: Number of frames
            num_agents: Number of agents

        Returns:
            Dict mapping (time_index, agent_id) -> agent_type_id
        """
        agent_type_map = {}

        for time_idx in range(num_frames):
            for agent_id in range(num_agents):
                if food_agent_idx >= 0 and agent_id == food_agent_idx:
                    agent_type_map[(time_idx, agent_id)] = 'food'
                else:
                    agent_type_map[(time_idx, agent_id)] = 'agent'

        return agent_type_map

    def _prepare_acceleration_properties(
        self,
        accelerations: np.ndarray,
        num_frames: int,
        num_agents: int,
        scene_size: float
    ) -> Dict[str, pd.Series]:
        """
        Prepare acceleration extended properties.

        Args:
            accelerations: [frames, agents, 2]
            num_frames: Number of frames
            num_agents: Number of agents
            scene_size: Scene size for scaling

        Returns:
            Dict of property_id -> Series with multi-index (time_index, agent_id)
        """
        # Scale accelerations
        accelerations_scaled = accelerations * scene_size
        accelerations_flat = accelerations_scaled.reshape(-1, 2)

        # Create multi-index
        time_indices = np.repeat(np.arange(num_frames), num_agents)
        agent_ids = np.tile(np.arange(num_agents), num_frames)
        idx = pd.MultiIndex.from_arrays([time_indices, agent_ids])

        return {
            'acceleration_x': pd.Series(accelerations_flat[:, 0], index=idx),
            'acceleration_y': pd.Series(accelerations_flat[:, 1], index=idx),
        }

    def _prepare_attention_properties(
        self,
        W_frames: List,  # List of (edge_index, edge_weight) tuples
        traj_idx: int,   # Trajectory index within batch
        num_frames: int,
        num_agents: int,
        food_agent_idx: int
    ) -> Dict[str, pd.Series]:
        """
        Prepare attention weight extended properties.

        Args:
            W_frames: List of (edge_index, edge_weight) tuples, one per frame
            traj_idx: Trajectory index within batch
            num_frames: Number of frames
            num_agents: Number of agents
            food_agent_idx: Index of food agent (-1 if none)

        Returns:
            Dict of property_id -> Series with multi-index (time_index, agent_id)
            Properties: attn_weight_self, attn_weight_boid, attn_weight_food
        """
        from collab_env.gnn.gnn import debatch_edge_index_weight

        # Initialize storage for attention decomposition
        attn_self_all = []   # [num_frames, num_agents]
        attn_boid_all = []   # [num_frames, num_agents]
        attn_food_all = []   # [num_frames, num_agents]

        for frame_idx in range(num_frames):
            edge_index, edge_weight = W_frames[frame_idx]

            # Handle empty edge_index (no edges in this frame)
            if edge_index.numel() == 0:
                # No edges, set all attention weights to zero
                attn_self_all.append(np.zeros(num_agents))
                attn_boid_all.append(np.zeros(num_agents))
                attn_food_all.append(np.zeros(num_agents))
                continue

            # Average across heads if multi-head
            if len(edge_weight.shape) > 1 and edge_weight.shape[1] > 1:
                edge_weight_avg = edge_weight.mean(dim=1, keepdim=True)  # [num_edges, 1]
            else:
                edge_weight_avg = edge_weight

            # Debatch to get NxN adjacency matrix for this trajectory
            # Note: debatch expects batch_size trajectories, we only want one
            batch_size = int(edge_index.max() // num_agents) + 1
            W_by_file, file_IDs = debatch_edge_index_weight(
                edge_index, edge_weight_avg, num_agents, np.arange(batch_size)
            )

            # Get adjacency matrix for our trajectory
            A = W_by_file[traj_idx]  # [num_agents, num_agents]

            # Decompose attention for each agent
            attn_self_frame = np.zeros(num_agents)
            attn_boid_frame = np.zeros(num_agents)
            attn_food_frame = np.zeros(num_agents)

            for agent_id in range(num_agents):
                # Self-attention
                attn_self_frame[agent_id] = A[agent_id, agent_id]

                boid_mask = np.ones(num_agents, dtype=bool)
                boid_mask[agent_id] = False  # Exclude self
                # Attention to other boids
                if food_agent_idx >= 0:
                    # Sum attention to all boids (excluding self and food)
                    boid_mask[food_agent_idx] = False  # Exclude food
                    # Attention to food
                    attn_food_frame[agent_id] = A[agent_id, food_agent_idx]
                else:
                    # No food, all non-self are boids
                    attn_food_frame[agent_id] = 0.0

                attn_boid_frame[agent_id] = np.sum(A[agent_id, boid_mask])

            attn_self_all.append(attn_self_frame)
            attn_boid_all.append(attn_boid_frame)
            attn_food_all.append(attn_food_frame)

        # Convert to [num_frames, num_agents] arrays
        attn_self_all = np.array(attn_self_all)  # [frames, agents]
        attn_boid_all = np.array(attn_boid_all)
        attn_food_all = np.array(attn_food_all)

        # Flatten and create multi-index
        time_indices = np.repeat(np.arange(num_frames), num_agents)
        agent_ids = np.tile(np.arange(num_agents), num_frames)
        idx = pd.MultiIndex.from_arrays([time_indices, agent_ids])

        return {
            'attn_weight_self': pd.Series(attn_self_all.flatten(), index=idx),
            'attn_weight_boid': pd.Series(attn_boid_all.flatten(), index=idx),
            'attn_weight_food': pd.Series(attn_food_all.flatten(), index=idx),
        }

    def _prepare_loss_observations(
        self,
        loss_values: List[float],  # List of scalars, one per frame
        num_frames: int,
        scene_size: float
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Prepare per-frame loss as 'env' agent observations.

        Args:
            loss_values: List of scalar loss values (one per frame)
            num_frames: Number of frames
            scene_size: Scene size for positioning env node at center

        Returns:
            observations: DataFrame with env agent observations at scene center
            extended_props: Dict with 'loss' property
        """
        # Verify we have the right number of loss values
        loss_array = np.array(loss_values)
        if len(loss_array) != num_frames:
            logger.warning(
                f"Loss array length ({len(loss_array)}) != num_frames ({num_frames}). "
                f"Truncating or padding with NaN."
            )
            # Pad with NaN or truncate as needed
            if len(loss_array) < num_frames:
                loss_array = np.pad(
                    loss_array,
                    (0, num_frames - len(loss_array)),
                    constant_values=np.nan
                )
            else:
                loss_array = loss_array[:num_frames]

        # Position env node at scene center
        scene_center = scene_size / 2.0  # 240.0 for default scene_size=480.0

        # Create observations with agent_type_id='env', agent_id=-1 (arbitrary, but distinct)
        observations = pd.DataFrame({
            'time_index': np.arange(num_frames),
            'agent_id': -1,  # Use -1 to distinguish from regular agent IDs (0, 1, 2, ...)
            'agent_type_id': 'env',
            'x': scene_center,
            'y': scene_center,
            'z': None,
            'v_x': None,  # NULL velocity for env node
            'v_y': None,
            'v_z': None
        })

        # Create extended property for loss
        idx = pd.MultiIndex.from_arrays([
            np.arange(num_frames),           # time_index
            np.full(num_frames, -1, dtype=int)  # agent_id=-1
        ])
        extended_props = {
            'loss': pd.Series(loss_array, index=idx)
        }

        return observations, extended_props

    def _load_trajectory_episode(
        self,
        session_id: str,
        episode_id: str,
        episode_number: int,
        positions: np.ndarray,  # [frames, agents, 2]
        accelerations: np.ndarray,  # [frames, agents, 2]
        W_frames: Optional[List] = None,  # List of (edge_index, edge_weight) tuples
        loss_frames: Optional[List] = None,  # List of scalar loss values (one per frame)
        traj_idx: int = 0,  # Trajectory index within batch
        num_frames: int = None,
        num_agents: int = None,
        episode_type: str = 'actual',  # 'actual' or 'predicted'
        food_agent_idx: int = -1,
        actual_food_positions: Optional[np.ndarray] = None,  # [frames, 2] - TRUE food positions from actual episode
        actual_positions: Optional[np.ndarray] = None,  # [frames, agents, 2] - Actual positions for prediction error
        metadata: Dict = None,
        scene_size: float = 480.0,
        conn=None
    ):
        """Load a single trajectory as an episode with accelerations, attention weights, loss, and prediction error."""

        # Infer dimensions if not provided
        if num_frames is None:
            num_frames = positions.shape[0]
        if num_agents is None:
            num_agents = positions.shape[1]

        # 1. Create agent type map
        agent_type_map = self._create_agent_type_map(
            food_agent_idx, num_frames, num_agents
        )

        # 2. Insert episode
        metadata_dict = metadata or {}
        query = """
        INSERT INTO episodes (episode_id, session_id, episode_number, num_frames,
                             num_agents, frame_rate, file_path)
        VALUES (:episode_id, :session_id, :episode_number, :num_frames,
                :num_agents, :frame_rate, :file_path)
        """
        self.db.execute(query, {
            'episode_id': episode_id,
            'session_id': session_id,
            'episode_number': episode_number,
            'num_frames': num_frames,
            'num_agents': num_agents,
            'frame_rate': 1.0,
            'file_path': metadata_dict.get('rollout_file', '')
        }, conn=conn)

        # 3. Prepare observations with agent types
        positions_scaled = positions * scene_size
        positions_flat = positions_scaled.reshape(-1, 2)

        # Compute velocities
        velocities = np.diff(positions_scaled, axis=0)
        last_velocity = velocities[-1:, :, :]
        velocities = np.vstack([velocities, last_velocity])
        velocities_flat = velocities.reshape(-1, 2)

        # Create index arrays
        time_indices = np.repeat(np.arange(num_frames), num_agents)
        agent_ids = np.tile(np.arange(num_agents), num_frames)

        # Map agent types
        agent_type_ids = [
            agent_type_map.get((t, a), 'agent')
            for t, a in zip(time_indices, agent_ids)
        ]

        # Create observations DataFrame
        observations = pd.DataFrame({
            'time_index': time_indices,
            'agent_id': agent_ids,
            'agent_type_id': agent_type_ids,
            'x': positions_flat[:, 0],
            'y': positions_flat[:, 1],
            'z': None,
            'v_x': velocities_flat[:, 0],
            'v_y': velocities_flat[:, 1],
            'v_z': None
        })

        self.load_observations_batch(observations, episode_id, conn=conn)

        # 3b. Load env observations for loss (predicted episodes only)
        if episode_type == 'predicted' and loss_frames is not None:
            env_observations, loss_props = self._prepare_loss_observations(
                loss_frames, num_frames, scene_size
            )
            self.load_observations_batch(env_observations, episode_id, conn=conn)
            logger.debug(f"Added {len(loss_props['loss'])} loss observations for env agent")

        # 4. Load accelerations as extended properties
        extended_props = self._prepare_acceleration_properties(
            accelerations, num_frames, num_agents, scene_size
        )

        # 4b. Merge loss properties if we loaded env observations
        if episode_type == 'predicted' and loss_frames is not None:
            extended_props.update(loss_props)

        # 5. Load attention weights as extended properties (if available)
        if W_frames is not None:
            attention_props = self._prepare_attention_properties(
                W_frames, traj_idx, num_frames, num_agents, food_agent_idx
            )
            extended_props.update(attention_props)

        # 6. Compute distance to food (if food agent exists)
        if food_agent_idx >= 0:
            # positions_scaled: [frames, agents, 2]
            time_indices_all = np.repeat(np.arange(num_frames), num_agents)
            agent_ids_all = np.tile(np.arange(num_agents), num_frames)
            idx = pd.MultiIndex.from_arrays([time_indices_all, agent_ids_all])

            if episode_type == 'actual':
                # ACTUAL EPISODE: Compute distance to actual food position
                food_positions = positions_scaled[:, food_agent_idx, :]  # [frames, 2]

                distances = []
                for time_idx in range(num_frames):
                    for agent_id in range(num_agents):
                        if agent_id == food_agent_idx:
                            distances.append(0.0)
                        else:
                            boid_x, boid_y = positions_scaled[time_idx, agent_id, :]
                            food_x, food_y = food_positions[time_idx, :]
                            distance = np.sqrt((boid_x - food_x)**2 + (boid_y - food_y)**2)
                            distances.append(distance)

                extended_props['distance_to_food'] = pd.Series(distances, index=idx)
                logger.debug(f"Computed distance_to_food for actual episode ({len(distances)} observations)")

            else:  # episode_type == 'predicted'
                # PREDICTED EPISODE: Compute TWO distance metrics
                # 1. distance_to_food_actual: Distance to TRUE food position (from actual episode)
                # 2. distance_to_food_predicted: Distance to PREDICTED food position

                predicted_food_positions = positions_scaled[:, food_agent_idx, :]  # [frames, 2]

                distances_to_actual_food = []
                distances_to_predicted_food = []

                for time_idx in range(num_frames):
                    for agent_id in range(num_agents):
                        if agent_id == food_agent_idx:
                            # Food agent itself
                            distances_to_actual_food.append(0.0)
                            distances_to_predicted_food.append(0.0)
                        else:
                            # Boid positions
                            boid_x, boid_y = positions_scaled[time_idx, agent_id, :]

                            # Distance to TRUE food (from actual episode)
                            if actual_food_positions is not None:
                                actual_food_x, actual_food_y = actual_food_positions[time_idx, :]
                                dist_to_actual = np.sqrt((boid_x - actual_food_x)**2 + (boid_y - actual_food_y)**2)
                                distances_to_actual_food.append(dist_to_actual)
                            else:
                                distances_to_actual_food.append(np.nan)

                            # Distance to PREDICTED food
                            pred_food_x, pred_food_y = predicted_food_positions[time_idx, :]
                            dist_to_predicted = np.sqrt((boid_x - pred_food_x)**2 + (boid_y - pred_food_y)**2)
                            distances_to_predicted_food.append(dist_to_predicted)

                extended_props['distance_to_food_actual'] = pd.Series(distances_to_actual_food, index=idx)
                extended_props['distance_to_food_predicted'] = pd.Series(distances_to_predicted_food, index=idx)

                logger.debug(f"Computed distance_to_food_actual and distance_to_food_predicted "
                           f"for predicted episode ({len(distances_to_actual_food)} observations)")

        # 7. Compute prediction error (predicted episodes only)
        if episode_type == 'predicted' and actual_positions is not None:
            # actual_positions: [frames, agents, 2] (normalized coordinates)
            # positions: [frames, agents, 2] (predicted, normalized coordinates)

            time_indices_all = np.repeat(np.arange(num_frames), num_agents)
            agent_ids_all = np.tile(np.arange(num_agents), num_frames)
            idx = pd.MultiIndex.from_arrays([time_indices_all, agent_ids_all])

            # Compute L2 distance in scene units
            actual_scaled = actual_positions * scene_size    # [frames, agents, 2]
            predicted_scaled = positions * scene_size        # [frames, agents, 2]

            prediction_errors = []
            for time_idx in range(num_frames):
                for agent_id in range(num_agents):
                    actual_pos = actual_scaled[time_idx, agent_id, :]
                    predicted_pos = predicted_scaled[time_idx, agent_id, :]
                    error = np.sqrt(np.sum((actual_pos - predicted_pos)**2))
                    prediction_errors.append(error)

            extended_props['prediction_error'] = pd.Series(prediction_errors, index=idx)
            logger.debug(f"Computed prediction_error for predicted episode ({len(prediction_errors)} observations)")

        if extended_props:
            self.load_extended_properties_batch(
                episode_id, extended_props, conn=conn
            )

    def _parse_rollout_filename(self, rollout_path: Path) -> Dict[str, Any]:
        """
        Parse rollout filename to extract metadata.

        Example: boid_food_strong_vpluspplus_a_n0_h1_vr0.5_s0_rollout_5.pkl
        """
        filename = rollout_path.stem

        # Extract dataset name (everything before the model parameters)
        # Pattern: {dataset}_{model}_n{noise}_h{heads}_vr{visual_range}_s{seed}_rollout_{frame}
        match = re.match(
            r'(.+?)_n(\d+(?:\.\d+)?)_h(\d+)_vr(\d+(?:\.\d+)?)_s(\d+)_rollout_(\d+)',
            filename
        )

        if not match:
            logger.warning(f"Could not parse rollout filename: {filename}, using defaults")
            return {
                'dataset': filename,
                'model_spec': 'unknown',
                'noise': 0,
                'heads': 1,
                'visual_range': 0.1,
                'seed': 0,
                'rollout_frame': 5
            }

        dataset_and_model = match.group(1)
        noise = float(match.group(2))
        heads = int(match.group(3))
        visual_range = float(match.group(4))
        seed = int(match.group(5))
        rollout_frame = int(match.group(6))

        # Try to separate dataset from model name (heuristic: look for common model names)
        # Common patterns: dataset_modelname or just dataset
        parts = dataset_and_model.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ['vpluspplus', 'vplus', 'basic', 'a', 'b', 'c']:
            dataset = parts[0]
            model_name = parts[1]
        else:
            dataset = dataset_and_model
            model_name = 'base'

        model_spec = f"{model_name}_n{noise}_h{heads}_vr{visual_range}_s{seed}"

        return {
            'dataset': dataset,
            'model_name': model_name,
            'model_spec': model_spec,
            'noise': noise,
            'heads': heads,
            'visual_range': visual_range,
            'seed': seed,
            'rollout_frame': rollout_frame
        }

    def _load_pickle(self, rollout_path: Path) -> Dict:
        """Load pickle file with CPU device mapping."""
        from collab_env.gnn.plotting_utility import DeviceUnpickler

        with open(rollout_path, 'rb') as f:
            # Use DeviceUnpickler to safely load CUDA tensors on CPU-only machines
            rollout_data = DeviceUnpickler(f, device="cpu").load()

        return rollout_data

    def load_rollout_file(self, rollout_path: Path, scene_size: float = 480.0, conn=None):
        """
        Load a single rollout pickle file.

        Args:
            rollout_path: Path to rollout .pkl file
            scene_size: Scene size for coordinate scaling
            conn: Optional database connection (for bulk loading in single transaction)
        """

        logger.info(f"Loading GNN rollout from: {rollout_path}")
        logger.info(f"Loader max_episodes setting: {self.max_episodes}")

        # 1. Parse filename
        metadata = self._parse_rollout_filename(rollout_path)

        # 2. Load pickle
        rollout_data = self._load_pickle(rollout_path)

        # 3. Create session
        session_id = f"rollout-{metadata['dataset']}-{metadata['model_spec']}"
        session_metadata = SessionMetadata(
            session_id=session_id,
            session_name=f"GNN Rollout: {metadata['dataset']}",
            category_id='boids_2d_rollout',
            config={**metadata, 'scene_size': scene_size},
            metadata={'rollout_file': str(rollout_path)}
        )

        # 4. Load in single transaction (or use provided connection)
        def _load_data(conn):
            """Internal function to load data with given connection."""
            self.load_session(session_metadata, conn=conn)

            trajectories_loaded = 0
            episode_counter = 0
            # Iterate epochs → batches → trajectories
            for epoch_id, epoch_data in rollout_data.items():
                for batch_id, batch_data in epoch_data.items():
                    # Extract numpy arrays
                    actual = np.array(batch_data['actual'])          # [frames, batch, agents, 2]
                    predicted = np.array(batch_data['predicted'])    # [frames, batch, agents, 2]
                    actual_acc = np.array(batch_data['actual_acc'])  # [frames, batch, agents, 2]
                    predicted_acc = np.array(batch_data['predicted_acc'])  # [frames, batch, agents, 2]

                    # Extract attention weights if available
                    W_frames = batch_data.get('W', None)  # List of (edge_index, edge_weight) tuples

                    # Extract loss values if available
                    loss_frames = batch_data.get('loss', None)  # List of scalar loss values (one per frame)

                    num_frames = actual.shape[0]
                    batch_size = actual.shape[1]
                    num_agents = actual.shape[2]

                    # 6. Process each trajectory in batch
                    for traj_idx in range(batch_size):
                        # Check if we've reached max episodes before loading this trajectory
                        # Each trajectory creates 2 episodes (actual + predicted)
                        if trajectories_loaded >= self.max_episodes:
                            logger.info(f"Reached max_episodes limit: {trajectories_loaded} >= {self.max_episodes}, stopping...")
                            break

                        # Extract data for this trajectory
                        actual_traj = actual[:, traj_idx, :, :]          # [frames, agents, 2]
                        predicted_traj = predicted[:, traj_idx, :, :]    # [frames, agents, 2]
                        actual_acc_traj = actual_acc[:, traj_idx, :, :]  # [frames, agents, 2]
                        predicted_acc_traj = predicted_acc[:, traj_idx, :, :]  # [frames, agents, 2]

                        # Detect food agent (same for actual and predicted)
                        food_agent_idx = self._detect_food_agent(metadata['dataset'], num_agents)

                        # Extract actual food positions for use in predicted episode
                        # This is the TRUE food location that should remain stationary
                        actual_food_positions = None
                        if food_agent_idx >= 0:
                            # Scale actual food positions to scene coordinates
                            actual_food_positions = actual_traj[:, food_agent_idx, :] * scene_size  # [frames, 2]

                        # Load ACTUAL episode (no attention weights for ground truth)
                        episode_id_actual = f"{session_id}-{trajectories_loaded:04d}-actual"
                        self._load_trajectory_episode(
                            session_id=session_id,
                            episode_id=episode_id_actual,
                            episode_number=episode_counter,
                            positions=actual_traj,
                            accelerations=actual_acc_traj,
                            W_frames=None,  # No attention weights for ground truth
                            traj_idx=traj_idx,
                            num_frames=num_frames,
                            num_agents=num_agents,
                            episode_type='actual',
                            food_agent_idx=food_agent_idx,
                            actual_food_positions=None,  # Not needed for actual episode
                            metadata={
                                'source_epoch': int(epoch_id),
                                'source_batch': int(batch_id),
                                'trajectory_index': int(traj_idx),
                                'rollout_file': str(rollout_path)
                            },
                            scene_size=scene_size,
                            conn=conn
                        )

                        # Load PREDICTED episode (with attention weights from model)
                        episode_id_predicted = f"{session_id}-{trajectories_loaded:04d}-predicted"
                        
                        episode_counter += 1
                        
                        self._load_trajectory_episode(
                            session_id=session_id,
                            episode_id=episode_id_predicted,
                            episode_number=episode_counter,
                            positions=predicted_traj,
                            accelerations=predicted_acc_traj,
                            W_frames=W_frames,  # Include attention weights for predictions
                            loss_frames=loss_frames,  # Include loss values per frame
                            traj_idx=traj_idx,
                            num_frames=num_frames,
                            num_agents=num_agents,
                            episode_type='predicted',
                            food_agent_idx=food_agent_idx,
                            actual_food_positions=actual_food_positions,  # Pass TRUE food positions
                            actual_positions=actual_traj,  # Pass actual positions for prediction error
                            metadata={
                                'source_epoch': int(epoch_id),
                                'source_batch': int(batch_id),
                                'trajectory_index': int(traj_idx),
                                'model_name': metadata.get('model_name'),
                                'model_params': {
                                    'noise': metadata.get('noise'),
                                    'heads': metadata.get('heads'),
                                    'visual_range': metadata.get('visual_range'),
                                    'seed': metadata.get('seed')
                                },
                                'rollout_file': str(rollout_path)
                            },
                            scene_size=scene_size,
                            conn=conn
                        )
                        
                        trajectories_loaded += 1
                        episode_counter += 1

            logger.info(f"Completed loading rollout: {session_id} ({trajectories_loaded} trajectories, {episode_counter} episodes)")

        # Use provided connection or create new transaction
        if conn is not None:
            _load_data(conn)
        else:
            with self.db.transaction() as new_conn:
                _load_data(new_conn)

    def load_rollouts_bulk(self, rollouts_dir: Path, scene_size: float = 480.0):
        """
        Load multiple rollout files from a directory in a single transaction.

        Each rollout file becomes a separate session. The max_episodes limit applies
        independently to each file/session.

        Args:
            rollouts_dir: Directory containing rollout .pkl files
            scene_size: Scene size for coordinate scaling
        """
        # Find all .pkl files
        rollout_files = sorted(rollouts_dir.glob("*.pkl"))

        if not rollout_files:
            raise ValueError(f"No .pkl files found in {rollouts_dir}")

        logger.info(f"Found {len(rollout_files)} rollout files in {rollouts_dir}")

        # Load all files in a single transaction
        with self.db.transaction() as conn:
            for idx, rollout_file in enumerate(rollout_files, 1):
                logger.info(f"[{idx}/{len(rollout_files)}] Loading {rollout_file.name}...")
                self.load_rollout_file(rollout_file, scene_size=scene_size, conn=conn)

        logger.info(f"Completed loading {len(rollout_files)} rollout files")


def main():
    """Command-line interface for data loader."""
    parser = argparse.ArgumentParser(
        description='Load tracking data into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load 3D boids simulation (uses environment variables for DB)
  python -m collab_env.data.db.db_loader --source boids3d --path simulated_data/hackathon

  # Load 2D boids dataset
  python -m collab_env.data.db.db_loader --source boids2d --path simulated_data/boid_dataset.pt

  # Load GNN rollout predictions
  python -m collab_env.data.db.db_loader --source boids2d_rollout --path trained_models/boid_food_strong/rollouts/rollout.pkl --scene-size 480.0

  # Load with specific backend
  python -m collab_env.data.db.db_loader --source boids3d --path simulated_data/hackathon --backend duckdb
        """
    )

    parser.add_argument(
        '--source',
        required=True,
        choices=['boids3d', 'boids2d', 'tracking', 'boids2d_rollout'],
        help='Data source type'
    )
    
    parser.add_argument(
        '--max-episodes-per-session',
        type=int,
        default=None,
        help='Maximum number of episodes to load per session (default: unlimited)'
    )

    parser.add_argument(
        '--path',
        required=True,
        type=Path,
        help='Path to data directory or file'
    )

    parser.add_argument(
        '--backend',
        choices=['postgres', 'duckdb'],
        help='Database backend (overrides DB_BACKEND env var)'
    )

    parser.add_argument(
        '--dbpath',
        help='DuckDB database path (overrides DUCKDB_PATH env var)'
    )

    parser.add_argument(
        '--scene-size',
        type=float,
        default=480.0,
        help='Scene size for coordinate scaling (default: 480.0, for boids2d_rollout source only)'
    )

    args = parser.parse_args()

    # Handle dbpath via environment variable if specified
    if args.dbpath:
        os.environ['DUCKDB_PATH'] = str(args.dbpath)

    # Load configuration
    config = get_db_config(backend=args.backend)

    # Create database connection
    db_conn = DatabaseConnection(config)
    db_conn.connect()

    try:
        # Convert None to np.inf for unlimited episodes
        max_eps = args.max_episodes_per_session if args.max_episodes_per_session is not None else np.inf
        logger.info(f"max_episodes_per_session argument: {args.max_episodes_per_session}")
        logger.info(f"Effective max_episodes: {max_eps}")

        # Load data based on source type
        if args.source == 'boids3d':
            loader = Boids3DLoader(db_conn, max_episodes=max_eps)

            # Check if path is a directory with multiple simulations or a single simulation
            if args.path.is_dir():
                # Check if this is a single simulation (has config.yaml) or parent directory
                if (args.path / "config.yaml").exists():
                    # Single simulation
                    logger.info("Loading single simulation...")
                    loader.load_simulation(args.path)
                else:
                    # Parent directory with multiple simulations
                    logger.info("Loading multiple simulations from parent directory...")
                    loader.load_simulations_bulk(args.path)
            else:
                raise ValueError(f"Path must be a directory: {args.path}")

        elif args.source == 'boids2d':
            loader = Boids2DLoader(db_conn, max_episodes=max_eps)

            # Check if path is a file or directory
            if args.path.is_file() and args.path.suffix == '.pt':
                # Single dataset file
                logger.info("Loading single dataset...")
                loader.load_dataset(args.path)
            elif args.path.is_dir():
                # Directory with multiple datasets
                logger.info("Loading multiple datasets from directory...")
                loader.load_datasets_bulk(args.path)
            else:
                raise ValueError(f"Path must be a .pt file or directory: {args.path}")

        elif args.source == 'tracking':
            loader = TrackingCSVLoader(db_conn, max_episodes=max_eps)

            # Check if path is a single session or parent directory
            if args.path.is_dir():
                # Check if this is a single session (has aligned_frames/)
                if (args.path / "aligned_frames").exists():
                    # Single session
                    logger.info("Loading single session...")
                    loader.load_tracking_session(args.path)
                else:
                    # Parent directory with multiple sessions
                    logger.info("Loading multiple sessions from parent directory...")
                    loader.load_sessions_bulk(args.path)
            else:
                raise ValueError(f"Path must be a directory: {args.path}")

        elif args.source == 'boids2d_rollout':
            loader = GNNRolloutLoader(db_conn, max_episodes=max_eps)

            if args.path.is_file() and args.path.suffix == '.pkl':
                # Single rollout file
                logger.info("Loading single rollout file...")
                loader.load_rollout_file(args.path, scene_size=args.scene_size)
            elif args.path.is_dir():
                # Directory with multiple rollout files
                logger.info("Loading multiple rollout files from directory...")
                loader.load_rollouts_bulk(args.path, scene_size=args.scene_size)
            else:
                raise ValueError(f"Path must be a .pkl file or directory: {args.path}")

        logger.info("Data loading completed successfully")

    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        raise
    finally:
        db_conn.close()


if __name__ == '__main__':
    main()
