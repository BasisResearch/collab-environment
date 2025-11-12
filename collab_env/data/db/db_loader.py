"""
Data loader for tracking analytics database.

Loads data from various sources into PostgreSQL or DuckDB:
- 3D Boids: Parquet files from collab_env.sim.boids
- 2D Boids: PyTorch .pt files from collab_env.sim.boids_gnn_temp
- Tracking CSV: Real-world tracking data from collab_env.tracking
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    def __init__(self, db_conn: DatabaseConnection):
        self.db = db_conn

    def load_session(self, metadata: SessionMetadata, conn=None):
        """Load session metadata into database.

        Args:
            metadata: Session metadata
            conn: Optional connection (for transactional usage)
        """
        # Convert config and metadata to JSON strings
        import json
        config_json = json.dumps(metadata.config)
        metadata_json = json.dumps(metadata.metadata) if metadata.metadata else None

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
        logger.info(f"Loading {len(episode_files)} episodes...")

        for episode_num, episode_file in enumerate(episode_files):
            self.load_episode_file(session_id, episode_num, episode_file, config, conn=conn)

        logger.info(f"Completed loading simulation: {session_id} ({len(episode_files)} episodes)")

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
        logger.info(f"Loading {len(episode_files)} episodes in single transaction...")

        with self.db.transaction() as conn:
            self.load_session(session_metadata, conn=conn)

            # Load each episode
            for episode_num, episode_file in enumerate(episode_files):
                self.load_episode_file(session_id, episode_num, episode_file, config, conn=conn)

        logger.info(f"Completed loading simulation: {session_id} ({len(episode_files)} episodes)")

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
                'num_samples': len(dataset),
                'scene_size': scene_size,
                'loader': 'Boids2DLoader'
            }
        )

        self.load_session(session_metadata, conn=conn)

        # Load each sample as an episode
        logger.info(f"Loading {len(dataset)} episodes...")
        for sample_idx in range(len(dataset)):
            self.load_sample(session_id, sample_idx, dataset[sample_idx], config, scene_size, conn=conn)

        logger.info(f"Completed loading dataset: {session_id} ({len(dataset)} episodes)")

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
                'num_samples': len(dataset),
                'scene_size': scene_size,
                'loader': 'Boids2DLoader'
            }
        )

        # Load all data in a single transaction for performance
        logger.info(f"Loading {len(dataset)} episodes in single transaction...")

        with self.db.transaction() as conn:
            self.load_session(session_metadata, conn=conn)

            # Load each sample as an episode
            for sample_idx in range(len(dataset)):
                self.load_sample(session_id, sample_idx, dataset[sample_idx], config, scene_size, conn=conn)

        logger.info(f"Completed loading dataset: {session_id} ({len(dataset)} episodes)")

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

        logger.info(f"Loading {len(camera_dirs)} camera episodes...")

        for episode_num, camera_dir in enumerate(camera_dirs):
            # Find CSV file (e.g., thermal_2_tracks.csv)
            csv_files = list(camera_dir.glob("*_tracks.csv"))
            if not csv_files:
                logger.warning(f"No CSV file found in {camera_dir}, skipping")
                continue

            csv_path = csv_files[0]
            episode_name = camera_dir.name  # e.g., "thermal_2"

            self.load_episode_csv(session_id, episode_num, episode_name, csv_path, conn=conn)

        logger.info(f"Completed loading session: {session_id} ({len(camera_dirs)} episodes)")

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


def main():
    """Command-line interface for data loader."""
    parser = argparse.ArgumentParser(
        description='Load tracking data into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load 3D boids simulation (uses environment variables for DB)
  python -m collab_env.data.db.db_loader --source boids3d --path simulated_data/hackathon

  # Load with specific backend
  python -m collab_env.data.db.db_loader --source boids3d --path simulated_data/hackathon --backend duckdb
        """
    )

    parser.add_argument(
        '--source',
        required=True,
        choices=['boids3d', 'boids2d', 'tracking'],
        help='Data source type'
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
        # Load data based on source type
        if args.source == 'boids3d':
            loader = Boids3DLoader(db_conn)

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
            loader = Boids2DLoader(db_conn)

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
            loader = TrackingCSVLoader(db_conn)

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

        logger.info("Data loading completed successfully")

    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        raise
    finally:
        db_conn.close()


if __name__ == '__main__':
    main()
