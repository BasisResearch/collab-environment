"""
Data loader for tracking analytics database.

Loads data from various sources into PostgreSQL or DuckDB:
- 3D Boids: Parquet files from collab_env.sim.boids
- 2D Boids: PyTorch .pt files from collab_env.sim.boids_gnn_temp
- Tracking CSV: Real-world tracking data from collab_env.tracking
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow.parquet as pq
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from collab_env.data.db.config import DBConfig, get_db_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Execute a query with named parameters."""
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

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """Insert DataFrame using pandas to_sql (much faster for bulk inserts)."""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False, method='multi')


class BaseDataLoader:
    """Base class for data loaders."""

    def __init__(self, db_conn: DatabaseConnection):
        self.db = db_conn

    def load_session(self, metadata: SessionMetadata):
        """Load session metadata into database."""
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
        })
        logger.info(f"Loaded session: {metadata.session_id}")

    def load_episode(self, metadata: EpisodeMetadata):
        """Load episode metadata into database."""
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
        })
        logger.info(f"Loaded episode: {metadata.episode_id}")

    def load_observations_batch(self, observations: pd.DataFrame, episode_id: str):
        """Load observations in batch using pandas to_sql."""
        # Ensure required columns exist
        required_cols = ['time_index', 'agent_id', 'x', 'y']
        for col in required_cols:
            if col not in observations.columns:
                raise ValueError(f"Missing required column: {col}")

        # Prepare DataFrame for insertion
        df = pd.DataFrame({
            'episode_id': episode_id,
            'time_index': observations['time_index'].astype(int),
            'agent_id': observations['agent_id'].astype(int),
            'agent_type_id': observations.get('agent_type_id', 'agent'),
            'x': observations['x'].astype(float),
            'y': observations['y'].astype(float),
            'z': observations['z'].astype(float) if 'z' in observations else None,
            'v_x': observations['v_x'].astype(float) if 'v_x' in observations else None,
            'v_y': observations['v_y'].astype(float) if 'v_y' in observations else None,
            'v_z': observations['v_z'].astype(float) if 'v_z' in observations else None,
            'confidence': observations['confidence'].astype(float) if 'confidence' in observations else None,
            'detection_class': observations.get('detection_class', None)
        })

        # Use pandas to_sql for fast bulk insert
        self.db.insert_dataframe(df, 'observations', if_exists='append')
        logger.info(f"Loaded {len(df)} observations for episode {episode_id}")

    def load_extended_properties_batch(
        self,
        episode_id: str,
        property_data: Dict[str, pd.Series]
    ):
        """
        Load extended properties in batch.

        Args:
            episode_id: Episode identifier
            property_data: Dict mapping property_id to Series with values
                          Series index should match observations (time_index, agent_id)
        """
        # Get observation IDs for this episode
        query = """
        SELECT observation_id, time_index, agent_id
        FROM observations
        WHERE episode_id = :episode_id
        ORDER BY time_index, agent_id
        """

        obs_rows = self.db.fetch_all(query, {'episode_id': episode_id})

        # Build mapping from (time_index, agent_id) to observation_id
        obs_id_map = {(row[1], row[2]): row[0] for row in obs_rows}

        # Prepare data for batch insert
        records = []
        for property_id, values in property_data.items():
            for (time_idx, agent_id), value in values.items():
                obs_id = obs_id_map.get((time_idx, agent_id))
                if obs_id is None:
                    logger.warning(f"No observation found for ({time_idx}, {agent_id})")
                    continue

                if pd.notna(value):
                    records.append({
                        'observation_id': obs_id,
                        'property_id': property_id,
                        'value_float': float(value) if isinstance(value, (int, float)) else None,
                        'value_text': str(value) if not isinstance(value, (int, float)) else None
                    })

        if not records:
            logger.info(f"No extended properties to load for episode {episode_id}")
            return

        # Use pandas to_sql for fast bulk insert
        df = pd.DataFrame(records)
        self.db.insert_dataframe(df, 'extended_properties', if_exists='append')
        logger.info(f"Loaded {len(records)} extended property values for episode {episode_id}")


class Boids3DLoader(BaseDataLoader):
    """Loader for 3D boids simulation data from parquet files."""

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

        self.load_session(session_metadata)

        # Load each episode
        episode_files = sorted(simulation_dir.glob("episode-*.parquet"))
        for episode_num, episode_file in enumerate(episode_files):
            self.load_episode_file(session_id, episode_num, episode_file, config)

        logger.info(f"Completed loading simulation: {session_id} ({len(episode_files)} episodes)")

    def load_episode_file(
        self,
        session_id: str,
        episode_number: int,
        file_path: Path,
        config: Dict[str, Any]
    ):
        """Load a single episode parquet file."""
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

        self.load_episode(episode_metadata)

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

        self.load_observations_batch(observations, episode_id)

        # Load extended properties if they exist
        import numpy as np
        extended_props = {}

        # Map actual parquet column names to property IDs
        # Distance to target center (may have suffix like _1)
        target_center_cols = [c for c in df.columns if c.startswith('distance_target_center')]
        if target_center_cols:
            extended_props['distance_to_target_center'] = df.set_index(['time', 'id'])[target_center_cols[0]]

        # Distance to target mesh (may have suffix like _1)
        target_mesh_cols = [c for c in df.columns if 'distance_to_target_mesh' in c]
        if target_mesh_cols:
            extended_props['distance_to_target_mesh'] = df.set_index(['time', 'id'])[target_mesh_cols[0]]

        # Distance to scene mesh
        if 'mesh_scene_distance' in df.columns:
            extended_props['distance_to_scene_mesh'] = df.set_index(['time', 'id'])['mesh_scene_distance']

        # Handle array-type closest point columns
        # Target mesh closest point (stored as array [x, y, z])
        # Match columns like 'target_mesh_closest_point_1' but NOT 'distance_to_target_mesh_closest_point_1'
        target_closest_cols = [c for c in df.columns if 'target_mesh_closest_point' in c and not c.startswith('distance')]
        if target_closest_cols:
            # Extract array column and filter out None values
            arr_col = df[target_closest_cols[0]]
            # Create mask for non-None values
            mask = arr_col.notna()
            filtered_df = df[mask]
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
        if 'mesh_scene_closest_point' in df.columns:
            # Extract array column and filter out None values
            arr_col = df['mesh_scene_closest_point']
            # Create mask for non-None values
            mask = arr_col.notna()
            filtered_df = df[mask]
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
            self.load_extended_properties_batch(episode_id, extended_props)


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
            loader.load_simulation(args.path)
        elif args.source == 'boids2d':
            raise NotImplementedError("2D boids loader not yet implemented")
        elif args.source == 'tracking':
            raise NotImplementedError("Tracking CSV loader not yet implemented")

        logger.info("Data loading completed successfully")

    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise
    finally:
        db_conn.close()


if __name__ == '__main__':
    main()
