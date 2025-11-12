"""
Pytest fixtures for database tests - with 2D boids fixture.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from collab_env.data.db.config import DBConfig, get_db_config
from collab_env.data.db.init_database import DatabaseBackend
from collab_env.data.db.db_loader import DatabaseConnection
from collab_env.data.file_utils import get_project_root
from collab_env.sim.boids_gnn_temp.animal_simulation import AnimalTrajectoryDataset


@pytest.fixture
def temp_duckdb() -> Generator[Path, None, None]:
    """Create a temporary DuckDB file path (file created by DuckDB)."""
    # Create temp file to get the name, then delete it so DuckDB can create it fresh
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=True) as f:
        db_path = Path(f.name)

    # File is deleted by tempfile, DuckDB will create it fresh
    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()
    # Also clean up .wal file if it exists
    wal_file = db_path.with_suffix('.duckdb.wal')
    if wal_file.exists():
        wal_file.unlink()


@pytest.fixture
def duckdb_config(temp_duckdb: Path) -> DBConfig:
    """DuckDB configuration for testing."""
    os.environ['DB_BACKEND'] = 'duckdb'
    os.environ['DUCKDB_PATH'] = str(temp_duckdb)
    return get_db_config('duckdb')


@pytest.fixture
def postgres_config() -> DBConfig:
    """PostgreSQL configuration from .env file."""
    os.environ['DB_BACKEND'] = 'postgres'
    return get_db_config('postgres')


@pytest.fixture
def duckdb_initialized(duckdb_config: DBConfig) -> DBConfig:
    """DuckDB with schema initialized."""
    backend = DatabaseBackend(duckdb_config)
    backend.connect()

    # Initialize schema
    project_root = get_project_root()
    schema_dir = project_root / 'schema'
    backend.execute_file(schema_dir / '01_core_tables.sql')
    backend.execute_file(schema_dir / '02_extended_properties.sql')
    backend.execute_file(schema_dir / '03_seed_data.sql')

    backend.close()
    return duckdb_config


@pytest.fixture
def postgres_initialized(postgres_config: DBConfig) -> DBConfig:
    """PostgreSQL with schema initialized (or skip if unavailable)."""
    backend = None
    try:
        backend = DatabaseBackend(postgres_config)
        backend.connect()

        # SAFETY CHECK: Ensure we're using a test database
        # Refuse to run tests on production databases
        db_name = postgres_config.postgres.dbname.lower()
        forbidden_names = ['tracking_analytics', 'production', 'prod', 'main']

        if db_name in forbidden_names:
            pytest.skip(
                f"SAFETY: Refusing to run tests on database '{db_name}'. "
                f"Please set POSTGRES_DB=tracking_analytics_test in your environment. "
                f"Tests drop all tables and should only run on dedicated test databases!"
            )

        # Require explicit '_test' suffix for safety
        if not db_name.endswith('_test'):
            pytest.skip(
                f"SAFETY: Database name '{db_name}' must end with '_test'. "
                f"Example: tracking_analytics_test. "
                f"This prevents accidentally running destructive tests on production."
            )

        # Clean up any existing test data
        # DROP TABLE IF EXISTS with CASCADE should always succeed
        backend.execute_query("DROP TABLE IF EXISTS extended_properties CASCADE")
        backend.execute_query("DROP TABLE IF EXISTS observations CASCADE")
        backend.execute_query("DROP TABLE IF EXISTS episodes CASCADE")
        backend.execute_query("DROP TABLE IF EXISTS sessions CASCADE")
        backend.execute_query("DROP TABLE IF EXISTS property_definitions CASCADE")
        backend.execute_query("DROP TABLE IF EXISTS categories CASCADE")
        backend.execute_query("DROP TABLE IF EXISTS agent_types CASCADE")

        # Initialize schema
        project_root = get_project_root()
        schema_dir = project_root / 'schema'
        backend.execute_file(schema_dir / '01_core_tables.sql')
        backend.execute_file(schema_dir / '02_extended_properties.sql')
        backend.execute_file(schema_dir / '03_seed_data.sql')

        backend.close()
        return postgres_config
    except Exception as e:
        if backend:
            try:
                backend.close()
            except:
                pass
        pytest.skip(f"PostgreSQL not available: {e}")


@pytest.fixture
def sample_boids_data(tmp_path: Path) -> Path:
    """Create a small sample boids dataset for testing."""
    # Create config.yaml
    config = {
        'frame_rate': 30.0,
        'num_agents': 10,
        'num_frames': 100,
        'scene_size': [100, 100, 100]
    }

    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Create sample episode data with both agents and environment entities
    num_agents = 5
    num_env_entities = 2  # Environment entities (walls, obstacles)
    num_frames = 10

    data = []
    for t in range(num_frames):
        # Agent entities
        for agent_id in range(num_agents):
            data.append({
                'time': t,
                'id': agent_id,
                'type': 'agent',
                'x': float(agent_id * 10 + t),
                'y': float(agent_id * 10 + t * 0.5),
                'z': float(agent_id * 5),
                'v_x': 1.0,
                'v_y': 0.5,
                'v_z': 0.0,
                'distance_to_target_center': float(100 - t * agent_id)
            })

        # Environment entities (can have same IDs as agents - now supported!)
        for env_id in range(num_env_entities):
            data.append({
                'time': t,
                'id': env_id,  # Same IDs as agents - no conflict with new schema
                'type': 'env',
                'x': float(500 + env_id * 100),
                'y': float(500 + env_id * 100),
                'z': 0.0,
                'v_x': 0.0,  # Environment entities don't move
                'v_y': 0.0,
                'v_z': 0.0,
                'distance_to_target_center': None
            })

    df = pd.DataFrame(data)
    episode_path = tmp_path / 'episode-0-test.parquet'
    df.to_parquet(episode_path)

    return tmp_path


@pytest.fixture(params=['duckdb', 'postgres'])
def backend_config(request):
    """Parametrized fixture that tests both backends."""
    if request.param == 'duckdb':
        # Get duckdb_initialized fixture
        return request.getfixturevalue('duckdb_initialized')
    else:
        # Get postgres_initialized fixture (may skip)
        return request.getfixturevalue('postgres_initialized')


@pytest.fixture
def sample_2d_boids_data(tmp_path: Path) -> Path:
    """Create a small sample 2D boids dataset for testing."""
    # Create config
    config = {
        'A': {
            'visual_range': 50,
            'centering_factor': 0.005,
            'min_distance': 15,
            'avoid_factor': 0.05,
            'matching_factor': 0.5,
            'margin': 5,
            'turn_factor': 10,
            'speed_limit': 7,
            'counts': 5,
            'independent': False
        },
        'scene_size': 480.0
    }

    config_path = tmp_path / 'test_2d_boids_config.pt'
    torch.save(config, config_path)

    # Create dataset with 3 samples
    # Use the actual dataset class from real data
    dataset = torch.load('simulated_data/boid_food_basic.pt', weights_only=False)

    # Create a simple test dataset with just the first 3 samples
    test_samples = [dataset[i] for i in range(3)]

    # Save as simple list
    data_path = tmp_path / 'test_2d_boids.pt'
    torch.save(test_samples, data_path)

    return data_path
