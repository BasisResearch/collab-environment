"""
Tests for data loading functionality.
"""

from pathlib import Path

import pytest
from collab_env.data.db.config import DBConfig
from collab_env.data.db.db_loader import (
    DatabaseConnection,
    Boids3DLoader,
    SessionMetadata,
    EpisodeMetadata
)


class TestDataLoader:
    """Test data loading pipeline."""

    def test_load_session(self, backend_config: DBConfig):
        """Test loading session metadata."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids3DLoader(db)
        metadata = SessionMetadata(
            session_id='test-session-1',
            session_name='Test Session',
            data_source='boids_3d',
            category='simulated',
            config={'frame_rate': 30.0},
            metadata={'test': True}
        )

        loader.load_session(metadata)

        # Verify session was loaded
        result = db.fetch_one(
            "SELECT session_name FROM sessions WHERE session_id = :sid",
            {'sid': 'test-session-1'}
        )
        assert result is not None
        assert result[0] == 'Test Session'

        # Cleanup
        db.execute("DELETE FROM sessions WHERE session_id = :sid", {'sid': 'test-session-1'})
        db.close()

    def test_load_episode(self, backend_config: DBConfig):
        """Test loading episode metadata."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids3DLoader(db)

        # First load session
        session_metadata = SessionMetadata(
            session_id='test-session-2',
            session_name='Test Session',
            data_source='boids_3d',
            category='simulated',
            config={}
        )
        loader.load_session(session_metadata)

        # Then load episode
        episode_metadata = EpisodeMetadata(
            episode_id='test-episode-1',
            session_id='test-session-2',
            episode_number=0,
            num_frames=100,
            num_agents=10,
            frame_rate=30.0,
            file_path='/tmp/test.parquet'
        )
        loader.load_episode(episode_metadata)

        # Verify episode was loaded
        result = db.fetch_one(
            "SELECT num_agents FROM episodes WHERE episode_id = :eid",
            {'eid': 'test-episode-1'}
        )
        assert result is not None
        assert result[0] == 10

        # Cleanup
        db.execute("DELETE FROM episodes WHERE episode_id = :eid", {'eid': 'test-episode-1'})
        db.execute("DELETE FROM sessions WHERE session_id = :sid", {'sid': 'test-session-2'})
        db.close()

    def test_load_boids_simulation(self, backend_config: DBConfig, sample_boids_data: Path):
        """Test loading a complete boids simulation."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids3DLoader(db)
        loader.load_simulation(sample_boids_data)

        # Verify session was created
        session_name = sample_boids_data.name
        result = db.fetch_one(
            "SELECT COUNT(*) FROM sessions WHERE session_name = :name",
            {'name': session_name}
        )
        assert result[0] >= 1

        # Verify episodes were created
        result = db.fetch_one(
            "SELECT COUNT(*) FROM episodes WHERE session_id LIKE :pattern",
            {'pattern': f'%{session_name}%'}
        )
        assert result[0] >= 1

        # Verify observations were loaded
        result = db.fetch_one("SELECT COUNT(*) FROM observations")
        assert result[0] > 0, "Should have loaded some observations"

        # Expected: 5 agents * 10 frames = 50 observations
        assert result[0] == 50, f"Expected 50 observations, found {result[0]}"

        # Cleanup
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE :pattern)", {'pattern': f'%test%'})
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': f'%test%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': f'%{session_name}%'})
        db.execute("DELETE FROM sessions WHERE session_name = :name", {'name': session_name})
        db.close()

    def test_observations_data_integrity(self, backend_config: DBConfig, sample_boids_data: Path):
        """Test that observations are loaded with correct data."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids3DLoader(db)
        loader.load_simulation(sample_boids_data)

        # Check specific observation values
        result = db.fetch_one("""
            SELECT x, y, z, v_x, v_y, v_z
            FROM observations
            WHERE agent_id = 0 AND time_index = 0
            ORDER BY episode_id DESC
            LIMIT 1
        """)

        assert result is not None, "Should find observation for agent 0 at time 0"
        x, y, z, v_x, v_y, v_z = result

        # Check values match our sample data generation
        assert x == 0.0, f"Expected x=0.0, got {x}"
        assert y == 0.0, f"Expected y=0.0, got {y}"
        assert z == 0.0, f"Expected z=0.0, got {z}"
        assert v_x == 1.0, f"Expected v_x=1.0, got {v_x}"
        assert v_y == 0.5, f"Expected v_y=0.5, got {v_y}"
        assert v_z == 0.0, f"Expected v_z=0.0, got {v_z}"

        # Cleanup
        session_name = sample_boids_data.name
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE :pattern)", {'pattern': f'%test%'})
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': f'%test%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': f'%{session_name}%'})
        db.execute("DELETE FROM sessions WHERE session_name = :name", {'name': session_name})
        db.close()

    def test_extended_properties_loaded(self, backend_config: DBConfig, sample_boids_data: Path):
        """Test that extended properties are loaded."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids3DLoader(db)
        loader.load_simulation(sample_boids_data)

        # Check if extended properties were loaded
        result = db.fetch_one("SELECT COUNT(*) FROM extended_properties")
        count = result[0]

        # We should have some extended properties (distance_to_target_center)
        assert count > 0, f"Expected some extended properties, found {count}"

        # Expected: 50 observations * 1 property each = 50 extended properties
        assert count == 50, f"Expected 50 extended properties, found {count}"

        # Cleanup
        session_name = sample_boids_data.name
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE :pattern)", {'pattern': f'%test%'})
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': f'%test%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': f'%{session_name}%'})
        db.execute("DELETE FROM sessions WHERE session_name = :name", {'name': session_name})
        db.close()


class TestDataLoaderPerformance:
    """Test data loading performance."""

    def test_bulk_insert_performance(self, backend_config: DBConfig, sample_boids_data: Path):
        """Test that bulk inserts are reasonably fast."""
        import time

        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids3DLoader(db)

        start = time.time()
        loader.load_simulation(sample_boids_data)
        elapsed = time.time() - start

        # Should load 50 observations in under 2 seconds
        assert elapsed < 2.0, f"Loading took {elapsed:.2f}s, expected < 2s"

        # Cleanup
        session_name = sample_boids_data.name
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE :pattern)", {'pattern': f'%test%'})
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': f'%test%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': f'%{session_name}%'})
        db.execute("DELETE FROM sessions WHERE session_name = :name", {'name': session_name})
        db.close()
