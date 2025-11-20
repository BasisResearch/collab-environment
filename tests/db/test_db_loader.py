"""
Tests for data loading functionality.
"""

from pathlib import Path

import pytest
from collab_env.data.db.config import DBConfig
from collab_env.data.db.db_loader import (
    DatabaseConnection,
    Boids3DLoader,
    GNNRolloutLoader,
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
            category_id='boids_3d',
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
            category_id='boids_3d',
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

        # Expected: (5 agents + 2 env entities) * 10 frames = 70 observations
        assert result[0] == 70, f"Expected 70 observations (5 agents + 2 env), found {result[0]}"

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

        # Check specific observation values (filter by agent_type_id to get agent, not env entity)
        result = db.fetch_one("""
            SELECT x, y, z, v_x, v_y, v_z
            FROM observations
            WHERE agent_id = 0 AND time_index = 0 AND agent_type_id = 'agent'
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

        # Should load 70 observations (5 agents + 2 env * 10 frames) in under 2 seconds
        assert elapsed < 2.0, f"Loading took {elapsed:.2f}s, expected < 2s"

        # Cleanup
        session_name = sample_boids_data.name
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE :pattern)", {'pattern': f'%test%'})
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': f'%test%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': f'%{session_name}%'})
        db.execute("DELETE FROM sessions WHERE session_name = :name", {'name': session_name})
        db.close()


class TestGNNRolloutLoader:
    """Test GNN rollout data loading."""

    @pytest.fixture
    def rollout_file(self):
        """Path to a sample rollout file."""
        rollout_path = Path("trained_models/boid_food_strong/rollouts/boid_food_strong_vpluspplus_a_n0_h1_vr0.5_s0_rollout_5.pkl")
        if not rollout_path.exists():
            pytest.skip(f"Rollout file not found: {rollout_path}")
        return rollout_path

    def test_load_gnn_rollout(self, backend_config: DBConfig, rollout_file: Path):
        """Test loading a GNN rollout file."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = GNNRolloutLoader(db, max_episodes=2)  # Load only 2 trajectories for speed
        loader.load_rollout_file(rollout_file, scene_size=480.0)

        # Verify session was created
        result = db.fetch_one(
            "SELECT COUNT(*) FROM sessions WHERE category_id = 'boids_2d_rollout'"
        )
        assert result[0] >= 1, "Should have created rollout session"

        # Verify episodes were created (2 episodes per trajectory: actual + predicted)
        result = db.fetch_one(
            "SELECT COUNT(*) FROM episodes WHERE session_id LIKE 'rollout-%'"
        )
        assert result[0] >= 2, f"Should have at least 2 episodes, found {result[0]}"

        # Verify observations were loaded
        result = db.fetch_one("SELECT COUNT(*) FROM observations")
        assert result[0] > 0, "Should have loaded some observations"

        # Cleanup
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE 'rollout-%')")
        db.execute("DELETE FROM observations WHERE episode_id LIKE 'rollout-%'")
        db.execute("DELETE FROM episodes WHERE session_id LIKE 'rollout-%'")
        db.execute("DELETE FROM sessions WHERE category_id = 'boids_2d_rollout'")
        db.close()

    def test_prediction_error_property(self, backend_config: DBConfig, rollout_file: Path):
        """Test that prediction_error is computed for predicted episodes."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = GNNRolloutLoader(db, max_episodes=2)
        loader.load_rollout_file(rollout_file, scene_size=480.0)

        # Find a predicted episode
        result = db.fetch_one(
            "SELECT episode_id FROM episodes WHERE episode_id LIKE '%-predicted' LIMIT 1"
        )
        assert result is not None, "Should have a predicted episode"
        predicted_episode_id = result[0]

        # Check that prediction_error property exists for predicted episode
        result = db.fetch_one("""
            SELECT COUNT(*)
            FROM extended_properties ep
            JOIN observations o ON ep.observation_id = o.observation_id
            WHERE o.episode_id = :eid
            AND ep.property_id = 'prediction_error'
            AND o.agent_type_id = 'agent'
        """, {'eid': predicted_episode_id})

        count = result[0]
        assert count > 0, f"Should have prediction_error for predicted episode, found {count}"

        # Verify prediction_error values are non-negative
        result = db.fetch_one("""
            SELECT MIN(ep.value_float), MAX(ep.value_float)
            FROM extended_properties ep
            JOIN observations o ON ep.observation_id = o.observation_id
            WHERE o.episode_id = :eid
            AND ep.property_id = 'prediction_error'
        """, {'eid': predicted_episode_id})

        min_val, max_val = result
        assert min_val >= 0, f"prediction_error should be non-negative, found min={min_val}"
        assert max_val < 480.0, f"prediction_error should be < scene_size, found max={max_val}"

        # Verify prediction_error does NOT exist in actual episodes
        result = db.fetch_one(
            "SELECT episode_id FROM episodes WHERE episode_id LIKE '%-actual' LIMIT 1"
        )
        assert result is not None, "Should have an actual episode"
        actual_episode_id = result[0]

        result = db.fetch_one("""
            SELECT COUNT(*)
            FROM extended_properties ep
            JOIN observations o ON ep.observation_id = o.observation_id
            WHERE o.episode_id = :eid
            AND ep.property_id = 'prediction_error'
        """, {'eid': actual_episode_id})

        assert result[0] == 0, "Should NOT have prediction_error for actual episodes"

        # Cleanup
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE 'rollout-%')")
        db.execute("DELETE FROM observations WHERE episode_id LIKE 'rollout-%'")
        db.execute("DELETE FROM episodes WHERE session_id LIKE 'rollout-%'")
        db.execute("DELETE FROM sessions WHERE category_id = 'boids_2d_rollout'")
        db.close()

    def test_loss_env_agent(self, backend_config: DBConfig, rollout_file: Path):
        """Test that loss is stored as env agent observations."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = GNNRolloutLoader(db, max_episodes=2)
        loader.load_rollout_file(rollout_file, scene_size=480.0)

        # Find a predicted episode
        result = db.fetch_one(
            "SELECT episode_id FROM episodes WHERE episode_id LIKE '%-predicted' LIMIT 1"
        )
        assert result is not None, "Should have a predicted episode"
        predicted_episode_id = result[0]

        # Check that env observations exist in predicted episode
        result = db.fetch_one("""
            SELECT COUNT(*)
            FROM observations
            WHERE episode_id = :eid
            AND agent_type_id = 'env'
            AND agent_id = -1
        """, {'eid': predicted_episode_id})

        env_obs_count = result[0]
        assert env_obs_count > 0, f"Should have env observations in predicted episode, found {env_obs_count}"

        # Verify env observations are at scene center with NULL velocities
        result = db.fetch_one("""
            SELECT x, y, v_x, v_y, v_z
            FROM observations
            WHERE episode_id = :eid
            AND agent_type_id = 'env'
            AND agent_id = -1
            LIMIT 1
        """, {'eid': predicted_episode_id})

        assert result is not None, "Should have env observation"
        x, y, v_x, v_y, v_z = result
        assert x == 240.0, f"Expected x=240.0 (scene center), got {x}"
        assert y == 240.0, f"Expected y=240.0 (scene center), got {y}"
        assert v_x is None, f"Expected v_x=NULL, got {v_x}"
        assert v_y is None, f"Expected v_y=NULL, got {v_y}"
        assert v_z is None, f"Expected v_z=NULL, got {v_z}"

        # Check that loss property exists for env observations
        result = db.fetch_one("""
            SELECT COUNT(*)
            FROM extended_properties ep
            JOIN observations o ON ep.observation_id = o.observation_id
            WHERE o.episode_id = :eid
            AND o.agent_type_id = 'env'
            AND ep.property_id = 'loss'
        """, {'eid': predicted_episode_id})

        loss_count = result[0]
        assert loss_count > 0, f"Should have loss property for env agent, found {loss_count}"
        assert loss_count == env_obs_count, f"Should have one loss per env observation, found {loss_count} vs {env_obs_count}"

        # Verify loss values are non-negative
        result = db.fetch_one("""
            SELECT MIN(ep.value_float), MAX(ep.value_float)
            FROM extended_properties ep
            JOIN observations o ON ep.observation_id = o.observation_id
            WHERE o.episode_id = :eid
            AND ep.property_id = 'loss'
        """, {'eid': predicted_episode_id})

        min_loss, max_loss = result
        assert min_loss >= 0, f"Loss should be non-negative, found min={min_loss}"

        # Verify env observations do NOT exist in actual episodes
        result = db.fetch_one(
            "SELECT episode_id FROM episodes WHERE episode_id LIKE '%-actual' LIMIT 1"
        )
        assert result is not None, "Should have an actual episode"
        actual_episode_id = result[0]

        result = db.fetch_one("""
            SELECT COUNT(*)
            FROM observations
            WHERE episode_id = :eid
            AND agent_type_id = 'env'
        """, {'eid': actual_episode_id})

        assert result[0] == 0, "Should NOT have env observations in actual episodes"

        # Cleanup
        db.execute("DELETE FROM extended_properties WHERE observation_id IN (SELECT observation_id FROM observations WHERE episode_id LIKE 'rollout-%')")
        db.execute("DELETE FROM observations WHERE episode_id LIKE 'rollout-%'")
        db.execute("DELETE FROM episodes WHERE session_id LIKE 'rollout-%'")
        db.execute("DELETE FROM sessions WHERE category_id = 'boids_2d_rollout'")
        db.close()
