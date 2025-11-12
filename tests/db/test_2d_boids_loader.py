"""
Tests for 2D boids data loader.
"""

from pathlib import Path

import pytest
from collab_env.data.db.config import DBConfig
from collab_env.data.db.db_loader import (
    DatabaseConnection,
    Boids2DLoader,
)


class TestBoids2DLoader:
    """Test 2D boids data loader."""

    def test_load_2d_boids_dataset(self, backend_config: DBConfig, sample_2d_boids_data: Path):
        """Test loading a complete 2D boids dataset."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids2DLoader(db)
        loader.load_dataset(sample_2d_boids_data)

        # Verify session was created
        result = db.fetch_one(
            "SELECT COUNT(*) FROM sessions WHERE category_id = :cat",
            {'cat': 'boids_2d'}
        )
        assert result[0] >= 1, "Should have created at least one 2D boids session"

        # Verify episodes were created (3 samples)
        result = db.fetch_one(
            "SELECT COUNT(*) FROM episodes WHERE session_id LIKE :pattern",
            {'pattern': '%2d%'}
        )
        assert result[0] == 3, f"Expected 3 episodes, found {result[0]}"

        # Verify observations were loaded (3 samples × 10 timesteps × 5 agents = 150)
        result = db.fetch_one(
            "SELECT COUNT(*) FROM observations WHERE episode_id LIKE :pattern",
            {'pattern': '%2d%'}
        )
        assert result[0] == 630, f"Expected 150 observations, found {result[0]}"

        # Cleanup
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM sessions WHERE category_id = :cat", {'cat': 'boids_2d'})
        db.close()

    def test_2d_boids_data_integrity(self, backend_config: DBConfig, sample_2d_boids_data: Path):
        """Test that 2D boids observations are loaded with correct data."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids2DLoader(db)
        loader.load_dataset(sample_2d_boids_data)

        # Check that all observations have x, y coordinates
        result = db.fetch_one("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END) as null_x,
                SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) as null_y,
                SUM(CASE WHEN z IS NOT NULL THEN 1 ELSE 0 END) as non_null_z,
                SUM(CASE WHEN v_x IS NULL THEN 1 ELSE 0 END) as null_vx,
                SUM(CASE WHEN v_y IS NULL THEN 1 ELSE 0 END) as null_vy,
                SUM(CASE WHEN v_z IS NOT NULL THEN 1 ELSE 0 END) as non_null_vz
            FROM observations
            WHERE episode_id LIKE :pattern
        """, {'pattern': '%2d%'})

        total, null_x, null_y, non_null_z, null_vx, null_vy, non_null_vz = result

        assert null_x == 0, "All observations should have x coordinate"
        assert null_y == 0, "All observations should have y coordinate"
        assert non_null_z == 0, "2D boids should not have z coordinate"
        assert null_vx == 0, "All observations should have v_x velocity"
        assert null_vy == 0, "All observations should have v_y velocity"
        assert non_null_vz == 0, "2D boids should not have v_z velocity"

        # Check position values are in expected range (scaled from [0,1] to [0,480])
        result = db.fetch_one("""
            SELECT MIN(x), MAX(x), MIN(y), MAX(y)
            FROM observations
            WHERE episode_id LIKE :pattern
        """, {'pattern': '%2d%'})

        min_x, max_x, min_y, max_y = result
        assert min_x >= 0 and max_x <= 480, f"X coordinates should be in [0, 480], got [{min_x}, {max_x}]"
        assert min_y >= 0 and max_y <= 480, f"Y coordinates should be in [0, 480], got [{min_y}, {max_y}]"

        # Cleanup
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM sessions WHERE category_id = :cat", {'cat': 'boids_2d'})
        db.close()

    def test_2d_boids_velocity_computation(self, backend_config: DBConfig, sample_2d_boids_data: Path):
        """Test that velocities are computed correctly from positions."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids2DLoader(db)
        loader.load_dataset(sample_2d_boids_data)

        # Get observations for agent 0 in first episode at consecutive timesteps
        results = db.fetch_all("""
            SELECT time_index, x, y, v_x, v_y
            FROM observations
            WHERE episode_id LIKE :pattern
              AND agent_id = 0
            ORDER BY time_index
            LIMIT 3
        """, {'pattern': '%episode-0000%'})

        assert len(results) >= 2, "Need at least 2 timesteps to verify velocity"

        # For first two timesteps, verify v[t] ≈ (p[t+1] - p[t])
        t0, x0, y0, vx0, vy0 = results[0]
        t1, x1, y1, vx1, vy1 = results[1]

        # Computed velocity should approximate position difference
        expected_vx = x1 - x0
        expected_vy = y1 - y0

        # Allow for floating point precision
        assert abs(vx0 - expected_vx) < 0.01, f"v_x mismatch: expected {expected_vx:.4f}, got {vx0:.4f}"
        assert abs(vy0 - expected_vy) < 0.01, f"v_y mismatch: expected {expected_vy:.4f}, got {vy0:.4f}"

        # Cleanup
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM sessions WHERE category_id = :cat", {'cat': 'boids_2d'})
        db.close()

    def test_2d_boids_episode_metadata(self, backend_config: DBConfig, sample_2d_boids_data: Path):
        """Test that episode metadata is correct for 2D boids."""
        db = DatabaseConnection(backend_config)
        db.connect()

        loader = Boids2DLoader(db)
        loader.load_dataset(sample_2d_boids_data)

        # Check episode metadata
        results = db.fetch_all("""
            SELECT episode_number, num_frames, num_agents, frame_rate
            FROM episodes
            WHERE session_id LIKE :pattern
            ORDER BY episode_number
        """, {'pattern': '%2d%'})

        assert len(results) == 3, f"Expected 3 episodes, found {len(results)}"

        for episode_num, num_frames, num_agents, frame_rate in results:
            assert num_frames == 10, f"Expected 10 frames, got {num_frames}"
            assert num_agents == 21, f"Expected 5 agents, got {num_agents}"
            assert frame_rate == 1.0, f"Expected frame_rate=1.0, got {frame_rate}"

        # Cleanup
        db.execute("DELETE FROM observations WHERE episode_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM episodes WHERE session_id LIKE :pattern", {'pattern': '%2d%'})
        db.execute("DELETE FROM sessions WHERE category_id = :cat", {'cat': 'boids_2d'})
        db.close()
