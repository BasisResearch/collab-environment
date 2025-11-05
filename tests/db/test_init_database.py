"""
Tests for database initialization (schema creation).
"""

import pytest
from collab_env.data.db.config import DBConfig
from collab_env.data.db.db_loader import DatabaseConnection


class TestDatabaseInitialization:
    """Test database schema creation and seed data."""

    def test_tables_created(self, backend_config: DBConfig):
        """Test that all 8 tables are created."""
        db = DatabaseConnection(backend_config)
        db.connect()

        # Query information schema for table count
        if backend_config.backend == 'postgres':
            query = """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """
        else:  # duckdb
            query = """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = 'main'
            """

        result = db.fetch_one(query)
        table_count = result[0]

        db.close()

        assert table_count == 8, f"Expected 8 tables, found {table_count}"

    def test_agent_types_seeded(self, backend_config: DBConfig):
        """Test that agent types seed data is loaded."""
        db = DatabaseConnection(backend_config)
        db.connect()

        result = db.fetch_one("SELECT COUNT(*) FROM agent_types")
        count = result[0]

        db.close()

        assert count == 5, f"Expected 5 agent types, found {count}"

    def test_property_definitions_seeded(self, backend_config: DBConfig):
        """Test that property definitions seed data is loaded."""
        db = DatabaseConnection(backend_config)
        db.connect()

        result = db.fetch_one("SELECT COUNT(*) FROM property_definitions")
        count = result[0]

        db.close()

        assert count == 18, f"Expected 18 property definitions, found {count}"

    def test_property_categories_seeded(self, backend_config: DBConfig):
        """Test that property categories seed data is loaded."""
        db = DatabaseConnection(backend_config)
        db.connect()

        result = db.fetch_one("SELECT COUNT(*) FROM property_categories")
        count = result[0]

        db.close()

        assert count == 4, f"Expected 4 property categories, found {count}"

    def test_foreign_key_relationships(self, backend_config: DBConfig):
        """Test that foreign key relationships work."""
        db = DatabaseConnection(backend_config)
        db.connect()

        # Test we can query type_id from agent_types
        result = db.fetch_one("SELECT type_id FROM agent_types WHERE type_id = 'agent'")
        assert result is not None, "agent_types table should have 'agent' type"

        # Test we can query property categories
        result = db.fetch_one("SELECT category_id FROM property_categories WHERE category_id = 'boids_3d'")
        assert result is not None, "property_categories should have 'boids_3d' category"

        db.close()

    def test_sessions_table_structure(self, backend_config: DBConfig):
        """Test that sessions table has correct columns."""
        db = DatabaseConnection(backend_config)
        db.connect()

        # Insert and retrieve a test session
        db.execute("""
            INSERT INTO sessions (session_id, session_name, data_source, category, config, metadata)
            VALUES (:sid, :name, :source, :cat, :config, :meta)
        """, {
            'sid': 'test-session',
            'name': 'Test Session',
            'source': 'boids_3d',
            'cat': 'simulated',
            'config': '{"test": true}',
            'meta': None
        })

        result = db.fetch_one("SELECT session_name FROM sessions WHERE session_id = :sid", {'sid': 'test-session'})
        assert result[0] == 'Test Session'

        # Cleanup
        db.execute("DELETE FROM sessions WHERE session_id = :sid", {'sid': 'test-session'})
        db.close()


class TestDuckDBSpecific:
    """DuckDB-specific tests."""

    def test_auto_increment_works(self, duckdb_initialized: DBConfig):
        """Test that DuckDB auto-increment for observation_id works."""
        db = DatabaseConnection(duckdb_initialized)
        db.connect()

        # First create required parent records
        db.execute("""
            INSERT INTO sessions (session_id, session_name, data_source, category, config)
            VALUES ('test-s', 'Test', 'boids_3d', 'simulated', '{}')
        """)

        db.execute("""
            INSERT INTO episodes (episode_id, session_id, episode_number, num_frames, num_agents, frame_rate, file_path)
            VALUES ('test-e', 'test-s', 0, 10, 5, 30.0, '/tmp/test')
        """)

        # Insert two observations without specifying observation_id
        db.execute("""
            INSERT INTO observations (episode_id, time_index, agent_id, agent_type_id, x, y)
            VALUES ('test-e', 0, 0, 'agent', 0.0, 0.0)
        """)

        db.execute("""
            INSERT INTO observations (episode_id, time_index, agent_id, agent_type_id, x, y)
            VALUES ('test-e', 1, 0, 'agent', 1.0, 1.0)
        """)

        # Check that auto-increment IDs were generated
        result = db.fetch_all("SELECT observation_id FROM observations WHERE episode_id = 'test-e' ORDER BY time_index")

        assert len(result) == 2, "Should have 2 observations"
        assert result[0][0] is not None, "First observation should have an ID"
        assert result[1][0] is not None, "Second observation should have an ID"
        assert result[0][0] != result[1][0], "IDs should be different"

        # Cleanup
        db.execute("DELETE FROM observations WHERE episode_id = 'test-e'")
        db.execute("DELETE FROM episodes WHERE episode_id = 'test-e'")
        db.execute("DELETE FROM sessions WHERE session_id = 'test-s'")
        db.close()


class TestPostgreSQLSpecific:
    """PostgreSQL-specific tests."""

    @pytest.mark.skip(reason="Requires PostgreSQL connection")
    def test_jsonb_support(self, postgres_initialized: DBConfig):
        """Test that PostgreSQL JSONB columns work."""
        db = DatabaseConnection(postgres_initialized)
        db.connect()

        # Insert session with JSON config
        db.execute("""
            INSERT INTO sessions (session_id, session_name, data_source, category, config)
            VALUES (:sid, :name, :source, :cat, :config::jsonb)
        """, {
            'sid': 'test-json',
            'name': 'Test',
            'source': 'boids_3d',
            'cat': 'simulated',
            'config': '{"frame_rate": 30}'
        })

        # Query with JSONB operators (PostgreSQL-specific)
        result = db.fetch_one("""
            SELECT config->>'frame_rate' as frame_rate
            FROM sessions
            WHERE session_id = :sid
        """, {'sid': 'test-json'})

        assert result[0] == '30'

        # Cleanup
        db.execute("DELETE FROM sessions WHERE session_id = :sid", {'sid': 'test-json'})
        db.close()
