"""
Tests for database initialization (schema creation).
"""

import pytest
from collab_env.data.db.config import DBConfig
from collab_env.data.db.db_loader import DatabaseConnection
from collab_env.data.db.init_database import DatabaseBackend


class TestDatabaseInitialization:
    """Test database schema creation and seed data."""

    def test_tables_created(self, backend_config: DBConfig):
        """Test that all 7 tables are created."""
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

        assert table_count == 7, f"Expected 7 tables, found {table_count}"

    def test_agent_types_seeded(self, backend_config: DBConfig):
        """Test that agent types seed data is loaded."""
        db = DatabaseConnection(backend_config)
        db.connect()

        result = db.fetch_one("SELECT COUNT(*) FROM agent_types")
        count = result[0]

        db.close()

        assert count == 6, f"Expected 6 agent types (agent, env, target, bird, rat, gerbil), found {count}"

    def test_property_definitions_seeded(self, backend_config: DBConfig):
        """Test that property definitions seed data is loaded."""
        db = DatabaseConnection(backend_config)
        db.connect()

        result = db.fetch_one("SELECT COUNT(*) FROM property_definitions")
        count = result[0]

        db.close()

        assert count == 20, f"Expected 20 property definitions, found {count}"

    def test_categories_seeded(self, backend_config: DBConfig):
        """Test that session categories seed data is loaded."""
        db = DatabaseConnection(backend_config)
        db.connect()

        result = db.fetch_one("SELECT COUNT(*) FROM categories")
        count = result[0]

        db.close()

        assert count == 3, f"Expected 3 categories (boids_3d, boids_2d, tracking_csv), found {count}"

    def test_foreign_key_relationships(self, backend_config: DBConfig):
        """Test that foreign key relationships work."""
        db = DatabaseConnection(backend_config)
        db.connect()

        # Test we can query type_id from agent_types
        result = db.fetch_one("SELECT type_id FROM agent_types WHERE type_id = 'agent'")
        assert result is not None, "agent_types table should have 'agent' type"

        # Test we can query categories
        result = db.fetch_one("SELECT category_id FROM categories WHERE category_id = 'boids_3d'")
        assert result is not None, "categories should have 'boids_3d' category"

        db.close()

    def test_sessions_table_structure(self, backend_config: DBConfig):
        """Test that sessions table has correct columns."""
        db = DatabaseConnection(backend_config)
        db.connect()

        # Insert and retrieve a test session
        db.execute("""
            INSERT INTO sessions (session_id, session_name, category_id, config, metadata)
            VALUES (:sid, :name, :cat, :config, :meta)
        """, {
            'sid': 'test-session',
            'name': 'Test Session',
            'cat': 'boids_3d',
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
            INSERT INTO sessions (session_id, session_name, category_id, config)
            VALUES ('test-s', 'Test', 'boids_3d', '{}')
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
            INSERT INTO sessions (session_id, session_name, category_id, config)
            VALUES (:sid, :name, :cat, :config::jsonb)
        """, {
            'sid': 'test-json',
            'name': 'Test',
            'cat': 'boids_3d',
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


class TestDatabaseBackendExecuteQuery:
    """Test DatabaseBackend.execute_query method specifically."""

    def test_execute_query_returns_select_results(self, backend_config: DBConfig):
        """Test that execute_query returns results from SELECT queries."""
        backend = DatabaseBackend(backend_config)
        backend.connect()

        # Query table count (this was the failing scenario)
        if backend_config.backend == 'postgres':
            query = """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """
        else:  # duckdb
            query = "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main'"

        result = backend.execute_query(query)

        # This would fail with "list index out of range" if commit happens before fetchall
        assert result is not None, "execute_query should return results for SELECT queries"
        assert len(result) > 0, "Should have at least one row"
        assert result[0][0] == 7, f"Expected 7 tables, got {result[0][0]}"

        backend.close()

    def test_execute_query_returns_multiple_rows(self, backend_config: DBConfig):
        """Test that execute_query returns all rows from multi-row SELECT."""
        backend = DatabaseBackend(backend_config)
        backend.connect()

        # Query all agent types (should return 6 rows)
        query = "SELECT type_id, type_name FROM agent_types ORDER BY type_id"
        result = backend.execute_query(query)

        assert result is not None, "Should return results"
        assert len(result) == 6, f"Expected 6 agent types, got {len(result)}"
        # Check specific types are present (alphabetically: agent, bird, env, gerbil, rat, target)
        type_ids = [row[0] for row in result]
        assert 'agent' in type_ids, "Should have 'agent' type"
        assert 'env' in type_ids, "Should have 'env' type"
        assert type_ids[0] == 'agent', "First type (alphabetically) should be 'agent'"

        backend.close()

    def test_execute_query_handles_ddl_statements(self, backend_config: DBConfig):
        """Test that execute_query handles DDL statements that don't return rows."""
        backend = DatabaseBackend(backend_config)
        backend.connect()

        # Create a temporary table
        if backend_config.backend == 'postgres':
            result = backend.execute_query("""
                CREATE TEMPORARY TABLE test_temp (id INTEGER, name VARCHAR)
            """)
        else:  # duckdb
            result = backend.execute_query("""
                CREATE TEMPORARY TABLE test_temp (id INTEGER, name VARCHAR)
            """)

        # DDL statements should return None (not an empty list)
        assert result is None or result == [], "DDL statements should not return rows"

        backend.close()

    def test_execute_query_with_aggregate_functions(self, backend_config: DBConfig):
        """Test execute_query with aggregate functions and GROUP BY."""
        backend = DatabaseBackend(backend_config)
        backend.connect()

        # Query count of property definitions by data type
        query = """
            SELECT data_type, COUNT(*) as count
            FROM property_definitions
            GROUP BY data_type
            ORDER BY data_type
        """
        result = backend.execute_query(query)

        assert result is not None, "Should return aggregate results"
        assert len(result) > 0, "Should have at least one row"
        # Verify structure: each row should have (data_type, count)
        for row in result:
            assert len(row) == 2, "Each row should have 2 columns"
            assert isinstance(row[1], int), "Count should be an integer"

        backend.close()
