#!/usr/bin/env python
"""
init_database.py
Initialize tracking_analytics database (PostgreSQL or DuckDB)
Reads configuration from environment variables or command-line args
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from collab_env.data.db.config import get_db_config, DBConfig
from collab_env.data.file_utils import get_project_root


# ANSI colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color


def print_header(msg: str):
    print(f"{Colors.GREEN}{'=' * 60}{Colors.NC}")
    print(f"{Colors.GREEN}{msg}{Colors.NC}")
    print(f"{Colors.GREEN}{'=' * 60}{Colors.NC}")


def print_info(msg: str):
    print(f"{Colors.YELLOW}[INFO]{Colors.NC} {msg}")


def print_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


class DatabaseBackend:
    """Unified database backend using SQLAlchemy"""

    def __init__(self, config: DBConfig):
        self.config = config
        self.engine: Optional[Engine] = None

    def connect(self):
        """Connect to database using SQLAlchemy"""
        try:
            url = self.config.sqlalchemy_url()
            self.engine = create_engine(url, echo=False)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.commit()

            if self.config.backend == 'postgres':
                print_success(f"Connected to PostgreSQL: {self.config.postgres.dbname}")
            else:
                print_success(f"Connected to DuckDB: {self.config.duckdb.dbpath}")
        except Exception as e:
            print_error(f"Failed to connect to database: {e}")
            raise

    def execute_file(self, filepath: Path):
        """Execute SQL file (with automatic dialect adaptation)"""
        try:
            with open(filepath, 'r') as f:
                sql_content = f.read()

            # DuckDB-specific adaptations
            if self.config.backend == 'duckdb':
                # For BIGSERIAL, we need to use a sequence
                sql_content = sql_content.replace(
                    'observation_id BIGSERIAL UNIQUE NOT NULL',
                    'observation_id BIGINT UNIQUE DEFAULT nextval(\'obs_id_seq\')'
                )
                sql_content = sql_content.replace('BIGSERIAL', 'BIGINT')
                sql_content = sql_content.replace('JSONB', 'JSON')
                sql_content = sql_content.replace('DOUBLE PRECISION', 'DOUBLE')
                # DuckDB doesn't support CASCADE in FK constraints
                sql_content = sql_content.replace(' ON DELETE CASCADE', '')
                # Remove ON CONFLICT clauses
                sql_content = re.sub(r'\s+ON CONFLICT[^;]+DO NOTHING', '', sql_content)

                # Create sequence if needed
                if 'observations' in sql_content.lower() and 'obs_id_seq' in sql_content:
                    try:
                        with self.engine.connect() as conn:
                            conn.execute(text("CREATE SEQUENCE IF NOT EXISTS obs_id_seq START 1"))
                            conn.commit()
                    except Exception:
                        pass  # Sequence may already exist

            # Execute the SQL content
            with self.engine.connect() as conn:
                conn.execute(text(sql_content))
                conn.commit()

            print_success(f"Executed {filepath.name}")
        except Exception as e:
            print_error(f"Failed to execute {filepath.name}: {e}")
            raise

    def execute_query(self, query: str):
        """Execute single query and return results (if any)"""
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            # Fetch results before commit (commit closes the result object)
            if result.returns_rows:
                rows = result.fetchall()
                conn.commit()
                return rows
            else:
                conn.commit()
                return None

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()


def get_schema_files(schema_dir: Path) -> list[Path]:
    """Get schema files in order"""
    files = [
        schema_dir / '01_core_tables.sql',
        schema_dir / '02_extended_properties.sql',
        schema_dir / '03_seed_data.sql',
    ]

    for f in files:
        if not f.exists():
            print_error(f"Schema file not found: {f}")
            sys.exit(1)

    return files


def verify_setup(backend: DatabaseBackend):
    """Verify database setup"""
    print_header("Verifying Setup")

    # Check table count
    if backend.config.backend == 'postgres':
        query = """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """
    else:  # DuckDB
        query = "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main';"

    result = backend.execute_query(query)
    table_count = result[0][0]

    if table_count == 8:
        print_success(f"All 8 tables created")
    else:
        print_error(f"Expected 8 tables, found {table_count}")
        return False

    # Check seed data
    result = backend.execute_query("SELECT count(*) FROM agent_types;")
    agent_count = result[0][0]
    if agent_count >= 5:
        print_success(f"Agent types loaded ({agent_count} rows)")
    else:
        print_error(f"Expected at least 5 agent types, found {agent_count}")
        return False

    result = backend.execute_query("SELECT count(*) FROM property_definitions;")
    prop_count = result[0][0]
    if prop_count >= 10:
        print_success(f"Property definitions loaded ({prop_count} rows)")
    else:
        print_error(f"Expected at least 10 property definitions, found {prop_count}")
        return False

    result = backend.execute_query("SELECT count(*) FROM property_categories;")
    cat_count = result[0][0]
    if cat_count == 4:
        print_success(f"Property categories loaded ({cat_count} rows)")
    else:
        print_error(f"Expected 4 property categories, found {cat_count}")
        return False

    return True


def print_summary(backend: DatabaseBackend):
    """Print summary"""
    print_header("Summary")

    if backend.config.backend == 'postgres':
        print(f"{Colors.GREEN}Database:{Colors.NC} {backend.config.postgres.dbname}")
        print(f"{Colors.GREEN}Host:{Colors.NC} {backend.config.postgres.host}:{backend.config.postgres.port}")
        print(f"{Colors.GREEN}User:{Colors.NC} {backend.config.postgres.user}")
    else:
        print(f"{Colors.GREEN}Database:{Colors.NC} {backend.config.duckdb.dbpath}")

    print()
    print_info("Tables created:")

    if backend.config.backend == 'postgres':
        result = backend.execute_query("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
    else:
        result = backend.execute_query("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name;
        """)

    for row in result:
        print(f"  - {row[0]}")

    print()
    print_success("Database initialization complete!")


def recreate_database(config: DBConfig):
    """Drop and recreate the database"""
    import os

    if config.backend == 'postgres':
        # For PostgreSQL, connect to 'postgres' database to drop/create target database
        temp_dbname = config.postgres.dbname

        # Connect to 'postgres' database
        config.postgres.dbname = 'postgres'
        temp_engine = create_engine(config.sqlalchemy_url(), isolation_level='AUTOCOMMIT')

        try:
            with temp_engine.connect() as conn:
                # Terminate existing connections to target database
                print_info(f"Terminating connections to {temp_dbname}...")
                conn.execute(text(f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{temp_dbname}'
                      AND pid <> pg_backend_pid()
                """))

                # Drop database if it exists
                print_info(f"Dropping database {temp_dbname}...")
                conn.execute(text(f"DROP DATABASE IF EXISTS {temp_dbname}"))

                # Create database
                print_success(f"Creating database {temp_dbname}...")
                conn.execute(text(f"CREATE DATABASE {temp_dbname}"))

        finally:
            temp_engine.dispose()
            # Restore original database name
            config.postgres.dbname = temp_dbname

    else:  # DuckDB
        # For DuckDB, just delete the file
        if os.path.exists(config.duckdb.dbpath):
            print_info(f"Removing existing DuckDB file: {config.duckdb.dbpath}")
            os.remove(config.duckdb.dbpath)
            # Also remove .wal file if it exists
            wal_file = config.duckdb.dbpath + '.wal'
            if os.path.exists(wal_file):
                os.remove(wal_file)
        print_success(f"Creating new DuckDB file: {config.duckdb.dbpath}")


def main():
    parser = argparse.ArgumentParser(
        description='Initialize tracking_analytics database (PostgreSQL or DuckDB)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # PostgreSQL (recreates database)
    python -m collab_env.data.init_database --backend postgres

    # DuckDB
    python -m collab_env.data.init_database --backend duckdb --dbpath tracking.duckdb

    # Custom PostgreSQL connection
    python -m collab_env.data.init_database --backend postgres --dbname mydb --user myuser --host localhost
        """
    )

    parser.add_argument('--backend', choices=['postgres', 'duckdb'], default=None,
                        help='Database backend (default: from DB_BACKEND env var or duckdb)')
    parser.add_argument('--dbname', default=None,
                        help='Database name (PostgreSQL, default: from POSTGRES_DB env or tracking_analytics)')
    parser.add_argument('--user', default=None,
                        help='Database user (PostgreSQL, default: from POSTGRES_USER env or current user)')
    parser.add_argument('--host', default=None,
                        help='Database host (PostgreSQL, default: from POSTGRES_HOST env or localhost)')
    parser.add_argument('--port', type=int, default=None,
                        help='Database port (PostgreSQL, default: from POSTGRES_PORT env or 5432)')
    parser.add_argument('--dbpath', default=None,
                        help='Database file path (DuckDB, default: from DUCKDB_PATH env or tracking.duckdb)')
    parser.add_argument('--schema-dir', type=Path, default=None,
                        help='Schema directory (default: <project_root>/schema)')
    parser.add_argument('--no-drop', action='store_true',
                        help='Do not drop existing database (only create tables, fails if tables exist)')

    args = parser.parse_args()

    # Load configuration from environment, override with command-line args
    print_info("Loading database configuration...")
    config = get_db_config(backend=args.backend)

    # Override config with command-line args if provided
    if args.backend:
        config.backend = args.backend

    if config.backend == 'postgres':
        # Override PostgreSQL settings
        if args.dbname:
            config.postgres.dbname = args.dbname
        if args.user:
            config.postgres.user = args.user
        if args.host:
            config.postgres.host = args.host
        if args.port:
            config.postgres.port = args.port
    else:
        # Override DuckDB settings
        if args.dbpath:
            config.duckdb.dbpath = args.dbpath

    # Get project root and schema directory
    project_root = get_project_root()
    schema_dir = args.schema_dir or (project_root / 'schema')

    if not schema_dir.exists():
        print_error(f"Schema directory not found: {schema_dir}")
        sys.exit(1)

    print_header("Tracking Analytics Database Initialization")
    print(f"Backend: {config.backend}")
    print(f"Configuration: {config}")
    print(f"Schema dir: {schema_dir}")
    print()

    # Get schema files
    schema_files = get_schema_files(schema_dir)
    print_info(f"Found {len(schema_files)} schema files")

    # Drop and recreate database (unless --no-drop specified)
    if not args.no_drop:
        print_header("Recreating Database")
        print_info("⚠️  This will drop the existing database and all data!")
        recreate_database(config)
        print()

    # Create unified backend with SQLAlchemy
    backend = DatabaseBackend(config)

    try:
        # Connect
        print_header("Connecting to Database")
        backend.connect()

        # Execute schema files
        print_header("Creating Schema")
        for schema_file in schema_files:
            print_info(f"Executing {schema_file.name}...")
            backend.execute_file(schema_file)

        # Verify
        if not verify_setup(backend):
            print_error("Verification failed")
            sys.exit(1)

        # Summary
        print_summary(backend)

        print()
        print_info("Next steps:")
        print("  1. Load data: python -m collab_env.data.db_loader")
        if config.backend == 'postgres':
            print(f"  2. Connect Grafana to: {config.postgres.connection_string(include_password=False)}")
            print(f"  3. Query: psql -h {config.postgres.host} -U {config.postgres.user} -d {config.postgres.dbname}")
        else:
            print(f"  2. Query: duckdb {config.duckdb.dbpath}")
            print(f"  3. Or in Python: import duckdb; conn = duckdb.connect('{config.duckdb.dbpath}')")

    except Exception as e:
        print_error(f"Initialization failed: {e}")
        sys.exit(1)
    finally:
        backend.close()


if __name__ == '__main__':
    main()
