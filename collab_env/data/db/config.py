"""
Database configuration from environment variables.
Supports PostgreSQL and DuckDB backends.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from collab_env.data.file_utils import expand_path, get_project_root


@dataclass
class PostgresConfig:
    """PostgreSQL connection configuration"""
    dbname: str
    user: str
    password: Optional[str]
    host: str
    port: int

    @classmethod
    def from_env(cls) -> 'PostgresConfig':
        """Load PostgreSQL config from environment variables"""
        return cls(
            dbname=os.getenv('POSTGRES_DB', 'tracking_analytics'),
            user=os.getenv('POSTGRES_USER', os.getenv('USER', 'postgres')),
            password=os.getenv('POSTGRES_PASSWORD'),  # None if not set
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432'))
        )

    def connection_string(self, include_password: bool = True) -> str:
        """Generate PostgreSQL connection string for SQLAlchemy"""
        if self.password and include_password:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        else:
            return f"postgresql://{self.user}@{self.host}:{self.port}/{self.dbname}"

    def psycopg2_params(self) -> dict:
        """Get parameters for psycopg2.connect() (for direct connections)"""
        params = {
            'dbname': self.dbname,
            'user': self.user,
            'host': self.host,
            'port': self.port
        }
        if self.password:
            params['password'] = self.password
        return params

    def sqlalchemy_url(self) -> str:
        """Get SQLAlchemy connection URL"""
        return self.connection_string(include_password=True)


@dataclass
class DuckDBConfig:
    """DuckDB connection configuration"""
    dbpath: str
    read_only: bool

    @classmethod
    def from_env(cls) -> 'DuckDBConfig':
        """Load DuckDB config from environment variables"""
        # Default to tracking.duckdb in project root or specified path
        default_path = os.getenv('DUCKDB_PATH', 'tracking.duckdb')

        # If relative path, make it absolute from project root
        if not os.path.isabs(default_path):
            project_root = get_project_root()
            default_path = str(project_root / default_path)

        return cls(
            dbpath=default_path,
            read_only=os.getenv('DUCKDB_READ_ONLY', 'false').lower() == 'true'
        )

    def connection_string(self) -> str:
        """Generate DuckDB connection string for SQLAlchemy"""
        return f"duckdb:///{self.dbpath}"

    def sqlalchemy_url(self) -> str:
        """Get SQLAlchemy connection URL"""
        return self.connection_string()


class DBConfig:
    """
    Main database configuration.
    Automatically detects backend from DB_BACKEND env var.
    """

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize database configuration.

        Parameters
        ----------
        backend : str, optional
            Database backend ('postgres' or 'duckdb').
            If None, reads from DB_BACKEND env var (defaults to 'duckdb')
        """
        self.backend = backend or os.getenv('DB_BACKEND', 'duckdb')

        if self.backend not in ('postgres', 'duckdb'):
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'postgres' or 'duckdb'")

        if self.backend == 'postgres':
            self.postgres = PostgresConfig.from_env()
            self.duckdb = None
        else:
            self.duckdb = DuckDBConfig.from_env()
            self.postgres = None

    @property
    def connection_string(self) -> str:
        """Get connection string for current backend"""
        if self.backend == 'postgres':
            return self.postgres.connection_string()
        else:
            return self.duckdb.connection_string()

    def sqlalchemy_url(self) -> str:
        """Get SQLAlchemy connection URL for current backend"""
        if self.backend == 'postgres':
            return self.postgres.sqlalchemy_url()
        else:
            return self.duckdb.sqlalchemy_url()

    def __repr__(self) -> str:
        if self.backend == 'postgres':
            return f"DBConfig(backend='postgres', dbname='{self.postgres.dbname}', host='{self.postgres.host}')"
        else:
            return f"DBConfig(backend='duckdb', dbpath='{self.duckdb.dbpath}')"


def load_dotenv_if_exists():
    """
    Load .env file if python-dotenv is available.
    This is optional - falls back to OS environment variables.
    """
    try:
        from dotenv import load_dotenv
        project_root = get_project_root()
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            return True
    except ImportError:
        pass
    return False


# Convenience function
def get_db_config(backend: Optional[str] = None) -> DBConfig:
    """
    Get database configuration with optional .env file loading.

    Parameters
    ----------
    backend : str, optional
        Override backend (otherwise uses DB_BACKEND env var)

    Returns
    -------
    DBConfig
        Database configuration

    Examples
    --------
    >>> config = get_db_config()
    >>> print(config.backend)
    'duckdb'

    >>> config = get_db_config('postgres')
    >>> print(config.postgres.dbname)
    'tracking_analytics'
    """
    load_dotenv_if_exists()
    return DBConfig(backend)


if __name__ == '__main__':
    # Test configuration loading
    import sys

    print("="*60)
    print("Database Configuration Test")
    print("="*60)

    # Try loading .env
    if load_dotenv_if_exists():
        print("✓ Loaded .env file")
    else:
        print("✗ No .env file found (using OS environment)")

    print()

    # Test both backends
    for backend in ['duckdb', 'postgres']:
        print(f"\n{backend.upper()} Configuration:")
        print("-"*60)

        try:
            config = DBConfig(backend)
            print(f"Backend: {config.backend}")
            print(f"Connection string: {config.connection_string}")

            if backend == 'postgres':
                print(f"Database: {config.postgres.dbname}")
                print(f"User: {config.postgres.user}")
                print(f"Host: {config.postgres.host}:{config.postgres.port}")
                print(f"Password set: {config.postgres.password is not None}")
            else:
                print(f"DB Path: {config.duckdb.dbpath}")
                print(f"Read-only: {config.duckdb.read_only}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*60)
