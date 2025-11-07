"""
Query backend for spatial analysis of tracking data.

Provides a high-level API for executing spatial analysis queries
on the tracking analytics database. Uses aiosql to manage SQL queries
in separate .sql files with driver-specific adapters.
"""

import logging
from pathlib import Path
from typing import Optional

import aiosql
import pandas as pd

from collab_env.data.db.config import DBConfig, get_db_config
from collab_env.data.db.db_loader import DatabaseConnection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryBackend:
    """
    Query interface for spatial analysis.

    Uses aiosql to load SQL queries from .sql files and execute them
    against the database using driver-specific adapters (psycopg2 for
    PostgreSQL, duckdb for DuckDB). All methods return pandas DataFrames.

    Examples
    --------
    >>> from collab_env.data.db.query_backend import QueryBackend
    >>>
    >>> # Initialize with default config (reads from environment)
    >>> query = QueryBackend()
    >>>
    >>> # Get sessions
    >>> sessions = query.get_sessions(category_id='boids_3d')
    >>> print(sessions)
    >>>
    >>> # Get episodes for first session
    >>> session_id = sessions.iloc[0]['session_id']
    >>> episodes = query.get_episodes(session_id)
    >>>
    >>> # Get spatial heatmap for first episode
    >>> episode_id = episodes.iloc[0]['episode_id']
    >>> heatmap = query.get_spatial_heatmap(episode_id, bin_size=10.0)
    >>>
    >>> # Close connection
    >>> query.close()
    """

    def __init__(self, config: Optional[DBConfig] = None, backend: Optional[str] = None):
        """
        Initialize QueryBackend.

        Parameters
        ----------
        config : DBConfig, optional
            Database configuration. If None, uses get_db_config()
        backend : str, optional
            Database backend ('postgres' or 'duckdb').
            If provided, overrides config.
        """
        if backend:
            self.config = DBConfig(backend=backend)
        else:
            self.config = config or get_db_config()

        # Initialize database connection
        self.db = DatabaseConnection(self.config)
        self.db.connect()

        # Load SQL queries using aiosql with driver-specific adapter
        queries_dir = Path(__file__).parent / "queries"
        if not queries_dir.exists():
            raise FileNotFoundError(f"Queries directory not found: {queries_dir}")

        # Choose adapter based on backend
        if self.config.backend == 'postgres':
            driver_adapter = "psycopg2"
        else:
            driver_adapter = "duckdb"

        self.queries = aiosql.from_path(str(queries_dir), driver_adapter)
        logger.info(f"Loaded queries from {queries_dir} using {driver_adapter} adapter")

    def close(self):
        """Close database connection."""
        self.db.close()

    def _execute_query(self, query_name: str, **params) -> pd.DataFrame:
        """
        Execute a query and return results as pandas DataFrame.

        Parameters
        ----------
        query_name : str
            Name of the aiosql query to execute
        **params
            Query parameters

        Returns
        -------
        pd.DataFrame
            Query results
        """
        try:
            # Get the cursor variant of the query from aiosql
            # aiosql provides both query_name() and query_name_cursor() variants
            cursor_func = getattr(self.queries, f"{query_name}_cursor")

            # Get raw database connection
            # aiosql works directly with psycopg2/duckdb connections
            raw_conn = self.db.engine.raw_connection()
            try:
                # Execute query using aiosql cursor method
                # The _cursor variant returns a context manager
                with cursor_func(raw_conn, **params) as cursor:
                    # Fetch all rows
                    rows = cursor.fetchall()

                    # Get column names from cursor description
                    if rows:
                        col_names = [desc[0] for desc in cursor.description]
                        return pd.DataFrame(rows, columns=col_names)
                    else:
                        # Return empty DataFrame with column names
                        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                        return pd.DataFrame(columns=col_names)
            finally:
                raw_conn.close()

        except Exception as e:
            logger.error(f"Query '{query_name}' execution failed: {e}")
            raise

    # ==================== Session/Episode Metadata ====================

    def get_sessions(self, category_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get list of all sessions, optionally filtered by category.

        Parameters
        ----------
        category_id : str, optional
            Filter by category ('boids_3d', 'boids_2d', 'tracking_csv')

        Returns
        -------
        pd.DataFrame
            Sessions with columns: session_id, session_name, category_id, created_at, config
        """
        return self._execute_query('get_sessions', category_id=category_id)

    def get_episodes(self, session_id: str) -> pd.DataFrame:
        """
        Get all episodes for a given session.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        pd.DataFrame
            Episodes with columns: episode_id, episode_number, num_frames,
            num_agents, frame_rate, file_path
        """
        return self._execute_query('get_episodes', session_id=session_id)

    def get_episode_metadata(self, episode_id: str) -> pd.DataFrame:
        """
        Get detailed metadata for a single episode.

        Parameters
        ----------
        episode_id : str
            Episode identifier

        Returns
        -------
        pd.DataFrame
            Episode metadata with columns: episode_id, session_id, episode_number,
            num_frames, num_agents, frame_rate, file_path, session_name,
            category_id, config
        """
        return self._execute_query('get_episode_metadata', episode_id=episode_id)

    def get_agent_types(self, episode_id: str) -> pd.DataFrame:
        """
        Get distinct agent types for an episode.

        Parameters
        ----------
        episode_id : str
            Episode identifier

        Returns
        -------
        pd.DataFrame
            Agent types with column: agent_type_id
        """
        return self._execute_query('get_agent_types', episode_id=episode_id)

    # ==================== Spatial Analysis ====================

    def get_spatial_heatmap(
        self,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        bin_size: float = 10.0,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        min_count: int = 1,
        **kwargs  # Accept extra parameters (e.g., window_size, min_samples from shared context)
    ) -> pd.DataFrame:
        """
        Compute spatial density heatmap with binned positions.

        Parameters
        ----------
        episode_id : str, optional
            Episode to analyze (mutually exclusive with session_id)
        session_id : str, optional
            Session to analyze - aggregates all episodes in session (mutually exclusive with episode_id)
        bin_size : float, default=10.0
            Spatial bin size in scene units
        start_time : int, optional
            Start time index (None = from beginning)
        end_time : int, optional
            End time index (None = to end)
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        min_count : int, default=1
            Minimum observations per bin to include
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Heatmap with columns: x_bin, y_bin, z_bin, density, avg_vx, avg_vy, avg_vz
        """
        if episode_id is None and session_id is None:
            raise ValueError("Either episode_id or session_id must be provided")
        if episode_id is not None and session_id is not None:
            raise ValueError("Cannot specify both episode_id and session_id")

        return self._execute_query(
            'get_spatial_heatmap',
            episode_id=episode_id,
            session_id=session_id,
            bin_size=bin_size,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type,
            min_count=min_count
        )

    def get_velocity_heatmap(
        self,
        episode_id: str,
        bin_size: float = 10.0,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        min_count: int = 5
    ) -> pd.DataFrame:
        """
        Compute velocity field on spatial grid (for quiver plots).

        Parameters
        ----------
        episode_id : str
            Episode to analyze
        bin_size : float, default=10.0
            Spatial bin size in scene units
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        min_count : int, default=5
            Minimum observations per bin to include

        Returns
        -------
        pd.DataFrame
            Velocity field with columns: x_bin, y_bin, count, avg_vx, avg_vy,
            avg_vz, avg_speed
        """
        return self._execute_query(
            'get_velocity_heatmap',
            episode_id=episode_id,
            bin_size=bin_size,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type,
            min_count=min_count
        )

    def get_velocity_distribution(
        self,
        episode_id: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent'
    ) -> pd.DataFrame:
        """
        Get raw velocity vectors for distribution analysis.

        Parameters
        ----------
        episode_id : str
            Episode to analyze
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')

        Returns
        -------
        pd.DataFrame
            Velocities with columns: agent_id, time_index, v_x, v_y, v_z, speed
        """
        return self._execute_query(
            'get_velocity_distribution',
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type
        )

    def get_speed_statistics(
        self,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        window_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        **kwargs  # Accept extra parameters (e.g., bin_size, min_samples from shared context)
    ) -> pd.DataFrame:
        """
        Compute speed statistics over time windows.

        Parameters
        ----------
        episode_id : str, optional
            Episode to analyze (mutually exclusive with session_id)
        session_id : str, optional
            Session to analyze - aggregates all episodes in session (mutually exclusive with episode_id)
        window_size : int, default=100
            Number of frames per window
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Speed statistics with columns: time_window, n_observations,
            avg_speed, std_speed, min_speed, max_speed, median_speed
        """
        if episode_id is None and session_id is None:
            raise ValueError("Either episode_id or session_id must be provided")
        if episode_id is not None and session_id is not None:
            raise ValueError("Cannot specify both episode_id and session_id")

        return self._execute_query(
            'get_speed_statistics',
            episode_id=episode_id,
            session_id=session_id,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type
        )

    def get_distance_to_target(
        self,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        window_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        **kwargs  # Accept extra parameters (e.g., bin_size, min_samples from shared context)
    ) -> pd.DataFrame:
        """
        Compute distance to target statistics over time windows.

        Parameters
        ----------
        episode_id : str, optional
            Episode to analyze (mutually exclusive with session_id)
        session_id : str, optional
            Session to analyze - aggregates all episodes in session (mutually exclusive with episode_id)
        window_size : int, default=100
            Number of frames per window
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Distance statistics with columns: time_window, n_observations,
            avg_distance, std_distance, min_distance, max_distance
        """
        if episode_id is None and session_id is None:
            raise ValueError("Either episode_id or session_id must be provided")
        if episode_id is not None and session_id is not None:
            raise ValueError("Cannot specify both episode_id and session_id")

        return self._execute_query(
            'get_distance_to_target',
            episode_id=episode_id,
            session_id=session_id,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type
        )

    def get_distance_to_boundary(
        self,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        window_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        **kwargs  # Accept extra parameters (e.g., bin_size, min_samples from shared context)
    ) -> pd.DataFrame:
        """
        Compute distance to scene boundary statistics over time windows.

        Parameters
        ----------
        episode_id : str, optional
            Episode to analyze (mutually exclusive with session_id)
        session_id : str, optional
            Session to analyze - aggregates all episodes in session (mutually exclusive with episode_id)
        window_size : int, default=100
            Number of frames per window
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Distance statistics with columns: time_window, n_observations,
            avg_distance, std_distance, min_distance, max_distance
        """
        if episode_id is None and session_id is None:
            raise ValueError("Either episode_id or session_id must be provided")
        if episode_id is not None and session_id is not None:
            raise ValueError("Cannot specify both episode_id and session_id")

        return self._execute_query(
            'get_distance_to_boundary',
            episode_id=episode_id,
            session_id=session_id,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type
        )

    # ==================== Correlations ====================

    def get_velocity_correlations(
        self,
        episode_id: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        min_samples: int = 100,
        agent_type: str = 'agent',
        **kwargs  # Accept extra parameters (e.g., bin_size, window_size from shared context)
    ) -> pd.DataFrame:
        """
        Compute pairwise velocity correlations between agents.

        Warning: O(n²) computation, can be slow for many agents.
        Note: Only supports episode-level analysis. Session-level correlation is disabled.

        Parameters
        ----------
        episode_id : str
            Episode to analyze
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        min_samples : int, default=100
            Minimum number of time points required for correlation
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Correlations with columns: agent_i, agent_j, v_x_correlation,
            v_y_correlation, v_z_correlation, n_samples
        """
        return self._execute_query(
            'get_velocity_correlations',
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            min_samples=min_samples,
            agent_type=agent_type
        )

    def get_distance_correlations(
        self,
        episode_id: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        min_samples: int = 100,
        **kwargs  # Accept extra parameters (e.g., bin_size, window_size from shared context)
    ) -> pd.DataFrame:
        """
        Compute pairwise distance-to-target correlations between agents.

        Warning: O(n²) computation, can be slow for many agents.
        Note: Only supports episode-level analysis. Session-level correlation is disabled.

        Parameters
        ----------
        episode_id : str
            Episode to analyze
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        min_samples : int, default=100
            Minimum number of time points required for correlation
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Correlations with columns: agent_i, agent_j, distance_correlation,
            n_samples
        """
        return self._execute_query(
            'get_distance_correlations',
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            min_samples=min_samples
        )


# Convenience function for quick testing
def main():
    """
    Test query backend with basic queries.
    """
    import sys

    print("="*60)
    print("Query Backend Test")
    print("="*60)

    try:
        # Initialize
        query = QueryBackend()
        print(f"✓ Connected to {query.config.backend}")
        print()

        # Get sessions
        print("Sessions:")
        sessions = query.get_sessions(category_id='boids_3d')
        print(f"  Found {len(sessions)} sessions")
        if len(sessions) > 0:
            print(f"  First session: {sessions.iloc[0]['session_name']}")
        print()

        # Get episodes
        if len(sessions) > 0:
            session_id = sessions.iloc[0]['session_id']
            print(f"Episodes for session {session_id}:")
            episodes = query.get_episodes(session_id)
            print(f"  Found {len(episodes)} episodes")

            # Test spatial heatmap
            if len(episodes) > 0:
                episode_id = episodes.iloc[0]['episode_id']
                print(f"\nSpatial heatmap for episode {episode_id}:")
                heatmap = query.get_spatial_heatmap(episode_id, bin_size=20.0)
                print(f"  Generated {len(heatmap)} bins")
                print(f"  Density range: {heatmap['density'].min():.0f} - {heatmap['density'].max():.0f}")

                # Display heatmap data
                print("\n  Heatmap sample (top 10 densest bins):")
                top_bins = heatmap.nlargest(10, 'density')[['x_bin', 'y_bin', 'z_bin', 'density', 'avg_vx', 'avg_vy', 'avg_vz']]
                print(top_bins.to_string(index=False))

                # Create and show 3D scatter plot
                print("\n  Creating 3D scatter plot...")
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D

                    # Create 3D plot
                    fig = plt.figure(figsize=(14, 10))
                    ax = fig.add_subplot(111, projection='3d')

                    # Create scatter plot with density as color and size
                    scatter = ax.scatter(
                        heatmap['x_bin'],
                        heatmap['y_bin'],
                        heatmap['z_bin'],
                        c=heatmap['density'],
                        s=heatmap['density'] / heatmap['density'].max() * 100,  # Scale size by density
                        cmap='viridis',
                        alpha=0.6,
                        edgecolors='w',
                        linewidth=0.5
                    )

                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
                    cbar.set_label('Density', rotation=270, labelpad=15)

                    # Labels and title
                    ax.set_xlabel('X Position (bins)')
                    ax.set_ylabel('Y Position (bins)')
                    ax.set_zlabel('Z Position (bins)')
                    ax.set_title(f'3D Spatial Density Heatmap\nEpisode: {episode_id}')

                    # Set viewing angle
                    ax.view_init(elev=20, azim=45)

                    plt.tight_layout()

                    # Save plot to file
                    output_file = '/tmp/spatial_heatmap_3d.png'
                    plt.savefig(output_file, dpi=150, bbox_inches='tight')
                    print(f"  ✓ 3D scatter plot saved to {output_file}")

                    # Show interactively if available
                    plt.show()
                except ImportError:
                    print("  ✗ matplotlib not available, skipping plot")
                except Exception as e:
                    print(f"  ✗ Error creating plot: {e}")

        # Close
        query.close()
        print("\n✓ All tests passed")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
