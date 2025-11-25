"""
Query backend for spatial analysis of tracking data.

Provides a high-level API for executing spatial analysis queries
on the tracking analytics database. Uses aiosql to manage SQL queries
in separate .sql files with driver-specific adapters.
"""

import time
from pathlib import Path
from typing import Optional

import aiosql
import pandas as pd
from loguru import logger

from collab_env.data.db.config import DBConfig, get_db_config
from collab_env.data.db.db_loader import DatabaseConnection


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
        logger.info("Closing database connection...")
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
        # Log query execution start
        start_time = time.time()
        logger.info(f"Executing query '{query_name}' with params: {params}")

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
                        result = pd.DataFrame(rows, columns=col_names)
                    else:
                        # Return empty DataFrame with column names
                        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                        result = pd.DataFrame(columns=col_names)

                    # Log successful query completion with execution time
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"Query '{query_name}' completed in {elapsed_time:.3f}s: "
                        f"{len(result)} rows returned"
                    )
                    return result
            finally:
                raw_conn.close()

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Query '{query_name}' execution failed after {elapsed_time:.3f}s: {e}")
            raise

    # ==================== Session/Episode Metadata ====================

    def get_categories(self) -> pd.DataFrame:
        """
        Get list of all categories.

        Returns
        -------
        pd.DataFrame
            Categories with columns: category_id, category_name, description
        """
        return self._execute_query('get_categories')

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

    def get_agent_types_for_session(self, session_id: str) -> pd.DataFrame:
        """
        Get distinct agent types across all episodes in a session.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        pd.DataFrame
            Agent types with column: agent_type_id
        """
        return self._execute_query('get_agent_types_for_session', session_id=session_id)

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

        PERFORMANCE NOTE: This method uses separate optimized queries for episode
        and session scopes to avoid query planner issues with NULL checks.

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

        # Use scope-specific query for optimal performance
        if episode_id is not None:
            return self._execute_query(
                'get_spatial_heatmap_episode',
                episode_id=episode_id,
                bin_size=bin_size,
                start_time=start_time,
                end_time=end_time,
                agent_type=agent_type,
                min_count=min_count
            )
        else:
            return self._execute_query(
                'get_spatial_heatmap_session',
                session_id=session_id,
                bin_size=bin_size,
                start_time=start_time,
                end_time=end_time,
                agent_type=agent_type,
                min_count=min_count
            )

    # ==================== Basic Data Viewer ====================

    def get_episode_tracks(
        self,
        episode_id: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        **kwargs
    ) -> pd.DataFrame:
        """
        Get position and velocity data for animation.

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
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Tracks with columns: agent_id, time_index, x, y, z, v_x, v_y, v_z, speed
        """
        return self._execute_query(
            'get_episode_tracks',
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type
        )

    def get_extended_properties_timeseries(
        self,
        episode_id: str,
        window_size: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        property_ids: Optional[list] = None,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get aggregated time series for extended properties.

        This is a property-agnostic query that returns windowed statistics
        for any extended properties. Filter by property_ids in Python after
        retrieval if needed.

        Parameters
        ----------
        episode_id : str
            Episode to analyze
        window_size : int, default=100
            Number of frames per window
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        property_ids : list, optional
            List of property IDs to filter (applied in Python after query)
        lower_quantile : float, default=0.10
            Lower quantile for uncertainty band (e.g., 0.10 for 10th percentile)
        upper_quantile : float, default=0.90
            Upper quantile for uncertainty band (e.g., 0.90 for 90th percentile)
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Time series with columns: time_window, property_id, n_observations,
            avg_value, std_value, min_value, max_value, median_value, q_lower, q_upper
            Filtered to property_ids if provided.
        """
        df = self._execute_query(
            'get_extended_properties_timeseries',
            episode_id=episode_id,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile
        )

        # Filter by property_ids if provided
        if property_ids is not None and len(df) > 0:
            df = df[df['property_id'].isin(property_ids)]

        return df

    def get_property_distributions(
        self,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        property_ids: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get raw property values for histogram generation.

        This is a property-agnostic query that returns individual property
        values for distribution analysis. Filter by property_ids in Python
        after retrieval if needed.

        Supports both episode-level and session-level analysis.

        PERFORMANCE NOTE: This method uses separate optimized queries for episode
        and session scopes to avoid query planner issues with NULL checks that
        caused 2x performance degradation.

        Parameters
        ----------
        episode_id : str, optional
            Episode to analyze (mutually exclusive with session_id)
        session_id : str, optional
            Session to analyze - aggregates all episodes in session (mutually exclusive with episode_id)
        start_time : int, optional
            Start time index
        end_time : int, optional
            End time index
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        property_ids : list, optional
            List of property IDs to filter (applied in Python after query)
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Property values with columns: property_id, value_float
            Filtered to property_ids if provided.
        """
        if episode_id is None and session_id is None:
            raise ValueError("Either episode_id or session_id must be provided")
        if episode_id is not None and session_id is not None:
            raise ValueError("Cannot specify both episode_id and session_id")

        # Use scope-specific query for optimal performance
        if episode_id is not None:
            df = self._execute_query(
                'get_property_distributions_episode',
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                agent_type=agent_type
            )
        else:
            df = self._execute_query(
                'get_property_distributions_session',
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                agent_type=agent_type
            )

        # Filter by property_ids if provided
        if property_ids is not None and len(df) > 0:
            df = df[df['property_id'].isin(property_ids)]

        return df

    def get_available_properties(
        self,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_type: str = 'agent',
        **kwargs
    ) -> pd.DataFrame:
        """
        Get list of available extended properties for an episode or session.

        Supports both episode-level and session-level analysis.

        PERFORMANCE NOTE: This method uses separate optimized queries for episode
        and session scopes to avoid query planner issues with NULL checks and OR
        conditions that caused 3x performance degradation.

        Parameters
        ----------
        episode_id : str, optional
            Episode to analyze (mutually exclusive with session_id)
        session_id : str, optional
            Session to analyze - aggregates all episodes in session (mutually exclusive with episode_id)
        agent_type : str, default='agent'
            Agent type to filter ('agent', 'target', 'all')
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Available properties with columns: property_id, property_name,
            description, unit, data_type
        """
        if episode_id is None and session_id is None:
            raise ValueError("Either episode_id or session_id must be provided")
        if episode_id is not None and session_id is not None:
            raise ValueError("Cannot specify both episode_id and session_id")

        # Use scope-specific query for optimal performance
        if episode_id is not None:
            return self._execute_query(
                'get_available_properties_episode',
                episode_id=episode_id,
                agent_type=agent_type
            )
        else:
            return self._execute_query(
                'get_available_properties_session',
                session_id=session_id,
                agent_type=agent_type
            )

    def get_extended_properties_raw(
        self,
        episode_id: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent_type: str = 'agent',
        property_ids: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get raw (unaggregated) extended property values with time indices.

        Returns individual observations for plotting agent trajectories
        as lines on time series plots. Data is in long format with
        property_id as a column for flexibility.

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
        property_ids : list, optional
            List of property IDs to filter (applied in Python after query)
        **kwargs
            Additional parameters (ignored)

        Returns
        -------
        pd.DataFrame
            Raw property values with columns: time_index, agent_id, property_id, value_float
            Filtered to property_ids if provided.
        """
        df = self._execute_query(
            'get_extended_properties_raw',
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            agent_type=agent_type
        )

        # Filter by property_ids if provided
        if property_ids is not None and len(df) > 0:
            df = df[df['property_id'].isin(property_ids)]

        return df

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
