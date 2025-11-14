"""
Analysis context for sharing state across widgets.

Provides shared query backend, scope, parameters, and status callbacks.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

from collab_env.data.db.query_backend import QueryBackend
from .query_scope import QueryScope


@dataclass
class AnalysisContext:
    """
    Shared context passed to all analysis widgets.

    Provides access to:
    - Query backend for data access
    - Query scope (what data to analyze)
    - Shared analysis parameters (bin sizes, thresholds, etc.)
    - Status callback hooks for UI updates

    Widgets access this context to query data and report status.

    Examples
    --------
    >>> context = AnalysisContext(
    ...     query_backend=backend,
    ...     scope=QueryScope.from_episode("ep_123"),
    ...     spatial_bin_size=10.0,
    ...     on_loading=lambda msg: print(f"Loading: {msg}")
    ... )
    >>> params = context.get_query_params()
    >>> df = context.query_backend.get_spatial_heatmap(**params)
    """

    # Core components
    query_backend: QueryBackend
    scope: QueryScope

    # Shared analysis parameters (commonly used across widgets)
    spatial_bin_size: float = 10.0      # For spatial discretization
    temporal_window_size: int = 10     # For time-windowed analyses

    # Additional shared parameters
    min_samples: int = 100              # Minimum samples for statistics (matches QueryBackend default)
    confidence_level: float = 0.95      # For confidence intervals

    # Callback hooks for status updates (optional)
    on_loading: Optional[Callable[[str], None]] = None
    on_success: Optional[Callable[[str], None]] = None
    on_error: Optional[Callable[[str], None]] = None

    def get_query_params(self, **extra_params) -> Dict[str, Any]:
        """
        Get combined query parameters from scope + shared params + extras.

        Merges:
        1. Scope parameters (episode_id, time range, agent filters)
        2. Shared analysis parameters (bin sizes, thresholds)
        3. Widget-specific overrides (passed as kwargs)

        Parameters
        ----------
        **extra_params
            Additional widget-specific parameters to merge

        Returns
        -------
        dict
            Complete set of query parameters

        Examples
        --------
        >>> # Widget uses shared params
        >>> params = context.get_query_params()
        >>> df = backend.get_spatial_heatmap(**params)

        >>> # Widget overrides specific param
        >>> params = context.get_query_params(bin_size=5.0)
        >>> df = backend.get_spatial_heatmap(**params)
        """
        # Start with scope parameters
        params = self.scope.to_query_params()

        # Add shared analysis parameters
        params['bin_size'] = self.spatial_bin_size
        params['window_size'] = self.temporal_window_size
        params['min_samples'] = self.min_samples

        # Widget-specific overrides
        params.update(extra_params)

        return params

    def report_loading(self, message: str):
        """Report loading status via callback."""
        if self.on_loading:
            self.on_loading(message)

    def report_success(self, message: str):
        """Report success status via callback."""
        if self.on_success:
            self.on_success(message)

    def report_error(self, message: str):
        """Report error status via callback."""
        if self.on_error:
            self.on_error(message)
