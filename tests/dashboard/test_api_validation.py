"""
API Validation Tests for SQL-Level Session Scope Implementation

Tests that QueryBackend methods correctly accept both episode_id and session_id
parameters, handle extra parameters gracefully, and validate inputs properly.
"""

import pytest
from unittest.mock import Mock
import pandas as pd

from collab_env.data.db.query_backend import QueryBackend
from collab_env.dashboard.widgets import QueryScope, AnalysisContext


@pytest.fixture
def mock_backend():
    """Create a QueryBackend with mocked _execute_query method."""
    backend = QueryBackend.__new__(QueryBackend)
    backend._execute_query = Mock(return_value=pd.DataFrame())
    return backend


@pytest.fixture
def shared_params():
    """Shared parameters that AnalysisContext.get_query_params() produces."""
    return {
        'episode_id': 'ep_123',
        'bin_size': 10.0,
        'window_size': 100,
        'min_samples': 100,
        'agent_type': 'agent',
        'start_time': 0,
        'end_time': 500
    }


# List of QueryBackend analysis methods supporting both episode_id and session_id
SESSION_SUPPORTED_METHODS = [
    'get_spatial_heatmap',
]

# List of correlation methods supporting only episode_id (session-level disabled)
EPISODE_ONLY_METHODS = [
    'get_velocity_correlations',
    'get_distance_correlations'
]

# All analysis methods combined
ANALYSIS_METHODS = SESSION_SUPPORTED_METHODS + EPISODE_ONLY_METHODS


class TestQueryBackendMethodSignatures:
    """Test that QueryBackend methods accept both episode_id and session_id."""

    @pytest.mark.parametrize("method_name", ANALYSIS_METHODS)
    def test_accepts_episode_id(self, mock_backend, method_name):
        """Test that all methods accept episode_id parameter."""
        method = getattr(mock_backend, method_name)

        # Should not raise
        if 'heatmap' in method_name:
            method(episode_id="ep_123", bin_size=10.0)
        elif 'correlation' in method_name:
            method(episode_id="ep_123", min_samples=100)
        else:
            method(episode_id="ep_123", window_size=100)

    @pytest.mark.parametrize("method_name", SESSION_SUPPORTED_METHODS)
    def test_accepts_session_id(self, mock_backend, method_name):
        """Test that session-supported methods accept session_id parameter."""
        method = getattr(mock_backend, method_name)

        # Should not raise
        if 'heatmap' in method_name:
            method(session_id="sess_456", bin_size=10.0)
        else:
            method(session_id="sess_456", window_size=100)


class TestExtraParameters:
    """Test that methods accept extra parameters without error."""

    def test_spatial_heatmap_accepts_extra_params(self, mock_backend, shared_params):
        """Spatial heatmap should accept extra params like window_size, min_samples."""
        # Should not raise TypeError
        mock_backend.get_spatial_heatmap(**shared_params)

    def test_velocity_correlations_accepts_extra_params(self, mock_backend, shared_params):
        """Velocity correlations should accept extra params like bin_size, window_size."""
        # Should not raise TypeError
        mock_backend.get_velocity_correlations(**shared_params)

    def test_distance_correlations_accepts_extra_params(self, mock_backend, shared_params):
        """Distance correlations should accept extra params like bin_size, window_size."""
        # Should not raise TypeError
        mock_backend.get_distance_correlations(**shared_params)


class TestParameterValidation:
    """Test that methods validate episode_id/session_id correctly."""

    @pytest.mark.parametrize("method_name", SESSION_SUPPORTED_METHODS)
    def test_rejects_missing_both_parameters(self, mock_backend, method_name):
        """Session-supported methods should raise ValueError when both episode_id and session_id are missing."""
        method = getattr(mock_backend, method_name)

        with pytest.raises(ValueError, match="Either episode_id or session_id must be provided"):
            method()

    @pytest.mark.parametrize("method_name", EPISODE_ONLY_METHODS)
    def test_correlation_methods_require_episode_id(self, mock_backend, method_name):
        """Correlation methods should raise TypeError when episode_id is missing (required parameter)."""
        method = getattr(mock_backend, method_name)

        with pytest.raises(TypeError):
            method()

    @pytest.mark.parametrize("method_name", SESSION_SUPPORTED_METHODS)
    def test_rejects_both_parameters(self, mock_backend, method_name):
        """Session-supported methods should raise ValueError when both episode_id and session_id are provided."""
        method = getattr(mock_backend, method_name)

        with pytest.raises(ValueError, match="Cannot specify both episode_id and session_id"):
            method(episode_id="ep_123", session_id="sess_456")


class TestAnalysisContextIntegration:
    """Test that AnalysisContext.get_query_params() produces compatible output."""

    def test_episode_scope_params(self, mock_backend):
        """Test that episode scope parameters work with all methods."""
        scope = QueryScope.from_episode(episode_id="ep_123", start_time=0, end_time=500)
        context = AnalysisContext(
            query_backend=mock_backend,
            scope=scope,
            spatial_bin_size=10.0,
            temporal_window_size=100,
            min_samples=100
        )

        params = context.get_query_params()

        # Should have episode_id
        assert 'episode_id' in params
        assert params['episode_id'] == "ep_123"

        # Should not have session_id
        assert 'session_id' not in params or params['session_id'] is None

        # All methods should accept these params
        mock_backend.get_spatial_heatmap(**params)
        mock_backend.get_velocity_correlations(**params)

    def test_session_scope_params(self, mock_backend):
        """Test that session scope parameters work with session-supported methods."""
        scope = QueryScope.from_session(session_id="sess_456")
        context = AnalysisContext(
            query_backend=mock_backend,
            scope=scope,
            spatial_bin_size=10.0,
            temporal_window_size=100,
            min_samples=100
        )

        params = context.get_query_params()

        # Should have session_id
        assert 'session_id' in params
        assert params['session_id'] == "sess_456"

        # Should not have episode_id
        assert 'episode_id' not in params or params['episode_id'] is None

        # Session-supported methods should accept these params
        mock_backend.get_spatial_heatmap(**params)
        # Note: Correlation methods do NOT support session scope

    def test_shared_parameters_included(self, mock_backend):
        """Test that shared parameters are included in query params."""
        scope = QueryScope.from_episode(episode_id="ep_123")
        context = AnalysisContext(
            query_backend=mock_backend,
            scope=scope,
            spatial_bin_size=15.0,
            temporal_window_size=200,
            min_samples=50
        )

        params = context.get_query_params()

        # Check shared parameters are present
        assert params['bin_size'] == 15.0
        assert params['window_size'] == 200
        assert params['min_samples'] == 50
        assert 'agent_type' in params
