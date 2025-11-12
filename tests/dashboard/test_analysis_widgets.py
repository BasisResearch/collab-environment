"""
Tests for widget refactoring.

Tests the modular widget system without database connection.
"""

import pytest
from pathlib import Path

from collab_env.dashboard.widgets import (
    QueryScope,
    ScopeType,
    AnalysisContext,
    BaseAnalysisWidget,
    WidgetRegistry
)


class TestImports:
    """Test that all modules import correctly."""

    def test_core_modules_import(self):
        """Test that core widget modules can be imported."""
        from collab_env.dashboard.widgets import (
            QueryScope,
            ScopeType,
            AnalysisContext,
            BaseAnalysisWidget,
            WidgetRegistry
        )
        # If we get here, imports succeeded
        assert QueryScope is not None
        assert ScopeType is not None
        assert AnalysisContext is not None
        assert BaseAnalysisWidget is not None
        assert WidgetRegistry is not None

    def test_widget_modules_import(self):
        """Test that all widget modules can be imported."""
        from collab_env.dashboard.widgets.velocity_widget import VelocityStatsWidget
        from collab_env.dashboard.widgets.distance_widget import DistanceStatsWidget
        from collab_env.dashboard.widgets.correlation_widget import CorrelationWidget

        assert VelocityStatsWidget is not None
        assert DistanceStatsWidget is not None
        assert CorrelationWidget is not None


class TestQueryScope:
    """Test QueryScope creation and parameter extraction."""

    def test_episode_scope_creation(self):
        """Test creating an episode scope."""
        scope = QueryScope.from_episode(
            episode_id="ep_123",
            start_time=0,
            end_time=500,
            agent_type="agent"
        )

        assert scope.scope_type == ScopeType.EPISODE
        assert scope.episode_id == "ep_123"
        assert scope.start_time == 0
        assert scope.end_time == 500

    def test_episode_scope_to_query_params(self):
        """Test converting episode scope to query parameters."""
        scope = QueryScope.from_episode(
            episode_id="ep_123",
            start_time=0,
            end_time=500,
            agent_type="agent"
        )

        params = scope.to_query_params()

        assert params['episode_id'] == "ep_123"
        assert params['start_time'] == 0
        assert params['end_time'] == 500
        assert params['agent_type'] == "agent"

    def test_session_scope_creation(self):
        """Test creating a session scope."""
        scope = QueryScope.from_session(
            session_id="sess_456",
            agent_type="target"
        )

        assert scope.scope_type == ScopeType.SESSION
        assert scope.session_id == "sess_456"

    def test_custom_scope_creation(self):
        """Test creating a custom scope."""
        scope = QueryScope.from_custom(
            session_id="sess_789",
            min_speed=5.0,
            max_distance=100.0
        )

        assert scope.scope_type == ScopeType.CUSTOM
        assert scope.custom_filters['min_speed'] == 5.0
        assert scope.custom_filters['max_distance'] == 100.0


@pytest.fixture(scope="module")
def init_holoviews():
    """Initialize HoloViews extensions once for all tests."""
    import holoviews as hv
    hv.extension("bokeh")


class TestWidgetRegistry:
    """Test widget registry loading from config."""

    def test_registry_loads_config(self, init_holoviews):
        """Test that WidgetRegistry loads config file."""
        # Find config file (relative to project root)
        config_path = Path(__file__).parent.parent.parent / "collab_env" / "dashboard" / "analysis_widgets.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found at {config_path}")

        registry = WidgetRegistry(str(config_path))
        widgets = registry.get_enabled_widgets()

        # Expected: BasicDataViewer, VelocityStats, DistanceStats, Correlation
        assert len(widgets) == 4, f"Expected 4 enabled widgets, got {len(widgets)}"

    def test_registry_loads_default_parameters(self, init_holoviews):
        """Test that WidgetRegistry loads default parameters."""
        config_path = Path(__file__).parent.parent.parent / "collab_env" / "dashboard" / "analysis_widgets.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found at {config_path}")

        registry = WidgetRegistry(str(config_path))
        defaults = registry.get_defaults()

        assert defaults['spatial_bin_size'] == 10.0
        assert defaults['temporal_window_size'] == 100
        assert defaults['min_samples'] == 100

    def test_widgets_have_correct_attributes(self, init_holoviews):
        """Test that loaded widgets have expected attributes."""
        config_path = Path(__file__).parent.parent.parent / "collab_env" / "dashboard" / "analysis_widgets.yaml"

        if not config_path.exists():
            pytest.skip(f"Config file not found at {config_path}")

        registry = WidgetRegistry(str(config_path))
        widgets = registry.get_enabled_widgets()

        for widget in widgets:
            assert hasattr(widget, 'widget_name')
            assert hasattr(widget, 'widget_category')
            assert widget.widget_name is not None
            assert widget.widget_category is not None


class TestWidgetInstantiation:
    """Test individual widget instantiation."""

    def test_velocity_widget_creation(self, init_holoviews):
        """Test creating a VelocityStatsWidget."""
        from collab_env.dashboard.widgets.velocity_widget import VelocityStatsWidget

        velocity = VelocityStatsWidget()

        assert velocity.widget_name == "Velocity Stats"
        assert velocity.load_btn is not None
        assert velocity.display_pane is not None

    def test_widget_has_ui_components(self, init_holoviews):
        """Test that widgets have required UI components."""
        from collab_env.dashboard.widgets.velocity_widget import VelocityStatsWidget

        widget = VelocityStatsWidget()

        # Check for standard components
        assert hasattr(widget, 'load_btn')
        assert hasattr(widget, 'display_pane')
        assert widget.load_btn is not None
        assert widget.display_pane is not None


class TestAnalysisContext:
    """Test AnalysisContext creation and parameter merging."""

    def test_context_creation_with_episode_scope(self):
        """Test creating AnalysisContext with episode scope."""
        scope = QueryScope.from_episode("ep_123", start_time=0, end_time=500)

        context = AnalysisContext(
            query_backend=None,  # Mock backend not needed for this test
            scope=scope,
            spatial_bin_size=5.0,
            temporal_window_size=50,
            min_samples=20
        )

        assert context.scope == scope
        assert context.spatial_bin_size == 5.0
        assert context.temporal_window_size == 50
        assert context.min_samples == 20

    def test_get_query_params_merges_scope_and_context(self):
        """Test that get_query_params() merges scope and context parameters."""
        scope = QueryScope.from_episode("ep_123", start_time=0, end_time=500)

        context = AnalysisContext(
            query_backend=None,
            scope=scope,
            spatial_bin_size=5.0,
            temporal_window_size=50,
            min_samples=20
        )

        params = context.get_query_params()

        # Scope parameters
        assert params['episode_id'] == "ep_123"
        assert params['start_time'] == 0
        assert params['end_time'] == 500

        # Context parameters
        assert params['bin_size'] == 5.0
        assert params['window_size'] == 50
        assert params['min_samples'] == 20

    def test_get_query_params_allows_overrides(self):
        """Test that get_query_params() allows parameter overrides."""
        scope = QueryScope.from_episode("ep_123")

        context = AnalysisContext(
            query_backend=None,
            scope=scope,
            spatial_bin_size=5.0,
            temporal_window_size=50,
            min_samples=20
        )

        # Override bin_size and add custom param
        params = context.get_query_params(bin_size=15.0, custom_param="test")

        assert params['bin_size'] == 15.0  # Overridden
        assert params['window_size'] == 50  # Not overridden
        assert params['min_samples'] == 20  # Not overridden
        assert params['custom_param'] == "test"  # Added

    def test_context_with_session_scope(self):
        """Test creating AnalysisContext with session scope."""
        scope = QueryScope.from_session(session_id="sess_456")

        context = AnalysisContext(
            query_backend=None,
            scope=scope,
            spatial_bin_size=10.0,
            temporal_window_size=100,
            min_samples=100
        )

        params = context.get_query_params()

        assert 'session_id' in params
        assert params['session_id'] == "sess_456"
        assert 'episode_id' not in params or params['episode_id'] is None
