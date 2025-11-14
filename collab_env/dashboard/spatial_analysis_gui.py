"""
Spatial Analysis GUI for 3D Boids Data - Refactored with Widget System.

Panel/HoloViz-based web dashboard for interactive spatial analysis using
modular widget architecture.
"""

import panel as pn
import param
import logging
from pathlib import Path

from collab_env.data.db.query_backend import QueryBackend
from collab_env.dashboard.widgets import (
    WidgetRegistry,
    AnalysisContext,
    QueryScope,
    ScopeType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable Panel extensions
pn.extension("tabulator", "plotly")
import holoviews as hv
hv.extension("plotly", "bokeh")  # plotly first for Scatter3D support


class SpatialAnalysisGUI(param.Parameterized):
    """
    Main GUI for spatial analysis of 3D boids data with modular widgets.

    Provides:
    - Flexible data scope selection (episode, session, custom)
    - Shared analysis parameters
    - Plugin-based analysis widgets loaded from config
    """

    # Reactive parameters for scope selection
    selected_category = param.String(default="", doc="Selected category ID")
    selected_session = param.String(default="", doc="Selected session ID")
    selected_episode = param.String(default="", doc="Selected episode ID")
    scope_type = param.Selector(
        default="Episode",
        objects=["Episode"],  # Session scope disabled - widgets require episode-level data
        doc="Analysis scope type"
    )

    def __init__(self, widget_config: str = None, **params):
        super().__init__(**params)

        # Initialize query backend
        try:
            self.query = QueryBackend()
            logger.info(f"Connected to {self.query.config.backend} database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

        # Load widgets from registry
        if widget_config is None:
            # Default config location
            config_path = Path(__file__).parent / "analysis_widgets.yaml"
            widget_config = str(config_path)

        self.registry = WidgetRegistry(widget_config)
        self.widgets = self.registry.get_enabled_widgets()
        logger.info(f"Loaded {len(self.widgets)} widgets")

        # Create UI components
        self._create_widgets()

        # Wire callbacks
        self._wire_callbacks()

        # Load initial data
        self._load_categories()
        self._load_sessions()

        # Initialize widget contexts
        self._update_widget_contexts()

    def _create_widgets(self):
        """Create all UI widgets."""

        # === Data Scope Selection ===
        # Session scope disabled - current widgets require episode-level data
        self.scope_type_select = pn.widgets.RadioButtonGroup(
            name="Analysis Scope",
            options=["Episode"],
            value="Episode",
            button_type="success",
            visible=False  # Hidden since only one option
        )

        self.category_select = pn.widgets.Select(
            name="Category",
            options=[],
            min_width=350,
            sizing_mode="stretch_width"
        )

        self.session_select = pn.widgets.Select(
            name="Session",
            options=[],
            min_width=350,
            sizing_mode="stretch_width"
        )

        self.episode_select = pn.widgets.Select(
            name="Episode",
            options=[],
            min_width=350,
            sizing_mode="stretch_width"
        )

        # === Agent Filtering ===
        self.agent_type_select = pn.widgets.RadioButtonGroup(
            name="Agent Type",
            options=["agent", "target", "all"],
            value="agent",
            button_type="success"
        )

        # === Time Range Controls ===
        self.start_time_slider = pn.widgets.IntSlider(
            name="Start Time",
            value=0,
            start=0,
            end=3000,
            step=10,
            width=350
        )

        self.end_time_slider = pn.widgets.IntSlider(
            name="End Time",
            value=3000,
            start=0,
            end=3000,
            step=10,
            width=350
        )

        # Quick time range buttons
        self.btn_before_500 = pn.widgets.Button(
            name="Before t=500",
            button_type="default",
            width=110
        )

        self.btn_after_500 = pn.widgets.Button(
            name="After t=500",
            button_type="default",
            width=110
        )

        self.btn_full_range = pn.widgets.Button(
            name="Full Range",
            button_type="default",
            width=110
        )

        # === Shared Analysis Parameters ===
        defaults = self.registry.get_defaults()

        self.spatial_bin_input = pn.widgets.FloatInput(
            name="Spatial Bin Size",
            value=defaults.get('spatial_bin_size', 10.0),
            start=1.0,
            end=100.0,
            step=1.0,
            width=350
        )

        self.temporal_window_input = pn.widgets.IntInput(
            name="Time Window Size",
            value=defaults.get('temporal_window_size', 100),
            start=10,
            end=1000,
            step=10,
            width=350
        )

        self.min_samples_input = pn.widgets.IntInput(
            name="Min Samples",
            value=defaults.get('min_samples', 10),
            start=1,
            end=1000,
            width=350
        )

        # === Status and Loading Indicators ===
        self.status_pane = pn.pane.HTML(
            "<p>Ready. Select a session to begin.</p>",
            sizing_mode="stretch_width"
        )

        self.loading_indicator = pn.indicators.LoadingSpinner(
            value=False,
            width=50,
            height=50
        )

    def _wire_callbacks(self):
        """Wire widget callbacks."""

        # Category/Session/Episode selection
        self.category_select.param.watch(self._on_category_change, "value")
        self.session_select.param.watch(self._on_session_change, "value")
        self.episode_select.param.watch(self._on_episode_change, "value")

        # Quick time range buttons
        self.btn_before_500.on_click(self._set_before_500)
        self.btn_after_500.on_click(self._set_after_500)
        self.btn_full_range.on_click(self._set_full_range)

        # Update contexts when scope parameters change
        self.scope_type_select.param.watch(lambda e: self._update_widget_contexts(), 'value')
        self.agent_type_select.param.watch(lambda e: self._update_widget_contexts(), 'value')
        self.start_time_slider.param.watch(lambda e: self._update_widget_contexts(), 'value')
        self.end_time_slider.param.watch(lambda e: self._update_widget_contexts(), 'value')

        # Shared parameter changes
        self.spatial_bin_input.param.watch(lambda e: self._update_widget_contexts(), 'value')
        self.temporal_window_input.param.watch(lambda e: self._update_widget_contexts(), 'value')
        self.min_samples_input.param.watch(lambda e: self._update_widget_contexts(), 'value')

    # ==================== Status Helpers ====================

    def _show_loading(self, message: str = "Loading..."):
        """Show loading indicator with message."""
        self.loading_indicator.value = True
        self.status_pane.object = f"<p><em>{message}</em></p>"

    def _hide_loading(self):
        """Hide loading indicator."""
        self.loading_indicator.value = False

    def _show_success(self, message: str):
        """Show success message."""
        self._hide_loading()
        self.status_pane.object = f"<p style='color:green'>✅ {message}</p>"

    def _show_error(self, message: str):
        """Show error message."""
        self._hide_loading()
        self.status_pane.object = f"<p style='color:red'>❌ Error: {message}</p>"

    # ==================== Data Loading ====================

    def _load_categories(self):
        """Load available categories."""
        try:
            categories_df = self.query.get_categories()
            if len(categories_df) == 0:
                self.category_select.options = [""]
                self.status_pane.object = "<p style='color:orange'>⚠️ No categories found in database.</p>"
            else:
                # Create display names (category_name)
                category_options = {row['category_name']: row['category_id']
                                   for _, row in categories_df.iterrows()}
                self.category_select.options = [""] + list(category_options.keys())
                self.category_select.param.trigger("options")
                self._category_map = category_options
                logger.info(f"Loaded {len(categories_df)} categories")
        except Exception as e:
            logger.error(f"Failed to load categories: {e}")
            self._show_error(f"Failed to load categories: {e}")

    def _load_sessions(self):
        """Load available sessions for the selected category."""
        # Get current category selection (None if not selected)
        category_id = self.selected_category if self.selected_category else None

        try:
            sessions_df = self.query.get_sessions(category_id=category_id)
            if len(sessions_df) == 0:
                self.session_select.options = [""]
                if category_id:
                    self.status_pane.object = f"<p style='color:orange'>⚠️ No sessions found for category '{category_id}'.</p>"
                else:
                    self.status_pane.object = "<p style='color:orange'>⚠️ No sessions found. Load data first.</p>"
            else:
                # Create display names (session_name)
                session_options = {row['session_name']: row['session_id']
                                   for _, row in sessions_df.iterrows()}
                self.session_select.options = [""] + list(session_options.keys())
                self.session_select.param.trigger("options")
                self._session_map = session_options
                logger.info(f"Loaded {len(sessions_df)} sessions")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            self._show_error(f"Failed to load sessions: {e}")

    def _on_category_change(self, event):
        """Handle category selection."""
        category_name = event.new
        if not category_name or category_name == "":
            self.session_select.options = [""]
            self.episode_select.options = [""]
            return

        try:
            # Get the category ID from the category map
            category_id = self._category_map[category_name]
            self.selected_category = category_id

            # Load sessions for this category
            self._load_sessions()

            # Clear episode selection
            self.episode_select.options = [""]

        except Exception as e:
            logger.error(f"Failed to load category: {e}")
            self._show_error(f"Failed to load category: {e}")

    def _on_session_change(self, event):
        """Handle session selection."""
        session_name = event.new
        if not session_name or session_name == "":
            self.episode_select.options = [""]
            return

        try:
            session_id = self._session_map[session_name]
            self.selected_session = session_id

            # Load episodes
            episodes_df = self.query.get_episodes(session_id)
            if len(episodes_df) == 0:
                self.episode_select.options = [""]
                self._show_error("No episodes found for this session")
            else:
                # Create display names (episode_id for clarity, especially for GNN rollouts)
                episode_options = {row['episode_id']: row['episode_id']
                                   for _, row in episodes_df.iterrows()}
                self.episode_select.options = [""] + list(episode_options.keys())
                self.episode_select.param.trigger("options")
                self._episode_map = episode_options
                self._show_success(f"Loaded {len(episodes_df)} episodes")
                logger.info(f"Loaded {len(episodes_df)} episodes")

                # Update widget contexts for session scope
                self._update_widget_contexts()

        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")
            self._show_error(f"Failed to load episodes: {e}")

    def _on_episode_change(self, event):
        """Handle episode selection."""
        episode_name = event.new
        if not episode_name or episode_name == "":
            return

        try:
            episode_id = self._episode_map[episode_name]
            self.selected_episode = episode_id

            # Get episode metadata to update time range
            metadata = self.query.get_episode_metadata(episode_id)
            if len(metadata) > 0:
                num_frames = int(metadata.iloc[0]['num_frames'])
                self.start_time_slider.end = num_frames
                self.end_time_slider.end = num_frames
                self.end_time_slider.value = num_frames

                # Get agent types from episode
                agent_types_df = self.query.get_agent_types(episode_id)
                if len(agent_types_df) > 0:
                    agent_types = agent_types_df['agent_type_id'].tolist()
                    # Always include 'all' option
                    agent_types_with_all = agent_types + ['all']
                    self.agent_type_select.options = agent_types_with_all
                    # Set default to first agent type
                    self.agent_type_select.value = agent_types[0] if agent_types else 'all'
                    logger.info(f"Loaded {len(agent_types)} agent types: {agent_types}")

                self._show_success(f"Episode selected ({num_frames} frames, {len(agent_types_df)} agent types)")
                logger.info(f"Selected episode {episode_id} ({num_frames} frames)")

                # Update widget contexts for episode scope
                self._update_widget_contexts()

        except Exception as e:
            logger.error(f"Failed to load episode metadata: {e}")
            self._show_error(f"Failed to load episode metadata: {e}")

    # ==================== Time Range Controls ====================

    def _set_before_500(self, event):
        """Set time range to before t=500."""
        self.start_time_slider.value = 0
        self.end_time_slider.value = 500

    def _set_after_500(self, event):
        """Set time range to after t=500."""
        self.start_time_slider.value = 500
        self.end_time_slider.value = self.end_time_slider.end

    def _set_full_range(self, event):
        """Set time range to full episode."""
        self.start_time_slider.value = 0
        self.end_time_slider.value = self.end_time_slider.end

    # ==================== Context Management ====================

    def _get_current_scope(self) -> QueryScope:
        """Build QueryScope from current UI state."""
        scope_type_str = self.scope_type_select.value.lower()

        if scope_type_str == "episode":
            if not self.selected_episode:
                return None

            return QueryScope.from_episode(
                episode_id=self.selected_episode,
                start_time=self.start_time_slider.value,
                end_time=self.end_time_slider.value,
                agent_type=self.agent_type_select.value
            )

        elif scope_type_str == "session":
            if not self.selected_session:
                return None

            return QueryScope.from_session(
                session_id=self.selected_session,
                agent_type=self.agent_type_select.value
            )

        return None

    def _get_context(self) -> AnalysisContext:
        """Build AnalysisContext from current UI state."""
        scope = self._get_current_scope()
        if scope is None:
            return None

        return AnalysisContext(
            query_backend=self.query,
            scope=scope,
            spatial_bin_size=self.spatial_bin_input.value,
            temporal_window_size=self.temporal_window_input.value,
            min_samples=self.min_samples_input.value,
            on_loading=self._show_loading,
            on_success=self._show_success,
            on_error=self._show_error
        )

    def _update_widget_contexts(self):
        """Update context for all widgets when scope changes."""
        context = self._get_context()
        if context:
            for widget in self.widgets:
                widget.context = context
            logger.debug(f"Updated widget contexts: {context.scope}")

    # ==================== Layout ====================

    def create_layout(self):
        """Create the dashboard layout with dynamic widget tabs."""

        # Sidebar with scope selection + shared parameters
        # Note: scope_type_select is hidden (only Episode scope available)
        sidebar = pn.Column(
            "## Episode Selection",
            self.category_select,
            self.session_select,
            self.episode_select,
            pn.layout.Divider(),

            "## Agent Type",
            self.agent_type_select,
            pn.layout.Divider(),

            "## Time Range",
            self.start_time_slider,
            self.end_time_slider,
            pn.Row(
                self.btn_before_500,
                self.btn_after_500,
                self.btn_full_range
            ),
            pn.layout.Divider(),

            "## Shared Parameters",
            self.spatial_bin_input,
            self.temporal_window_input,
            self.min_samples_input,

            width=400,
            sizing_mode="stretch_height"
        )

        # Main content: Dynamic tabs from widgets
        tabs = pn.Tabs(
            *[(w.widget_name, w.get_tab_content()) for w in self.widgets],
            sizing_mode="stretch_both"
        )

        # Status bar with loading indicator
        status_bar = pn.Row(
            self.loading_indicator,
            self.status_pane,
            sizing_mode="stretch_width"
        )

        # Create template
        template = pn.template.MaterialTemplate(
            title="CIS Analysis Dashboard",
            sidebar=[sidebar],
            main=[status_bar, tabs],
            header_background="#2596be",
            sidebar_width=400,
        )

        return template


def create_app():
    """
    Create and return the spatial analysis app.

    Returns
    -------
    pn.template.MaterialTemplate
        Panel template ready to serve
    """
    gui = SpatialAnalysisGUI()
    return gui.create_layout()
