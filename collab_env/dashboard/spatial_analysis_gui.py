"""
Spatial Analysis GUI for 3D Boids Data.

Panel/HoloViz-based web dashboard for interactive spatial analysis.
"""

import panel as pn
import param
import pandas as pd
import holoviews as hv
from holoviews import opts
import logging

from collab_env.data.db.query_backend import QueryBackend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable Panel extensions
pn.extension("tabulator", "plotly")
hv.extension("plotly", "bokeh")  # plotly first for Scatter3D support


class SpatialAnalysisGUI(param.Parameterized):
    """
    Main GUI for spatial analysis of 3D boids data.

    Provides interactive controls for selecting sessions/episodes and
    visualizing spatial statistics (heatmaps, velocities, distances, correlations).
    """

    # Reactive parameters
    selected_session = param.String(default="", doc="Selected session ID")
    selected_episode = param.String(default="", doc="Selected episode ID")
    agent_type = param.Selector(
        default="agent",
        objects=["agent", "target", "all"],
        doc="Agent type filter"
    )

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize query backend
        try:
            self.query = QueryBackend()
            logger.info(f"Connected to {self.query.config.backend} database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

        # Create UI components
        self._create_widgets()

        # Wire callbacks
        self._wire_callbacks()

        # Load initial data
        self._load_sessions()

    def _create_widgets(self):
        """Create all UI widgets."""

        # Session/Episode selectors
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

        self.agent_type_select = pn.widgets.RadioButtonGroup(
            name="Agent Type",
            options=["agent", "target", "all"],
            value="agent",
            button_type="success"
        )

        # Time range controls
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

        # Analysis parameters
        self.bin_size_input = pn.widgets.FloatInput(
            name="Spatial Bin Size",
            value=10.0,
            start=1.0,
            end=100.0,
            step=1.0,
            width=350
        )

        self.window_size_input = pn.widgets.IntInput(
            name="Time Window Size",
            value=100,
            start=10,
            end=1000,
            step=10,
            width=350
        )

        # Load buttons for each analysis type
        self.load_heatmap_btn = pn.widgets.Button(
            name="Load Heatmap",
            button_type="primary",
            width=200
        )

        self.load_velocity_btn = pn.widgets.Button(
            name="Load Velocity Stats",
            button_type="primary",
            width=200
        )

        self.load_distances_btn = pn.widgets.Button(
            name="Load Distances",
            button_type="primary",
            width=200
        )

        self.load_correlations_btn = pn.widgets.Button(
            name="Load Correlations",
            button_type="primary",
            width=200
        )

        # Display panes
        self.heatmap_pane = pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=500),
            sizing_mode="stretch_both"
        )

        self.velocity_plot_pane = pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=400),
            sizing_mode="stretch_both"
        )

        self.distance_plot_pane = pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=400),
            sizing_mode="stretch_both"
        )

        self.correlation_pane = pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=500),
            sizing_mode="stretch_both"
        )

        # Status and loading indicators
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

        # Session/Episode selection
        self.session_select.param.watch(self._on_session_change, "value")
        self.episode_select.param.watch(self._on_episode_change, "value")

        # Quick time range buttons
        self.btn_before_500.on_click(self._set_before_500)
        self.btn_after_500.on_click(self._set_after_500)
        self.btn_full_range.on_click(self._set_full_range)

        # Load buttons
        self.load_heatmap_btn.on_click(self._load_heatmap)
        self.load_velocity_btn.on_click(self._load_velocity_stats)
        self.load_distances_btn.on_click(self._load_distances)
        self.load_correlations_btn.on_click(self._load_correlations)

    def _show_loading(self, message: str = "Loading..."):
        """Show loading indicator with message."""
        self.loading_indicator.value = True
        self.status_pane.object = f"<p><em>{message}</em></p>"

    def _hide_loading(self):
        """Hide loading indicator."""
        self.loading_indicator.value = False

    def _show_success(self, message: str):
        """Show success message."""
        self.status_pane.object = f"<p style='color:green'>✅ {message}</p>"

    def _show_error(self, message: str):
        """Show error message."""
        self.status_pane.object = f"<p style='color:red'>❌ Error: {message}</p>"

    # ==================== Data Loading ====================

    def _load_sessions(self):
        """Load available sessions."""
        try:
            sessions_df = self.query.get_sessions(category_id='boids_3d')
            if len(sessions_df) == 0:
                self.session_select.options = [""]
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
                # Create display names (episode_number)
                episode_options = {f"Episode {row['episode_number']}": row['episode_id']
                                   for _, row in episodes_df.iterrows()}
                self.episode_select.options = [""] + list(episode_options.keys())
                self.episode_select.param.trigger("options")
                self._episode_map = episode_options
                self._show_success(f"Loaded {len(episodes_df)} episodes")
                logger.info(f"Loaded {len(episodes_df)} episodes")
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

    # ==================== Analysis Loading ====================

    def _load_heatmap(self, event):
        """Load and display 3D spatial heatmap."""
        if not self.selected_episode:
            self._show_error("Please select an episode first")
            return

        self._show_loading("Loading 3D spatial heatmap...")
        try:
            # Query heatmap data
            df = self.query.get_spatial_heatmap(
                episode_id=self.selected_episode,
                bin_size=self.bin_size_input.value,
                start_time=self.start_time_slider.value,
                end_time=self.end_time_slider.value,
                agent_type=self.agent_type_select.value
            )

            if len(df) == 0:
                self._hide_loading()
                self._show_error("No data found for selected parameters")
                return

            # Create 3D scatter plot with density as color and size
            scatter = hv.Scatter3D(
                df,
                kdims=['x_bin', 'y_bin', 'z_bin'],
                vdims='density'
            ).opts(
                color='density',
                cmap='viridis',
                size='density',
                width=800,
                height=600,
                colorbar=True,
                title=f'3D Spatial Density (bin size={self.bin_size_input.value})'
            )

            self.heatmap_pane.object = scatter
            self._hide_loading()
            self._show_success(f"3D heatmap loaded ({len(df)} bins)")
            logger.info(f"Loaded 3D heatmap with {len(df)} bins")

        except Exception as e:
            self._hide_loading()
            logger.error(f"Failed to load heatmap: {e}")
            self._show_error(f"Failed to load heatmap: {e}")

    def _load_velocity_stats(self, event):
        """Load velocity statistics."""
        if not self.selected_episode:
            self._show_error("Please select an episode first")
            return

        self._show_loading("Loading velocity statistics...")
        try:
            # Query speed statistics
            df = self.query.get_speed_statistics(
                episode_id=self.selected_episode,
                window_size=self.window_size_input.value,
                start_time=self.start_time_slider.value,
                end_time=self.end_time_slider.value,
                agent_type=self.agent_type_select.value
            )

            if len(df) == 0:
                self._hide_loading()
                self._show_error("No data found for selected parameters")
                return

            # Create line plot with error bands
            curve = hv.Curve(df, kdims='time_window', vdims='avg_speed', label='Mean Speed')
            curve.opts(
                opts.Curve(
                    color='blue',
                    line_width=2,
                    width=700,
                    height=400,
                    xlabel='Time',
                    ylabel='Speed',
                    title=f'Speed Over Time (window={self.window_size_input.value})',
                    tools=['hover'],
                    backend='bokeh'
                )
            )

            # Add std deviation as error bars if available
            if 'std_speed' in df.columns:
                df['upper'] = df['avg_speed'] + df['std_speed']
                df['lower'] = df['avg_speed'] - df['std_speed']
                area = hv.Area(df, kdims='time_window', vdims=['lower', 'upper'])
                area.opts(opts.Area(alpha=0.3, color='blue', backend='bokeh'))
                plot = curve * area
            else:
                plot = curve

            self.velocity_plot_pane.object = plot
            self._hide_loading()
            self._show_success(f"Velocity stats loaded ({len(df)} windows)")
            logger.info(f"Loaded velocity stats with {len(df)} windows")

        except Exception as e:
            self._hide_loading()
            logger.error(f"Failed to load velocity stats: {e}")
            self._show_error(f"Failed to load velocity stats: {e}")

    def _load_distances(self, event):
        """Load distance statistics."""
        if not self.selected_episode:
            self._show_error("Please select an episode first")
            return

        self._show_loading("Loading distance statistics...")
        try:
            # Query distance to target
            df_target = self.query.get_distance_to_target(
                episode_id=self.selected_episode,
                window_size=self.window_size_input.value,
                start_time=self.start_time_slider.value,
                end_time=self.end_time_slider.value,
                agent_type=self.agent_type_select.value
            )

            # Query distance to boundary
            df_boundary = self.query.get_distance_to_boundary(
                episode_id=self.selected_episode,
                window_size=self.window_size_input.value,
                start_time=self.start_time_slider.value,
                end_time=self.end_time_slider.value,
                agent_type=self.agent_type_select.value
            )

            if len(df_target) == 0 and len(df_boundary) == 0:
                self._hide_loading()
                self._show_error("No distance data found")
                return

            # Create overlay plot
            curves = []

            if len(df_target) > 0:
                curve_target = hv.Curve(df_target, kdims='time_window', vdims='avg_distance', label='Distance to Target')
                curve_target.opts(opts.Curve(color='orange', line_width=2, backend='bokeh'))
                curves.append(curve_target)

            if len(df_boundary) > 0:
                curve_boundary = hv.Curve(df_boundary, kdims='time_window', vdims='avg_distance', label='Distance to Boundary')
                curve_boundary.opts(opts.Curve(color='purple', line_width=2, backend='bokeh'))
                curves.append(curve_boundary)

            if curves:
                overlay = hv.Overlay(curves)
                overlay.opts(
                    opts.Overlay(
                        width=700,
                        height=400,
                        xlabel='Time',
                        ylabel='Distance',
                        title=f'Distances Over Time (window={self.window_size_input.value})',
                        tools=['hover'],
                        legend_position='top_right',
                        backend='bokeh'
                    )
                )
                self.distance_plot_pane.object = overlay
            else:
                self.distance_plot_pane.object = hv.Curve([]).opts(width=700, height=400)

            self._hide_loading()
            self._show_success(f"Distance stats loaded")
            logger.info(f"Loaded distance stats (target: {len(df_target)}, boundary: {len(df_boundary)} windows)")

        except Exception as e:
            self._hide_loading()
            logger.error(f"Failed to load distances: {e}")
            self._show_error(f"Failed to load distances: {e}")

    def _load_correlations(self, event):
        """Load and visualize correlation statistics as 3D heatmap."""
        if not self.selected_episode:
            self._show_error("Please select an episode first")
            return

        self._show_loading("Loading correlations (this may take a while)...")
        try:
            # Query velocity correlations with lower min_samples threshold
            df = self.query.get_velocity_correlations(
                episode_id=self.selected_episode,
                start_time=self.start_time_slider.value,
                end_time=self.end_time_slider.value,
                min_samples=10,  # Reduced from 100 to 10
                agent_type=self.agent_type_select.value
            )

            logger.info(f"Query returned {len(df)} correlation pairs")

            if len(df) == 0:
                self._hide_loading()
                self._show_error("No correlation data found. Try adjusting time range or agent type.")
                return

            # Calculate average correlation magnitude across all dimensions
            df['avg_correlation'] = df[['v_x_correlation', 'v_y_correlation', 'v_z_correlation']].abs().mean(axis=1)

            logger.info(f"Correlation range: {df['avg_correlation'].min():.3f} to {df['avg_correlation'].max():.3f}")

            # Scale correlation for better visibility (normalize to 0-1 then scale for size)
            # Use constant marker size for better visibility
            marker_size = 8

            # Create 3D scatter plot where each point represents an agent pair
            # Use agent_i and agent_j as spatial coordinates, correlation as z-axis and color
            scatter = hv.Scatter3D(
                df,
                kdims=['agent_i', 'agent_j', 'avg_correlation'],
                vdims=['v_x_correlation', 'v_y_correlation', 'v_z_correlation', 'n_samples']
            ).opts(
                color='avg_correlation',
                cmap='Viridis',  # Yellow-green-blue colormap for better visibility
                size=marker_size,  # Fixed size for visibility
                width=800,
                height=600,
                colorbar=True,
                title=f'Velocity Correlations (3D: agent_i × agent_j × avg_corr) - {len(df)} pairs',
                zlabel='Average Correlation',
                xlim=(df['agent_i'].min()-1, df['agent_i'].max()+1),
                ylim=(df['agent_j'].min()-1, df['agent_j'].max()+1)
            )

            self.correlation_pane.object = scatter
            self._hide_loading()
            self._show_success(f"Correlations loaded ({len(df)} agent pairs)")
            logger.info(f"Loaded correlations for {len(df)} agent pairs")

        except Exception as e:
            self._hide_loading()
            logger.error(f"Failed to load correlations: {e}")
            self._show_error(f"Failed to load correlations: {e}")

    # ==================== Layout ====================

    def create_layout(self):
        """Create the dashboard layout."""

        # Sidebar with controls
        sidebar = pn.Column(
            "## Session Selection",
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

            "## Analysis Parameters",
            self.bin_size_input,
            self.window_size_input,

            width=400,
            sizing_mode="stretch_height"
        )

        # Main content with tabs
        content = pn.Tabs(
            ("Heatmap", pn.Column(
                self.load_heatmap_btn,
                self.heatmap_pane,
                sizing_mode="stretch_both"
            )),
            ("Velocity Stats", pn.Column(
                self.load_velocity_btn,
                self.velocity_plot_pane,
                sizing_mode="stretch_both"
            )),
            ("Distances", pn.Column(
                self.load_distances_btn,
                self.distance_plot_pane,
                sizing_mode="stretch_both"
            )),
            ("Correlations", pn.Column(
                self.load_correlations_btn,
                self.correlation_pane,
                sizing_mode="stretch_both"
            )),
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
            title="Spatial Analysis Dashboard - 3D Boids",
            sidebar=[sidebar],
            main=[status_bar, content],
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
