"""
Basic Data Viewer widget - comprehensive episode visualization.

Provides a 4-panel synchronized view combining:
- Animation panel: Animated 2D/3D scatter plot of agent tracks
- Spatial heatmap panel: Density heatmap of agent positions
- Time series panel: Multi-property time series with sync indicator
- Histogram panel: Dynamic row of histograms for selected properties

All panels are synchronized by time window and current time.
Property selection affects time series and histograms.
"""

import logging
from typing import Optional, List

import param
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool, Span
from bokeh.plotting import figure
from bokeh.palettes import Category10_10

from .base_analysis_widget import BaseAnalysisWidget
from .query_scope import ScopeType

logger = logging.getLogger(__name__)

# Initialize HoloViews extension (plotly for 3D support, bokeh for 2D)
hv.extension('plotly', 'bokeh')


class BasicDataViewerWidget(BaseAnalysisWidget):
    """
    Comprehensive episode viewer with 4 synchronized panels.

    Features:
    - Animation of agent tracks with configurable playback
    - 3D spatial density heatmap
    - Multi-property time series with current time indicator
    - Dynamic histograms for selected properties

    All panels respect time window (start_time/end_time) and synchronize
    current playback time.
    """

    widget_name = "Basic Data Viewer"
    widget_description = "Comprehensive episode visualization with animation, heatmaps, and time series"
    widget_category = "visualization"

    # Playback controls
    current_time = param.Integer(default=0, bounds=(0, None), doc="Current playback time index")
    playback_speed = param.Number(default=1.0, bounds=(0.1, 10.0), doc="Playback speed multiplier")
    is_playing = param.Boolean(default=False, doc="Whether animation is playing")
    trail_length = param.Integer(default=50, bounds=(0, 500), doc="Number of frames to show in trail")

    # Visualization options
    show_agent_ids = param.Boolean(default=True, doc="Show agent ID labels")
    color_scale = param.Selector(
        default='viridis',
        objects=['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
        doc="Color scale for heatmap"
    )

    # Property selection (for time series and histograms)
    selected_properties = param.ListSelector(default=[], objects=[], doc="Selected properties to display")

    def __init__(self, **params):
        # Initialize data storage
        self.tracks_df = None
        self.properties_ts_df = None
        self.properties_dist_df = None
        self.available_props_df = None

        # UI components (will be created in _create_ui)
        self.animation_pane = None
        self.heatmap_pane = None
        self.timeseries_pane = None
        self.histogram_pane = None
        self.play_button = None

        # Playback state
        self._playback_callback = None  # Track periodic callback for playback

        # Data dimensionality (detected from loaded data)
        self.is_3d = True  # Default to 3D, will be updated based on data

        super().__init__(**params)

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create playback controls and property selection."""

        # Playback controls
        self.play_button = pn.widgets.Button(name="▶ Play", button_type="success", width=80)
        self.play_button.on_click(self._toggle_playback)

        speed_slider = pn.widgets.FloatSlider.from_param(
            self.param.playback_speed,
            name="Speed",
            width=150
        )

        time_slider = pn.widgets.IntSlider.from_param(
            self.param.current_time,
            name="Time",
            width=300
        )

        trail_slider = pn.widgets.IntSlider.from_param(
            self.param.trail_length,
            name="Trail",
            width=150
        )

        show_ids_checkbox = pn.widgets.Checkbox.from_param(
            self.param.show_agent_ids,
            name="IDs",
            width=60
        )

        color_select = pn.widgets.Select.from_param(
            self.param.color_scale,
            name="Colors",
            width=150
        )

        # Property selection (will be populated after data load)
        property_selector = pn.widgets.CheckBoxGroup.from_param(
            self.param.selected_properties,
            name="Properties",
            inline=True,
            width=400
        )

        # Compact single-row layout
        return pn.Column(
            pn.Row(
                self.play_button,
                speed_slider,
                time_slider,
                trail_slider,
                show_ids_checkbox,
                color_select,
                property_selector,
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create 4-panel layout for synchronized visualization."""

        # Create persistent panes that we'll update via .object property
        # This avoids recreating widgets every frame (much more efficient)
        self.animation_viz_pane = pn.pane.HoloViews(
            sizing_mode="stretch_both",
            min_height=400
        )

        self.heatmap_viz_pane = pn.pane.HoloViews(
            sizing_mode="stretch_both",
            min_height=400
        )

        self.timeseries_viz_pane = pn.pane.Bokeh(
            sizing_mode="stretch_width",
            height=300
        )

        self.histogram_viz_pane = pn.pane.HoloViews(
            sizing_mode="stretch_width",
            height=250
        )

        # Wrap in columns with loading messages
        self.animation_pane = pn.Column(
            pn.pane.Markdown("**Animation Panel**\n\nClick 'Load Basic Data Viewer' to load data."),
            self.animation_viz_pane,
            sizing_mode="stretch_both"
        )

        self.heatmap_pane = pn.Column(
            pn.pane.Markdown("**Spatial Heatmap Panel**\n\nClick 'Load Basic Data Viewer' to load data."),
            self.heatmap_viz_pane,
            sizing_mode="stretch_both"
        )

        self.timeseries_pane = pn.Column(
            pn.pane.Markdown("**Time Series Panel**\n\nClick 'Load Basic Data Viewer' to load data."),
            self.timeseries_viz_pane,
            sizing_mode="stretch_width"
        )

        self.histogram_pane = pn.Column(
            pn.pane.Markdown("**Histogram Panel**\n\nClick 'Load Basic Data Viewer' to load data."),
            self.histogram_viz_pane,
            sizing_mode="stretch_width"
        )

        # Layout: 2x2 grid for top panels, full-width for bottom panels
        return pn.Column(
            pn.Row(
                self.animation_pane,
                self.heatmap_pane,
                sizing_mode="stretch_both"
            ),
            self.timeseries_pane,
            self.histogram_pane,
            sizing_mode="stretch_both"
        )

    def _detect_dimensionality(self):
        """Detect if loaded data is 2D or 3D based on z-coordinate variance."""
        if self.tracks_df is None or len(self.tracks_df) == 0:
            self.is_3d = True
            return

        # Check if z column exists
        if 'z' not in self.tracks_df.columns:
            self.is_3d = False
            return

        # Check if z values are all NULL
        if self.tracks_df['z'].isna().all():
            self.is_3d = False
            return

        # Check if z has meaningful variance (not all same value)
        z_std = self.tracks_df['z'].std()
        z_range = self.tracks_df['z'].max() - self.tracks_df['z'].min()

        # Consider it 2D if z has very low variance (< 0.01) or range
        self.is_3d = (z_std > 0.01) and (z_range > 0.01)

    def load_data(self) -> None:
        """Load all data for the 4 panels."""

        # Validate this is episode scope only
        if self.context.scope.scope_type != ScopeType.EPISODE:
            raise ValueError("Basic Data Viewer only supports episode-level analysis")

        logger.info("Loading episode tracks...")
        self.tracks_df = self.query_with_context('get_episode_tracks')

        if len(self.tracks_df) == 0:
            raise ValueError("No track data found for selected episode")

        # Detect if data is 2D or 3D
        self._detect_dimensionality()

        # Compute and store fixed spatial bounds for consistent axis limits
        self.x_range = (float(self.tracks_df['x'].min()), float(self.tracks_df['x'].max()))
        self.y_range = (float(self.tracks_df['y'].min()), float(self.tracks_df['y'].max()))
        if self.is_3d:
            self.z_range = (float(self.tracks_df['z'].min()), float(self.tracks_df['z'].max()))
        else:
            self.z_range = None

        logger.info(f"Loaded {len(self.tracks_df)} track observations for {self.tracks_df['agent_id'].nunique()} agents")
        logger.info(f"Data dimensionality: {'3D' if self.is_3d else '2D'}")
        logger.info(f"Spatial bounds - X: {self.x_range}, Y: {self.y_range}" + (f", Z: {self.z_range}" if self.is_3d else ""))

        # Get available properties
        logger.info("Loading available properties...")
        self.available_props_df = self.query_with_context('get_available_properties')

        if len(self.available_props_df) > 0:
            # Update property selector options
            prop_list = self.available_props_df['property_id'].tolist()
            self.param.selected_properties.objects = prop_list

            # Default to first 2-3 properties if available
            default_props = prop_list[:min(3, len(prop_list))]
            self.selected_properties = default_props

            logger.info(f"Found {len(prop_list)} properties: {prop_list}")

            # Load property time series
            logger.info("Loading property time series...")
            self.properties_ts_df = self.query_with_context(
                'get_extended_properties_timeseries',
                property_ids=None  # Get all, filter later
            )

            # Load property distributions
            logger.info("Loading property distributions...")
            self.properties_dist_df = self.query_with_context(
                'get_property_distributions',
                property_ids=None  # Get all, filter later
            )
        else:
            logger.warning("No extended properties found for this episode")
            self.properties_ts_df = pd.DataFrame()
            self.properties_dist_df = pd.DataFrame()

        # Update time slider bounds based on data
        min_time = int(self.tracks_df['time_index'].min())
        max_time = int(self.tracks_df['time_index'].max())
        self.param.current_time.bounds = (min_time, max_time)
        self.current_time = min_time

        logger.info(f"Time range: {min_time} to {max_time}")

        # Convert placeholder panes to proper visualization panes
        self._init_visualization_panes()

        # Update all panels
        self._update_animation_panel()
        self._update_heatmap_panel()
        self._update_timeseries_panel()
        self._update_histogram_panel()

        logger.info("Basic Data Viewer loaded successfully")

    def _init_visualization_panes(self):
        """Remove loading messages, keep visualization panes."""
        # Remove loading markdown messages (first element in each Column)
        # The viz panes are persistent and will be updated via .object property
        if len(self.animation_pane.objects) > 1:
            self.animation_pane.pop(0)  # Remove markdown
        if len(self.heatmap_pane.objects) > 1:
            self.heatmap_pane.pop(0)
        if len(self.timeseries_pane.objects) > 1:
            self.timeseries_pane.pop(0)
        if len(self.histogram_pane.objects) > 1:
            self.histogram_pane.pop(0)

    def _update_animation_panel(self):
        """Update animation panel with current time and trails."""
        if self.tracks_df is None or len(self.tracks_df) == 0:
            return

        # Filter to current time window (current_time - trail_length to current_time)
        start_time = max(self.current_time - self.trail_length, self.tracks_df['time_index'].min())
        end_time = self.current_time

        window_df = self.tracks_df[
            (self.tracks_df['time_index'] >= start_time) &
            (self.tracks_df['time_index'] <= end_time)
        ]

        if len(window_df) == 0:
            # Set to None to show empty state
            self.animation_viz_pane.object = None
            return

        # Create scatter plot with trails (2D or 3D based on data)
        # Current positions (at current_time)
        current_df = window_df[window_df['time_index'] == self.current_time]

        if self.is_3d:
            # 3D visualization
            if len(current_df) > 0:
                scatter = hv.Scatter3D(
                    current_df,
                    kdims=['x', 'y', 'z'],
                    vdims=['agent_id', 'speed']
                ).opts(
                    color='agent_id',
                    cmap='Category20',
                    size=5,
                    width=450,
                    height=450,
                    title=f'Agent Positions (t={self.current_time})',
                    colorbar=False
                )
            else:
                scatter = hv.Scatter3D([]).opts(
                    width=450,
                    height=450,
                    title='Animation'
                )

            # Add trails if trail_length > 0
            if self.trail_length > 0 and len(window_df) > 0:
                paths = []
                for agent_id in window_df['agent_id'].unique():
                    agent_trail = window_df[window_df['agent_id'] == agent_id].sort_values('time_index')
                    if len(agent_trail) > 1:
                        path = hv.Path3D(
                            agent_trail,
                            kdims=['x', 'y', 'z']
                        ).opts(color='gray')
                        paths.append(path)

                if paths:
                    trails = hv.Overlay(paths)
                    scatter = trails * scatter

            # Apply fixed axis limits to the final overlay
            scatter = scatter.opts(
                xlim=self.x_range,
                ylim=self.y_range,
                zlim=self.z_range
            )
        else:
            # 2D visualization (use bokeh backend for 2D)
            # Note: Scatter requires single kdim (x) and vdims (y, other data) for bokeh backend
            if len(current_df) > 0:
                scatter = hv.Scatter(
                    current_df,
                    kdims=['x'],
                    vdims=['y', 'agent_id', 'speed']
                ).opts(
                    color='agent_id',
                    cmap='Category20',
                    size=8,
                    width=450,
                    height=450,
                    title=f'Agent Positions (t={self.current_time})',
                    colorbar=False,
                    tools=['hover'],
                    backend='bokeh'
                )
            else:
                scatter = hv.Scatter([]).opts(
                    width=450,
                    height=450,
                    title='Animation',
                    backend='bokeh'
                )

            # Add trails if trail_length > 0
            if self.trail_length > 0 and len(window_df) > 0:
                paths = []
                for agent_id in window_df['agent_id'].unique():
                    agent_trail = window_df[window_df['agent_id'] == agent_id].sort_values('time_index')
                    if len(agent_trail) > 1:
                        path = hv.Path(
                            agent_trail,
                            kdims=['x', 'y']
                        ).opts(color='gray', alpha=0.5, line_width=1, backend='bokeh')
                        paths.append(path)

                if paths:
                    trails = hv.Overlay(paths)
                    scatter = trails * scatter

            # Apply fixed axis limits to the final overlay
            scatter = scatter.opts(
                xlim=self.x_range,
                ylim=self.y_range,
                backend='bokeh'
            )

        # Update pane content (efficient - just updates object, doesn't recreate widget)
        self.animation_viz_pane.object = scatter

    def _update_heatmap_panel(self):
        """Update spatial heatmap panel."""
        # Query spatial heatmap using shared context
        df = self.query_with_context('get_spatial_heatmap')

        if len(df) == 0:
            self.heatmap_viz_pane.object = None
            return

        # Convert bin lower bounds to bin centers
        # Note: x_bin, y_bin, z_bin from the SQL query are already coordinates (lower bounds),
        # not indices. The query does: floor(x / bin_size) * bin_size
        # So x_bin is the left edge of the bin, and we just add half the bin size for center
        bin_size = self.context.spatial_bin_size
        df = df.copy()

        # Calculate bin centers from bin lower bounds
        df['x_center'] = df['x_bin'] + (bin_size / 2)
        df['y_center'] = df['y_bin'] + (bin_size / 2)

        # Debug logging
        logger.info(f"Heatmap data ranges - X: ({df['x_center'].min()}, {df['x_center'].max()}), "
                   f"Y: ({df['y_center'].min()}, {df['y_center'].max()})")
        logger.info(f"Axis limits being applied - X: {self.x_range}, Y: {self.y_range}")

        if self.is_3d:
            df['z_center'] = df['z_bin'] + (bin_size / 2)
            logger.info(f"Z: ({df['z_center'].min()}, {df['z_center'].max()}), Z_limit: {self.z_range}")

            # 3D scatter plot with density
            viz = hv.Scatter3D(
                df,
                kdims=['x_center', 'y_center', 'z_center'],
                vdims='density'
            ).opts(
                color='density',
                cmap=self.color_scale,
                size='density',
                width=450,
                height=450,
                colorbar=True,
                title=f'Spatial Density (bin={bin_size})',
                xlim=self.x_range,
                ylim=self.y_range,
                zlim=self.z_range
            )
        else:
            # 2D heatmap using QuadMesh for proper gridded visualization
            # QuadMesh expects (x, y, z) where x and y are bin edges/centers
            # Create a proper 2D grid from the binned data

            # Reshape data into grid format for QuadMesh
            x_unique = sorted(df['x_center'].unique())
            y_unique = sorted(df['y_center'].unique())

            # Create a 2D array for density values
            density_grid = np.zeros((len(y_unique), len(x_unique)))
            for _, row in df.iterrows():
                x_idx = x_unique.index(row['x_center'])
                y_idx = y_unique.index(row['y_center'])
                density_grid[y_idx, x_idx] = row['density']

            # Create QuadMesh with proper bounds
            viz = hv.QuadMesh((x_unique, y_unique, density_grid)).opts(
                cmap=self.color_scale,
                width=450,
                height=450,
                colorbar=True,
                title=f'Spatial Density (bin={bin_size})',
                tools=['hover'],
                xlim=self.x_range,
                ylim=self.y_range,
                backend='bokeh'
            )

        # Update pane content (efficient - just updates object, doesn't recreate widget)
        self.heatmap_viz_pane.object = viz

    def _update_timeseries_panel(self):
        """Update time series panel with selected properties."""
        if self.properties_ts_df is None or len(self.properties_ts_df) == 0:
            # Create empty bokeh plot
            p = figure(title="Time Series (no properties available)", width=800, height=300)
            self.timeseries_viz_pane.object = p
            return

        # Filter to selected properties
        if len(self.selected_properties) == 0:
            p = figure(title="Time Series (select properties to display)", width=800, height=300)
            self.timeseries_viz_pane.object = p
            return

        df = self.properties_ts_df[self.properties_ts_df['property_id'].isin(self.selected_properties)]

        if len(df) == 0:
            p = figure(title="Time Series (no data for selected properties)", width=800, height=300)
            self.timeseries_viz_pane.object = p
            return

        # Create bokeh plot with multiple lines
        p = figure(
            title="Property Time Series",
            x_axis_label="Time Window",
            y_axis_label="Value",
            width=800,
            height=300,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        for i, prop_id in enumerate(self.selected_properties):
            prop_df = df[df['property_id'] == prop_id].sort_values('time_window')

            if len(prop_df) == 0:
                continue

            color = Category10_10[i % len(Category10_10)]

            # Plot mean line
            p.line(
                prop_df['time_window'],
                prop_df['avg_value'],
                legend_label=prop_id,
                color=color,
                line_width=2
            )

            # Add shaded band for std (if available and not NaN)
            if 'std_value' in prop_df.columns and prop_df['std_value'].notna().any():
                upper = prop_df['avg_value'] + prop_df['std_value']
                lower = prop_df['avg_value'] - prop_df['std_value']

                p.varea(
                    x=prop_df['time_window'],
                    y1=lower,
                    y2=upper,
                    alpha=0.2,
                    color=color
                )

        # Add vertical line at current time
        vline = Span(
            location=self.current_time,
            dimension='height',
            line_color='red',
            line_width=2,
            line_dash='dashed'
        )
        p.add_layout(vline)

        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

        # Update pane content (efficient - just updates object, doesn't recreate widget)
        self.timeseries_viz_pane.object = p

    def _update_histogram_panel(self):
        """Update histogram panel with selected properties."""
        if self.properties_dist_df is None or len(self.properties_dist_df) == 0:
            self.histogram_viz_pane.object = None
            return

        if len(self.selected_properties) == 0:
            self.histogram_viz_pane.object = None
            return

        # Filter to selected properties
        df = self.properties_dist_df[self.properties_dist_df['property_id'].isin(self.selected_properties)]

        if len(df) == 0:
            self.histogram_viz_pane.object = None
            return

        # Create histogram for each property
        histograms = []
        for prop_id in self.selected_properties:
            prop_df = df[df['property_id'] == prop_id]

            if len(prop_df) == 0:
                continue

            # Create histogram using HoloViews
            hist = hv.Histogram(
                np.histogram(prop_df['value_float'], bins=30)
            ).opts(
                title=prop_id,
                xlabel='Value',
                ylabel='Count',
                width=300,
                height=200
            )

            histograms.append(hist)

        # Update pane content (efficient - just updates object, doesn't recreate widget)
        if histograms:
            # Layout histograms in a row
            layout = hv.Layout(histograms).cols(len(histograms))
            self.histogram_viz_pane.object = layout
        else:
            self.histogram_viz_pane.object = None

    def _toggle_playback(self, event):
        """Toggle animation playback."""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_button.name = "⏸ Pause"
            self.play_button.button_type = "warning"
            # Start playback loop using periodic callback
            self._start_playback()
        else:
            self.play_button.name = "▶ Play"
            self.play_button.button_type = "success"
            # Stop playback loop
            self._stop_playback()

    def _start_playback(self):
        """Start periodic playback updates."""
        # Always stop any existing callback first to prevent pile-up
        self._stop_playback()

        if self.tracks_df is not None:
            # Calculate period in milliseconds based on playback_speed
            # playback_speed = frames per second, so period = 1000ms / fps
            period = int(1000 / self.playback_speed)
            self._playback_callback = pn.state.add_periodic_callback(
                self._advance_frame,
                period=period
            )

    def _stop_playback(self):
        """Stop periodic playback updates."""
        if self._playback_callback is not None:
            try:
                self._playback_callback.stop()  # Call .stop() on the callback object
            except Exception as e:
                logger.warning(f"Error stopping playback callback: {e}")
            finally:
                self._playback_callback = None

    def _advance_frame(self):
        """Advance animation by one frame (called by periodic callback)."""
        # Safety check: stop callback if playback is no longer active
        if not self.is_playing or self.tracks_df is None:
            self._stop_playback()
            return

        # Advance current time
        max_time = int(self.tracks_df['time_index'].max())
        if self.current_time >= max_time:
            self.current_time = int(self.tracks_df['time_index'].min())
        else:
            self.current_time += 1

    @param.depends('playback_speed', watch=True)
    def _on_speed_change(self):
        """Handle playback speed changes (restart callback if playing)."""
        if self.is_playing:
            self._stop_playback()
            self._start_playback()

    @param.depends('current_time', watch=True)
    def _on_time_change(self):
        """Handle current time changes (from slider or playback)."""
        if self.tracks_df is not None:
            self._update_animation_panel()
            self._update_timeseries_panel()

    @param.depends('selected_properties', watch=True)
    def _on_properties_change(self):
        """Handle property selection changes."""
        if self.properties_ts_df is not None:
            self._update_timeseries_panel()
            self._update_histogram_panel()

    @param.depends('trail_length', watch=True)
    def _on_trail_length_change(self):
        """Handle trail length changes."""
        if self.tracks_df is not None:
            self._update_animation_panel()

    @param.depends('color_scale', watch=True)
    def _on_color_scale_change(self):
        """Handle color scale changes."""
        if self.tracks_df is not None:
            self._update_heatmap_panel()
