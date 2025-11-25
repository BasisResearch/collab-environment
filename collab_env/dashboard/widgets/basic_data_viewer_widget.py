"""
Basic Data Viewer widget - episode visualization with animation and heatmap.

Provides a 2-panel synchronized view combining:
- Animation panel: Animated 2D/3D scatter plot of agent tracks with trails
- Spatial heatmap panel: Density heatmap of agent positions

Both panels are synchronized by time window and current time.
"""

import logging
import json
from typing import Optional, List
from pathlib import Path

import param
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool, Span
from bokeh.plotting import figure
from bokeh.palettes import Category10_10, Category20_20

from collab_env.data.file_utils import get_project_root
from .base_analysis_widget import BaseAnalysisWidget
from .query_scope import ScopeType

logger = logging.getLogger(__name__)

# Initialize HoloViews extension (plotly for 3D support, bokeh for 2D)
hv.extension('plotly', 'bokeh')


class BasicDataViewerWidget(BaseAnalysisWidget):
    """
    Data viewer with animation (episode only) and spatial heatmap (episode or session).

    Features:
    - Animation of agent tracks with configurable playback and trails (episode scope only)
    - 2D/3D spatial density heatmap with configurable colormap (episode or session scope)
    - Synchronized time control across both panels

    Scope Support:
    - Episode scope: Both animation and heatmap available
    - Session scope: Only heatmap available (aggregated across all episodes)
    """

    widget_name = "Basic Data Viewer"
    widget_description = "Animated agent tracks (episode) and spatial density heatmap (episode or session)"
    widget_category = "visualization"

    # Playback controls
    current_time = param.Integer(default=0, bounds=(0, None), doc="Current playback time index")
    playback_speed = param.Number(default=1.0, bounds=(0.1, 10.0), doc="Default playback speed for modal viewer")
    trail_length = param.Integer(default=50, bounds=(0, 500), doc="Number of frames to show in trail")

    # Visualization options
    show_agent_ids = param.Boolean(default=True, doc="Show agent ID labels")
    color_scale = param.Selector(
        default='viridis',
        objects=['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
        doc="Color scale for heatmap"
    )

    def __init__(self, **params):
        # Initialize data storage
        self.tracks_df = None

        # UI components (will be created in _create_ui)
        self.animation_pane = None
        self.heatmap_pane = None
        self.play_button = None

        # Data dimensionality (detected from loaded data)
        self.is_3d = True  # Default to 3D, will be updated based on data

        # Animation modal - will hold the HTML overlay when shown
        self.animation_modal_pane = pn.Column(sizing_mode='stretch_both')

        super().__init__(**params)

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create playback controls and visualization options."""

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
            name="Colormap",
            width=150
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
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create 2-panel layout for animation and heatmap."""

        # Create persistent panes that we'll update via .object property
        # This avoids recreating widgets every frame (much more efficient)
        self.animation_viz_pane = pn.pane.HoloViews(
            sizing_mode="stretch_both",
            min_height=500
        )

        self.heatmap_viz_pane = pn.pane.HoloViews(
            sizing_mode="stretch_both",
            min_height=500
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

        # Layout: side-by-side panels
        return pn.Row(
            self.animation_pane,
            self.heatmap_pane,
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
        """Load data for animation and heatmap panels.

        Animation is only available for episode scope.
        Heatmap is available for both episode and session scope.
        """
        scope_type = self.context.scope.scope_type
        scope_label = "episode" if scope_type == ScopeType.EPISODE else "session"
        logger.info(f"Loading Basic Data Viewer for {scope_label} scope")

        if scope_type == ScopeType.EPISODE:
            # Episode scope: Load tracks for animation + heatmap
            logger.info("Loading episode tracks for animation and heatmap...")
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

            # Create color mapping for agent IDs (consistent colors across all frames)
            unique_agents = sorted(self.tracks_df['agent_id'].unique())
            self.agent_color_map = {agent_id: Category20_20[i % len(Category20_20)]
                                    for i, agent_id in enumerate(unique_agents)}

            logger.info(f"Loaded {len(self.tracks_df)} track observations for {self.tracks_df['agent_id'].nunique()} agents")
            logger.info(f"Data dimensionality: {'3D' if self.is_3d else '2D'}")
            logger.info(f"Spatial bounds - X: {self.x_range}, Y: {self.y_range}" + (f", Z: {self.z_range}" if self.is_3d else ""))

            # Update time slider bounds based on data
            min_time = int(self.tracks_df['time_index'].min())
            max_time = int(self.tracks_df['time_index'].max())
            self.param.current_time.bounds = (min_time, max_time)
            self.current_time = min_time

            logger.info(f"Time range: {min_time} to {max_time}")

            # Convert placeholder panes to proper visualization panes
            self._init_visualization_panes()

            # Restore animation pane structure (in case it was replaced by session scope)
            self.animation_pane.objects = [
                pn.pane.Markdown("**Animation Panel**"),
                self.animation_viz_pane
            ]

            # Update both panels
            self._update_animation_panel()
            self._update_heatmap_panel()

            logger.info("Basic Data Viewer loaded successfully (episode scope)")

        else:
            # Session scope: Only heatmap available (no animation tracks)
            logger.info("Loading session scope heatmap (animation not available)")

            # Clear track data (not available for session)
            self.tracks_df = None
            self.agent_color_map = None

            # Load heatmap data to determine spatial bounds and dimensionality
            heatmap_df = self.query_with_context('get_spatial_heatmap')

            if len(heatmap_df) == 0:
                raise ValueError("No heatmap data found for selected session")

            # Detect dimensionality from heatmap data
            # Check if z_bin has valid (non-NaN) values
            has_valid_z = heatmap_df['z_bin'].notna().any()
            if has_valid_z:
                z_variance = heatmap_df['z_bin'].std()
                z_range_span = heatmap_df['z_bin'].max() - heatmap_df['z_bin'].min()
                self.is_3d = (z_variance > 0.01) and (z_range_span > 0.01)
            else:
                self.is_3d = False

            # Compute spatial bounds from heatmap bins
            bin_size = self.context.spatial_bin_size
            self.x_range = (float(heatmap_df['x_bin'].min()), float(heatmap_df['x_bin'].max() + bin_size))
            self.y_range = (float(heatmap_df['y_bin'].min()), float(heatmap_df['y_bin'].max() + bin_size))

            if self.is_3d:
                self.z_range = (float(heatmap_df['z_bin'].min()), float(heatmap_df['z_bin'].max() + bin_size))
            else:
                self.z_range = None

            logger.info(f"Detected {'3D' if self.is_3d else '2D'} data for session heatmap")

            # Convert placeholder panes to proper visualization panes
            self._init_visualization_panes()

            # Update animation panel to show "not available" message
            self.animation_pane.objects = [
                pn.pane.Markdown(
                    "**Animation Panel**\n\n"
                    "_Animation is only available for episode-level analysis._\n\n"
                    "Switch to 'Episode' scope to view animated agent tracks.",
                    styles={'color': '#666', 'font-style': 'italic'}
                )
            ]

            # Restore heatmap pane structure
            self.heatmap_pane.objects = [
                pn.pane.Markdown("**Spatial Heatmap Panel**"),
                self.heatmap_viz_pane
            ]

            # Update heatmap panel
            self._update_heatmap_panel()

            logger.info("Basic Data Viewer loaded successfully (session scope - heatmap only)")

    def get_animation_data_json(self):
        """Return JSON-serializable animation data for client-side rendering.

        This method prepares all data needed for the client-side viewer.
        No HTTP requests needed - data is embedded directly in the modal HTML.
        """
        if self.tracks_df is None:
            return {}

        # Convert agent_color_map keys to native Python int (from numpy.int64)
        agent_colors_serializable = {
            int(agent_id): color
            for agent_id, color in self.agent_color_map.items()
        }

        return {
            'tracks': self.tracks_df.to_dict('records'),  # List of all track points
            'settings': {
                'trail_length': int(self.trail_length),
                'show_agent_ids': bool(self.show_agent_ids),
                'playback_speed': float(self.playback_speed),
            },
            'bounds': {
                'x_range': [float(self.x_range[0]), float(self.x_range[1])],
                'y_range': [float(self.y_range[0]), float(self.y_range[1])],
                'z_range': [float(self.z_range[0]), float(self.z_range[1])] if self.z_range else None,
            },
            'is_3d': bool(self.is_3d),
            'agent_colors': agent_colors_serializable,
            'time_range': [
                int(self.tracks_df['time_index'].min()),
                int(self.tracks_df['time_index'].max())
            ]
        }

    def _init_visualization_panes(self):
        """Remove loading messages, keep visualization panes."""
        # Remove loading markdown messages (first element in each Column)
        # The viz panes are persistent and will be updated via .object property
        if len(self.animation_pane.objects) > 1:
            self.animation_pane.pop(0)  # Remove markdown
        if len(self.heatmap_pane.objects) > 1:
            self.heatmap_pane.pop(0)

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
                # For 3D Plotly plots, map agent IDs to actual color hex values
                current_df = current_df.copy()
                current_df['agent_color'] = current_df['agent_id'].map(self.agent_color_map)

                scatter = hv.Scatter3D(
                    current_df,
                    kdims=['x', 'y', 'z'],
                    vdims=['agent_id', 'agent_color', 'speed']
                ).opts(
                    color='agent_color',
                    size=5,
                    width=450,
                    height=450,
                    title=f'Agent Positions (t={self.current_time})',
                    colorbar=False,
                    xlim=self.x_range,
                    ylim=self.y_range,
                    zlim=self.z_range
                )
            else:
                scatter = hv.Scatter3D([]).opts(
                    width=450,
                    height=450,
                    title='Animation',
                    xlim=self.x_range,
                    ylim=self.y_range,
                    zlim=self.z_range
                )

            # Add trails if trail_length > 0
            if self.trail_length > 0 and len(window_df) > 0:
                paths = []
                for agent_id in window_df['agent_id'].unique():
                    agent_trail = window_df[window_df['agent_id'] == agent_id].sort_values('time_index')
                    if len(agent_trail) > 1:
                        # Ensure continuous trail (no gaps/jumps in time)
                        # Only include consecutive time points to avoid "looping back"
                        time_indices = agent_trail['time_index'].values
                        continuous_mask = np.ones(len(time_indices), dtype=bool)
                        continuous_mask[1:] = (np.diff(time_indices) == 1)

                        # Find continuous segments
                        segment_starts = np.where(~continuous_mask)[0]
                        if len(segment_starts) > 0:
                            # Only use the most recent continuous segment
                            last_break = segment_starts[-1]
                            agent_trail = agent_trail.iloc[last_break:]

                        if len(agent_trail) > 1:
                            agent_color = self.agent_color_map.get(agent_id, 'gray')
                            path = hv.Path3D(
                                agent_trail,
                                kdims=['x', 'y', 'z']
                            ).opts(color=agent_color, line_width=2)
                            paths.append(path)

                if paths:
                    trails = hv.Overlay(paths)
                    scatter = trails * scatter
        else:
            # 2D visualization (use PLOTLY backend for consistency)
            if len(current_df) > 0:
                # Map agent IDs to actual color hex values (same as 3D approach)
                current_df = current_df.copy()
                current_df['agent_color'] = current_df['agent_id'].map(self.agent_color_map)

                scatter = hv.Scatter(
                    current_df,
                    kdims=['x', 'y'],
                    vdims=['agent_id', 'agent_color', 'speed']
                ).opts(
                    color='agent_color',
                    size=8,
                    # width=450,
                    # height=450,
                    title=f'Agent Positions (t={self.current_time})',
                    colorbar=False,
                    xlim=self.x_range,
                    ylim=self.y_range
                )
            else:
                scatter = hv.Scatter([]).opts(
                    # width=450,
                    # height=450,
                    title='Animation',
                    xlim=self.x_range,
                    ylim=self.y_range
                )

            # Add trails if trail_length > 0
            if self.trail_length > 0 and len(window_df) > 0:
                paths = []
                for agent_id in window_df['agent_id'].unique():
                    agent_trail = window_df[window_df['agent_id'] == agent_id].sort_values('time_index')
                    if len(agent_trail) > 1:
                        # Ensure continuous trail (no gaps/jumps in time)
                        # Only include consecutive time points to avoid "looping back"
                        time_indices = agent_trail['time_index'].values
                        continuous_mask = np.ones(len(time_indices), dtype=bool)
                        continuous_mask[1:] = (np.diff(time_indices) == 1)

                        # Find continuous segments
                        segment_starts = np.where(~continuous_mask)[0]
                        if len(segment_starts) > 0:
                            # Only use the most recent continuous segment
                            last_break = segment_starts[-1]
                            agent_trail = agent_trail.iloc[last_break:]

                        if len(agent_trail) > 1:
                            agent_color = self.agent_color_map.get(agent_id, 'gray')
                            path = hv.Path(
                                agent_trail,
                                kdims=['x', 'y']
                            ).opts(color=agent_color, line_width=2)
                            paths.append(path)

                if paths:
                    trails = hv.Overlay(paths)
                    scatter = trails * scatter

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

        if self.is_3d:
            df['z_center'] = df['z_bin'] + (bin_size / 2)

            # 3D scatter plot with density (PLOTLY)
            # Use fixed moderate opacity to help with overlapping points
            viz = hv.Scatter3D(
                df,
                kdims=['x_center', 'y_center', 'z_center'],
                vdims='density'
            ).opts(
                color='density',
                cmap=self.color_scale,
                size=5,
                alpha=0.6,
                # width=450,
                # height=450,
                colorbar=True,
                title=f'Spatial Density (bin={bin_size})',
                xlim=self.x_range,
                ylim=self.y_range,
                zlim=self.z_range
            )
        else:
            # 2D heatmap using Image (PLOTLY for consistency)
            # Create a proper 2D grid from the binned data
            x_unique = sorted(df['x_center'].unique())
            y_unique = sorted(df['y_center'].unique())

            # Create a 2D array for density values
            density_grid = np.zeros((len(y_unique), len(x_unique)))
            for _, row in df.iterrows():
                x_idx = x_unique.index(row['x_center'])
                y_idx = y_unique.index(row['y_center'])
                density_grid[y_idx, x_idx] = row['density']

            # Create Image with proper bounds
            # Flip y axis for heatmap (so y goes from max to min, as in image coordinates)
            viz = hv.Image(
                np.flipud(density_grid),
                bounds=(min(x_unique), max(y_unique), max(x_unique), min(y_unique))  # flip y axis
            ).opts(
                cmap=self.color_scale,
                # width=450,
                # height=450,
                colorbar=True,
                title=f'Spatial Density (bin={bin_size})',
                xlim=self.x_range,
                ylim=self.y_range
            )

        # Update pane content (efficient - just updates object, doesn't recreate widget)
        self.heatmap_viz_pane.object = viz

    def _toggle_playback(self, event):
        """Open modal dialog for client-side animation."""
        if self.tracks_df is None:
            logger.warning("No track data available for animation")
            return

        # Generate animation data from in-memory tracks
        animation_data = self.get_animation_data_json()

        # Create and show modal
        self._show_animation_modal(animation_data)

    def _show_animation_modal(self, animation_data):
        """Create and display full-screen animation modal with client-side playback.

        The modal is injected directly into document.body via JavaScript, creating a
        true overlay that works outside of Panel's component tree. All track data is
        embedded as JSON in the HTML, enabling smooth 60fps client-side animation
        with no server round-trips during playback.

        Features:
        - Canvas 2D rendering with requestAnimationFrame for smooth animation
        - Playback controls: play/pause, stop, speed (0.1x-10x)
        - Keyboard shortcuts: Space (play/pause), Escape (close)
        - Dynamic frame indicator and speed slider
        - Close button to remove modal from DOM
        """
        # Validate data
        if not animation_data or not animation_data.get('tracks'):
            logger.warning("No track data available for animation")
            return

        # Check data size and warn if large
        num_points = len(animation_data['tracks'])
        if num_points > 10000:
            logger.warning(f"Large dataset: {num_points} track points - animation may be slow")

        # Serialize data to JSON string for embedding
        try:
            animation_json = json.dumps(animation_data)
        except Exception as e:
            logger.error(f"Failed to serialize animation data: {e}")
            return

        # Load template from file using project root for robust path resolution
        template_path = get_project_root() / 'collab_env' / 'dashboard' / 'templates' / 'animation_modal.html'
        try:
            template_content = template_path.read_text()
        except Exception as e:
            logger.error(f"Failed to load animation modal template: {e}")
            return

        # Render template with variables
        html_content = template_content.format(
            animation_json=animation_json,
            playback_speed=animation_data['settings']['playback_speed'],
            max_frame=animation_data['time_range'][1]
        )

        # Create HTML pane - let it size naturally
        viewer_pane = pn.pane.HTML(html_content, sizing_mode='stretch_both')

        # Add to modal pane (which is part of the widget layout)
        self.animation_modal_pane.clear()
        self.animation_modal_pane.append(viewer_pane)

        logger.info("Animation modal displayed")

    @param.depends('current_time', watch=True)
    def _on_time_change(self):
        """Handle current time changes (from slider or playback)."""
        if self.tracks_df is not None:
            self._update_animation_panel()

    @param.depends('trail_length', watch=True)
    def _on_trail_length_change(self):
        """Handle trail length changes."""
        if self.tracks_df is not None:
            self._update_animation_panel()

    @param.depends('color_scale', watch=True)
    def _on_color_scale_change(self):
        """Handle color scale changes."""
        if self.tracks_df is not None:
            # Force re-render by clearing the object first
            self.heatmap_viz_pane.object = None
            self._update_heatmap_panel()

    def get_tab_content(self) -> pn.Column:
        """
        Override to add modal pane to component tree.

        Modal overlay will be injected here when Play is clicked.
        """
        # Get base components from parent class
        base_content = super().get_tab_content()

        # Add modal pane to the column (empty by default, filled when Play is clicked)
        base_content.append(self.animation_modal_pane)

        return base_content
