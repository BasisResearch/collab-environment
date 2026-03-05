"""
Base class for analysis widgets.

Provides common infrastructure for widget lifecycle, error handling,
and interaction with the analysis context.
"""

import json
from html import escape
from typing import Optional
import logging

import param
import panel as pn
import pandas as pd

from .analysis_context import AnalysisContext
from .query_scope import ScopeType

logger = logging.getLogger(__name__)


class BaseAnalysisWidget(param.Parameterized):
    """
    Abstract base class for analysis widgets.

    Subclasses must implement:
    - create_custom_controls(): Widget-specific parameter controls
    - create_display_pane(): Visualization pane
    - load_data(): Query and visualization logic

    The base class provides:
    - Load button with error handling
    - Context validation
    - Helper method for querying with context
    - Tab content layout

    Examples
    --------
    >>> class MyWidget(BaseAnalysisWidget):
    ...     widget_name = "My Analysis"
    ...     widget_category = "custom"
    ...
    ...     threshold = param.Number(default=0.5)
    ...
    ...     def create_custom_controls(self):
    ...         return pn.Column(
    ...             pn.widgets.FloatSlider.from_param(self.param.threshold)
    ...         )
    ...
    ...     def create_display_pane(self):
    ...         return pn.pane.HoloViews(hv.Curve([]))
    ...
    ...     def load_data(self):
    ...         df = self.query_with_context('get_my_data')
    ...         self.display_pane.object = hv.Curve(df)
    """

    # Metadata (subclasses should override)
    # Note: Using widget_name instead of name to avoid conflict with param.Parameterized.name
    widget_name: str = ""  # Display name for tab
    widget_description: str = ""  # Widget description
    widget_category: str = "general"  # For grouping/filtering

    # Shared context (injected by main GUI)
    context: Optional[AnalysisContext] = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        self._load_total_rows = 0
        self._load_agents = set()
        self._create_ui()

    # ========== Abstract methods (must implement) ==========

    def create_custom_controls(self) -> Optional[pn.Column]:
        """
        Create widget-specific parameter controls.

        These are controls that are NOT in the shared context.
        Return None if no custom controls needed.

        Returns
        -------
        pn.Column or None
            Column of custom controls, or None

        Examples
        --------
        >>> def create_custom_controls(self):
        ...     return pn.Column(
        ...         "### Custom Parameters",
        ...         pn.widgets.Select.from_param(self.param.color_scale),
        ...         pn.widgets.Checkbox.from_param(self.param.show_grid)
        ...     )
        """
        raise NotImplementedError("Subclasses must implement create_custom_controls()")

    def create_display_pane(self) -> pn.pane.PaneBase:
        """
        Create the visualization pane (empty state).

        Returns the pane that will be updated by load_data().

        Returns
        -------
        pn.pane.PaneBase
            Empty visualization pane

        Examples
        --------
        >>> def create_display_pane(self):
        ...     return pn.pane.HoloViews(
        ...         hv.Curve([]).opts(width=700, height=500),
        ...         sizing_mode="stretch_both"
        ...     )
        """
        raise NotImplementedError("Subclasses must implement create_display_pane()")

    def load_data(self) -> None:
        """
        Query data using self.context and update visualization.

        Access shared parameters via:
        - self.context.scope (QueryScope)
        - self.context.spatial_bin_size
        - self.context.temporal_window_size
        - etc.

        Access widget-specific parameters via self attributes.

        Should update self.display_pane.object with new visualization.

        Raises
        ------
        ValueError
            If no data found or invalid parameters

        Examples
        --------
        >>> def load_data(self):
        ...     # Query using shared context parameters
        ...     df = self.query_with_context('get_spatial_heatmap')
        ...
        ...     if len(df) == 0:
        ...         raise ValueError("No data found")
        ...
        ...     # Create visualization
        ...     scatter = hv.Scatter3D(df, kdims=['x', 'y', 'z'], vdims='density')
        ...     self.display_pane.object = scatter
        """
        raise NotImplementedError("Subclasses must implement load_data()")

    # ========== Concrete methods (provided by base) ==========

    def _create_ui(self):
        """Create UI components (called by __init__)."""
        # Scope display (shows current session/episode)
        self.scope_display = pn.pane.Markdown(
            "**No data loaded**",
            sizing_mode="stretch_width",
            styles={"background": "#f0f0f0", "padding": "10px", "border-radius": "5px"},
        )

        # Info button for config/metadata popup
        self.info_btn = pn.widgets.Button(
            name="Info", button_type="light", width=70, height=35
        )
        self.info_btn.on_click(self._toggle_info_popup)

        # Info popup (hidden by default)
        self._info_content = pn.pane.HTML("", sizing_mode="stretch_width")
        self._info_close_btn = pn.widgets.Button(
            name="Close", button_type="light", width=70
        )
        self._info_close_btn.on_click(lambda e: self._hide_info_popup())

        self.info_popup = pn.Column(
            pn.Row(
                pn.pane.Markdown("### Session & Episode Info"),
                self._info_close_btn,
                sizing_mode="stretch_width",
            ),
            self._info_content,
            visible=False,
            sizing_mode="stretch_width",
            styles={
                "background": "white",
                "border": "2px solid #2596be",
                "border-radius": "8px",
                "padding": "15px",
                "box-shadow": "0 4px 12px rgba(0, 0, 0, 0.15)",
                "margin": "10px 0",
            },
        )

        # Load button (standard for all widgets)
        self.load_btn = pn.widgets.Button(
            name=f"Load {self.widget_name}", button_type="primary", width=200
        )
        self.load_btn.on_click(self._on_load_click)

        # Display pane (subclass creates)
        self.display_pane = self.create_display_pane()

        # Custom controls (subclass creates)
        self.custom_controls = self.create_custom_controls()

    def _on_load_click(self, event):
        """Handle load button click (with error handling)."""
        if not self._validate_context():
            return

        try:
            self.context.report_loading(f"Loading {self.widget_name}...")

            # Reset counters before load (tracked by query_with_context)
            self._load_total_rows = 0
            self._load_agents = set()

            self.load_data()

            # Update scope display after successful load
            self._update_scope_display()

            self.context.report_success(f"{self.widget_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load {self.widget_name}: {e}", exc_info=True)
            self.context.report_error(f"Failed to load {self.widget_name}: {e}")

    def _validate_context(self) -> bool:
        """
        Validate that context has required data scope.

        Returns
        -------
        bool
            True if context is valid, False otherwise
        """
        if not self.context:
            logger.error("No context set")
            return False

        scope = self.context.scope

        if scope.scope_type == ScopeType.EPISODE and not scope.episode_id:
            self.context.report_error("Please select an episode first")
            return False

        if scope.scope_type == ScopeType.SESSION and not scope.session_id:
            self.context.report_error("Please select a session first")
            return False

        return True

    def _update_scope_display(self):
        """Update the scope display with current session/episode information."""
        if self.context and self.context.scope:
            scope = self.context.scope
            scope_str = str(scope)

            # Use counts tracked during load_data via query_with_context
            num_rows = getattr(self, "_load_total_rows", 0)
            num_agents = len(getattr(self, "_load_agents", set()))
            if num_rows > 0:
                scope_str += (
                    f" | {num_agents} agents, {num_rows:,} data points"
                )

            self.scope_display.object = f"**Current Scope:** {scope_str}"
        else:
            self.scope_display.object = "**No data loaded**"

    def get_tab_content(self) -> pn.Column:
        """
        Return complete tab content (controls + display).

        Layout:
        - Scope display with info button (current session/episode)
        - Info popup (hidden by default)
        - Load button
        - Custom controls (if any)
        - Display pane

        Returns
        -------
        pn.Column
            Complete widget content for tab
        """
        scope_row = pn.Row(
            self.scope_display,
            self.info_btn,
            sizing_mode="stretch_width",
            align="center",
        )
        components = [scope_row, self.info_popup, self.load_btn]

        if self.custom_controls:
            components.append(pn.layout.Divider())
            components.append(self.custom_controls)

        components.append(self.display_pane)

        return pn.Column(*components, sizing_mode="stretch_both")

    # ========== Info popup methods ==========

    def _toggle_info_popup(self, event=None):
        """Toggle the config/metadata info popup."""
        if self.info_popup.visible:
            self._hide_info_popup()
        else:
            self._show_info_popup()

    def _show_info_popup(self):
        """Show popup with session config and metadata JSONs."""
        if not self.context or not self.context.scope:
            return

        scope = self.context.scope
        config_json = ""
        metadata_json = ""

        try:
            if scope.scope_type == ScopeType.EPISODE and scope.episode_id:
                meta_df = self.context.query_backend.get_episode_metadata(
                    scope.episode_id
                )
                if len(meta_df) > 0:
                    row = meta_df.iloc[0]
                    config_data = row.get("config", {})
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)
                    config_json = json.dumps(config_data, indent=2, default=str)

                    metadata = {
                        k: row.get(k)
                        for k in [
                            "episode_id",
                            "session_id",
                            "session_name",
                            "category_id",
                            "episode_number",
                            "num_frames",
                            "num_agents",
                            "frame_rate",
                            "file_path",
                        ]
                    }
                    metadata_json = json.dumps(metadata, indent=2, default=str)

            elif scope.scope_type == ScopeType.SESSION and scope.session_id:
                sessions_df = self.context.query_backend.get_sessions()
                session_row = sessions_df[
                    sessions_df["session_id"] == scope.session_id
                ]
                if len(session_row) > 0:
                    row = session_row.iloc[0]
                    config_data = row.get("config", {})
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)
                    config_json = json.dumps(config_data, indent=2, default=str)

                    metadata = {
                        k: row.get(k)
                        for k in [
                            "session_id",
                            "session_name",
                            "category_id",
                            "created_at",
                        ]
                    }
                    metadata_json = json.dumps(metadata, indent=2, default=str)

                # Also include episode summary
                episodes_df = self.context.query_backend.get_episodes(
                    scope.session_id
                )
                if len(episodes_df) > 0:
                    episodes_summary = episodes_df[
                        ["episode_id", "episode_number", "num_frames", "num_agents"]
                    ].to_dict("records")
                    base = json.loads(metadata_json) if metadata_json else {}
                    base["episodes"] = episodes_summary
                    metadata_json = json.dumps(base, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to load info: {e}")
            config_json = config_json or f"Error: {e}"
            metadata_json = metadata_json or f"Error: {e}"

        pre_style = (
            "background: #f8f8f8; padding: 10px; border-radius: 4px; "
            "overflow-x: auto; max-height: 400px; overflow-y: auto; "
            "font-size: 12px; white-space: pre-wrap; word-break: break-word;"
        )
        html = (
            f"<div>"
            f"<h4>Config</h4>"
            f"<pre style='{pre_style}'>{escape(config_json)}</pre>"
            f"<h4>Metadata</h4>"
            f"<pre style='{pre_style}'>{escape(metadata_json)}</pre>"
            f"</div>"
        )
        self._info_content.object = html
        self.info_popup.visible = True

    def _hide_info_popup(self):
        """Hide the config/metadata info popup."""
        self.info_popup.visible = False

    # ========== Helper methods ==========

    def query_with_context(self, query_method: str, **extra_params) -> pd.DataFrame:
        """
        Helper to query backend with merged parameters.

        Supports both episode-level and session-level queries.
        Session aggregation is handled at the SQL level in QueryBackend.

        Merges context parameters (scope + shared) with widget-specific
        parameters and calls the specified query method.

        Parameters
        ----------
        query_method : str
            Name of QueryBackend method to call
        **extra_params
            Widget-specific parameters to add/override

        Returns
        -------
        pd.DataFrame
            Query results

        Examples
        --------
        >>> # Use shared parameters only
        >>> df = self.query_with_context('get_spatial_heatmap')

        >>> # Override specific parameter
        >>> df = self.query_with_context(
        ...     'get_spatial_heatmap',
        ...     bin_size=5.0  # Override shared bin size
        ... )

        >>> # Add widget-specific parameter
        >>> df = self.query_with_context(
        ...     'get_velocity_correlations',
        ...     method=self.correlation_method
        ... )
        """
        assert self.context is not None
        query_fn = getattr(self.context.query_backend, query_method)
        params = self.context.get_query_params(**extra_params)
        result = query_fn(**params)

        # Track loaded data counts
        self._load_total_rows += len(result)
        if "agent_id" in result.columns:
            self._load_agents.update(result["agent_id"].unique())

        return result
