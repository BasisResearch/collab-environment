"""
Base class for analysis widgets.

Provides common infrastructure for widget lifecycle, error handling,
and interaction with the analysis context.
"""

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
    widget_name: str = ""              # Display name for tab
    widget_description: str = ""       # Widget description
    widget_category: str = "general"   # For grouping/filtering

    # Shared context (injected by main GUI)
    context: Optional[AnalysisContext] = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)
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
        # Load button (standard for all widgets)
        self.load_btn = pn.widgets.Button(
            name=f"Load {self.widget_name}",
            button_type="primary",
            width=200
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

            self.load_data()

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

    def get_tab_content(self) -> pn.Column:
        """
        Return complete tab content (controls + display).

        Layout:
        - Load button
        - Custom controls (if any)
        - Display pane

        Returns
        -------
        pn.Column
            Complete widget content for tab
        """
        components = [self.load_btn]

        if self.custom_controls:
            components.append(pn.layout.Divider())
            components.append(self.custom_controls)

        components.append(self.display_pane)

        return pn.Column(*components, sizing_mode="stretch_both")

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
        query_fn = getattr(self.context.query_backend, query_method)
        params = self.context.get_query_params(**extra_params)
        return query_fn(**params)

