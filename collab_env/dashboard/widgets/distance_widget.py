"""
Distance statistics widget.

Displays distance to target and boundary over time.
"""

import logging
from typing import Optional

import param
import panel as pn
import holoviews as hv
from holoviews import opts

from .base_analysis_widget import BaseAnalysisWidget

logger = logging.getLogger(__name__)


class DistanceStatsWidget(BaseAnalysisWidget):
    """
    Distance statistics visualization.

    Displays two distance metrics over time:
    - Distance to target (orange line)
    - Distance to boundary (purple line)

    Uses time-windowed aggregation with window size from context.
    """

    widget_name = "Distances"
    widget_description = "Distance to target and boundary over time"
    widget_category = "spatial"

    # Widget-specific parameters
    show_target = param.Boolean(
        default=True,
        doc="Show distance to target"
    )

    show_boundary = param.Boolean(
        default=True,
        doc="Show distance to boundary"
    )

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create widget-specific controls for display options."""
        return pn.Column(
            "### Display Options",
            pn.widgets.Checkbox.from_param(
                self.param.show_target,
                name="Show Distance to Target"
            ),
            pn.widgets.Checkbox.from_param(
                self.param.show_boundary,
                name="Show Distance to Boundary"
            )
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create empty plot pane."""
        return pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=400),
            sizing_mode="stretch_both"
        )

    def load_data(self) -> None:
        """Load and visualize distance statistics."""
        curves = []

        # Query distance to target if requested
        if self.show_target:
            df_target = self.query_with_context('get_distance_to_target')

            if len(df_target) > 0:
                curve_target = hv.Curve(
                    df_target,
                    kdims='time_window',
                    vdims='avg_distance',
                    label='Distance to Target'
                )
                curve_target.opts(
                    opts.Curve(color='orange', line_width=2, backend='bokeh')
                )
                curves.append(curve_target)

        # Query distance to boundary if requested
        if self.show_boundary:
            df_boundary = self.query_with_context('get_distance_to_boundary')

            if len(df_boundary) > 0:
                curve_boundary = hv.Curve(
                    df_boundary,
                    kdims='time_window',
                    vdims='avg_distance',
                    label='Distance to Boundary'
                )
                curve_boundary.opts(
                    opts.Curve(color='purple', line_width=2, backend='bokeh')
                )
                curves.append(curve_boundary)

        if not curves:
            raise ValueError("No distance data found")

        # Create overlay plot
        overlay = hv.Overlay(curves)
        overlay.opts(
            opts.Overlay(
                width=700,
                height=400,
                xlabel='Time',
                ylabel='Distance',
                title=f'Distances Over Time (window={self.context.temporal_window_size})',
                tools=['hover'],
                legend_position='top_right',
                backend='bokeh'
            )
        )

        self.display_pane.object = overlay
        logger.info(f"Loaded distance stats ({len(curves)} metrics)")
