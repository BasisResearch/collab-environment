"""
Velocity statistics widget.

Displays speed statistics over time with error bands.
"""

import logging
from typing import Optional

import param
import panel as pn
import holoviews as hv
from holoviews import opts

from .base_analysis_widget import BaseAnalysisWidget

logger = logging.getLogger(__name__)


class VelocityStatsWidget(BaseAnalysisWidget):
    """
    Speed statistics visualization over time.

    Displays mean speed with optional standard deviation bands.
    Uses time-windowed aggregation with window size from context.
    """

    widget_name = "Velocity Stats"
    widget_description = "Speed statistics over time"
    widget_category = "temporal"

    # Widget-specific parameters
    show_std = param.Boolean(
        default=True,
        doc="Show standard deviation as shaded area"
    )

    line_color = param.Selector(
        default='blue',
        objects=['blue', 'green', 'red', 'orange', 'purple'],
        doc="Line color"
    )

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create widget-specific controls for visualization options."""
        return pn.Column(
            "### Visualization Options",
            pn.widgets.Checkbox.from_param(
                self.param.show_std,
                name="Show Std Deviation"
            ),
            pn.widgets.Select.from_param(
                self.param.line_color,
                name="Line Color",
                width=200
            )
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create empty plot pane."""
        return pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=400),
            sizing_mode="stretch_both"
        )

    def load_data(self) -> None:
        """Load and visualize velocity statistics."""
        # Query speed statistics using shared temporal_window_size
        df = self.query_with_context('get_speed_statistics')

        if len(df) == 0:
            raise ValueError("No data found for selected parameters")

        # Create line plot with mean speed
        curve = hv.Curve(
            df,
            kdims='time_window',
            vdims='avg_speed',
            label='Mean Speed'
        )
        curve.opts(
            opts.Curve(
                color=self.line_color,  # Widget-specific parameter
                line_width=2,
                width=700,
                height=400,
                xlabel='Time',
                ylabel='Speed',
                title=f'Speed Over Time (window={self.context.temporal_window_size})',
                tools=['hover'],
                backend='bokeh'
            )
        )

        # Add std deviation as error bands if available and requested
        if self.show_std and 'std_speed' in df.columns:
            df['upper'] = df['avg_speed'] + df['std_speed']
            df['lower'] = df['avg_speed'] - df['std_speed']
            area = hv.Area(df, kdims='time_window', vdims=['lower', 'upper'])
            area.opts(opts.Area(alpha=0.3, color=self.line_color, backend='bokeh'))
            plot = curve * area
        else:
            plot = curve

        self.display_pane.object = plot
        logger.info(f"Loaded velocity stats with {len(df)} windows")
