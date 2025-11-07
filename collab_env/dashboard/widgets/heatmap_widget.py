"""
3D spatial density heatmap widget.

Visualizes spatial occupancy as a 3D scatter plot with density encoding.
"""

import logging
from typing import Optional

import param
import panel as pn
import holoviews as hv

from .base_analysis_widget import BaseAnalysisWidget

logger = logging.getLogger(__name__)


class HeatmapWidget(BaseAnalysisWidget):
    """
    3D spatial density heatmap visualization.

    Displays spatial occupancy using a 3D scatter plot where:
    - Position: 3D bin centers (x, y, z)
    - Color: Density (number of observations)
    - Size: Density (scaled for visibility)

    Uses shared `spatial_bin_size` from context for binning.
    """

    widget_name = "Heatmap"
    widget_description = "3D spatial density visualization"
    widget_category = "spatial"

    # Widget-specific parameters
    color_scale = param.Selector(
        default='viridis',
        objects=['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
        doc="Color scale for density"
    )

    show_grid = param.Boolean(default=True, doc="Show grid lines")

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create widget-specific controls for visualization options."""
        return pn.Column(
            "### Visualization Options",
            pn.widgets.Select.from_param(
                self.param.color_scale,
                name="Color Scale",
                width=200
            ),
            pn.widgets.Checkbox.from_param(
                self.param.show_grid,
                name="Show Grid"
            )
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create empty 3D plot pane."""
        return pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=500),
            sizing_mode="stretch_both"
        )

    def load_data(self) -> None:
        """Load and visualize 3D spatial heatmap."""
        # Query using shared context parameters
        # bin_size comes from context.spatial_bin_size
        df = self.query_with_context('get_spatial_heatmap')

        if len(df) == 0:
            raise ValueError("No data found for selected parameters")

        # Create 3D scatter plot with density as color and size
        scatter = hv.Scatter3D(
            df,
            kdims=['x_bin', 'y_bin', 'z_bin'],
            vdims='density'
        ).opts(
            color='density',
            cmap=self.color_scale,  # Widget-specific parameter
            size='density',
            width=800,
            height=600,
            colorbar=True,
            show_grid=self.show_grid,  # Widget-specific parameter
            title=f'3D Spatial Density (bin size={self.context.spatial_bin_size})'
        )

        self.display_pane.object = scatter
        logger.info(f"Loaded 3D heatmap with {len(df)} bins")
