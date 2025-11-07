"""
Velocity correlation widget.

Displays agent velocity correlations as 3D scatter plot.
"""

import logging
from typing import Optional

import param
import panel as pn
import holoviews as hv

from .base_analysis_widget import BaseAnalysisWidget
from .query_scope import ScopeType

logger = logging.getLogger(__name__)


class CorrelationWidget(BaseAnalysisWidget):
    """
    Velocity correlation visualization.

    Displays pairwise agent velocity correlations as 3D scatter:
    - X axis: Agent i index
    - Y axis: Agent j index
    - Z axis: Average correlation magnitude
    - Color: Correlation strength
    """

    widget_name = "Correlations"
    widget_description = "Agent velocity correlations"
    widget_category = "behavioral"

    # Widget-specific parameters
    correlation_method = param.Selector(
        default='pearson',
        objects=['pearson', 'spearman', 'kendall'],
        doc="Correlation method (future use)"
    )

    min_correlation = param.Number(
        default=0.0,
        bounds=(0.0, 1.0),
        doc="Minimum correlation magnitude to display"
    )

    marker_size = param.Integer(
        default=8,
        bounds=(1, 20),
        doc="Marker size for scatter points"
    )

    color_map = param.Selector(
        default='Viridis',
        objects=['Viridis', 'Plasma', 'Inferno', 'Magma', 'RdYlBu'],
        doc="Color map for correlation values"
    )

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create widget-specific controls for correlation parameters."""
        return pn.Column(
            "### Correlation Parameters",
            pn.widgets.Select.from_param(
                self.param.correlation_method,
                name="Method",
                width=200
            ),
            pn.widgets.FloatSlider.from_param(
                self.param.min_correlation,
                name="Min Correlation",
                width=200
            ),
            pn.layout.Divider(),
            "### Visualization",
            pn.widgets.IntSlider.from_param(
                self.param.marker_size,
                name="Marker Size",
                width=200
            ),
            pn.widgets.Select.from_param(
                self.param.color_map,
                name="Color Map",
                width=200
            )
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create empty 3D plot pane."""
        return pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=500),
            sizing_mode="stretch_both"
        )

    def load_data(self) -> None:
        """Load and visualize velocity correlations."""
        # Check if scope is SESSION - correlations only support episode-level
        if self.context.scope.scope_type == ScopeType.SESSION:
            raise ValueError(
                "Session-level correlation is not supported. "
                "Correlation analysis is only available for single episodes. "
                "Please select an episode scope instead."
            )

        # Query velocity correlations
        # min_samples comes from context, but we can override if needed
        df = self.query_with_context('get_velocity_correlations')

        logger.info(f"Query returned {len(df)} correlation pairs")

        if len(df) == 0:
            raise ValueError(
                "No correlation data found. Try adjusting time range or agent type."
            )

        # Calculate average correlation magnitude across all dimensions
        df['avg_correlation'] = df[
            ['v_x_correlation', 'v_y_correlation', 'v_z_correlation']
        ].abs().mean(axis=1)

        # Filter by minimum correlation threshold
        df_filtered = df[df['avg_correlation'] >= self.min_correlation]

        if len(df_filtered) == 0:
            raise ValueError(
                f"No correlations above threshold {self.min_correlation}. "
                f"Total pairs: {len(df)}, max correlation: {df['avg_correlation'].max():.3f}"
            )

        logger.info(
            f"Correlation range: {df_filtered['avg_correlation'].min():.3f} to "
            f"{df_filtered['avg_correlation'].max():.3f}"
        )

        # Create 3D scatter plot where each point represents an agent pair
        # Use agent_i and agent_j as spatial coordinates, correlation as z-axis and color
        scatter = hv.Scatter3D(
            df_filtered,
            kdims=['agent_i', 'agent_j', 'avg_correlation'],
            vdims=['v_x_correlation', 'v_y_correlation', 'v_z_correlation', 'n_samples']
        ).opts(
            color='avg_correlation',
            cmap=self.color_map,  # Widget-specific parameter
            size=self.marker_size,  # Widget-specific parameter
            width=800,
            height=600,
            colorbar=True,
            title=f'Velocity Correlations (3D: agent_i × agent_j × avg_corr) - {len(df_filtered)} pairs',
            zlabel='Average Correlation',
            xlim=(df_filtered['agent_i'].min()-1, df_filtered['agent_i'].max()+1),
            ylim=(df_filtered['agent_j'].min()-1, df_filtered['agent_j'].max()+1)
        )

        self.display_pane.object = scatter
        logger.info(f"Loaded correlations for {len(df_filtered)} agent pairs")
