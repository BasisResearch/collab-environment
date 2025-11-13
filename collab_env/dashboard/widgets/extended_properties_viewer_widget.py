"""
Extended Properties Viewer widget - time series and histogram visualization.

Provides a 2-panel view for exploring extended properties:
- Time series panel: Multi-property time series showing trends over time
- Histogram panel: Dynamic row of histograms for selected properties

Property selection affects both time series and histograms.
"""

import logging
from typing import Optional, List

import param
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts

from .base_analysis_widget import BaseAnalysisWidget
from .query_scope import ScopeType

logger = logging.getLogger(__name__)


class ExtendedPropertiesViewerWidget(BaseAnalysisWidget):
    """
    Extended properties viewer with time series and histograms.

    Features:
    - Multi-property time series showing trends over time
    - Dynamic histograms for selected properties

    Both panels update based on property selection.
    """

    widget_name = "Extended Properties Viewer"
    widget_description = "Time series and histogram visualization of extended properties"
    widget_category = "visualization"

    # Property selection (for time series and histograms)
    selected_properties = param.ListSelector(default=[], objects=[], doc="Selected properties to display")

    def __init__(self, **params):
        # Initialize data storage
        self.properties_ts_df = None
        self.properties_dist_df = None
        self.available_props_df = None

        # Color palette (shared between time series and histograms)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.property_color_map = {}  # Maps property_id -> color

        # UI components (will be created in _create_ui)
        self.timeseries_pane = None
        self.histogram_pane = None

        super().__init__(**params)

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create property selection controls."""

        # Property selection (will be populated after data load)
        property_selector = pn.widgets.CheckBoxGroup.from_param(
            self.param.selected_properties,
            name="Select Properties",
            inline=True,
            width=800
        )

        # Single-row layout
        return pn.Column(
            pn.Row(
                property_selector,
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create 2-panel layout for time series and histograms."""

        # Return a Column that we'll update with content after data loads
        # This matches the pattern used by velocity_widget
        return pn.Column(
            pn.pane.Markdown('**Extended Properties Viewer**\n\nClick "Load Extended Properties Viewer" to load data.'),
            sizing_mode="stretch_both"
        )

    def load_data(self) -> None:
        """Load extended properties data for time series and histograms."""

        # Validate this is episode scope only
        if self.context.scope.scope_type != ScopeType.EPISODE:
            raise ValueError("Extended Properties Viewer only supports episode-level analysis")

        logger.info("Loading available properties...")
        self.available_props_df = self.query_with_context('get_available_properties')

        if len(self.available_props_df) == 0:
            raise ValueError("No extended properties found for this episode")

        # Update property selector options
        prop_list = self.available_props_df['property_id'].tolist()
        self.param.selected_properties.objects = prop_list

        # Default to first 3-5 properties if available
        default_props = prop_list[:min(5, len(prop_list))]
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

        # Create the initial visualizations
        timeseries_plot = self._create_timeseries_plot()
        histogram_layout = self._create_histogram_layout()

        # Update display pane with plots
        # histogram_layout can be either a Panel GridBox or a HoloViews element (for empty states)
        if isinstance(histogram_layout, pn.GridBox):
            histogram_pane = histogram_layout
        else:
            # It's a HoloViews element (empty placeholder)
            histogram_pane = pn.pane.HoloViews(histogram_layout, sizing_mode="stretch_width", min_height=300)

        self.display_pane.objects = [
            pn.pane.Markdown("## Time Series Panel"),
            pn.pane.HoloViews(timeseries_plot, sizing_mode="stretch_width", height=400),
            pn.pane.Markdown("## Histogram Panel"),
            histogram_pane
        ]

        logger.info("Extended Properties Viewer loaded successfully")

    def _create_timeseries_plot(self):
        """Create time series plot with selected properties."""
        if self.properties_ts_df is None or len(self.properties_ts_df) == 0:
            # Create empty placeholder
            return hv.Curve([]).opts(
                width=800,
                height=400,
                title="Time Series (no properties available)"
            )

        # Filter to selected properties
        if len(self.selected_properties) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=400,
                title="Time Series (select properties to display)"
            )

        df = self.properties_ts_df[self.properties_ts_df['property_id'].isin(self.selected_properties)]

        if len(df) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=400,
                title="Time Series (no data for selected properties)"
            )

        # Build color map for selected properties (used by both time series and histograms)
        self.property_color_map = {
            prop_id: self.colors[i % len(self.colors)]
            for i, prop_id in enumerate(self.selected_properties)
        }

        # Build overlay of curves and spreads
        overlay_elements = []
        has_data = False

        for i, prop_id in enumerate(self.selected_properties):
            prop_df = df[df['property_id'] == prop_id].sort_values('time_window')

            if len(prop_df) == 0:
                continue

            # Filter out rows with NaN/inf in avg_value
            prop_df = prop_df[np.isfinite(prop_df['avg_value'])]

            if len(prop_df) == 0:
                logger.warning(f"No valid time series data for property {prop_id}")
                continue

            has_data = True
            color = self.property_color_map[prop_id]

            # Create mean line curve
            curve = hv.Curve(
                (prop_df['time_window'], prop_df['avg_value']),
                kdims='Time Window',
                vdims='Value',
                label=prop_id
            ).opts(
                color=color,
                line_width=2
            )

            overlay_elements.append(curve)

            # Add shaded band for std (if available and not NaN)
            if 'std_value' in prop_df.columns:
                # Filter to rows where std_value is finite and positive
                std_df = prop_df[np.isfinite(prop_df['std_value']) & (prop_df['std_value'] > 0)]
                if len(std_df) > 0:
                    # Compute error bands for Spread element
                    neg_err = std_df['std_value']
                    pos_err = std_df['std_value']

                    # Add spread without showing in legend
                    spread = hv.Spread(
                        (std_df['time_window'], std_df['avg_value'], neg_err, pos_err),
                        kdims='Time Window',
                        vdims=['Value', 'neg_err', 'pos_err']
                    ).opts(
                        color=color,
                        show_legend=False  # Explicitly hide from legend
                        # alpha not supported in plotly backend
                    )

                    overlay_elements.append(spread)

        if not has_data:
            logger.warning("No valid data for any selected properties in time series")
            return hv.Curve([]).opts(
                width=800,
                height=400,
                title="Time Series (no valid data)"
            )

        def legend_hook(plot, element):
            """Position legend inside plot for plotly backend."""
            fig = plot.state
            fig["layout"]["legend"] = dict(
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.98
            )

        # Create overlay and apply options
        return hv.Overlay(overlay_elements).opts(
            width=800,
            height=400,
            title="Property Time Series",
            show_legend=True,
            hooks=[legend_hook]
        )

    def _create_histogram_layout(self):
        """Create histogram layout with selected properties."""
        if self.properties_dist_df is None or len(self.properties_dist_df) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Histograms (no properties available)"
            )

        if len(self.selected_properties) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Histograms (select properties to display)"
            )

        # Filter to selected properties
        df = self.properties_dist_df[self.properties_dist_df['property_id'].isin(self.selected_properties)]

        if len(df) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Histograms (no data for selected properties)"
            )

        # Ensure color map is built (same as in time series)
        if not hasattr(self, 'property_color_map') or set(self.property_color_map.keys()) != set(self.selected_properties):
            self.property_color_map = {
                prop_id: self.colors[i % len(self.colors)]
                for i, prop_id in enumerate(self.selected_properties)
            }

        # Create histogram for each property using HoloViews
        histogram_plots = []
        for prop_id in self.selected_properties:
            prop_df = df[df['property_id'] == prop_id]

            if len(prop_df) == 0:
                continue

            # Get values and clean them (remove NaN/inf)
            values = prop_df['value_float'].values
            values = values[np.isfinite(values)]

            if len(values) == 0:
                logger.warning(f"No valid values for property {prop_id}")
                continue

            # Create histogram using HoloViews
            # Use the same color as the time series for this property
            color = self.property_color_map.get(prop_id, 'navy')

            try:
                frequencies, edges = np.histogram(values, bins=30)
                hist = hv.Histogram((edges, frequencies)).opts(
                    opts.Histogram(
                        color=color,
                        width=400,
                        height=300,
                        xlabel='Value',
                        ylabel='Count',
                        title=prop_id
                        # alpha not supported in plotly backend
                    )
                )
                histogram_plots.append(hist)
            except Exception as e:
                logger.error(f"Failed to create histogram for {prop_id}: {e}")
                continue

        # Return layout
        if histogram_plots:
            # For plotly backend, use Panel GridBox instead of HoloViews Layout
            # This ensures each histogram has independent axes
            # Arrange in rows of 3 columns
            return pn.GridBox(*[pn.pane.HoloViews(h) for h in histogram_plots], ncols=3)
        else:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Histograms (no valid data)"
            )

    @param.depends('selected_properties', watch=True)
    def _on_properties_change(self):
        """Handle property selection changes."""
        if self.properties_ts_df is not None and len(self.display_pane.objects) > 1:
            # Recreate both plots
            timeseries_plot = self._create_timeseries_plot()
            histogram_layout = self._create_histogram_layout()

            # Prepare histogram pane
            if isinstance(histogram_layout, pn.GridBox):
                histogram_pane = histogram_layout
            else:
                histogram_pane = pn.pane.HoloViews(histogram_layout, sizing_mode="stretch_width", min_height=300)

            # Rebuild the entire objects list to trigger Panel update
            self.display_pane.objects = [
                pn.pane.Markdown("## Time Series Panel"),
                pn.pane.HoloViews(timeseries_plot, sizing_mode="stretch_width", height=400),
                pn.pane.Markdown("## Histogram Panel"),
                histogram_pane
            ]
