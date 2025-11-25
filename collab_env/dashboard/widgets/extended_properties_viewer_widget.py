"""
Extended Properties Viewer widget - time series and histogram visualization.

Provides a 2-panel view for exploring extended properties:
- Time series panel: Multi-property time series with median + configurable quantile bands
  (default: 10th-90th percentile) and optional individual agent trajectories
- Histogram panel: Dynamic row of histograms for selected properties

Property selection affects both time series and histograms.
"""

import logging
import textwrap
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
    - Multi-property time series with median + configurable quantile bands (default: 10th-90th percentile)
    - Optional individual agent trajectories with configurable opacity and markers
    - Dynamic histograms for selected properties

    Both panels update based on property selection and quantile settings.
    """

    widget_name = "Extended Properties Viewer"
    widget_description = "Time series and histogram visualization of extended properties"
    widget_category = "visualization"

    # Property selection (for time series and histograms)
    selected_properties = param.ListSelector(default=[], objects=[], doc="Selected properties to display")

    # Quantile band parameters
    lower_quantile = param.Number(default=0.10, bounds=(0.01, 0.49), doc="Lower quantile for uncertainty band (default: 10th percentile)")
    upper_quantile = param.Number(default=0.90, bounds=(0.51, 0.99), doc="Upper quantile for uncertainty band (default: 90th percentile)")

    # Raw datapoints display options
    show_raw_lines = param.Boolean(default=False, doc="Show individual agent trajectories as lines")
    raw_line_opacity = param.Number(default=0.3, bounds=(0.05, 1.0), doc="Opacity of individual agent lines")
    raw_marker_size = param.Integer(default=4, bounds=(1, 12), doc="Size of markers on individual agent lines")
    max_agents_to_plot = param.Integer(default=20, bounds=(1, 100), doc="Maximum number of agent lines to display")

    # Normalization option
    normalize = param.Boolean(default=False, doc="Z-score normalize all properties to display on same scale")

    # Histogram/violin plot options
    bin_count = param.Integer(default=30, bounds=(10, 100), doc="Number of bins for histograms")
    plot_type = param.Selector(default="Histogram", objects=["Histogram", "Violin Plot (separate)", "Violin Plot (merged)"],
                              doc="Distribution plot type")

    def __init__(self, **params):
        # Initialize data storage
        self.properties_ts_df = None
        self.properties_dist_df = None
        self.properties_raw_df = None  # NEW: raw data for agent lines
        self.available_props_df = None

        # Color palette (shared between time series and histograms)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.property_color_map = {}  # Maps property_id -> color

        # UI components (will be created in _create_ui)
        self.timeseries_pane = None
        self.histogram_pane = None

        # Two-stage loading UI components
        self.load_names_btn = None
        self.load_data_btn = None
        self.property_selector = None
        self.property_status = None

        super().__init__(**params)

    def _create_ui(self):
        """Override to hide the base class load button."""
        super()._create_ui()
        # Hide the base load button to avoid confusion (we have our own 2-button workflow)
        self.load_btn.visible = False

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create property selection controls with two-stage loading."""

        # === Stage 1: Load Property Names ===
        self.load_names_btn = pn.widgets.Button(
            name="1. Load Property Names",
            button_type="primary",
            width=200
        )
        self.load_names_btn.on_click(self._on_load_names_click)

        self.property_status = pn.pane.Markdown(
            "_Click 'Load Property Names' to see available properties_",
            styles={'color': '#666', 'font-style': 'italic'}
        )

        load_names_section = pn.Column(
            "### Step 1: Load Property Names",
            pn.Row(self.load_names_btn),
            self.property_status
        )

        # === Stage 2: Select and Load Properties ===
        # Property selection (disabled until names are loaded)
        self.property_selector = pn.widgets.CheckBoxGroup.from_param(
            self.param.selected_properties,
            name="Select Properties to Load",
            inline=True,
            width=800,
            disabled=True
        )

        self.load_data_btn = pn.widgets.Button(
            name="2. Load Selected Properties",
            button_type="success",
            width=200,
            disabled=True
        )
        self.load_data_btn.on_click(self._on_load_data_click)

        property_selection_section = pn.Column(
            "### Step 2: Select Properties",
            self.property_selector,
            pn.Row(
                self.load_data_btn,
                pn.pane.Markdown("_Load only selected properties_", styles={'color': '#666', 'font-style': 'italic'})
            ),
            pn.pane.Markdown(
                "**Note:** Time range filtering is controlled by the Start/End Time fields in the main panel (not widget-specific).",
                styles={'color': '#666', 'font-size': '0.85em', 'font-style': 'italic', 'margin-top': '10px'}
            )
        )

        # === Visualization Options ===
        # Quantile controls
        lower_quantile_slider = pn.widgets.FloatSlider.from_param(
            self.param.lower_quantile,
            name="Lower Quantile",
            width=150,
            step=0.05
        )

        upper_quantile_slider = pn.widgets.FloatSlider.from_param(
            self.param.upper_quantile,
            name="Upper Quantile",
            width=150,
            step=0.05
        )

        # Normalize checkbox
        normalize_checkbox = pn.widgets.Checkbox.from_param(
            self.param.normalize,
            name="Z-Score Normalize"
        )

        # Raw datapoints controls
        show_raw_checkbox = pn.widgets.Checkbox.from_param(
            self.param.show_raw_lines,
            name="Show Agent Lines"
        )

        opacity_slider = pn.widgets.FloatSlider.from_param(
            self.param.raw_line_opacity,
            name="Opacity",
            width=120
        )

        marker_size_slider = pn.widgets.IntSlider.from_param(
            self.param.raw_marker_size,
            name="Marker Size",
            width=120
        )

        max_agents_slider = pn.widgets.IntSlider.from_param(
            self.param.max_agents_to_plot,
            name="Max Agents",
            width=120
        )

        # Distribution plot controls
        bin_count_slider = pn.widgets.IntSlider.from_param(
            self.param.bin_count,
            name="Histogram Bins",
            width=150
        )

        plot_type_selector = pn.widgets.Select.from_param(
            self.param.plot_type,
            name="Plot Type",
            width=150
        )

        viz_options = pn.Column(
            "### Visualization Options",
            pn.Row(
                lower_quantile_slider,
                upper_quantile_slider,
                normalize_checkbox,
                sizing_mode="stretch_width"
            ),
            pn.Row(
                show_raw_checkbox,
                opacity_slider,
                marker_size_slider,
                max_agents_slider,
                sizing_mode="stretch_width"
            ),
            pn.Row(
                bin_count_slider,
                plot_type_selector,
                sizing_mode="stretch_width"
            )
        )

        # === Complete Layout ===
        return pn.Column(
            load_names_section,
            pn.layout.Divider(),
            property_selection_section,
            pn.layout.Divider(),
            viz_options,
            sizing_mode="stretch_width"
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create 2-panel layout for time series and histograms."""

        # Return a Column that we'll update with content after data loads
        # This matches the pattern used by velocity_widget
        return pn.Column(
            pn.pane.Markdown('**Extended Properties Viewer**\n\nClick "1. Load Property Names" to begin.'),
            sizing_mode="stretch_both"
        )

    def _on_load_names_click(self, event):
        """Handle Stage 1: Load property names only (no data)."""
        if not self._validate_context():
            return

        try:
            self.context.report_loading("Loading property names...")

            # Determine scope label
            scope_label = "episode" if self.context.scope.scope_type == ScopeType.EPISODE else "session"
            logger.info(f"Loading available properties for {scope_label}...")

            # Query for available property names
            self.available_props_df = self.query_with_context('get_available_properties')

            if len(self.available_props_df) == 0:
                raise ValueError(f"No extended properties found for this {scope_label}")

            # Update property selector options
            prop_list = self.available_props_df['property_id'].tolist()
            self.param.selected_properties.objects = prop_list

            # Preserve previous selection if possible
            previous_selection = list(self.selected_properties) if self.selected_properties else []
            preserved_selection = [prop for prop in previous_selection if prop in prop_list]
            self.selected_properties = preserved_selection

            # Enable UI controls
            self.property_selector.disabled = False
            self.load_data_btn.disabled = False

            # Update status
            status_msg = f"✓ Found **{len(prop_list)}** properties"
            if preserved_selection:
                status_msg += f" ({len(preserved_selection)} previously selected)"
            status_msg += ". Select properties and click 'Load Selected Properties'."
            self.property_status.object = status_msg

            self.context.report_success(f"Loaded {len(prop_list)} property names")
            logger.info(f"Property names loaded: {prop_list}")

        except Exception as e:
            logger.error(f"Failed to load property names: {e}", exc_info=True)
            self.context.report_error(f"Failed to load property names: {e}")

    def _on_load_data_click(self, event):
        """Handle Stage 2: Load selected property data."""
        if not self._validate_context():
            return

        # Validate that property names are loaded
        if self.available_props_df is None or len(self.available_props_df) == 0:
            self.context.report_error("Please load property names first (Step 1)")
            return

        # Validate that at least one property is selected
        if not self.selected_properties or len(self.selected_properties) == 0:
            self.context.report_error("Please select at least one property to load")
            return

        try:
            self.context.report_loading(f"Loading {len(self.selected_properties)} selected properties...")

            # Call load_data with selected properties
            self.load_data(property_ids=list(self.selected_properties))

            # Update scope display after successful load
            self._update_scope_display()

            self.context.report_success(f"Loaded {len(self.selected_properties)} properties successfully")

        except Exception as e:
            logger.error(f"Failed to load properties: {e}", exc_info=True)
            self.context.report_error(f"Failed to load properties: {e}")

    def _reset_loading_state(self):
        """Reset UI to initial state (used when scope changes)."""
        self.available_props_df = None
        self.properties_ts_df = None
        self.properties_dist_df = None
        self.properties_raw_df = None
        self.param.selected_properties.objects = []
        self.selected_properties = []

        if self.property_selector:
            self.property_selector.disabled = True
        if self.load_data_btn:
            self.load_data_btn.disabled = True
        if self.property_status:
            self.property_status.object = "_Click 'Load Property Names' to see available properties_"

        # Reset display pane
        self.display_pane.objects = [
            pn.pane.Markdown('**Extended Properties Viewer**\n\nClick "1. Load Property Names" to begin.')
        ]

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

    def load_data(self, property_ids: Optional[List[str]] = None) -> None:
        """
        Load extended properties data for time series and histograms.

        Parameters
        ----------
        property_ids : list of str, optional
            Specific properties to load. If None, uses selected_properties.
            If no properties selected and None, raises error.
        """
        # Validate scope type (support both episode and session)
        if self.context.scope.scope_type not in [ScopeType.EPISODE, ScopeType.SESSION]:
            raise ValueError("Extended Properties Viewer only supports episode and session scopes")

        # Use selected properties if not specified
        if property_ids is None:
            if not self.selected_properties:
                raise ValueError(
                    "No properties selected. Please select properties first, "
                    "or use 'Load Property Names' to see available options."
                )
            property_ids = list(self.selected_properties)

        scope_label = "episode" if self.context.scope.scope_type == ScopeType.EPISODE else "session"
        logger.info(f"Loading {len(property_ids)} properties for {scope_label}: {property_ids}")

        # Conditional data loading based on scope
        if self.context.scope.scope_type == ScopeType.EPISODE:
            # Episode scope: Load time series, distributions, and optionally raw data
            logger.info(f"Loading property time series (quantiles: {self.lower_quantile:.0%} - {self.upper_quantile:.0%})...")
            self.properties_ts_df = self.query_with_context(
                'get_extended_properties_timeseries',
                property_ids=property_ids,  # Load ONLY selected properties
                lower_quantile=self.lower_quantile,
                upper_quantile=self.upper_quantile
            )

            logger.info("Loading property distributions...")
            self.properties_dist_df = self.query_with_context(
                'get_property_distributions',
                property_ids=property_ids  # Load ONLY selected properties
            )

            # Load raw property data if show_raw_lines is enabled
            if self.show_raw_lines:
                logger.info("Loading raw property data for agent lines...")
                self.properties_raw_df = self.query_with_context(
                    'get_extended_properties_raw',
                    property_ids=property_ids  # Load ONLY selected properties
                )
                logger.info(f"Loaded {len(self.properties_raw_df)} raw observations")
            else:
                self.properties_raw_df = None

            # Create both time series and distribution visualizations
            timeseries_plot = self._create_timeseries_plot()
            histogram_layout = self._create_histogram_layout()

            # Update display pane with both panels
            histogram_pane = self._wrap_distribution_pane(histogram_layout)

            self.display_pane.objects = [
                pn.pane.Markdown("## Time Series Panel"),
                pn.pane.HoloViews(timeseries_plot, sizing_mode="stretch_width", height=400),
                pn.pane.Markdown("## Distribution Panel"),
                histogram_pane
            ]

        else:  # SESSION scope
            # Session scope: Load only distributions (aggregated across all episodes in session)
            # Use time range from context.scope (shared with main UI)
            start_time = self.context.scope.start_time
            end_time = self.context.scope.end_time

            time_filter_msg = ""
            if start_time is not None or end_time is not None:
                time_filter_msg = f" (time range: {start_time or 0}-{end_time or '∞'})"
            logger.info(f"Loading property distributions (session-level aggregation{time_filter_msg})...")

            self.properties_ts_df = None
            self.properties_raw_df = None
            self.properties_dist_df = self.query_with_context(
                'get_property_distributions',
                property_ids=property_ids  # Load ONLY selected properties
                # start_time and end_time automatically passed via query_with_context from scope
            )

            # Create only distribution visualization
            histogram_layout = self._create_histogram_layout()

            # Update display pane with single panel
            histogram_pane = self._wrap_distribution_pane(histogram_layout)

            session_header = "## Distribution Panel (Session-Level Aggregation)"
            if time_filter_msg:
                session_header += f"\n_Filtered{time_filter_msg}_"

            self.display_pane.objects = [
                pn.pane.Markdown(session_header),
                pn.pane.Markdown("_Time series plots are not available for session-level analysis_"),
                histogram_pane
            ]

        logger.info(f"Extended Properties Viewer loaded successfully ({scope_label} scope, {len(property_ids)} properties)")

    def _get_normalization_stats(self, prop_id: str):
        """
        Compute mean and std for a property from distribution data.

        Parameters
        ----------
        prop_id : str
            Property ID to compute stats for

        Returns
        -------
        tuple
            (mean, std) for z-score normalization
        """
        if self.properties_dist_df is None:
            return 0.0, 1.0

        # Get all values for this property
        prop_values = self.properties_dist_df[
            self.properties_dist_df['property_id'] == prop_id
        ]['value_float'].values

        # Filter out NaN/inf
        prop_values = prop_values[np.isfinite(prop_values)]

        if len(prop_values) == 0:
            return 0.0, 1.0

        mean = np.mean(prop_values)
        std = np.std(prop_values)

        # Avoid division by zero
        if std == 0 or not np.isfinite(std):
            std = 1.0

        return mean, std

    def _normalize_values(self, values: np.ndarray, mean: float, std: float) -> np.ndarray:
        """
        Apply z-score normalization to values.

        Parameters
        ----------
        values : np.ndarray
            Values to normalize
        mean : float
            Mean for normalization
        std : float
            Standard deviation for normalization

        Returns
        -------
        np.ndarray
            Normalized values
        """
        return (values - mean) / std

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
            prop_df = prop_df[np.isfinite(prop_df['avg_value'])].copy()

            if len(prop_df) == 0:
                logger.warning(f"No valid time series data for property {prop_id}")
                continue

            has_data = True
            color = self.property_color_map[prop_id]

            # Get normalization stats if needed
            if self.normalize:
                mean, std = self._get_normalization_stats(prop_id)
            else:
                mean, std = 0.0, 1.0

            # Add individual agent lines FIRST (so they render behind aggregated lines)
            if self.show_raw_lines and self.properties_raw_df is not None:
                agent_lines = self._create_agent_lines(prop_id, color)
                overlay_elements.extend(agent_lines)

            # Add shaded band for quantile range if available
            if 'q_lower' in prop_df.columns and 'q_upper' in prop_df.columns:
                # Filter to rows where quantiles are finite
                quantile_df = prop_df[
                    np.isfinite(prop_df['median_value']) &
                    np.isfinite(prop_df['q_lower']) &
                    np.isfinite(prop_df['q_upper'])
                ].copy()
                if len(quantile_df) > 0:
                    # Apply normalization if needed
                    if self.normalize:
                        quantile_df['median_value'] = self._normalize_values(quantile_df['median_value'].values, mean, std)
                        quantile_df['q_lower'] = self._normalize_values(quantile_df['q_lower'].values, mean, std)
                        quantile_df['q_upper'] = self._normalize_values(quantile_df['q_upper'].values, mean, std)

                    # Compute error bands for Spread element (distance from median to quantiles)
                    neg_err = quantile_df['median_value'] - quantile_df['q_lower']
                    pos_err = quantile_df['q_upper'] - quantile_df['median_value']

                    # Add quantile spread without showing in legend
                    spread = hv.Spread(
                        (quantile_df['time_window'], quantile_df['median_value'], neg_err, pos_err),
                        kdims='Time Window',
                        vdims=['Value', 'neg_err', 'pos_err']
                    ).opts(
                        color=color,
                        show_legend=False  # Explicitly hide from legend
                    )

                    overlay_elements.append(spread)

            # Prepare median values for curve
            median_values = prop_df['median_value'].values
            if self.normalize:
                median_values = self._normalize_values(median_values, mean, std)

            # Create median line curve (on top of agent lines and IQR spread)
            curve = hv.Curve(
                (prop_df['time_window'], median_values),
                kdims='Time Window',
                vdims='Value',
                label=prop_id
            ).opts(
                color=color,
                line_width=2
            )

            overlay_elements.append(curve)

        if not has_data:
            logger.warning("No valid data for any selected properties in time series")
            return hv.Curve([]).opts(
                width=800,
                height=400,
                title="Time Series (no valid data)"
            )

        def layout_hook(plot, element):
            """Position legend and style title for plotly backend."""
            fig = plot.state
            # Position legend inside plot
            fig["layout"]["legend"] = dict(
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.98
            )
            # Reduce title font size for long scope strings
            # Title might be a string or dict, so we need to handle both
            current_title = fig["layout"].get("title", "")
            if isinstance(current_title, str):
                title_text = current_title
            else:
                title_text = current_title.get("text", "")

            fig["layout"]["title"] = dict(
                text=title_text,
                font=dict(size=9)
            )

        # Create overlay and apply options
        scope_str = str(self.context.scope) if self.context and self.context.scope else "Unknown Scope"
        # Wrap title text for long scope strings using <br> for Plotly
        full_title = f"Property Time Series - {scope_str}"
        if len(full_title) > 80:
            # Split at a reasonable point (prefer after " - ")
            parts = textwrap.wrap(full_title, width=80)
            wrapped_title = "<br>".join(parts)
        else:
            wrapped_title = full_title

        return hv.Overlay(overlay_elements).opts(
            width=800,
            height=400,
            title=wrapped_title,
            show_legend=True,
            hooks=[layout_hook]
        )

    def _create_histogram_layout(self):
        """Create histogram layout with selected properties."""
        if self.properties_dist_df is None or len(self.properties_dist_df) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Distribution plots (no properties available)"
            )

        if len(self.selected_properties) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Distribution plots (select properties to display)"
            )

        # Filter to selected properties
        df = self.properties_dist_df[self.properties_dist_df['property_id'].isin(self.selected_properties)]

        if len(df) == 0:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Distribution plots (no data for selected properties)"
            )

        # Ensure color map is built (same as in time series)
        if not hasattr(self, 'property_color_map') or set(self.property_color_map.keys()) != set(self.selected_properties):
            self.property_color_map = {
                prop_id: self.colors[i % len(self.colors)]
                for i, prop_id in enumerate(self.selected_properties)
            }

        # Dispatch to appropriate helper method based on plot_type
        if self.plot_type == "Histogram":
            return self._create_histograms(df)
        elif self.plot_type == "Violin Plot (separate)":
            return self._create_separate_violins(df)
        else:  # "Violin Plot (merged)"
            return self._create_merged_violin(df)

    def _create_histograms(self, df: pd.DataFrame):
        """Create separate histogram for each property."""
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

            # Apply normalization if needed
            if self.normalize:
                mean, std = self._get_normalization_stats(prop_id)
                values = self._normalize_values(values, mean, std)

            # Use the same color as the time series for this property
            color = self.property_color_map.get(prop_id, 'navy')

            try:
                # Create histogram with configurable bin count
                frequencies, edges = np.histogram(values, bins=self.bin_count)
                plot = hv.Histogram((edges, frequencies)).opts(
                    opts.Histogram(
                        color=color,
                        width=400,
                        height=300,
                        xlabel='Value',
                        ylabel='Count',
                        title=prop_id
                    )
                )
                histogram_plots.append(plot)
            except Exception as e:
                logger.error(f"Failed to create histogram for {prop_id}: {e}")
                continue

        # Return layout
        if histogram_plots:
            # Use Panel GridBox for independent axes, arranged in rows of 3 columns
            return pn.GridBox(*[pn.pane.HoloViews(h) for h in histogram_plots], ncols=3)
        else:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Histograms (no valid data)"
            )

    def _create_separate_violins(self, df: pd.DataFrame):
        """Create separate violin plot for each property."""
        violin_plots = []
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

            # Apply normalization if needed
            if self.normalize:
                mean, std = self._get_normalization_stats(prop_id)
                values = self._normalize_values(values, mean, std)

            # Use the same color as the time series for this property
            color = self.property_color_map.get(prop_id, 'navy')

            try:
                # Create violin plot using HoloViews Violin element
                violin_df = pd.DataFrame({
                    'Property': [prop_id] * len(values),
                    'Value': values
                })
                plot = hv.Violin(violin_df, kdims=['Property'], vdims=['Value']).opts(
                    color=color,
                    width=400,
                    height=300,
                    xlabel='Property',
                    ylabel='Value',
                    title=prop_id,
                    backend='plotly'
                )
                violin_plots.append(plot)
            except Exception as e:
                logger.error(f"Failed to create violin plot for {prop_id}: {e}")
                continue

        # Return layout
        if violin_plots:
            # Use Panel GridBox for independent axes, arranged in rows of 3 columns
            return pn.GridBox(*[pn.pane.HoloViews(h) for h in violin_plots], ncols=3)
        else:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Violin plots (no valid data)"
            )

    def _create_merged_violin(self, df: pd.DataFrame):
        """Create a single merged violin plot with all selected properties."""
        # Collect all property values into a single DataFrame
        all_values = []
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

            # Apply normalization if needed
            if self.normalize:
                mean, std = self._get_normalization_stats(prop_id)
                values = self._normalize_values(values, mean, std)

            # Add rows with property label
            for val in values:
                all_values.append({'Property': prop_id, 'Value': val})

        if not all_values:
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Merged violin plot (no valid data)"
            )

        # Create merged DataFrame
        merged_df = pd.DataFrame(all_values)

        # Create single violin plot with all properties
        # Use matplotlib backend for simpler, less cluttered visualization
        try:
            plot = hv.Violin(merged_df, kdims=['Property'], vdims=['Value']).opts(
                width=800,
                height=400,
                xlabel='Property',
                ylabel='Value',
                title='Property Distributions (Merged View)',
                backend='matplotlib',
                show_legend=False
            )

            # Apply color mapping via hooks
            def color_hook(plot, element):
                """Apply per-property colors to violin plot."""
                try:
                    # Try to get axis from handles (different keys may be used)
                    ax = None
                    if hasattr(plot, 'handles'):
                        # Try common handle keys
                        for key in ['axis', 'axes', 'ax']:
                            if key in plot.handles:
                                ax = plot.handles[key]
                                break

                    # Fallback: get current axis from plot state
                    if ax is None and hasattr(plot, 'state'):
                        # plot.state might be the figure, get current axes
                        import matplotlib.pyplot as plt
                        ax = plt.gca()

                    if ax is None:
                        logger.warning("Could not access matplotlib axis for color customization")
                        return

                    # Get violin parts (PolyCollection objects)
                    if hasattr(ax, 'collections') and len(ax.collections) > 0:
                        for i, (collection, prop_id) in enumerate(zip(ax.collections, self.selected_properties)):
                            if prop_id in self.property_color_map:
                                color = self.property_color_map[prop_id]
                                collection.set_facecolor(color)
                                collection.set_edgecolor(color)
                                collection.set_alpha(0.6)
                except Exception as e:
                    logger.warning(f"Failed to apply colors to merged violin plot: {e}")

            plot = plot.opts(hooks=[color_hook])

            return plot
        except Exception as e:
            logger.error(f"Failed to create merged violin plot: {e}")
            return hv.Curve([]).opts(
                width=800,
                height=300,
                title="Merged violin plot (error during creation)"
            )

    def _hex_to_rgba(self, hex_color: str, opacity: float) -> str:
        """
        Convert hex color to rgba string for Plotly.

        Parameters
        ----------
        hex_color : str
            Hex color like '#1f77b4'
        opacity : float
            Opacity value 0.0-1.0

        Returns
        -------
        str
            RGBA color string like 'rgba(31, 119, 180, 0.3)'
        """
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')

        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return f'rgba({r}, {g}, {b}, {opacity})'

    def _create_agent_lines(self, prop_id: str, base_color: str) -> List[hv.Curve]:
        """
        Create individual agent trajectory lines for a property.

        Parameters
        ----------
        prop_id : str
            Property ID to plot
        base_color : str
            Base color for this property (will be used with opacity)

        Returns
        -------
        List[hv.Curve]
            List of curve elements, one per agent (up to max_agents_to_plot)
        """
        if self.properties_raw_df is None:
            return []

        # Filter to this property
        prop_raw = self.properties_raw_df[self.properties_raw_df['property_id'] == prop_id].copy()

        if len(prop_raw) == 0:
            return []

        # Filter out NaN/inf values
        prop_raw = prop_raw[np.isfinite(prop_raw['value_float'])].copy()

        if len(prop_raw) == 0:
            return []

        # Apply normalization if needed
        if self.normalize:
            mean, std = self._get_normalization_stats(prop_id)
            prop_raw['value_float'] = self._normalize_values(prop_raw['value_float'].values, mean, std)

        # Get unique agents
        agents = prop_raw['agent_id'].unique()

        # Limit number of agents to plot for performance
        if len(agents) > self.max_agents_to_plot:
            logger.info(f"Limiting agent lines from {len(agents)} to {self.max_agents_to_plot} agents")
            # Sample agents deterministically (same agents each time)
            agents = np.sort(agents)[:self.max_agents_to_plot]

        # Convert base color to rgba with opacity (Plotly doesn't support alpha parameter)
        rgba_color = self._hex_to_rgba(base_color, self.raw_line_opacity)

        # Create a curve for each agent
        agent_curves = []
        for agent_id in agents:
            agent_data = prop_raw[prop_raw['agent_id'] == agent_id].sort_values('time_index')

            if len(agent_data) < 2:
                # Need at least 2 points for a line
                continue

            # Create curve with markers (Plotly-compatible)
            curve = hv.Curve(
                (agent_data['time_index'], agent_data['value_float']),
                kdims='Time Window',
                vdims='Value'
            ).opts(
                color=rgba_color,  # Use rgba color with opacity
                line_width=1,
                show_legend=False
            )

            # Add markers using Scatter overlay
            scatter = hv.Scatter(
                (agent_data['time_index'], agent_data['value_float']),
                kdims='Time Window',
                vdims='Value'
            ).opts(
                color=rgba_color,
                size=self.raw_marker_size,
                show_legend=False
            )

            # Combine line and markers
            agent_curves.append(curve * scatter)

        logger.info(f"Created {len(agent_curves)} agent lines for property {prop_id}")
        return agent_curves

    @param.depends('selected_properties', watch=True)
    def _on_properties_change(self):
        """Handle property selection changes."""
        # Update plots if any data is loaded (time series OR distributions)
        if (self.properties_ts_df is not None or self.properties_dist_df is not None) and len(self.display_pane.objects) > 1:
            self._update_plots()

    @param.depends('show_raw_lines', watch=True)
    def _on_show_raw_lines_change(self):
        """Handle show_raw_lines toggle - reload data if needed."""
        if self.properties_ts_df is None:
            return  # No data loaded yet

        # If enabling raw lines and we don't have the data, query it
        if self.show_raw_lines and self.properties_raw_df is None:
            logger.info("Loading raw property data for agent lines...")
            # Use selected properties (not all properties)
            property_ids = list(self.selected_properties) if self.selected_properties else None
            self.properties_raw_df = self.query_with_context(
                'get_extended_properties_raw',
                property_ids=property_ids
            )
            logger.info(f"Loaded {len(self.properties_raw_df)} raw observations")

        # Update plots
        if len(self.display_pane.objects) > 1:
            self._update_plots()

    @param.depends('raw_line_opacity', 'raw_marker_size', 'max_agents_to_plot', watch=True)
    def _on_raw_params_change(self):
        """Handle changes to raw line parameters."""
        if self.show_raw_lines and self.properties_raw_df is not None and len(self.display_pane.objects) > 1:
            self._update_plots()

    @param.depends('lower_quantile', 'upper_quantile', watch=True)
    def _on_quantile_change(self):
        """Handle quantile parameter changes - reload aggregated data."""
        if self.properties_ts_df is None:
            return  # No data loaded yet

        logger.info(f"Reloading time series with new quantiles: {self.lower_quantile:.0%} - {self.upper_quantile:.0%}")

        # Reload aggregated data with new quantiles
        # Use selected properties (not all properties)
        property_ids = list(self.selected_properties) if self.selected_properties else None
        self.properties_ts_df = self.query_with_context(
            'get_extended_properties_timeseries',
            property_ids=property_ids,
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile
        )

        # Update plots
        if len(self.display_pane.objects) > 1:
            self._update_plots()

    @param.depends('normalize', watch=True)
    def _on_normalize_change(self):
        """Handle normalization toggle - update plots with z-scored values."""
        if self.properties_ts_df is None and self.properties_dist_df is None:
            return  # No data loaded yet

        logger.info(f"Normalization {'enabled' if self.normalize else 'disabled'}")

        # Update plots (normalization is applied client-side)
        if len(self.display_pane.objects) > 1:
            self._update_plots()

    @param.depends('bin_count', watch=True)
    def _on_bin_count_change(self):
        """Handle bin count changes - update histogram plots."""
        if self.properties_dist_df is None:
            return  # No data loaded yet

        logger.info(f"Histogram bin count changed to {self.bin_count}")

        # Update plots (only affects histograms)
        if len(self.display_pane.objects) > 1:
            self._update_plots()

    @param.depends('plot_type', watch=True)
    def _on_plot_type_change(self):
        """Handle plot type changes - switch between histogram and violin."""
        if self.properties_dist_df is None:
            return  # No data loaded yet

        logger.info(f"Plot type changed to {self.plot_type}")

        # Update plots (only affects distribution panel)
        if len(self.display_pane.objects) > 1:
            self._update_plots()

    def _wrap_distribution_pane(self, histogram_layout):
        """Wrap distribution plot layout in appropriate Panel pane."""
        if isinstance(histogram_layout, pn.GridBox):
            # GridBox is already a Panel component
            return histogram_layout
        else:
            # For merged violin (matplotlib) or empty plots, wrap in HoloViews pane
            # Use fixed height for matplotlib backend to prevent rendering issues
            if self.plot_type == "Violin Plot (merged)":
                return pn.pane.HoloViews(histogram_layout, sizing_mode="stretch_width", height=500)
            else:
                return pn.pane.HoloViews(histogram_layout, sizing_mode="stretch_width", min_height=300)

    def _update_plots(self):
        """Helper to update plots based on current scope."""
        # Determine scope type
        if self.context.scope.scope_type == ScopeType.EPISODE:
            # Episode scope: Update both time series and distribution panels
            timeseries_plot = self._create_timeseries_plot()
            histogram_layout = self._create_histogram_layout()

            # Prepare histogram pane
            histogram_pane = self._wrap_distribution_pane(histogram_layout)

            # Rebuild the entire objects list to trigger Panel update
            self.display_pane.objects = [
                pn.pane.Markdown("## Time Series Panel"),
                pn.pane.HoloViews(timeseries_plot, sizing_mode="stretch_width", height=400),
                pn.pane.Markdown("## Distribution Panel"),
                histogram_pane
            ]

        else:  # SESSION scope
            # Session scope: Update only distribution panel
            histogram_layout = self._create_histogram_layout()

            # Prepare histogram pane
            histogram_pane = self._wrap_distribution_pane(histogram_layout)

            # Rebuild the objects list for session scope
            self.display_pane.objects = [
                pn.pane.Markdown("## Distribution Panel (Session-Level Aggregation)"),
                pn.pane.Markdown("_Time series plots are not available for session-level analysis_"),
                histogram_pane
            ]
