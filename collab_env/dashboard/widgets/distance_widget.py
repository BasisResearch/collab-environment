"""
Distance statistics widget.

Displays relative locations (pairwise distances) between agents:
- Histogram of ||x_i - x_j|| for all pairs i<j
- Time series with median and IQR (25th-75th percentile)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import param
import panel as pn
import holoviews as hv
from holoviews import opts

from .base_analysis_widget import BaseAnalysisWidget

logger = logging.getLogger(__name__)


class DistanceStatsWidget(BaseAnalysisWidget):
    """
    Relative location distance statistics.

    Displays pairwise distances ||x_i - x_j|| between agents:
    - Histogram over entire episode
    - Time series with median and IQR (25th-75th percentile) bands
    """

    widget_name = "Spatial Analysis"
    widget_description = "Relative locations (pairwise distances)"
    widget_category = "spatial"

    # Widget-specific parameters
    bin_count = param.Integer(
        default=30,
        bounds=(10, 100),
        doc="Number of bins for histogram"
    )

    def create_custom_controls(self) -> Optional[pn.Column]:
        """Create widget-specific controls."""
        return pn.Column(
            "### Visualization Options",
            pn.widgets.IntSlider.from_param(
                self.param.bin_count,
                name="Histogram Bins",
                width=200
            )
        )

    def create_display_pane(self) -> pn.pane.PaneBase:
        """Create empty plot pane."""
        # Create a placeholder curve (empty data)
        placeholder = hv.Curve([]).opts(
            width=800,
            height=300,
            title='Click "Load Data" to display distance statistics'
        )
        return pn.pane.HoloViews(
            placeholder,
            sizing_mode="stretch_both"
        )

    def load_data(self) -> None:
        """Load and visualize relative location statistics."""
        # Get raw episode tracks (positions for all agents at all times)
        df_tracks = self.query_with_context('get_episode_tracks')

        if len(df_tracks) == 0:
            raise ValueError("No data found for selected parameters")

        # Compute pairwise distances
        rel_dist_data = self._compute_relative_distances(df_tracks)

        if len(rel_dist_data) == 0:
            raise ValueError("No pairwise distance data computed")

        # Create histogram
        hist = self._create_histogram(rel_dist_data['relative_distance'].values)

        # Create time series with mean ± std
        ts = self._create_time_series(rel_dist_data)

        # Arrange plots side by side
        layout = (hist + ts).opts(title="Relative Locations (||x_i - x_j||)")

        self.display_pane.object = layout
        logger.info(f"Loaded distance stats with {len(rel_dist_data)} pairwise distances")

    def _to_numeric_array(self, series: pd.Series) -> np.ndarray:
        """Convert pandas series to clean numeric array, replacing None/NaN with 0.0."""
        # Convert to numeric (handles None, converts to NaN)
        numeric = pd.to_numeric(series, errors='coerce')
        # Replace NaN with 0.0
        return numeric.fillna(0.0).values

    def _create_histogram(self, data: np.ndarray) -> hv.Histogram:
        """Create histogram of relative distances."""
        frequencies, edges = np.histogram(data, bins=self.bin_count)
        hist = hv.Histogram((edges, frequencies))
        hist.opts(
            opts.Histogram(
                color='darkviolet',
                width=400,
                height=300,
                xlabel='Distance ||x_i - x_j||',
                ylabel='Count',
                title='Distribution of Pairwise Distances'
            )
        )
        return hist

    def _create_time_series(self, df: pd.DataFrame) -> hv.Overlay:
        """Create time series with median and IQR (25th-75th percentile) for relative distances."""
        logger.info("⭐ USING NEW VERSION: Creating distance time series with Spread element")
        window_size = self.context.temporal_window_size

        # Compute windowed statistics (median and quartiles)
        df['time_window'] = (df['time_index'] // window_size) * window_size
        stats = df.groupby('time_window')['relative_distance'].agg([
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).reset_index()

        # Compute errors for Spread element
        stats['neg_err'] = stats['median'] - stats['q25']
        stats['pos_err'] = stats['q75'] - stats['median']

        # Create median line
        curve = hv.Curve(
            (stats['time_window'], stats['median']),
            kdims='Time',
            vdims='Distance',
            label='Median'
        ).opts(
            color='darkviolet',
            line_width=2
        )

        # Create IQR spread
        spread = hv.Spread(
            (stats['time_window'], stats['median'], stats['neg_err'], stats['pos_err']),
            kdims='Time',
            vdims=['Distance', 'neg_err', 'pos_err'],
            label='IQR (25th-75th)'
        ).opts(
            color='plum'
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

        return (spread * curve).opts(
            width=400,
            height=300,
            title='Pairwise Distance Over Time',
            show_legend=True,
            hooks=[legend_hook]
        )

    def _compute_relative_distances(self, df_tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise distance magnitudes ||x_i - x_j|| for all i<j.
        """
        results = []

        for time_idx in df_tracks['time_index'].unique():
            df_t = df_tracks[df_tracks['time_index'] == time_idx].copy()

            agents = df_t['agent_id'].values
            # Handle None/NaN values in all position components
            x = self._to_numeric_array(df_t['x'])
            y = self._to_numeric_array(df_t['y'])
            z = self._to_numeric_array(df_t['z']) if 'z' in df_t.columns else np.zeros(len(df_t))

            # Compute all pairwise distances (i < j)
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    delta_x = x[i] - x[j]
                    delta_y = y[i] - y[j]
                    delta_z = z[i] - z[j]

                    rel_dist = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

                    results.append({
                        'time_index': time_idx,
                        'agent_i': agents[i],
                        'agent_j': agents[j],
                        'relative_distance': rel_dist
                    })

        return pd.DataFrame(results)
