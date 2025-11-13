"""
Velocity statistics widget.

Displays three groups of velocity statistics:
1. Individual agent speed (histogram + time series with median and IQR)
2. Mean velocity magnitude at each timestamp (histogram + time series)
3. Relative velocity magnitude between pairs (histogram + time series with median and IQR)
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


class VelocityStatsWidget(BaseAnalysisWidget):
    """
    Comprehensive velocity statistics visualization.

    Displays three groups of velocity metrics:
    - Individual agent speeds (from observations) - median + IQR bands
    - Mean velocity magnitude (normalized velocity vectors, then magnitude of mean)
    - Relative velocity magnitudes (pairwise ||v_i - v_j||) - median + IQR bands
    """

    widget_name = "Velocity Stats"
    widget_description = "Comprehensive velocity statistics"
    widget_category = "temporal"

    # Widget-specific parameters (minimal, since layout is fixed)
    bin_count = param.Integer(
        default=30,
        bounds=(10, 100),
        doc="Number of bins for histograms"
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
        # Return a Column that we can update with content
        return pn.Column(
            pn.pane.Markdown('Click "Load Data" to display velocity statistics'),
            sizing_mode="stretch_both"
        )

    def load_data(self) -> None:
        """Load and visualize velocity statistics."""
        # Get raw episode tracks (positions + velocities for all agents at all times)
        df_tracks = self.query_with_context('get_episode_tracks')

        if len(df_tracks) == 0:
            raise ValueError("No data found for selected parameters")

        # Compute individual agent speeds (already in df_tracks as 'speed')
        # Group 1a: Individual agent speed
        # Clean speed data first
        speed_data = self._to_numeric_array(df_tracks['speed'])
        speed_hist = self._create_histogram(
            speed_data,
            "1a. Individual Agent Speed - Distribution",
            "Speed",
            "darkblue"
        )
        speed_ts = self._create_speed_time_series(df_tracks, "1a. Individual Agent Speed - Time Series")

        # Group 1b: Mean velocity magnitude
        mean_vel_mag_data = self._compute_mean_velocity_magnitude(df_tracks)
        if len(mean_vel_mag_data) == 0:
            raise ValueError("Mean velocity magnitude computation resulted in no data. All agents may have zero velocity.")
        if mean_vel_mag_data['mean_velocity_magnitude'].isna().all():
            raise ValueError("Mean velocity magnitude computation resulted in all NaN values. Check velocity data quality.")
        mean_vel_mag_hist = self._create_histogram(
            mean_vel_mag_data['mean_velocity_magnitude'].values,
            "1b. Mean Velocity Magnitude - Distribution",
            "Magnitude",
            "darkgreen"
        )
        mean_vel_mag_ts = self._create_simple_time_series(
            mean_vel_mag_data,
            'mean_velocity_magnitude',
            "1b. Mean Velocity Magnitude - Time Series",
            "green"
        )

        # Group 1c: Relative velocity magnitude
        rel_vel_mag_data = self._compute_relative_velocity_magnitude(df_tracks)
        rel_vel_mag_hist = self._create_histogram(
            rel_vel_mag_data['relative_velocity_magnitude'].values,
            "1c. Relative Velocity Magnitude - Distribution",
            "||v_i - v_j||",
            "darkorange"
        )
        rel_vel_mag_ts = self._create_relative_time_series(
            rel_vel_mag_data,
            'relative_velocity_magnitude',
            "1c. Relative Velocity Magnitude - Time Series",
            "orange"
        )

        # Arrange plots in 3 rows x 2 columns layout
        # Each row has a header + histogram (left) + time series (right)
        self.display_pane.objects = [
            pn.pane.Markdown("## 1a. Individual Agent Speed"),
            pn.pane.HoloViews(speed_hist + speed_ts),
            pn.pane.Markdown("## 1b. Mean Velocity Magnitude"),
            pn.pane.HoloViews(mean_vel_mag_hist + mean_vel_mag_ts),
            pn.pane.Markdown("## 1c. Relative Velocity Magnitude (pairwise)"),
            pn.pane.HoloViews(rel_vel_mag_hist + rel_vel_mag_ts)
        ]
        logger.info(f"Loaded velocity stats with {len(df_tracks)} observations")

    def _to_numeric_array(self, series: pd.Series) -> np.ndarray:
        """Convert pandas series to clean numeric array, replacing None/NaN with 0.0."""
        # Convert to numeric (handles None, converts to NaN)
        numeric = pd.to_numeric(series, errors='coerce')
        # Replace NaN with 0.0
        return numeric.fillna(0.0).values

    def _create_histogram(
        self,
        data: np.ndarray,
        title: str,
        xlabel: str,
        color: str
    ) -> hv.Histogram:
        """Create a histogram plot."""
        # Remove NaN and infinite values
        clean_data = data[np.isfinite(data)]
        if len(clean_data) == 0:
            raise ValueError(f"No valid data for histogram: {title}")

        frequencies, edges = np.histogram(clean_data, bins=self.bin_count)
        hist = hv.Histogram((edges, frequencies))
        hist.opts(
            opts.Histogram(
                color=color,
                width=500,
                height=300,
                xlabel=xlabel,
                ylabel='Count',
                title=title
            )
        )
        return hist

    def _create_speed_time_series(self, df_tracks: pd.DataFrame, title: str = "Speed Over Time") -> hv.Overlay:
        """Create time series with median and IQR (25th-75th percentile) for individual agent speeds."""
        logger.info("⭐ USING NEW VERSION: Creating speed time series with Spread element")
        window_size = self.context.temporal_window_size

        # Clean speed data first (handle None/NaN values)
        df_tracks = df_tracks.copy()
        df_tracks['speed'] = self._to_numeric_array(df_tracks['speed'])

        # Compute windowed statistics (median and quartiles)
        df_tracks['time_window'] = (df_tracks['time_index'] // window_size) * window_size
        stats = df_tracks.groupby('time_window')['speed'].agg([
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).reset_index()

        # Compute errors for Spread element (distance from median to quantiles)
        stats['neg_err'] = stats['median'] - stats['q25']
        stats['pos_err'] = stats['q75'] - stats['median']

        # Create median line
        curve = hv.Curve(
            (stats['time_window'], stats['median']),
            kdims='Time',
            vdims='Speed',
            label='Median'
        ).opts(
            color='darkblue',
            line_width=2
        )

        # Create IQR spread
        spread = hv.Spread(
            (stats['time_window'], stats['median'], stats['neg_err'], stats['pos_err']),
            kdims='Time',
            vdims=['Speed', 'neg_err', 'pos_err'],
            label='IQR (25th-75th)'
        ).opts(
            color='lightblue',
            alpha=0.3
        )

        return (spread * curve).opts(
            width=500,
            height=300,
            title=title,
            show_legend=True
        )

    def _create_simple_time_series(
        self,
        df: pd.DataFrame,
        value_col: str,
        title: str,
        color: str
    ) -> hv.Curve:
        """Create simple time series with windowing (no std bands)."""
        window_size = self.context.temporal_window_size

        # Filter out NaN values
        df_clean = df.dropna(subset=[value_col]).copy()
        if len(df_clean) == 0:
            raise ValueError(f"No valid data for time series: {title}")

        # Apply windowing and compute mean per window
        df_clean['time_window'] = (df_clean['time_index'] // window_size) * window_size
        stats = df_clean.groupby('time_window')[value_col].mean().reset_index()

        return hv.Curve(
            stats,
            kdims='time_window',
            vdims=value_col
        ).opts(
            color=color,
            line_width=2,
            width=500,
            height=300,
            xlabel='Time',
            ylabel='Magnitude',
            title=title
        )

    def _create_relative_time_series(
        self,
        df: pd.DataFrame,
        value_col: str,
        title: str,
        color: str
    ) -> hv.Overlay:
        """Create time series with median and IQR (25th-75th percentile) for relative quantities."""
        logger.info(f"⭐ USING NEW VERSION: Creating relative time series with Spread element (color={color})")
        window_size = self.context.temporal_window_size

        # Compute windowed statistics (median and quartiles)
        df['time_window'] = (df['time_index'] // window_size) * window_size
        stats = df.groupby('time_window')[value_col].agg([
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).reset_index()

        # Compute errors for Spread element
        stats['neg_err'] = stats['median'] - stats['q25']
        stats['pos_err'] = stats['q75'] - stats['median']

        # Create line plot (median) - use darker shade for line
        dark_color = {'green': 'darkgreen', 'orange': 'darkorange'}.get(color, color)
        light_color = {'green': 'lightgreen', 'orange': 'lightsalmon'}.get(color, color)

        curve = hv.Curve(
            (stats['time_window'], stats['median']),
            kdims='Time',
            vdims='Magnitude',
            label='Median'
        ).opts(
            color=dark_color,
            line_width=2
        )

        # Create IQR spread
        spread = hv.Spread(
            (stats['time_window'], stats['median'], stats['neg_err'], stats['pos_err']),
            kdims='Time',
            vdims=['Magnitude', 'neg_err', 'pos_err'],
            label='IQR (25th-75th)'
        ).opts(
            color=light_color,
            alpha=0.3
        )

        return (spread * curve).opts(
            width=500,
            height=300,
            title=title,
            show_legend=True
        )

    def _compute_mean_velocity_magnitude(self, df_tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean velocity magnitude at each timestamp.

        At each time: normalize velocities v_i/||v_i||, compute mean vector, take norm.
        """
        results = []

        for time_idx in df_tracks['time_index'].unique():
            df_t = df_tracks[df_tracks['time_index'] == time_idx].copy()

            # Compute velocity magnitudes (handle 2D data where v_z might be None/NaN or missing)
            v_x = self._to_numeric_array(df_t['v_x'])
            v_y = self._to_numeric_array(df_t['v_y'])
            if 'v_z' in df_t.columns:
                v_z_vals = self._to_numeric_array(df_t['v_z'])
            else:
                v_z_vals = np.zeros(len(df_t))
            v_mag = np.sqrt(v_x**2 + v_y**2 + v_z_vals**2)

            # Normalize velocities (avoid division by zero)
            mask = v_mag > 1e-10
            if mask.sum() == 0:
                continue

            v_x_norm = np.zeros_like(v_x)
            v_y_norm = np.zeros_like(v_y)
            v_z_norm = np.zeros(len(df_t))

            v_x_norm[mask] = v_x[mask] / v_mag[mask]
            v_y_norm[mask] = v_y[mask] / v_mag[mask]
            v_z_norm[mask] = v_z_vals[mask] / v_mag[mask]

            # Compute mean normalized vector
            mean_v_x = v_x_norm[mask].mean()
            mean_v_y = v_y_norm[mask].mean()
            mean_v_z = v_z_norm[mask].mean()

            # Compute magnitude of mean vector
            mean_vel_mag = np.sqrt(mean_v_x**2 + mean_v_y**2 + mean_v_z**2)

            results.append({
                'time_index': time_idx,
                'mean_velocity_magnitude': mean_vel_mag
            })

        return pd.DataFrame(results)

    def _compute_relative_velocity_magnitude(self, df_tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise relative velocity magnitudes ||v_i - v_j|| for all i<j.
        """
        results = []

        for time_idx in df_tracks['time_index'].unique():
            df_t = df_tracks[df_tracks['time_index'] == time_idx].copy()

            agents = df_t['agent_id'].values
            # Handle None/NaN values in all velocity components
            v_x = self._to_numeric_array(df_t['v_x'])
            v_y = self._to_numeric_array(df_t['v_y'])
            v_z = self._to_numeric_array(df_t['v_z']) if 'v_z' in df_t.columns else np.zeros(len(df_t))

            # Compute all pairwise differences (i < j)
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    delta_vx = v_x[i] - v_x[j]
                    delta_vy = v_y[i] - v_y[j]
                    delta_vz = v_z[i] - v_z[j]

                    rel_vel_mag = np.sqrt(delta_vx**2 + delta_vy**2 + delta_vz**2)

                    results.append({
                        'time_index': time_idx,
                        'agent_i': agents[i],
                        'agent_j': agents[j],
                        'relative_velocity_magnitude': rel_vel_mag
                    })

        return pd.DataFrame(results)
