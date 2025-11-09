# Dashboard System: Design & Implementation Guide

**Last Updated:** 2025-11-08
**Status:** Phase 6 Complete ✅ | Phase 7 Planned (Unified Widgets)

> **Note**: This document covers the **query and visualization layer** of the system.
> For database schema and data loading, see [docs/data/db/README.md](../data/db/README.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Current Status](#current-status)
4. [Planned Features](#planned-features)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Technical Reference](#technical-reference)
7. [Usage Examples](#usage-examples)

---

## Overview

### What is the Dashboard System?

The Dashboard System provides interactive web-based visualization and analysis for 3D boids simulation data and animal tracking data. It consists of two major components:

1. **Data Layer** (QueryBackend): SQL-based query interface for efficient data retrieval
2. **GUI Layer** (Analysis Widgets): Modular, plugin-based visualization framework

### Key Features

**Current (Production Ready):**
- Session/episode management and selection
- 3D spatial density heatmaps
- Speed statistics over time
- Distance to target/boundary analysis
- Velocity correlations (episode-level)
- Episode and session-level aggregation (SQL-optimized)
- Modular widget architecture

**Planned (Phase 7):**
- Unified comprehensive viewers (3 widgets replacing 6+)
- Animated track visualization with synchronized property time series
- Pairwise interaction analysis
- Property correlation analysis with dual modes (windowed/lagged)
- Property computation framework (speed, acceleration)

### Design Goals

1. **Extensibility**: Add new analyses without modifying core code
2. **Performance**: SQL-level aggregation for efficiency
3. **Flexibility**: Support multiple data scopes (episode, session, custom)
4. **Usability**: Clear, intuitive interface with real-time feedback
5. **Maintainability**: Modular architecture with clear separation of concerns

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Spatial Analysis GUI                        │
│              (Panel/HoloViz Web Interface)                   │
│                                                               │
│  Scope: Episode | Session | Custom                          │
│  Shared: Bin Size | Time Window | Agent Type                │
│  Tabs: [Dynamic Widget Registry]                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ AnalysisContext (shared state)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     QueryBackend                             │
│          (collab_env/data/db/query_backend.py)              │
│                                                               │
│  Session Queries  │  Spatial Analysis  │  Correlations      │
│  - get_sessions   │  - get_heatmap     │  - velocity        │
│  - get_episodes   │  - get_velocities  │  - distance        │
│  - get_metadata   │  - get_distances   │                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ aiosql (driver-specific adapters)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   SQL Query Files (.sql)                     │
│             (collab_env/data/db/queries/)                   │
│                                                               │
│  - session_metadata.sql    (3 queries)                      │
│  - spatial_analysis.sql    (6 queries)                      │
│  - correlations.sql        (2 queries)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Executed via psycopg2/duckdb
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Database (PostgreSQL or DuckDB)                 │
│                                                               │
│  Tables: observations, extended_properties, episodes, ...   │
└─────────────────────────────────────────────────────────────┘
```

### Data Layer: QueryBackend

**Purpose**: Provide clean Python API for database queries, abstracting SQL details

**Key Features**:
- **aiosql Integration**: SQL queries in separate `.sql` files for easy modification
- **Driver-Specific Adapters**: Direct use of psycopg2/duckdb (not SQLAlchemy)
- **Dual Backend Support**: PostgreSQL and DuckDB with same API
- **Session-Level Aggregation**: Efficient SQL subqueries for multi-episode analysis
- **Parameterized Queries**: Named parameters (`:episode_id`, `:bin_size`, etc.)

**File Structure**:
```
collab_env/data/db/
├── config.py                   # Database configuration
├── init_database.py            # Schema initialization
├── db_loader.py                # Data loading utilities
├── query_backend.py            # Main QueryBackend class ✅
└── queries/                    # SQL query files ✅
    ├── session_metadata.sql
    ├── spatial_analysis.sql
    └── correlations.sql
```

### GUI Layer: Widget System

**Purpose**: Modular, extensible analysis visualization framework

**Key Components**:

1. **SpatialAnalysisGUI** (`spatial_analysis_gui.py`)
   - Main application class
   - Scope selection (episode/session/custom)
   - Shared parameter controls
   - Widget registry management
   - Status indicators and error handling

2. **AnalysisContext** (`widgets/analysis_context.py`)
   - Shared state container
   - QueryBackend instance
   - QueryScope (what to analyze)
   - Shared parameters (bin_size, window_size, etc.)
   - Status callbacks (loading, success, error)

3. **BaseAnalysisWidget** (`widgets/base_analysis_widget.py`)
   - Abstract base class for all widgets
   - Standard lifecycle: create controls → create display → load data
   - Error handling and loading indicators
   - Context-aware query helpers

4. **Concrete Widgets** (`widgets/*_widget.py`)
   - HeatmapWidget: 3D spatial density
   - VelocityStatsWidget: Speed over time
   - DistanceStatsWidget: Distance to target/boundary
   - CorrelationWidget: Velocity correlations

5. **Widget Registry** (`analysis_widgets.yaml`)
   - YAML-based widget configuration
   - Enable/disable widgets
   - Control display order
   - Set default parameters

**File Structure**:
```
collab_env/dashboard/
├── spatial_analysis_gui.py          # Main GUI (~300 lines) ✅
├── spatial_analysis_app.py          # Entry point ✅
├── analysis_widgets.yaml            # Widget configuration ✅
└── widgets/
    ├── __init__.py
    ├── query_scope.py               # QueryScope + ScopeType ✅
    ├── analysis_context.py          # AnalysisContext ✅
    ├── base_analysis_widget.py      # Abstract base ✅
    ├── widget_registry.py           # Config loader ✅
    ├── heatmap_widget.py            # Spatial density ✅
    ├── velocity_widget.py           # Speed stats ✅
    ├── distance_widget.py           # Distance stats ✅
    └── correlation_widget.py        # Correlations ✅
```

---

## Current Status

### ✅ Production Ready (Phase 6 Complete)

**Data Layer:**
- [x] QueryBackend class with aiosql integration
- [x] 11 SQL queries across 3 files
- [x] Session-level aggregation (4 spatial queries)
- [x] Episode-only correlations (2 queries)
- [x] PostgreSQL and DuckDB support
- [x] Comprehensive API validation tests (45 tests passing)

**GUI Layer:**
- [x] Modular widget architecture
- [x] 4 analysis widgets (heatmap, velocity, distance, correlation)
- [x] YAML-based widget registry
- [x] Episode and session scope support
- [x] Shared parameter context
- [x] Loading indicators and error handling
- [x] All 16 widget tests passing

**Running the Dashboard:**
```bash
# Development mode with autoreload
panel serve collab_env/dashboard/spatial_analysis_app.py --dev --show --port 5008

# Production mode
panel serve collab_env/dashboard/spatial_analysis_app.py --port 5008
```

### Known Limitations

1. **Correlation Analysis**: Episode-level only (session scope disabled)
   - Correlations only meaningful within single episodes
   - Attempting session scope shows clear error message

2. **Current Widgets**: Specialized, not property-agnostic
   - Hardcoded for specific metrics (speed, distance)
   - New properties require new widget code

3. **No Animation**: Static visualizations only
   - No temporal animation of tracks
   - No synchronized time series views

---

## Planned Features

### Phase 7: Unified Widget Architecture (24-32 hours)

**Problem**: Current specialized widgets (heatmap, velocity, distance) are inflexible and hardcoded for specific metrics. Adding new properties requires writing new widget code.

**Solution**: Replace 6+ specialized widgets with **3 comprehensive general-purpose viewers** that work with any extended property.

#### Query Layer Simplification

The unified widget architecture will **simplify and generalize** the QueryBackend:

**Current Approach** (Specialized):

- Separate queries for each metric: `get_speed_statistics()`, `get_distance_to_target()`, `get_distance_to_boundary()`
- Hardcoded property IDs in SQL
- Each widget has custom query logic
- Adding new properties requires new SQL queries + new Python methods

**New Approach** (General-Purpose):

- **5 general queries replace dozens of specialized ones:**
  1. `get_episode_tracks` - Positions/velocities for animation (any agent type)
  2. `get_extended_properties_timeseries` - **Any** property time series (aggregated)
  3. `get_property_distributions` - **Any** property histogram data
  4. `get_available_properties` - List available properties for UI
  5. `get_agent_property_timeseries` - Per-agent time series for correlations
- **Properties are data, not code**: Query accepts `property_ids` parameter
- **Automatic support**: New properties work immediately without code changes
- **Simpler maintenance**: Fewer query methods, cleaner API

**Example - Old vs New:**

```python
# OLD: Specialized queries
speed_stats = query.get_speed_statistics(episode_id, window_size=100)
distance_stats = query.get_distance_to_target(episode_id, window_size=100)
boundary_stats = query.get_distance_to_boundary(episode_id, window_size=100)

# NEW: Single general query
properties = query.get_extended_properties_timeseries(
    episode_id=episode_id,
    property_ids=['speed', 'distance_to_target_center', 'distance_to_scene_mesh']
)
# Returns unified DataFrame with all properties
```

**Benefits:**

- ✅ Fewer query methods (5 instead of 10+)
- ✅ Property-agnostic architecture
- ✅ Easier testing (test once, works for all properties)
- ✅ Cleaner codebase (less duplication)
- ✅ Future-proof (new properties automatically supported)

#### Widget 1: Basic Data Viewer (12-16 hours)

**Purpose**: Comprehensive episode-level visualization combining animation, spatial analysis, time series, and distributions.

**Features**:
- **4-Panel Synchronized Layout**:
  1. **Animation Panel** (top-left)
     - Animated 2D/3D scatter plot of agent tracks
     - Playback controls: play/pause, speed (0.1x-10x), time scrubbing
     - Trails with configurable length
     - Toggle agent IDs

  2. **Spatial Heatmap Panel** (top-right)
     - Density heatmap of agent positions over time window
     - 2D/3D matching animation mode
     - Updates with time window changes

  3. **Time Series Panel** (middle)
     - Multi-property time series (user-selected extended properties)
     - Mean ± std bands for aggregated properties
     - Vertical line indicator @ current animation time
     - Click to jump animation to that time

  4. **Histogram Panel** (bottom)
     - Dynamic row of histograms for selected properties
     - One histogram per property
     - Statistics overlay: mean, median, quartiles

- **Property Selection**: Unified checkbox list affecting time series + histograms
- **Full Synchronization**:
  - Time window: All panels respect start_time/end_time
  - Current time: Animation ↔ time series ↔ slider (bidirectional)
  - Property selection: Updates time series + histograms

**Why This Matters**: Any extended property (speed, distance_to_target, acceleration, custom properties) automatically works without code changes.

#### Widget 2: Relative Quantities Viewer (6-8 hours)

**Purpose**: Pairwise interaction analysis showing relative positions and velocities between agents.

**Features**:
- **3-Panel Layout**:
  1. **Relative Position Heatmap**
     - Heatmap of pairwise Δ(x,y,z) distributions
     - Shows spatial separation patterns

  2. **Relative Velocity Heatmap**
     - Heatmap of pairwise Δ(v_x, v_y, v_z) distributions
     - Shows velocity coordination patterns

  3. **Scalar Histograms**
     - Relative distance magnitudes: √(Δx² + Δy² + Δz²)
     - Relative speed magnitudes: √(Δv_x² + Δv_y² + Δv_z²)

- **2D/3D Mode Toggle**: Applies to both heatmaps
- **Episode-Only**: O(n²) complexity limits session aggregation

**Why This Matters**: Reveals agent-agent interaction patterns not visible in single-agent statistics.

#### Widget 3: Correlation Viewer (8-10 hours)

**Purpose**: Analyze relationships between agent properties with dual modes for different questions.

**Strategy**: **Per-Agent-Pair Analysis**
- For each agent i: compute corr(agent_i.property_1, agent_i.property_2)
- Aggregate across agents: **mean and std** of correlation coefficients
- Shows: "On average, how correlated are these properties **within individual agents**?"

**Two Complementary Modes**:

1. **Windowed Mode** (Temporal Evolution)
   - **Plot**: Time series with uncertainty bands
     - X-axis: Window center time
     - Y-axis: Correlation coefficient
     - Line: Mean correlation across agents
     - Shaded band: Mean ± std (inter-agent variability)
   - **Use Case**: "How does the speed-distance relationship evolve over the episode?"
   - **Example**: "Correlation strong early (0.8±0.1), weakens after t=500 (0.3±0.2)"

2. **Lagged Mode** (Temporal Dependencies)
   - **Plot**: 2D heatmap (window_start × lag)
     - X-axis: Lag (frames, e.g., -50 to +50)
     - Y-axis: Window start time
     - Color: Mean correlation coefficient across agents
     - Colormap: RdBu_r (diverging, centered at 0)
     - Toggle: Show std instead of mean
   - **Use Case**: "Does speed at time t predict distance at time t+lag? Does this change over the episode?"
   - **Interpretation**:
     - Vertical band at lag=k: Property_1 consistently leads by k frames
     - Horizontal band at time=t: Lag relationships change at this time
     - Diagonal patterns: Lag relationship evolves linearly

**Variable Agent Handling**:
- Per-window filtering: Only include agents present ≥80% of window
- Minimum sample size: Require ≥3 agents per window/cell
- Track n_agents for data quality monitoring

**Why This Matters**:
- Reveals temporal dependencies (e.g., "speed leads distance by 10 frames")
- Shows how relationships change over episode (e.g., "correlation breaks down at t=500 when target appears")
- Quantifies behavior heterogeneity (high std = agents behave differently)

### Phase 8: Property Computation Framework (6-8 hours)

**Purpose**: Automated computation and storage of derived properties (speed, acceleration, etc.)

**Features**:
- Property computation functions (speed, acceleration components/magnitude)
- Integration with data import pipeline (optional flag)
- Standalone CLI tool for batch computation
- Proper handling of edge cases (2D data, missing velocities, first/last frames)

**Deliverables**:
- `collab_env/data/db/property_computations.py`
- `collab_env/data/db/compute_properties_cli.py`
- Integration with `db_loader.py`
- Comprehensive tests

---

## Implementation Roadmap

### Phase 7.1: Basic Data Viewer (12-16 hours)

**Step 1: SQL Queries (2-3 hours)**
- [ ] Create `queries/basic_data_viewer.sql`
- [ ] Implement 4 queries:
  - `get_episode_tracks`: Position/velocity data for animation
  - `get_extended_properties_timeseries`: Aggregated property time series
  - `get_property_distributions`: Raw property values for histograms
  - `get_available_properties`: List available properties
- [ ] Test queries with real episode data
- [ ] Verify performance (<2s for typical episode)

**Step 2: QueryBackend Integration (2 hours)**
- [ ] Add 4 methods to `QueryBackend`
- [ ] Handle array parameters for `property_ids`
- [ ] Test with various parameter combinations

**Step 3: Widget Core & Layout (3-4 hours)**
- [ ] Create `BasicDataViewerWidget` class skeleton
- [ ] Implement 4-panel layout (animation, heatmap, timeseries, histograms)
- [ ] Implement `load_data()` to fetch all data
- [ ] Add playback controls

**Step 4: Panel Visualizations (4-5 hours)**
- [ ] Animation panel: HoloViews DynamicMap with trails
- [ ] Heatmap panel: Reuse existing heatmap code
- [ ] Time series panel: Bokeh multi-line with sync indicator
- [ ] Histogram panel: Dynamic row of hvPlot histograms

**Step 5: Synchronization (2-3 hours)**
- [ ] Time window synchronization (all panels)
- [ ] Current time synchronization (animation ↔ time series)
- [ ] Property selection synchronization (time series + histograms)
- [ ] Handle edge cases (no properties, empty data)

**Step 6: Testing & Documentation (2 hours)**
- [ ] Create `tests/dashboard/test_basic_data_viewer.py`
- [ ] Test with 2D and 3D episodes
- [ ] Test property selection
- [ ] Update documentation

### Phase 7.2: Relative Quantities Viewer (6-8 hours)

**Step 1: SQL Queries (2-3 hours)**
- [ ] Create `queries/pairwise_analysis.sql`
- [ ] Implement pairwise distance query (O(n²) warning)
- [ ] Implement pairwise velocity query
- [ ] Test performance with different agent counts

**Step 2: QueryBackend Integration (1 hour)**
- [ ] Add `get_pairwise_distances()` method
- [ ] Add `get_pairwise_relative_velocities()` method

**Step 3: Widget Implementation (2-3 hours)**
- [ ] Create `RelativeQuantitiesViewerWidget` class
- [ ] Implement 3-panel layout
- [ ] Bin pairwise vectors for heatmaps
- [ ] Compute scalar magnitudes for histograms

**Step 4: Testing (1-2 hours)**
- [ ] Verify correct number of pairs: n(n-1)/2
- [ ] No self-pairs or duplicates
- [ ] Accurate distance/velocity calculations

### Phase 7.3: Correlation Viewer (8-10 hours)

**Step 1: SQL Queries (2 hours)**
- [ ] Extend `queries/correlations.sql`
- [ ] Implement `get_agent_property_timeseries` query
- [ ] Returns per-agent time series for property pairs

**Step 2: Correlation Computation (3-4 hours)**
- [ ] Implement `_compute_windowed_correlations()` method
  - Per-agent correlations with presence filtering
  - Aggregate mean/std across agents
  - Return DataFrame: [window_center, mean_corr, std_corr, n_agents]
- [ ] Implement `_compute_lagged_correlations()` method
  - Nested loops over windows and lags
  - Per-agent correlations at each (window, lag)
  - Return DataFrame: [window_start, lag, mean_corr, std_corr, n_agents]

**Step 3: Visualization (2-3 hours)**
- [ ] Windowed mode: Bokeh line plot with error bands
- [ ] Lagged mode: HoloViews 2D heatmap with diverging colormap
- [ ] Statistics panel: n_agents tracking, warnings

**Step 4: Testing (1-2 hours)**
- [ ] Test with known correlation patterns
- [ ] Verify agent filtering logic
- [ ] Test edge cases (few agents, sparse data)

### Phase 8: Property Computation (6-8 hours)

**Step 1: Core Functions (2-3 hours)**
- [ ] Implement `compute_speed()` function
- [ ] Implement `compute_acceleration()` function
- [ ] Handle 2D vs 3D data
- [ ] Edge case handling (missing data, first/last frames)

**Step 2: Integration (2 hours)**
- [ ] Extend `db_loader.py` with `compute_derived_properties()` method
- [ ] Add optional flag to `load_episode()`
- [ ] Proper storage in extended_properties table

**Step 3: CLI Tool (2-3 hours)**
- [ ] Create `compute_properties_cli.py`
- [ ] Support episode and session selection
- [ ] Progress bars for large datasets
- [ ] Dry-run mode

**Step 4: Testing (1-2 hours)**
- [ ] Unit tests for computation functions
- [ ] Integration tests with database
- [ ] CLI argument validation

---

## Technical Reference

### QueryBackend API

#### Session Metadata

```python
from collab_env.data.db.query_backend import QueryBackend

query = QueryBackend()  # Auto-detects backend from environment

# List sessions
sessions = query.get_sessions(category_id='boids_3d')
# Returns: session_id, session_name, category_id, created_at

# List episodes for session
episodes = query.get_episodes(session_id='...')
# Returns: episode_id, episode_number, num_frames, num_agents, frame_rate

# Get episode metadata
metadata = query.get_episode_metadata(episode_id='...')
# Returns: episode_id, session_id, episode_number, num_frames, config
```

#### Spatial Analysis

```python
# Spatial heatmap (supports episode OR session)
heatmap = query.get_spatial_heatmap(
    episode_id='episode-0-...',  # OR session_id='session-...'
    bin_size=10.0,
    start_time=500,
    end_time=3000,
    agent_type='agent'
)
# Returns: x_bin, y_bin, density, avg_vx, avg_vy

# Speed statistics over time windows
speed_stats = query.get_speed_statistics(
    episode_id='...',
    window_size=100,
    agent_type='agent'
)
# Returns: time_window, n_observations, avg_speed, std_speed,
#          median_speed, min_speed, max_speed

# Distance to target
dist_target = query.get_distance_to_target(
    episode_id='...',
    window_size=100,
    agent_type='agent'
)
# Returns: time_window, avg_distance, std_distance, min_distance, max_distance

# Distance to boundary
dist_boundary = query.get_distance_to_boundary(
    episode_id='...',
    window_size=100,
    agent_type='agent'
)
```

#### Correlations (Episode-Only)

```python
# Velocity correlations between agents
vel_corr = query.get_velocity_correlations(
    episode_id='...',
    start_time=500,
    end_time=3000
)
# Returns: agent_i, agent_j, v_x_correlation, v_y_correlation,
#          v_z_correlation, n_samples

# Distance correlations
dist_corr = query.get_distance_correlations(
    episode_id='...',
    start_time=500,
    end_time=3000
)
# Returns: agent_i, agent_j, distance_correlation, n_samples
```

### Creating a Custom Widget

```python
from collab_env.dashboard.widgets.base_analysis_widget import BaseAnalysisWidget
import param
import panel as pn
import holoviews as hv

class MyCustomWidget(BaseAnalysisWidget):
    """Example custom analysis widget."""

    # Metadata
    widget_name = "My Analysis"
    widget_description = "Custom analysis visualization"
    widget_category = "custom"

    # Widget-specific parameters
    threshold = param.Number(default=0.5, bounds=(0, 1), doc="Analysis threshold")
    color_scale = param.Selector(default='viridis',
                                  objects=['viridis', 'plasma', 'inferno'])

    def create_custom_controls(self):
        """Create widget-specific controls."""
        return pn.Column(
            "### Custom Parameters",
            pn.widgets.FloatSlider.from_param(self.param.threshold),
            pn.widgets.Select.from_param(self.param.color_scale)
        )

    def create_display_pane(self):
        """Create visualization pane (empty state)."""
        return pn.pane.HoloViews(
            hv.Curve([]).opts(width=700, height=500),
            sizing_mode="stretch_both"
        )

    def load_data(self):
        """Query data and update visualization."""
        # Access shared parameters via self.context
        # Access widget parameters via self
        df = self.query_with_context(
            'get_my_custom_data',
            threshold=self.threshold  # Widget-specific parameter
        )

        if len(df) == 0:
            raise ValueError("No data found for selected parameters")

        # Create visualization
        curve = hv.Curve(df, kdims='x', vdims='y').opts(
            color=self.color_scale,
            width=800,
            height=600,
            title=f'My Analysis (threshold={self.threshold})'
        )

        self.display_pane.object = curve
        logger.info(f"Loaded {len(df)} data points")
```

**Register in `analysis_widgets.yaml`:**
```yaml
  - class: my_package.widgets.my_custom_widget.MyCustomWidget
    enabled: true
    order: 10
    category: custom
    description: "Custom analysis visualization"
```

---

## Usage Examples

### Example 1: Compare Before/After Target Appears

```python
from collab_env.data.db.query_backend import QueryBackend
import matplotlib.pyplot as plt

query = QueryBackend()

# Get first episode
sessions = query.get_sessions(category_id='boids_3d')
episode_id = query.get_episodes(sessions.iloc[0]['session_id']).iloc[0]['episode_id']

# Get speed before target (t < 500)
speed_before = query.get_speed_statistics(episode_id, window_size=50)
speed_before = speed_before[speed_before['time_window'] < 500]

# Get speed after target (t >= 500)
speed_after = query.get_speed_statistics(episode_id, window_size=50)
speed_after = speed_after[speed_after['time_window'] >= 500]

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(speed_before['time_window'], speed_before['avg_speed'])
ax1.fill_between(speed_before['time_window'],
                 speed_before['avg_speed'] - speed_before['std_speed'],
                 speed_before['avg_speed'] + speed_before['std_speed'],
                 alpha=0.3)
ax1.set_title('Speed Before Target (t<500)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Speed')

ax2.plot(speed_after['time_window'], speed_after['avg_speed'])
ax2.fill_between(speed_after['time_window'],
                 speed_after['avg_speed'] - speed_after['std_speed'],
                 speed_after['avg_speed'] + speed_after['std_speed'],
                 alpha=0.3)
ax2.set_title('Speed After Target (t>=500)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Speed')

plt.tight_layout()
plt.show()

query.close()
```

### Example 2: Session-Level Aggregation

```python
query = QueryBackend()

# Get session
sessions = query.get_sessions(category_id='boids_3d')
session_id = sessions.iloc[0]['session_id']

# Get aggregated heatmap across ALL episodes in session
# (SQL efficiently aggregates at database level)
heatmap = query.get_spatial_heatmap(
    session_id=session_id,
    bin_size=20.0,
    agent_type='agent'
)

# Plot
pivot = heatmap.pivot(index='y_bin', columns='x_bin', values='density')
plt.imshow(pivot, origin='lower', cmap='viridis')
plt.colorbar(label='Density (aggregated across all episodes)')
plt.title(f'Session-Level Spatial Distribution')
plt.show()

query.close()
```

### Example 3: Agent-Type Comparison

```python
query = QueryBackend()
episode_id = '...'

# Get heatmaps for agents and targets
heatmap_agents = query.get_spatial_heatmap(
    episode_id=episode_id,
    bin_size=20.0,
    agent_type='agent'
)
heatmap_targets = query.get_spatial_heatmap(
    episode_id=episode_id,
    bin_size=20.0,
    agent_type='target'
)

# Pivot and plot side-by-side
grid_agents = heatmap_agents.pivot(index='y_bin', columns='x_bin', values='density').fillna(0)
grid_targets = heatmap_targets.pivot(index='y_bin', columns='x_bin', values='density').fillna(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(grid_agents, origin='lower', cmap='viridis')
ax1.set_title('Agents')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(grid_targets, origin='lower', cmap='plasma')
ax2.set_title('Targets')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
```

---

## Appendix

### Database Schema

See [`collab_env/data/db/README.md`](../../collab_env/data/db/README.md) for complete schema documentation.

**Key Tables:**
- `sessions`: Session metadata
- `episodes`: Episode metadata and configuration
- `observations`: Position, velocity, agent type per timestep
- `extended_properties`: Derived properties (speed, distance, etc.)
- `property_definitions`: Property metadata

### Dependencies

**Core:**
- `sqlalchemy` - Database toolkit
- `psycopg2-binary` - PostgreSQL driver
- `duckdb` - DuckDB driver
- `aiosql>=9.0` - SQL query management
- `pandas` - Data manipulation

**GUI:**
- `panel>=1.3.0` - Dashboard framework
- `holoviews>=1.18.0` - Visualization library
- `bokeh>=3.3.0` - Plotting backend
- `param` - Parameterized classes

### Running Tests

```bash
# All dashboard tests
pytest tests/dashboard/ -v

# Specific test modules
pytest tests/dashboard/test_analysis_widgets.py -v
pytest tests/dashboard/test_api_validation.py -v

# With coverage
pytest tests/dashboard/ --cov=collab_env.dashboard --cov-report=html
```

### Performance Notes

**Expected Query Performance** (DuckDB, ~90K observations per episode):
- Heatmap: < 1 second
- Velocity distribution: < 2 seconds
- Speed statistics: < 1 second
- Distance statistics: < 2 seconds
- Correlations: < 5 seconds (O(n²) pairwise)

**Session Aggregation** (10 episodes):
- Heatmap: 2-3 seconds (vs 10-15s Python-level)
- Speed statistics: 3-4 seconds (vs 20-30s Python-level)

**Scalability**:
- Episodes: Tested up to 100 episodes per session
- Agents: Tested up to 50 agents per episode
- Correlation queries slow beyond 30 agents (O(n²))

---

**End of Document**
