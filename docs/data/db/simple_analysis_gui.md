# Spatial Analysis GUI: Implementation Guide

## Overview

This document describes the implementation of Phase 5 (Query Backend) and Phase 6 (Simple Analysis GUI) for the tracking analytics database layer. The goal is to provide:

1. **Query Backend**: A Python interface for executing spatial analysis queries on 3D boids data
2. **Analysis GUI**: A Panel/HoloViz-based web dashboard for interactive data exploration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Spatial Analysis GUI                      │
│              (Panel/HoloViz Web Interface)                   │
│                                                               │
│  Tabs: Heatmap | Velocity | Distances | Correlations        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Python API calls
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     QueryBackend Class                       │
│          (collab_env/data/db/query_backend.py)              │
│                                                               │
│  Methods: get_spatial_heatmap(), get_velocity_stats(), ...  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Loads SQL via aiosql
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   SQL Query Files (.sql)                     │
│             (collab_env/data/db/queries/)                   │
│                                                               │
│  - spatial_analysis.sql    (heatmaps, velocities)           │
│  - correlations.sql        (correlation queries)            │
│  - session_metadata.sql    (session/episode queries)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Executed via SQLAlchemy
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Database (PostgreSQL or DuckDB)                 │
│                                                               │
│  Tables: observations, extended_properties, episodes, ...   │
└─────────────────────────────────────────────────────────────┘
```

## SQL Query Management with aiosql

### Using aiosql with Driver-Specific Adapters

We use **aiosql** with driver-specific adapters (psycopg2 for PostgreSQL, duckdb for DuckDB) to manage SQL queries separately from Python code:

- **Pure SQL**: Queries written in `.sql` files, not Python strings
- **Easy Inspection**: View all queries without opening Python code
- **Easy Modification**: Edit queries without touching application logic
- **Parameterization**: Named parameters like `:episode_id`, `:bin_size`
- **Database Driver Direct**: Works directly with psycopg2/duckdb, not SQLAlchemy
- **Industry Standard**: aiosql is widely used for SQL file management
- **Version Control**: SQL changes clearly visible in git diffs
- **Reusable**: Same queries usable from notebooks, CLI tools, dashboards

**Why driver-specific adapters?** We use aiosql with `psycopg2` and `duckdb` adapters directly rather than SQLAlchemy. This makes the query layer independent of SQLAlchemy, providing flexibility to change ORMs or use raw connections in the future.

### Query File Structure

```
collab_env/data/db/queries/
├── __init__.py                # Empty or with helper functions
├── spatial_analysis.sql       # Heatmaps, velocities, distances (6 queries)
├── correlations.sql           # Velocity and distance correlations (2 queries)
└── session_metadata.sql       # Session/episode metadata (3 queries)
```

### Query File Format

Each `.sql` file contains multiple named queries:

```sql
-- name: get_spatial_heatmap
-- Get spatial density heatmap with configurable binning
SELECT
    floor(x / :bin_size) * :bin_size as x_bin,
    floor(y / :bin_size) * :bin_size as y_bin,
    count(*) as density,
    avg(v_x) as avg_vx,
    avg(v_y) as avg_vy
FROM observations
WHERE episode_id = :episode_id
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
GROUP BY x_bin, y_bin
ORDER BY x_bin, y_bin;

-- name: get_velocity_distribution
-- Get raw velocity vectors for distribution analysis
SELECT
    agent_id,
    time_index,
    v_x, v_y, v_z,
    sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0)) as speed
FROM observations
WHERE episode_id = :episode_id
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
  AND v_x IS NOT NULL
ORDER BY time_index, agent_id;
```

### How to Add/Modify Queries

1. **Add New Query**: Edit appropriate `.sql` file, add new `-- name:` section
2. **Modify Existing Query**: Edit SQL directly in `.sql` file
3. **Add Method to QueryBackend**: Add Python wrapper method calling `_execute_query()`
4. **No Code Changes Needed**: For simple SQL modifications, just edit `.sql` file and restart

**Example: Adding a new query**
```sql
-- In spatial_analysis.sql:
-- name: get_acceleration_statistics
-- Compute acceleration statistics over time windows
SELECT
    floor(time_index / :window_size) * :window_size as time_window,
    avg(sqrt(a_x*a_x + a_y*a_y)) as avg_acceleration
FROM observations
WHERE episode_id = :episode_id
GROUP BY time_window;
```

Then add to QueryBackend:
```python
def get_acceleration_statistics(self, episode_id: str, window_size: int = 100) -> pd.DataFrame:
    """Get acceleration statistics."""
    return self._execute_query('get_acceleration_statistics',
                                episode_id=episode_id,
                                window_size=window_size)
```

## Query Reference

### Session Metadata Queries

#### `get_sessions(category_id=None)`

**Purpose**: List all available sessions, optionally filtered by category

**Parameters**:
- `category_id` (str, optional): Filter by category ('boids_3d', 'boids_2d', 'tracking_csv')

**Returns**: DataFrame with columns:
- `session_id`, `session_name`, `category_id`, `created_at`

**SQL**: See `session_metadata.sql`

#### `get_episodes(session_id)`

**Purpose**: List all episodes for a given session

**Parameters**:
- `session_id` (str, required): Session identifier

**Returns**: DataFrame with columns:
- `episode_id`, `episode_number`, `num_frames`, `num_agents`, `frame_rate`, `file_path`

**SQL**: See `session_metadata.sql`

#### `get_episode_metadata(episode_id)`

**Purpose**: Get detailed metadata for a single episode

**Parameters**:
- `episode_id` (str, required): Episode identifier

**Returns**: DataFrame with columns:
- `episode_id`, `session_id`, `episode_number`, `num_frames`, `num_agents`, `frame_rate`, `session_name`, `config`

**SQL**: See `session_metadata.sql`

---

### Spatial Analysis Queries

#### `get_spatial_heatmap(episode_id, bin_size, start_time, end_time, agent_type)`

**Purpose**: Compute spatial density heatmap with binned positions

**Parameters**:
- `episode_id` (str, required): Episode to analyze
- `bin_size` (float, default=10.0): Spatial bin size in scene units
- `start_time` (int, optional): Start time index (None = from beginning)
- `end_time` (int, optional): End time index (None = to end)
- `agent_type` (str, default='agent'): Agent type to filter ('agent', 'target', 'all')

**Returns**: DataFrame with columns:
- `x_bin`, `y_bin`: Bin center coordinates
- `density`: Count of observations in bin
- `avg_vx`, `avg_vy`: Average velocities in bin

**Use Cases**:
- Position heatmaps (visited locations)
- Before/after t=500 comparison
- Per-agent-type spatial distributions

**SQL**: See `spatial_analysis.sql`

#### `get_velocity_heatmap(episode_id, bin_size, start_time, end_time, agent_type)`

**Purpose**: Compute velocity field on spatial grid (for quiver plots)

**Parameters**: Same as `get_spatial_heatmap`

**Returns**: DataFrame with columns:
- `x_bin`, `y_bin`: Bin center coordinates
- `count`: Number of observations in bin
- `avg_vx`, `avg_vy`, `avg_vz`: Average velocity components
- `avg_speed`: Average speed magnitude

**Use Cases**:
- Quiver plots (velocity arrows on spatial grid)
- Flow field visualization

**SQL**: See `spatial_analysis.sql`

#### `get_velocity_distribution(episode_id, start_time, end_time, agent_type)`

**Purpose**: Get raw velocity vectors for distribution analysis

**Parameters**:
- `episode_id` (str, required)
- `start_time` (int, optional)
- `end_time` (int, optional)
- `agent_type` (str, default='agent')

**Returns**: DataFrame with columns:
- `agent_id`, `time_index`
- `v_x`, `v_y`, `v_z`: Velocity components
- `speed`: Speed magnitude

**Use Cases**:
- Velocity vector distributions (scatter plots, histograms)
- Speed distributions
- Raw data for custom analysis

**SQL**: See `spatial_analysis.sql`

#### `get_speed_statistics(episode_id, window_size, agent_type)`

**Purpose**: Compute speed statistics over time windows

**Parameters**:
- `episode_id` (str, required)
- `window_size` (int, default=100): Number of frames per window
- `agent_type` (str, default='agent')

**Returns**: DataFrame with columns:
- `time_window`: Window start time
- `n_observations`: Count of observations in window
- `avg_speed`, `std_speed`: Speed mean and standard deviation
- `median_speed`, `min_speed`, `max_speed`: Speed quantiles

**Use Cases**:
- Speed over time plots
- Moving window statistics
- Identify speed changes (e.g., at t=500 when target appears)

**SQL**: See `spatial_analysis.sql`

#### `get_distance_to_target(episode_id, window_size, agent_type)`

**Purpose**: Compute distance to target statistics over time windows

**Parameters**:
- `episode_id` (str, required)
- `window_size` (int, default=100): Number of frames per window
- `agent_type` (str, default='agent')

**Returns**: DataFrame with columns:
- `time_window`: Window start time
- `avg_distance`, `std_distance`: Distance mean and standard deviation
- `min_distance`, `max_distance`: Distance range
- `n_observations`: Count of observations

**Use Cases**:
- Distance to target over time
- Approach behavior analysis
- Before/after t=500 comparison

**SQL**: See `spatial_analysis.sql`

#### `get_distance_to_boundary(episode_id, window_size, agent_type)`

**Purpose**: Compute distance to scene boundary statistics over time windows

**Parameters**: Same as `get_distance_to_target`

**Returns**: Same schema as `get_distance_to_target`

**Use Cases**:
- Boundary avoidance analysis
- Wall proximity over time

**SQL**: See `spatial_analysis.sql`

---

### Correlation Queries

#### `get_velocity_correlations(episode_id, start_time, end_time)`

**Purpose**: Compute pairwise velocity correlations between agents

**Parameters**:
- `episode_id` (str, required)
- `start_time` (int, optional)
- `end_time` (int, optional)

**Returns**: DataFrame with columns:
- `agent_i`, `agent_j`: Agent pair
- `v_x_correlation`, `v_y_correlation`, `v_z_correlation`: Component correlations
- `n_samples`: Number of time points used

**Use Cases**:
- Velocity alignment analysis
- Flocking behavior quantification
- Agent-agent interaction strength

**SQL**: See `correlations.sql`

**Note**: This query can be slow for many agents (O(n²) pairs). Consider filtering or limiting to subset of agents.

#### `get_distance_correlations(episode_id, start_time, end_time)`

**Purpose**: Compute pairwise distance-to-target correlations between agents

**Parameters**: Same as `get_velocity_correlations`

**Returns**: DataFrame with columns:
- `agent_i`, `agent_j`: Agent pair
- `distance_correlation`: Correlation of distances to target
- `n_samples`: Number of time points used

**Use Cases**:
- Coordinated approach behavior
- Leader-follower detection

**SQL**: See `correlations.sql`

---

## QueryBackend API

### Class: `QueryBackend`

```python
from collab_env.data.db.query_backend import QueryBackend

# Initialize (reads DB_BACKEND from environment)
query = QueryBackend()

# Or specify backend
query = QueryBackend(backend='duckdb')

# Session/episode queries
sessions_df = query.get_sessions(category_id='boids_3d')
episodes_df = query.get_episodes(session_id='...')
metadata_df = query.get_episode_metadata(episode_id='...')

# Spatial analysis
heatmap_df = query.get_spatial_heatmap(
    episode_id='episode-0-...',
    bin_size=10.0,
    start_time=500,  # After target appears
    end_time=3000,
    agent_type='agent'
)

velocity_df = query.get_velocity_distribution(
    episode_id='episode-0-...',
    start_time=0,
    end_time=500,  # Before target
    agent_type='agent'
)

speed_stats_df = query.get_speed_statistics(
    episode_id='episode-0-...',
    window_size=100,
    agent_type='agent'
)

# Distance analysis
dist_target_df = query.get_distance_to_target(
    episode_id='episode-0-...',
    window_size=100,
    agent_type='agent'
)

dist_boundary_df = query.get_distance_to_boundary(
    episode_id='episode-0-...',
    window_size=100,
    agent_type='agent'
)

# Correlations
vel_corr_df = query.get_velocity_correlations(
    episode_id='episode-0-...',
    start_time=500,
    end_time=3000
)

dist_corr_df = query.get_distance_correlations(
    episode_id='episode-0-...',
    start_time=500,
    end_time=3000
)

# Close connection when done
query.close()
```

### Usage from Notebooks

```python
import pandas as pd
import matplotlib.pyplot as plt
from collab_env.data.db.query_backend import QueryBackend

# Initialize
query = QueryBackend()

# Get sessions
sessions = query.get_sessions(category_id='boids_3d')
print(sessions)

# Pick first session, get episodes
session_id = sessions.iloc[0]['session_id']
episodes = query.get_episodes(session_id)
print(episodes)

# Pick first episode, get heatmap
episode_id = episodes.iloc[0]['episode_id']
heatmap = query.get_spatial_heatmap(episode_id, bin_size=20.0)

# Plot heatmap
pivot = heatmap.pivot(index='y_bin', columns='x_bin', values='density')
plt.imshow(pivot, origin='lower', cmap='viridis')
plt.colorbar(label='Density')
plt.title('Spatial Heatmap')
plt.show()

query.close()
```

---

## Spatial Analysis GUI

### Overview

Panel/HoloViz-based web dashboard for interactive spatial analysis of 3D boids data.

### Running the Dashboard

```bash
# Install dependencies
pip install -r requirements-db.txt

# Development mode with autoreload
panel serve collab_env/dashboard/spatial_analysis_app.py --dev --show --port 5008

# Production mode
panel serve collab_env/dashboard/spatial_analysis_app.py --port 5008
```

Then open browser to: http://localhost:5008

### GUI Components

#### Sidebar Controls

**Session Selection:**
- **Session Selector**: Dropdown to select session (auto-populated from database)
- **Episode Selector**: Dropdown to select episode (populated after session selection)
- **Agent Type Filter**: Radio buttons for 'agent', 'target', 'all'

**Time Range:**
- **Start Time Slider**: Adjust start of time window (0 to num_frames)
- **End Time Slider**: Adjust end of time window (0 to num_frames)
- **Quick Buttons**: "Before t=500", "After t=500", "Full Range"

**Analysis Parameters:**
- **Bin Size**: Float input for spatial binning (1-100 scene units)
- **Window Size**: Integer input for time window size (10-1000 frames)

#### Main Content Tabs

**Tab 1: Heatmap**
- **Load Button**: Triggers query execution
- **Visualization**: HoloViews heatmap showing spatial density
- **Colormap**: Viridis (density)
- **Interactivity**: Pan, zoom, hover tooltips

**Tab 2: Velocity Stats**
- **Load Button**: Triggers query execution
- **Plot 1**: Speed over time (line plot with error bands)
- **Plot 2**: Velocity component distributions (histograms)
- **Interactivity**: Linked brushing between plots

**Tab 3: Distances**
- **Load Button**: Triggers query execution
- **Plot 1**: Distance to target over time
- **Plot 2**: Distance to boundary over time
- **Both plots**: Mean ± std deviation bands

**Tab 4: Correlations**
- **Load Button**: Triggers query execution
- **Velocity Correlations**: Table showing top correlated agent pairs
- **Distance Correlations**: Table showing distance correlation pairs
- **Sortable**: Click column headers to sort

#### Status Indicators

- **Loading Modal**: Appears during query execution
- **Status Bar**: Shows success/error messages with icons
- **Progress**: "Loading..." or "✅ Data loaded" or "❌ Error: ..."

### GUI Architecture

```python
class SpatialAnalysisGUI(param.Parameterized):
    """Main GUI class."""

    # Reactive parameters
    selected_session = param.String(default="")
    selected_episode = param.String(default="")
    agent_type = param.Selector(default="agent", objects=["agent", "target", "all"])
    start_time = param.Integer(default=0, bounds=(0, 10000))
    end_time = param.Integer(default=3000, bounds=(0, 10000))

    def __init__(self):
        # Initialize QueryBackend
        self.query = QueryBackend()

        # Create widgets
        self._create_widgets()

        # Wire callbacks
        self._wire_callbacks()

        # Load initial data
        self._load_sessions()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Selectors, sliders, buttons, etc.

    def _wire_callbacks(self):
        """Wire widget callbacks."""
        self.session_select.param.watch(self._on_session_change, "value")
        self.episode_select.param.watch(self._on_episode_change, "value")
        self.load_heatmap_btn.on_click(self._load_heatmap)
        # etc.

    def _load_heatmap(self, event):
        """Load and display heatmap."""
        self._show_loading("Loading heatmap...")
        try:
            df = self.query.get_spatial_heatmap(...)
            # Create HoloViews visualization
            heatmap = hv.HeatMap(df, kdims=['x_bin', 'y_bin'], vdims='density')
            self.heatmap_pane.object = heatmap
            self._hide_loading()
            self.status_pane.object = "<p>✅ Heatmap loaded</p>"
        except Exception as e:
            self._hide_loading()
            self.status_pane.object = f"<p style='color:red'>❌ Error: {e}</p>"

    def create_layout(self):
        """Create Panel layout."""
        sidebar = pn.Column(...)
        content = pn.Tabs(...)
        return pn.template.MaterialTemplate(...)
```

### Visualization Examples

**Heatmap (HoloViews):**
```python
import holoviews as hv
from holoviews import opts

heatmap = hv.HeatMap(df, kdims=['x_bin', 'y_bin'], vdims='density')
heatmap.opts(
    opts.HeatMap(
        cmap='viridis',
        width=700,
        height=500,
        colorbar=True,
        tools=['hover'],
        title='Spatial Density'
    )
)
```

**Line Plot with Error Bands:**
```python
import holoviews as hv

# Speed over time
curve = hv.Curve(df, kdims='time_window', vdims='avg_speed', label='Mean Speed')
area = hv.Area(df, kdims='time_window', vdims=['lower', 'upper'], label='±1 std')
overlay = curve * area
overlay.opts(
    opts.Curve(color='blue', line_width=2),
    opts.Area(alpha=0.3, color='blue'),
    width=700,
    height=400
)
```

**Correlation Table (Tabulator):**
```python
import panel as pn

table = pn.widgets.Tabulator(
    df,
    width=800,
    height=400,
    page_size=20,
    sorters=[{'column': 'v_x_correlation', 'dir': 'desc'}],
    show_index=False
)
```

---

## Development Roadmap

### Phase 5: Query Backend ✅ (Current)

**Status**: Implementation in progress

**Deliverables**:
- [x] SQL query files with aiosql
- [x] QueryBackend class
- [ ] Test with loaded 3D boids data
- [ ] Document query performance

**Queries Implemented**:
- Session/episode metadata (3 queries)
- Spatial analysis (6 queries)
- Correlations (2 queries)

### Phase 6: Simple GUI ✅ (Current)

**Status**: Implementation in progress

**Deliverables**:
- [ ] SpatialAnalysisGUI class
- [ ] Panel app entry point
- [ ] 4 analysis tabs (Heatmap, Velocity, Distances, Correlations)
- [ ] Basic visualizations

**Features**:
- Session/episode selection
- Agent type filtering
- Time range controls
- Loading indicators
- HoloViews visualizations

### Phase 7: Advanced Features ⏳ (Future)

**Planned Features**:
1. **Quiver Plots**: Velocity arrows on spatial grid
2. **3D Visualizations**: VTK integration for 3D trajectories
3. **Multi-Episode Comparison**: Overlay multiple episodes
4. **Animation**: Playback with time slider
5. **Export**: Save plots as PNG/SVG, data as CSV
6. **Clustering Statistics**: Once Tom defines requirements
7. **Aggregation**: Combine multiple episodes from same config

### Phase 8: Integration ⏳ (Future)

**Planned Integration**:
1. **Main Dashboard**: Link from GCS dashboard to spatial analysis
2. **Jupyter Notebooks**: Example notebooks using QueryBackend
3. **CLI Tool**: Command-line interface for batch analysis
4. **Grafana**: PostgreSQL-based dashboards (requires different queries)

---

## Requirements

### Python Dependencies

Add to `requirements-db.txt`:
```
aiosql>=9.0
panel>=1.3.0
holoviews>=1.18.0
bokeh>=3.3.0
```

Existing dependencies already include:
- sqlalchemy
- pandas
- psycopg2-binary (PostgreSQL)
- duckdb (DuckDB)

### Database

- Initialized database with schema (see [Database README](README.md))
- Loaded 3D boids data (at least one session with episodes)

### Environment

- `.env` file with database configuration (optional, defaults to DuckDB)
- See `.env.example` for template

---

## Testing

### Manual Testing

**Test Query Backend:**
```python
from collab_env.data.db.query_backend import QueryBackend

query = QueryBackend()

# Test session listing
sessions = query.get_sessions(category_id='boids_3d')
assert len(sessions) > 0, "No sessions found"
print(f"Found {len(sessions)} sessions")

# Test episode listing
session_id = sessions.iloc[0]['session_id']
episodes = query.get_episodes(session_id)
assert len(episodes) > 0, "No episodes found"
print(f"Found {len(episodes)} episodes")

# Test heatmap query
episode_id = episodes.iloc[0]['episode_id']
heatmap = query.get_spatial_heatmap(episode_id, bin_size=10.0)
assert len(heatmap) > 0, "Empty heatmap"
print(f"Heatmap has {len(heatmap)} bins")

query.close()
print("✅ All tests passed")
```

**Test GUI:**
```bash
# Run dashboard
panel serve collab_env/dashboard/spatial_analysis_app.py --dev --show

# Manual checks:
# 1. Sessions dropdown populates
# 2. Episodes dropdown populates after session selection
# 3. Load Heatmap button triggers query and displays plot
# 4. Time sliders work and update queries
# 5. Agent type filter works
```

### Performance Testing

**Query Performance:**
```python
import time
from collab_env.data.db.query_backend import QueryBackend

query = QueryBackend()

# Get test episode
sessions = query.get_sessions(category_id='boids_3d')
episode_id = query.get_episodes(sessions.iloc[0]['session_id']).iloc[0]['episode_id']

# Time heatmap query
start = time.time()
heatmap = query.get_spatial_heatmap(episode_id, bin_size=10.0)
duration = time.time() - start
print(f"Heatmap query: {duration:.2f}s, {len(heatmap)} bins")

# Time velocity query
start = time.time()
velocity = query.get_velocity_distribution(episode_id)
duration = time.time() - start
print(f"Velocity query: {duration:.2f}s, {len(velocity)} rows")

query.close()
```

**Expected Performance** (DuckDB, 90K observations per episode):
- Heatmap: < 1 second
- Velocity distribution: < 2 seconds
- Speed statistics: < 1 second
- Distance statistics: < 2 seconds (joins extended_properties)
- Correlations: < 5 seconds (pairwise computation)

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'aiosql'`

**Solution**: Install dependencies
```bash
pip install -r requirements-db.txt
```

---

**Issue**: `FileNotFoundError: queries/spatial_analysis.sql not found`

**Solution**: aiosql looks for queries relative to script location. Ensure queries are in correct path:
- Queries should be in: `collab_env/data/db/queries/`
- Run scripts from project root: `/Users/dima/git/collab-environment`

---

**Issue**: `No sessions found` when running QueryBackend

**Solution**: Load 3D boids data first:
```bash
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/hackathon-boid-small-200-sim_run-started-20250926-220926
```

---

**Issue**: GUI shows "Loading..." but never completes

**Solution**: Check browser console for errors. Likely causes:
1. Database connection failed (check `.env` configuration)
2. SQL query error (check Python console for traceback)
3. Empty result (verify data loaded in database)

---

**Issue**: Correlation queries very slow (> 30 seconds)

**Solution**: Correlations compute O(n²) agent pairs. For 30 agents, that's 435 pairs.
- Reduce time range (`start_time`, `end_time`)
- Consider limiting to subset of agents
- Use PostgreSQL instead of DuckDB (better optimizer)

---

## Integration with Existing Database Layer

### Compatibility

**Uses Existing Infrastructure**:
- `collab_env.data.db.config.DBConfig` - Database configuration
- `collab_env.data.db.db_loader.DatabaseConnection` - SQLAlchemy connection
- Same schema created by `init_database.py`
- Same data loaded by `db_loader.py`

**No Schema Changes Required**: All queries use existing tables.

**Backend Agnostic**: Works with both PostgreSQL and DuckDB (via SQLAlchemy).

### File Structure After Implementation

```
collab-environment/
├── collab_env/data/db/
│   ├── __init__.py
│   ├── config.py                   # ✅ Existing
│   ├── init_database.py            # ✅ Existing
│   ├── db_loader.py                # ✅ Existing
│   ├── query_backend.py            # 🆕 NEW (Phase 5)
│   └── queries/                    # 🆕 NEW (Phase 5)
│       ├── __init__.py
│       ├── spatial_analysis.sql
│       ├── correlations.sql
│       └── session_metadata.sql
│
├── collab_env/dashboard/
│   ├── __init__.py
│   ├── app.py                      # ✅ Existing (GCS dashboard)
│   ├── dashboard_app.py            # ✅ Existing (GCS dashboard entry point)
│   ├── spatial_analysis_gui.py     # 🆕 NEW (Phase 6)
│   └── spatial_analysis_app.py     # 🆕 NEW (Phase 6)
│
├── docs/data/db/
│   ├── README.md                   # ✅ Existing
│   ├── implementation_progress.md  # ✅ Existing
│   └── simple_analysis_gui.md      # 🆕 NEW (this file)
│
└── requirements-db.txt             # 🔄 UPDATED (add aiosql, panel, holoviews)
```

---

## Example Workflows

### Workflow 1: Analyze Single Episode

```python
from collab_env.data.db.query_backend import QueryBackend
import matplotlib.pyplot as plt

query = QueryBackend()

# Get first episode
sessions = query.get_sessions(category_id='boids_3d')
episode_id = query.get_episodes(sessions.iloc[0]['session_id']).iloc[0]['episode_id']

# Compare before/after target appears (t=500)
speed_before = query.get_speed_statistics(episode_id, window_size=50)
speed_before = speed_before[speed_before['time_window'] < 500]

speed_after = query.get_speed_statistics(episode_id, window_size=50)
speed_after = speed_after[speed_after['time_window'] >= 500]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(speed_before['time_window'], speed_before['avg_speed'])
ax1.set_title('Speed Before Target (t<500)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Speed')

ax2.plot(speed_after['time_window'], speed_after['avg_speed'])
ax2.set_title('Speed After Target (t>=500)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Speed')

plt.tight_layout()
plt.show()

query.close()
```

### Workflow 2: Compare Agent Types

```python
query = QueryBackend()

# Get episode
sessions = query.get_sessions(category_id='boids_3d')
episode_id = query.get_episodes(sessions.iloc[0]['session_id']).iloc[0]['episode_id']

# Get heatmaps for agents and targets separately
heatmap_agents = query.get_spatial_heatmap(episode_id, bin_size=20.0, agent_type='agent')
heatmap_targets = query.get_spatial_heatmap(episode_id, bin_size=20.0, agent_type='target')

# Compare densities
import numpy as np

# Pivot to 2D grids
grid_agents = heatmap_agents.pivot(index='y_bin', columns='x_bin', values='density').fillna(0)
grid_targets = heatmap_targets.pivot(index='y_bin', columns='x_bin', values='density').fillna(0)

# Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(grid_agents, origin='lower', cmap='viridis')
ax1.set_title('Agents')
plt.colorbar(im1, ax=ax1, label='Density')

im2 = ax2.imshow(grid_targets, origin='lower', cmap='plasma')
ax2.set_title('Targets')
plt.colorbar(im2, ax=ax2, label='Density')

plt.tight_layout()
plt.show()

query.close()
```

### Workflow 3: Aggregate Across Episodes

```python
query = QueryBackend()

# Get all episodes for a session
sessions = query.get_sessions(category_id='boids_3d')
session_id = sessions.iloc[0]['session_id']
episodes = query.get_episodes(session_id)

# Aggregate speed statistics across all episodes
all_speed_stats = []
for episode_id in episodes['episode_id']:
    speed_stats = query.get_speed_statistics(episode_id, window_size=100)
    speed_stats['episode_id'] = episode_id
    all_speed_stats.append(speed_stats)

# Combine
import pandas as pd
combined = pd.concat(all_speed_stats, ignore_index=True)

# Compute mean across episodes
mean_speed = combined.groupby('time_window')['avg_speed'].mean()
std_speed = combined.groupby('time_window')['avg_speed'].std()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(mean_speed.index, mean_speed.values, label='Mean Speed')
plt.fill_between(mean_speed.index,
                 mean_speed - std_speed,
                 mean_speed + std_speed,
                 alpha=0.3,
                 label='±1 std')
plt.xlabel('Time Window')
plt.ylabel('Speed')
plt.title(f'Speed Over Time (Averaged Across {len(episodes)} Episodes)')
plt.legend()
plt.show()

query.close()
```

---

## References

- [Database Layer README](README.md) - Main database documentation
- [Implementation Progress](implementation_progress.md) - Phase-by-phase status
- [Schema Documentation](../../../schema/README.md) - Database schema details
- [Spatial Analysis Requirements](../../dashboard/spatial_analysis.md) - Analysis requirements
- [Panel Documentation](https://panel.holoviz.org/) - Dashboard framework
- [HoloViews Documentation](https://holoviews.org/) - Visualization library
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/) - Database toolkit

---

**Last Updated**: 2025-11-06
**Status**: Phase 5-6 Implementation in Progress
