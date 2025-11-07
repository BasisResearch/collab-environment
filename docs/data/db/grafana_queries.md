# Grafana SQL Queries for Tracking Analytics

Curated collection of SQL queries for visualizing boid simulation data in Grafana.

## Quick Reference

- **Database**: `tracking_analytics`
- **Primary Tables**: `observations`, `episodes`, `extended_properties`, `property_definitions`
- **Sample Episode**: `episode-0-episode-0-completed-20250926-221003`
- **Dataset**: 10 episodes × 3,001 frames × 30 agents = ~900K observations
- **Frame Rate**: 30 FPS (frames per second)

## Table of Contents

1. [Time Series Queries](#time-series-queries)
2. [Spatial Statistics](#spatial-statistics)
3. [Extended Properties](#extended-properties)
4. [Time-Windowed Aggregations](#time-windowed-aggregations)
5. [Multi-Episode Comparisons](#multi-episode-comparisons)
6. [Grafana-Specific Tips](#grafana-specific-tips)

---

## Time Series Queries

### 1.1 Agent Speed Over Time

**Use Case**: Track individual agent speeds as time-series lines
**Panel Type**: Time series (line chart)
**Visualization**: One line per agent

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0)) as speed,
    o.agent_id::text as metric
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
ORDER BY o.time_index, o.agent_id
```

**Grafana Variables**:
- `$episode_id`: Episode selector (see Variable Queries section)

**Notes**:
- Returns speed in scene units per frame
- `metric` column tells Grafana to create separate series per agent
- Use `$__timeFilter(time)` for Grafana time range filtering

---

### 1.2 Average Speed Across All Agents

**Use Case**: Population-level speed trend
**Panel Type**: Time series (single line) or Stat panel with sparkline

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed,
    stddev(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as speed_stddev
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
GROUP BY o.time_index, e.frame_rate
ORDER BY o.time_index
```

**Notes**:
- Returns both mean and standard deviation
- Useful for detecting emergent behavior patterns

---

### 1.3 Velocity Components (X, Y, Z)

**Use Case**: Analyze directional movement patterns
**Panel Type**: Time series (multi-line)

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(o.v_x) as velocity_x,
    avg(o.v_y) as velocity_y,
    avg(o.v_z) as velocity_z
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
GROUP BY o.time_index, e.frame_rate
ORDER BY o.time_index
```

---

### 1.4 Current Speed Statistics

**Use Case**: Latest speed metrics
**Panel Type**: Stat panel (single value)

```sql
SELECT
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed
FROM observations o
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
  AND o.time_index = (
    SELECT MAX(time_index)
    FROM observations
    WHERE episode_id = $episode_id
  )
```

---

## Spatial Statistics

### 2.1 Position Heatmap

**Use Case**: Agent density visualization
**Panel Type**: Heatmap

```sql
SELECT
    floor(x / 10) * 10 as x_bin,
    floor(y / 10) * 10 as y_bin,
    count(*) as value
FROM observations
WHERE episode_id = $episode_id
  AND agent_type_id = 'agent'
  AND time_index BETWEEN $start_frame AND $end_frame
GROUP BY x_bin, y_bin
ORDER BY x_bin, y_bin
```

**Grafana Variables**:
- `$start_frame`: Start frame number (default: 0)
- `$end_frame`: End frame number (default: 3000)

**Notes**:
- Bin size of 10 scene units (adjust `/10` for finer/coarser resolution)
- For Grafana heatmap, set format to "Time series buckets" or use "Table" format with Transform

---

### 2.2 Position Scatter Plot

**Use Case**: Agent positions at specific time
**Panel Type**: Scatter plot or Table

```sql
SELECT
    o.x,
    o.y,
    o.z,
    sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0)) as speed,
    o.agent_id
FROM observations o
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND o.time_index = $frame_number
ORDER BY o.agent_id
```

**Notes**:
- Color by speed for velocity field visualization
- Use with frame slider variable

---

### 2.3 Speed Distribution (Histogram Data)

**Use Case**: Velocity distribution analysis
**Panel Type**: Bar chart or Histogram

```sql
SELECT
    floor(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0)) / 0.5) * 0.5 as speed_bin,
    count(*) as count
FROM observations
WHERE episode_id = $episode_id
  AND agent_type_id = 'agent'
  AND v_x IS NOT NULL
  AND time_index BETWEEN $start_frame AND $end_frame
GROUP BY speed_bin
ORDER BY speed_bin
```

**Notes**:
- Bin width: 0.5 scene units per frame
- Adjust binning: change `/0.5` and `*0.5` values

---

### 2.4 Agent Trajectories (Path Traces)

**Use Case**: Movement paths over time
**Panel Type**: Table or export for external plotting

```sql
SELECT
    o.agent_id,
    o.time_index,
    o.x,
    o.y,
    o.z
FROM observations o
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND o.agent_id = $selected_agent_id
ORDER BY o.time_index
```

---

## Extended Properties

### 3.1 Distance to Target Center Over Time

**Use Case**: Track target approach behavior
**Panel Type**: Time series

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(ep.value_float) as avg_distance,
    min(ep.value_float) as min_distance,
    max(ep.value_float) as max_distance
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND pd.property_name = 'Distance to Target Center'
GROUP BY o.time_index, e.frame_rate
ORDER BY o.time_index
```

**Notes**:
- Shows min/avg/max distance trends
- Useful for detecting convergence behavior

---

### 3.2 Distance to Target Mesh (Individual Agents)

**Use Case**: Per-agent target proximity
**Panel Type**: Time series (multi-line)

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    ep.value_float as distance,
    o.agent_id::text as metric
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND pd.property_name = 'Distance to Target Mesh'
ORDER BY o.time_index, o.agent_id
```

---

### 3.3 Distance to Scene Boundary

**Use Case**: Wall avoidance analysis
**Panel Type**: Time series

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(ep.value_float) as avg_boundary_distance,
    min(ep.value_float) as min_boundary_distance
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND pd.property_name = 'Distance to Scene Mesh'
GROUP BY o.time_index, e.frame_rate
ORDER BY o.time_index
```

**Notes**:
- Low values indicate boundary proximity
- Useful for collision avoidance analysis

---

### 3.4 All Extended Properties for Single Agent

**Use Case**: Detailed agent state inspection
**Panel Type**: Table

```sql
SELECT
    o.time_index,
    pd.property_name,
    ep.value_float,
    pd.unit
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = $episode_id
  AND o.agent_id = $selected_agent_id
  AND o.agent_type_id = 'agent'
ORDER BY o.time_index, pd.property_name
```

---

## Time-Windowed Aggregations

### 4.1 Speed Statistics per 100-Frame Window

**Use Case**: Temporal dynamics at coarser resolution
**Panel Type**: Bar chart or Time series

```sql
SELECT
    floor(o.time_index / 100) * 100 as time_window,
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed,
    stddev(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as speed_stddev,
    count(*) as n_observations
FROM observations o
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
GROUP BY time_window
ORDER BY time_window
```

**Notes**:
- Window size: 100 frames (~3.3 seconds at 30 FPS)
- Adjust window: change `/100` and `*100`

---

### 4.2 Before/After t=500 Comparison

**Use Case**: Compare early vs. late simulation dynamics
**Panel Type**: Stat panels (side-by-side)

**Before t=500**:
```sql
SELECT
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed_early
FROM observations
WHERE episode_id = $episode_id
  AND agent_type_id = 'agent'
  AND v_x IS NOT NULL
  AND time_index < 500
```

**After t=500**:
```sql
SELECT
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed_late
FROM observations
WHERE episode_id = $episode_id
  AND agent_type_id = 'agent'
  AND v_x IS NOT NULL
  AND time_index >= 500
```

---

### 4.3 Distance to Target per Time Window

**Use Case**: Target approach trends over time
**Panel Type**: Time series

```sql
SELECT
    floor(o.time_index / 100) * 100 as time_window,
    avg(ep.value_float) as avg_distance,
    count(*) as n_observations
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = $episode_id
  AND o.agent_type_id = 'agent'
  AND pd.property_name = 'Distance to Target Center'
  AND o.time_index > 500
GROUP BY time_window
ORDER BY time_window
```

---

## Multi-Episode Comparisons

### 5.1 Average Speed Across All Episodes

**Use Case**: Compare behavior consistency across runs
**Panel Type**: Bar chart or Table

```sql
SELECT
    o.episode_id,
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed,
    stddev(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as speed_stddev
FROM observations o
WHERE o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
GROUP BY o.episode_id
ORDER BY o.episode_id
```

---

### 5.2 Episode Summary Statistics

**Use Case**: Episode metadata overview
**Panel Type**: Table

```sql
SELECT
    e.episode_id,
    e.episode_number,
    e.num_frames,
    e.num_agents,
    e.frame_rate,
    s.session_name,
    c.category_name
FROM episodes e
JOIN sessions s ON e.session_id = s.session_id
JOIN categories c ON s.category_id = c.category_id
ORDER BY e.episode_number
```

---

### 5.3 Distance to Target: Multi-Episode Comparison

**Use Case**: Compare target approach across runs
**Panel Type**: Time series (one line per episode)

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(ep.value_float) as avg_distance,
    o.episode_id as metric
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.agent_type_id = 'agent'
  AND pd.property_name = 'Distance to Target Center'
GROUP BY o.episode_id, o.time_index, e.frame_rate
ORDER BY o.time_index, o.episode_id
```

**Warning**: High cardinality query - may be slow with many episodes.

---

## Grafana-Specific Tips

### Variable Queries

#### Episode Selector Variable
```sql
SELECT episode_id as __text, episode_id as __value
FROM episodes
ORDER BY episode_number
```

#### Frame Number Slider Variable
- Type: Interval
- Values: `0,100,500,1000,1500,2000,2500,3000`

#### Agent ID Selector Variable
```sql
SELECT DISTINCT agent_id::text as __text, agent_id as __value
FROM observations
WHERE episode_id = '$episode_id'
  AND agent_type_id = 'agent'
ORDER BY agent_id
```

---

### Time Range Filtering

Grafana provides `$__timeFilter(column)` macro for automatic time range filtering:

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    -- your metrics here
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE $__timeFilter(to_timestamp(o.time_index * (1.0 / e.frame_rate)))
  AND o.episode_id = $episode_id
```

---

### Panel-Specific Formatting

#### For Time Series Panels
- Ensure first column is named `time` (timestamp type)
- Use `as metric` for series names
- Use `$__timeFilter()` for time range support

#### For Heatmap Panels
- Format: "Time series buckets" or "Table"
- X-axis: Categorical (bin values)
- Y-axis: Time or categorical
- Value: Aggregation count/sum

#### For Stat Panels
- Single value query (one row, one column)
- Add sparkline by including time series data

#### For Table Panels
- Any column structure works
- Use column aliases for readable headers

---

### Performance Optimization

1. **Always filter by `episode_id`** - uses index
2. **Filter by `agent_type_id`** early - reduces scan
3. **Limit time ranges** - use `time_index BETWEEN`
4. **Use time windows** - aggregate with `floor(time_index / N)`
5. **Avoid SELECT \*** - specify needed columns only
6. **Test queries in psql first** - validate before Grafana

---

### Connection Settings

**Data Source Configuration**:
- Host: `localhost:5432`
- Database: `tracking_analytics`
- User: `postgres`
- Password: `password` (from Docker setup)
- SSL Mode: `disable` (local development)
- Version: PostgreSQL 17 with TimescaleDB

---

## Available Properties Reference

Current extended properties in the database:

| Property Name | Data Type | Unit | Description |
|---------------|-----------|------|-------------|
| Distance to Target Center | float | scene_units | Euclidean distance to target centroid |
| Distance to Target Mesh | float | scene_units | Distance to nearest point on target mesh |
| Distance to Scene Mesh | float | scene_units | Distance to scene boundary |
| Target Mesh Closest Point X | float | scene_units | X coordinate of closest target mesh point |
| Target Mesh Closest Point Y | float | scene_units | Y coordinate of closest target mesh point |
| Target Mesh Closest Point Z | float | scene_units | Z coordinate of closest target mesh point |
| Scene Mesh Closest Point X | float | scene_units | X coordinate of closest scene mesh point |
| Scene Mesh Closest Point Y | float | scene_units | Y coordinate of closest scene mesh point |
| Scene Mesh Closest Point Z | float | scene_units | Z coordinate of closest scene mesh point |
| Speed | float | scene_units/frame | Magnitude of velocity vector |
| Acceleration X/Y/Z | float | scene_units/frame² | Acceleration components |
| Acceleration Magnitude | float | scene_units/frame² | Magnitude of acceleration |
| Bounding Box X1/X2/Y1/Y2 | float | pixels | Tracking bounding boxes (tracking_csv only) |

---

## Example Dashboard Structure

Recommended dashboard layout:

```
Row 1: Overview Statistics
├─ [Stat] Current Avg Speed
├─ [Stat] Total Agents
├─ [Stat] Avg Distance to Target
└─ [Stat] Frames Loaded

Row 2: Time Series Analysis
├─ [Time Series] Agent Speed Over Time (multi-line)
└─ [Time Series] Distance to Target (avg/min/max)

Row 3: Spatial Analysis
├─ [Heatmap] Position Density
└─ [Scatter] Current Agent Positions

Row 4: Distribution Analysis
├─ [Bar Chart] Speed Distribution
└─ [Bar Chart] Distance Distribution

Variables:
- Episode ID (dropdown)
- Start Frame (slider: 0-3000)
- End Frame (slider: 0-3000)
- Selected Agent ID (dropdown)
```

---

## Troubleshooting

**Query returns no data**:
- Verify episode_id exists: `SELECT * FROM episodes`
- Check time_index range: episodes have 0-3000 frames
- Verify agent_type_id filter: use `'agent'` for boids

**Slow queries**:
- Add `LIMIT` for testing
- Use time windows instead of raw observations
- Filter by episode_id first (indexed)
- Avoid joining extended_properties if not needed

**Grafana time formatting issues**:
- Ensure `time` column is timestamp type
- Use `to_timestamp()` function
- Check timezone settings in Grafana

---

**Last Updated**: 2025-11-06
**Database Version**: PostgreSQL 17 + TimescaleDB
**Schema Version**: Phase 4 Complete (3D Boids)
