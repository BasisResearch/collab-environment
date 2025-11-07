# Grafana Integration Guide

Complete guide for visualizing boid simulation data from the `tracking_analytics` PostgreSQL database in Grafana.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Source Configuration](#data-source-configuration)
4. [Creating Your First Dashboard](#creating-your-first-dashboard)
5. [Dashboard 1: Time Series Overview](#dashboard-1-time-series-overview)
6. [Dashboard 2: Spatial Analysis](#dashboard-2-spatial-analysis)
7. [Dashboard 3: Time-Windowed Statistics](#dashboard-3-time-windowed-statistics)
8. [Importing JSON Dashboard Template](#importing-json-dashboard-template)
9. [Troubleshooting](#troubleshooting)
10. [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

- ✅ **PostgreSQL server running** on `localhost:5432`
- ✅ **Database initialized** with tracking_analytics schema
- ✅ **Data loaded** - at least one simulation loaded via `db_loader.py`
- ✅ **Network access** to PostgreSQL from Grafana

**Verify prerequisites**:
```bash
# Check PostgreSQL is running
PGPASSWORD=password psql -h localhost -U postgres tracking_analytics -c "SELECT COUNT(*) FROM episodes;"

# Should return a count > 0
```

---

## Installation

### macOS

```bash
# Install Grafana via Homebrew
brew install grafana

# Start Grafana service
brew services start grafana

# Verify Grafana is running
brew services info grafana
# Should show: Running: true

# Access Grafana
open http://localhost:3000
```

**Default credentials**:
- Username: `admin`
- Password: `admin`
- You'll be prompted to change the password on first login

### Linux (Ubuntu/Debian)

```bash
# Add Grafana repository
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -

# Install Grafana
sudo apt-get update
sudo apt-get install grafana

# Start Grafana service
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# Verify status
sudo systemctl status grafana-server

# Access Grafana
# Open browser to http://localhost:3000
```

### Docker (Alternative)

```bash
# Run Grafana in Docker
docker run -d \
  --name=grafana \
  -p 3000:3000 \
  --network=host \
  grafana/grafana-oss:latest

# Access Grafana
open http://localhost:3000
```

**Note**: Use `--network=host` to access PostgreSQL on localhost. Alternatively, use Docker networking if PostgreSQL is also in Docker.

---

## Data Source Configuration

### Step 1: Access Data Sources

1. Log in to Grafana at http://localhost:3000
2. Click **⚙️ Configuration** → **Data sources** (left sidebar)
3. Click **Add data source**
4. Select **PostgreSQL**

### Step 2: Configure PostgreSQL Connection

Fill in the following settings:

| Field | Value | Notes |
|-------|-------|-------|
| **Name** | `tracking_analytics` | Data source display name |
| **Host** | `localhost:5432` | PostgreSQL server address |
| **Database** | `tracking_analytics` | Database name |
| **User** | `postgres` | Database username |
| **Password** | `password` | Database password (from Docker setup) |
| **SSL Mode** | `disable` | For local development |
| **Version** | `17` | PostgreSQL version |
| **TimescaleDB** | ☑️ Enabled | Check this box (optional optimization) |

### Step 3: Test Connection

1. Scroll down and click **Save & test**
2. You should see: ✅ **"Database Connection OK"**

**If connection fails**:
- Verify PostgreSQL is running: `brew services list | grep postgresql`
- Check credentials: `psql -h localhost -U postgres -d tracking_analytics`
- Check firewall settings if remote connection

---

## Creating Your First Dashboard

### Basic Dashboard Setup

1. Click **+** → **Create** → **Dashboard** (left sidebar)
2. Click **Add visualization**
3. Select **tracking_analytics** data source
4. Start building panels!

### Adding Dashboard Variables

Variables enable dynamic filtering across all panels.

#### Create Episode Selector Variable

1. Click **⚙️ Dashboard settings** → **Variables**
2. Click **Add variable**
3. Configure:
   - **Name**: `episode_id`
   - **Type**: `Query`
   - **Data source**: `tracking_analytics`
   - **Query**:
     ```sql
     SELECT episode_id as __text, episode_id as __value
     FROM episodes
     ORDER BY episode_number
     ```
   - **Multi-value**: ❌ (unchecked)
   - **Include All**: ❌ (unchecked)
4. Click **Apply**

Now you'll see an **Episode ID** dropdown at the top of your dashboard!

#### Create Frame Range Variables (Optional)

**Start Frame Variable**:
- **Name**: `start_frame`
- **Type**: `Interval`
- **Values**: `0,100,500,1000,1500,2000,2500,3000`
- **Default**: `0`

**End Frame Variable**:
- **Name**: `end_frame`
- **Type**: `Interval`
- **Values**: `0,100,500,1000,1500,2000,2500,3000`
- **Default**: `3000`

---

## Dashboard 1: Time Series Overview

This dashboard shows agent speeds and distances over time.

### Panel 1.1: Average Speed Over Time

**Type**: Time series (line chart)

1. Add new panel
2. Enter query:

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
GROUP BY o.time_index, e.frame_rate
ORDER BY o.time_index
```

3. **Panel settings**:
   - **Title**: "Average Agent Speed Over Time"
   - **Unit**: Custom → `scene_units/frame`
   - **Legend**: Show
   - **Y-axis label**: "Speed"

4. Click **Apply**

### Panel 1.2: Individual Agent Speeds

**Type**: Time series (multi-line)

Query:
```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0)) as speed,
    o.agent_id::text as metric
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
ORDER BY o.time_index, o.agent_id
```

**Panel settings**:
- **Title**: "Individual Agent Speeds"
- **Legend**: Show (one line per agent)
- **Series override**: Optional - limit to first 10 agents for readability

### Panel 1.3: Distance to Target Center

**Type**: Time series with multi-line (avg/min/max)

Query:
```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(ep.value_float) as "Average Distance",
    min(ep.value_float) as "Min Distance",
    max(ep.value_float) as "Max Distance"
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND pd.property_name = 'Distance to Target Center'
GROUP BY o.time_index, e.frame_rate
ORDER BY o.time_index
```

**Panel settings**:
- **Title**: "Distance to Target Over Time"
- **Unit**: `scene_units`
- **Fill opacity**: 20% for area chart effect

### Panel 1.4: Current Speed Statistics

**Type**: Stat panel (single value)

Query:
```sql
SELECT
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed
FROM observations o
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
  AND o.time_index = (
    SELECT MAX(time_index)
    FROM observations
    WHERE episode_id = '$episode_id'
  )
```

**Panel settings**:
- **Title**: "Current Average Speed"
- **Value**: Show last value
- **Unit**: Custom → `scene_units/frame`
- **Sparkline**: Enabled (for mini trend)

### Dashboard Layout (Row 1)

Arrange panels in this layout:
```
┌─────────────────────────────────────────────────────────────┐
│ Dashboard Title: "Time Series Overview"                     │
│ Variables: [Episode ID ▼]                                   │
├────────────┬───────────┬───────────┬────────────────────────┤
│ [Stat]     │ [Stat]    │ [Stat]    │ [Stat]                 │
│ Avg Speed  │ Min Dist  │ Max Dist  │ Total Agents           │
│ 0.85       │ 145.3     │ 301.1     │ 30                     │
├────────────────────────────────────┬────────────────────────┤
│ [Time Series]                      │ [Time Series]          │
│ Average Agent Speed Over Time      │ Distance to Target     │
│                                    │                        │
├────────────────────────────────────┴────────────────────────┤
│ [Time Series - Multi-line]                                  │
│ Individual Agent Speeds                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Save the dashboard**: Click **💾 Save dashboard** → Name it "Time Series Overview"

---

## Dashboard 2: Spatial Analysis

This dashboard visualizes spatial distributions and movement patterns.

### Panel 2.1: Position Heatmap

**Type**: Heatmap

Query:
```sql
SELECT
    floor(x / 10) * 10 as x_bin,
    floor(y / 10) * 10 as y_bin,
    count(*) as value
FROM observations
WHERE episode_id = '$episode_id'
  AND agent_type_id = 'agent'
  AND time_index BETWEEN $start_frame AND $end_frame
GROUP BY x_bin, y_bin
ORDER BY x_bin, y_bin
```

**Panel settings**:
- **Title**: "Agent Position Density Heatmap"
- **Format**: Table
- **Calculation**: Total
- **Color scheme**: Choose Spectral or YlOrRd
- **Data format**: Set X-axis to `x_bin`, Y-axis to `y_bin`

**Note**: Grafana's heatmap can be tricky. Alternative: use a scatter plot with color by density.

### Panel 2.2: Speed Distribution Histogram

**Type**: Bar chart

Query:
```sql
SELECT
    floor(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0)) / 0.5) * 0.5 as speed_bin,
    count(*) as count
FROM observations
WHERE episode_id = '$episode_id'
  AND agent_type_id = 'agent'
  AND v_x IS NOT NULL
  AND time_index BETWEEN $start_frame AND $end_frame
GROUP BY speed_bin
ORDER BY speed_bin
```

**Panel settings**:
- **Title**: "Speed Distribution"
- **X-axis**: `speed_bin` (speed ranges)
- **Y-axis**: `count` (frequency)
- **Bar width**: Auto

### Panel 2.3: Agent Positions at Current Frame

**Type**: Table (can export for external plotting)

Query:
```sql
SELECT
    o.agent_id,
    o.x,
    o.y,
    o.z,
    sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0)) as speed
FROM observations o
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND o.time_index = $end_frame
ORDER BY o.agent_id
```

**Panel settings**:
- **Title**: "Agent Positions (Frame $end_frame)"
- **Table columns**: Show all
- **Color by value**: Enable for speed column

### Panel 2.4: Velocity Quiver Data

**Type**: Table (for export to Python/external tools)

Query:
```sql
SELECT
    o.x,
    o.y,
    o.v_x,
    o.v_y
FROM observations o
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND o.time_index = $end_frame
  AND o.v_x IS NOT NULL
ORDER BY o.agent_id
```

**Usage**: Export this data for quiver plots in matplotlib/plotly.

---

## Dashboard 3: Time-Windowed Statistics

This dashboard shows aggregated metrics over time windows.

### Panel 3.1: Speed Statistics per 100-Frame Window

**Type**: Time series (bar chart)

Query:
```sql
SELECT
    to_timestamp((floor(o.time_index / 100) * 100) * (1.0 / 30)) as time,
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed,
    stddev(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as speed_stddev
FROM observations o
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND o.v_x IS NOT NULL
GROUP BY floor(o.time_index / 100)
ORDER BY floor(o.time_index / 100)
```

**Panel settings**:
- **Title**: "Average Speed per Time Window (100 frames)"
- **Style**: Bars
- **Fill**: 80%

### Panel 3.2: Before/After t=500 Comparison

**Type**: Two Stat panels side-by-side

**Panel 3.2a - Before t=500**:
```sql
SELECT
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed_early
FROM observations
WHERE episode_id = '$episode_id'
  AND agent_type_id = 'agent'
  AND v_x IS NOT NULL
  AND time_index < 500
```

**Panel 3.2b - After t=500**:
```sql
SELECT
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed_late
FROM observations
WHERE episode_id = '$episode_id'
  AND agent_type_id = 'agent'
  AND v_x IS NOT NULL
  AND time_index >= 500
```

**Panel settings**:
- **Title**: "Early Phase (t<500)" and "Late Phase (t≥500)"
- **Color**: Different colors for comparison
- **Sparkline**: Disabled

### Panel 3.3: Distance Convergence Over Time

**Type**: Time series

Query:
```sql
SELECT
    to_timestamp((floor(o.time_index / 100) * 100) * (1.0 / 30)) as time,
    avg(ep.value_float) as avg_distance
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = '$episode_id'
  AND o.agent_type_id = 'agent'
  AND pd.property_name = 'Distance to Target Center'
  AND o.time_index > 500
GROUP BY floor(o.time_index / 100)
ORDER BY floor(o.time_index / 100)
```

**Panel settings**:
- **Title**: "Distance to Target (Post t=500, 100-frame windows)"
- **Unit**: `scene_units`

### Panel 3.4: Agent Type Summary

**Type**: Table

Query:
```sql
SELECT
    o.agent_type_id as "Agent Type",
    count(DISTINCT o.agent_id) as "Unique Agents",
    count(*) as "Total Observations",
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as "Avg Speed"
FROM observations o
WHERE o.episode_id = '$episode_id'
  AND o.v_x IS NOT NULL
GROUP BY o.agent_type_id
ORDER BY o.agent_type_id
```

**Panel settings**:
- **Title**: "Agent Type Statistics"
- **Column formatting**: Apply number formatting to statistics

---

## Importing JSON Dashboard Template

Instead of creating dashboards manually, you can import a pre-built template.

### Step 1: Locate Template File

The dashboard template is located at:
```
docs/data/db/grafana_dashboard_template.json
```

### Step 2: Import Dashboard

1. In Grafana, click **+** (Create) → **Import** (left sidebar)
2. Click **Upload JSON file**
3. Select `grafana_dashboard_template.json` from `docs/data/db/`
4. **Important**: You'll see a datasource selector dropdown:
   - Look for your PostgreSQL datasource (should be named something like "PostgreSQL" or "tracking_analytics")
   - Select it from the dropdown
   - If you don't see it, go back to [Data Source Configuration](#data-source-configuration) and create it first
5. Click **Import**

**Troubleshooting Import Issues**:

- **Error: "datasource ${DS_TRACKING_ANALYTICS} not found"**
  - This means the datasource selector step was skipped
  - Solution: Make sure you select your PostgreSQL datasource from the dropdown before clicking Import

- **No datasource appears in dropdown**
  - You need to create a PostgreSQL datasource first
  - Go to Configuration → Data sources → Add data source → PostgreSQL
  - Follow the configuration steps in [Data Source Configuration](#data-source-configuration)

- **Dashboard imports but shows "No data"**
  - The datasource might not be properly selected
  - Edit any panel and check that the datasource is set to your PostgreSQL connection
  - Or delete the dashboard and re-import, making sure to select the datasource

### Step 3: Verify Import

After importing, you should see:

- A dashboard titled "Boid Simulation Analytics - Complete Overview"
- An episode selector dropdown at the top
- 9 panels arranged in rows
- Some panels may show "No data" until you select an episode

### Step 4: Customize

After importing:

- Select an episode from the dropdown variable at the top
- Adjust panel positions and sizes if desired
- Modify queries for your use case
- Add/remove panels as needed
- Save with a new name if making significant changes

---

## Troubleshooting

### Issue: "Database Connection Failed"

**Cause**: PostgreSQL not accessible from Grafana

**Solutions**:
1. Verify PostgreSQL is running:
   ```bash
   brew services list | grep postgresql
   # Or
   docker ps | grep timescaledb
   ```

2. Test connection manually:
   ```bash
   PGPASSWORD=password psql -h localhost -U postgres tracking_analytics -c "SELECT 1;"
   ```

3. Check firewall/network settings if PostgreSQL is remote

4. Verify credentials in `.env` or Data Source settings

---

### Issue: Query Returns No Data

**Cause**: Episode ID variable not set or wrong filter

**Solutions**:
1. Check variable is selected (top of dashboard)

2. Verify episode exists:
   ```sql
   SELECT episode_id FROM episodes;
   ```

3. Check time_index range (episodes have 0-3000 frames)

4. Verify agent_type_id filter: `'agent'` not `'env'`

---

### Issue: Slow Query Performance

**Cause**: Large dataset, missing indexes, or inefficient query

**Solutions**:
1. Add `LIMIT 1000` for testing:
   ```sql
   SELECT ... LIMIT 1000
   ```

2. Use time windows instead of raw observations

3. Filter by `episode_id` early (indexed)

4. Check query execution plan:
   ```sql
   EXPLAIN ANALYZE <your query>
   ```

---

### Issue: Heatmap Not Rendering

**Cause**: Grafana heatmap requires specific data format

**Solutions**:
1. Ensure query returns exactly 3 columns: X, Y, value

2. Format as **Table** in panel settings

3. Alternative: Use scatter plot with color gradient instead

---

### Issue: Time Series Not Showing

**Cause**: Time column format issue

**Solutions**:
1. Ensure first column is named `time`:
   ```sql
   SELECT ... as time, ...
   ```

2. Verify timestamp conversion:
   ```sql
   to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time
   ```

3. Check timezone settings in Grafana (Settings → Preferences)

---

## Next Steps

### 1. Advanced Visualizations

Explore additional panel types:
- **Gauge panels**: Show current metrics with min/max
- **Pie charts**: Agent type distribution
- **Geo map**: If you add lat/lon to observations
- **Node graph**: Visualize agent interactions

### 2. Alerts

Set up alerts for anomalous behavior:
- Speed exceeds threshold
- Distance to boundary too small (collision risk)
- Agent count changes unexpectedly

### 3. Multi-Episode Analysis

Create dashboards comparing multiple episodes:
- Use `IN ($episode_ids)` with multi-select variable
- Add episode comparison bar charts
- Track metric evolution across runs

### 4. Export and Share

- **Snapshots**: Create shareable static views
- **PDF export**: Install Grafana Image Renderer
- **API access**: Query Grafana data via REST API
- **Embed**: Embed panels in external dashboards

### 5. TimescaleDB Optimizations

If using TimescaleDB (enabled in PostgreSQL setup):

```sql
-- Create hypertable for time-series optimization
SELECT create_hypertable('observations', 'time_index', chunk_time_interval => 500);

-- Add compression
ALTER TABLE observations SET (timescaledb.compress);
SELECT add_compression_policy('observations', INTERVAL '7 days');
```

### 6. Query Materialized Views

For frequently-accessed aggregations:

```sql
-- Create materialized view for speed statistics
CREATE MATERIALIZED VIEW mv_agent_speed_stats AS
SELECT
    episode_id,
    floor(time_index / 100) * 100 as time_window,
    agent_id,
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed
FROM observations
WHERE agent_type_id = 'agent'
  AND v_x IS NOT NULL
GROUP BY episode_id, time_window, agent_id;

-- Refresh periodically
REFRESH MATERIALIZED VIEW mv_agent_speed_stats;
```

Then query the view instead of raw observations for faster performance.

### 7. Integration with Analysis Pipeline

Connect Grafana to your broader workflow:
- **Trigger alerts** → Run analysis scripts
- **Export dashboard data** → Feed into ML models
- **Link to video** → Add annotations with video timestamps
- **Correlate with logs** → Join with experiment metadata

---

## Additional Resources

- **Grafana Documentation**: https://grafana.com/docs/grafana/latest/
- **PostgreSQL Data Source**: https://grafana.com/docs/grafana/latest/datasources/postgres/
- **Query Library**: [grafana_queries.md](grafana_queries.md)
- **Schema Documentation**: [../../../schema/README.md](../../../schema/README.md)
- **Database Setup**: [README.md](README.md)

---

## Example Gallery

### Time Series Dashboard
![Expected: Multi-line time series chart showing agent speeds over time]

### Spatial Heatmap Dashboard
![Expected: Heatmap showing agent position density]

### Statistics Dashboard
![Expected: Stat panels with key metrics and comparisons]

**Note**: Add screenshots after creating dashboards for visual reference.

---

## Feedback and Improvements

Have suggestions for additional visualizations or found issues?

1. Check existing issues: [GitHub Issues](https://github.com/your-org/collab-environment/issues)
2. Create a new issue with tag `grafana`
3. Contribute dashboard templates to `docs/data/db/dashboards/`

---

**Last Updated**: 2025-11-06
**Grafana Version Tested**: 12.2.1
**PostgreSQL Version**: 17 + TimescaleDB
**Database Schema**: Phase 4 Complete (3D Boids)
