# Tracking Analytics Database Schema

## Overview

PostgreSQL-based database schema for unified time-series animal tracking data from three sources:

1. **3D Boids Simulations**: Parquet files from `collab_env.sim.boids`
2. **2D Boids Simulations**: PyTorch `.pt` datasets from `collab_env.sim.boids_gnn_temp`
3. **Real-World Tracking**: CSV files from `collab_env.tracking` video analysis

**Version**: 1.0
**Database**: PostgreSQL (can be ported to DuckDB)

## Schema Design

### Core Principle: EAV with Flat Property List

- **Core observations**: Positions and velocities in main `observations` table (universal data only)
- **Extended properties**: Flexible EAV (Entity-Attribute-Value) pattern for all other data
- **Flat property definitions**: All properties in a simple list, no categorization
- **Property discovery**: Query what actually exists in the data, not what's defined
- **No hardcoded columns**: All extended properties defined in `property_definitions` table

### Entity-Relationship Structure

```
categories (session data source types only)
  └─> sessions (ON DELETE CASCADE)
       └─> episodes (ON DELETE CASCADE)
            └─> observations (PK: episode_id, time_index, agent_id, agent_type_id) (ON DELETE CASCADE)
                 └─> extended_properties (PK: observation_id, property_id) (ON DELETE CASCADE)

agent_types
  └─> observations (FK: agent_type_id)

property_definitions (flat list)
  └─> extended_properties (FK: property_id)
```

**Cascading Deletes**: Deleting a category, session, or episode automatically deletes all child records. See [cascading_deletes.md](../docs/data/db/cascading_deletes.md) for details and safety guidelines.

## Quick Start

### 1. Initialize Database

```bash
# Using the unified Python script (PostgreSQL or DuckDB)
source .venv-310/bin/activate

# For DuckDB (local analytics)
python -m collab_env.data.init_database --backend duckdb --dbpath tracking.duckdb

# For PostgreSQL (requires running server)
python -m collab_env.data.init_database --backend postgres

# Or manually with psql
createdb tracking_analytics
psql tracking_analytics < schema/01_core_tables.sql
psql tracking_analytics < schema/02_extended_properties.sql
psql tracking_analytics < schema/03_seed_data.sql
```

### 2. Verify Setup

```bash
# PostgreSQL
psql tracking_analytics

tracking_analytics=# \dt
# Should show: sessions, episodes, agent_types, observations,
#              categories, property_definitions, extended_properties

tracking_analytics=# SELECT * FROM categories;
# Should show: boids_3d, boids_2d, tracking_csv

# DuckDB
duckdb tracking.duckdb

D> SHOW TABLES;
D> SELECT * FROM categories;
```

### 3. Connect Grafana

1. Add PostgreSQL data source in Grafana
2. Connection string: `host=localhost dbname=tracking_analytics user=youruser`
3. Use example queries from `04_views_examples.sql`

## Schema Files

| File | Purpose | Required |
|------|---------|----------|
| `01_core_tables.sql` | Dimension and fact tables | Yes |
| `02_extended_properties.sql` | EAV schema for flexible properties | Yes |
| `03_seed_data.sql` | Default agent types and property definitions | Yes |
| `04_views_examples.sql` | Example queries and view templates | No (reference) |

## Table Descriptions

### Dimension Tables

#### `sessions`
Top-level container for related episodes (simulation run or fieldwork session).

**Primary Key**: `session_id` VARCHAR

**Columns**:
- `session_name`: Human-readable name
- `category_id`: Data source type (boids_3d, boids_2d, tracking_csv) - references categories table
- `config`: Full configuration as JSONB
- `metadata`: Additional metadata (environment, mesh paths, notes)

#### `episodes`
Single simulation run or video tracking session.

**Primary Key**: `episode_id` VARCHAR

**Columns**:
- `session_id`: Foreign key to parent session
- `episode_number`: Sequence number within session
- `num_frames`: Total timesteps/frames
- `num_agents`: Number of agents/tracks
- `frame_rate`: Frames per second (default 30)
- `file_path`: Original source file

#### `agent_types`
Agent/track type definitions.

**Primary Key**: `type_id` VARCHAR

**Examples**: agent, target, bird, rat, gerbil

### Fact Tables

#### `observations`
Core time-series data: positions and velocities.

**Primary Key**: Composite `(episode_id, time_index, agent_id)`
**Surrogate Key**: `observation_id` BIGSERIAL (for foreign key references)

**Core Columns**:
- `x, y, z`: Spatial coordinates (z NULL for 2D)
- `v_x, v_y, v_z`: Velocity components (may be NULL)

**Note**: Tracking metadata (confidence, detection_class) moved to extended_properties table

**Why composite PK?**
- Ensures uniqueness: One observation per (episode, time, agent)
- Natural query pattern: Filter by episode and time
- observation_id available for FK references

### Extended Properties (EAV)

#### `property_definitions`
Defines all available extended properties.

**Columns**:
- `property_id`: Unique identifier
- `property_name`: Display name
- `data_type`: float, vector, string
- `unit`: Measurement unit (scene_units, pixels, m/s^2, etc.)

**Examples**:
- `distance_to_target_center`: Distance to target
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: Bounding boxes
- `acceleration_x, acceleration_y`: Computed accelerations
- `confidence`: Detection confidence (tracking data)
- `detection_class`: Detected object class (tracking data)

#### `extended_properties`
EAV table storing property values.

**Primary Key**: Composite `(observation_id, property_id)`

**Columns**:
- `value_float`: For numeric properties
- `value_text`: For strings, arrays (JSON), etc.

## Query Patterns

### 1. Get Available Properties for a Session (Property Discovery)

```sql
-- Discover which properties actually exist for a session
SELECT DISTINCT
    pd.property_id,
    pd.property_name,
    pd.unit
FROM property_definitions pd
WHERE pd.property_id IN (
    SELECT DISTINCT ep.property_id
    FROM extended_properties ep
    JOIN observations o ON ep.observation_id = o.observation_id
    JOIN episodes e ON o.episode_id = e.episode_id
    WHERE e.session_id = 'session-...'
);
```

### 2. Query Observations with Extended Properties

```sql
-- EAV format (property_name in rows)
SELECT
    o.time_index,
    o.agent_id,
    o.x, o.y, o.z,
    pd.property_name,
    ep.value_float as property_value
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = 'episode-0-...'
ORDER BY o.time_index, o.agent_id, pd.property_name;
```

### 3. Pivot Properties to Columns

```sql
-- Denormalized format (properties as columns)
SELECT
    o.time_index,
    o.agent_id,
    o.x, o.y, o.z,
    MAX(CASE WHEN ep.property_id = 'distance_to_target_center' THEN ep.value_float END) as distance_to_target,
    MAX(CASE WHEN ep.property_id = 'speed' THEN ep.value_float END) as speed
FROM observations o
LEFT JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE o.episode_id = 'episode-0-...'
GROUP BY o.observation_id, o.time_index, o.agent_id, o.x, o.y, o.z;
```

### 4. Spatial Heatmap

```sql
SELECT
    floor(x / 10) * 10 as x_bin,
    floor(y / 10) * 10 as y_bin,
    count(*) as density
FROM observations
WHERE episode_id = 'episode-0-...'
  AND time_index BETWEEN 500 AND 1000
GROUP BY x_bin, y_bin;
```

### 5. Time-Series for Grafana

```sql
-- Speed over time
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0)) as speed
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE o.episode_id = $episode_id
ORDER BY o.time_index;
```

See `04_views_examples.sql` for more query templates.

## Session Categories Reference

Categories define data source types (used only for `sessions.category_id`):

### boids_3d
3D boid simulation sessions. Common properties include:
- `distance_to_target_center`
- `distance_to_target_mesh`
- `distance_to_scene_mesh`
- `target_mesh_closest_x/y/z`
- `scene_mesh_closest_x/y/z`
- Computed: `speed`, `acceleration_x/y/z`

### boids_2d
2D boid simulation sessions. Common properties include:
- Computed: `speed`, `acceleration_x/y`

### tracking_csv
Real-world tracking sessions from video data. Common properties include:
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2` (bounding boxes)
- `confidence` (detection confidence)
- `detection_class` (object class label)

## Adding New Properties

### Example: Add a new property "distance_to_boundary"

```sql
-- 1. Define the property
INSERT INTO property_definitions (property_id, property_name, data_type, description, unit)
VALUES ('distance_to_boundary', 'Distance to Boundary', 'float', 'Minimum distance to any boundary', 'scene_units');

-- 2. Insert values (during data loading)
INSERT INTO extended_properties (observation_id, property_id, value_float)
SELECT observation_id, 'distance_to_boundary', computed_distance_value
FROM observations o
WHERE ...;

-- 3. Discover where it's used
SELECT DISTINCT s.session_name, s.category_id
FROM sessions s
JOIN episodes e ON s.session_id = e.session_id
JOIN observations o ON e.episode_id = o.episode_id
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE ep.property_id = 'distance_to_boundary';
```

## Performance Considerations

### Indexes

Current indexes (bare-bones):
- `observations(episode_id, time_index)` - Time-slice queries
- `observations(episode_id)` - Episode scans
- `extended_properties(observation_id)` - Property lookups
- `extended_properties(property_id)` - Property-centric queries

### Query Optimization Tips

1. **Always filter by episode_id**: Uses primary index
2. **Limit property pivoting**: Only pivot properties you need
3. **Use CTEs for complex queries**: Improves readability and may help planner
4. **Consider materialized views**: For frequently-accessed aggregations

### Materialized Views (Optional)

For common query patterns, create materialized views:

```sql
CREATE MATERIALIZED VIEW boids_3d_flat AS
SELECT
    o.*,
    MAX(CASE WHEN ep.property_id = 'distance_to_target_center' THEN ep.value_float END) as distance_to_target
FROM observations o
LEFT JOIN extended_properties ep ON o.observation_id = ep.observation_id
GROUP BY o.observation_id, ... (all columns);

CREATE INDEX ON boids_3d_flat(episode_id, time_index);

-- Refresh after loading new data
REFRESH MATERIALIZED VIEW boids_3d_flat;
```

## Migration to DuckDB

To use DuckDB instead of PostgreSQL:

1. Convert data types:
   - `BIGSERIAL` → `BIGINT AUTO_INCREMENT`
   - `DOUBLE PRECISION` → `DOUBLE`
   - `JSONB` → `JSON`

2. DuckDB can query PostgreSQL directly:
   ```sql
   INSTALL postgres;
   LOAD postgres;
   SELECT * FROM postgres_scan('postgresql://...', 'observations');
   ```

3. Or export/import via parquet:
   ```sql
   -- PostgreSQL: Export
   COPY observations TO '/tmp/observations.parquet' (FORMAT parquet);

   -- DuckDB: Import
   CREATE TABLE observations AS SELECT * FROM '/tmp/observations.parquet';
   ```

## Troubleshooting

### Issue: UNIQUE constraint violation on observations PK

**Cause**: Duplicate (episode_id, time_index, agent_id) in source data

**Solution**: Check source parquet/CSV for duplicates before loading

### Issue: Slow queries on extended_properties

**Cause**: Large number of properties per observation

**Solutions**:
1. Filter by property_id in WHERE clause
2. Create materialized view for specific category
3. Add index on (property_id, observation_id) if querying by property first

### Issue: NULL observation_id in extended_properties

**Cause**: Inserting extended properties before observations

**Solution**: Load in correct order: sessions → episodes → observations → extended_properties

## Implementation Status

### Complete ✅

1. **Data Ingestion**: `collab_env/data/db_loader.py`
   - 3D Boids: Parquet files with extended properties
   - 2D Boids: PyTorch .pt files with velocity computation
   - Multi-session bulk loading with single transaction
2. **Query Interface**: `collab_env/data/db/query_backend.py` (see [docs/data/db/README.md](../docs/data/db/README.md))
3. **Grafana Dashboards**: Time-series visualizations with query library (see [docs/dashboard/grafana/](../docs/dashboard/grafana/))
4. **Cascading Deletes**: Automatic cleanup on PostgreSQL (DuckDB limitation documented)

### TODO ⏳

1. **Tracking CSV Loader**: Real-world video tracking data ingestion
2. **Computed Properties**: Pipeline to compute accelerations, speeds
3. **Pairwise Interactions**: Implement if needed (commented out in design)
4. **PostgreSQL COPY**: 10-100x faster bulk loading (see [data_loader_plan.md](../docs/data/db/data_loader_plan.md))

## References

- **data_formats.md**: Source format documentation
- **spatial_analysis.md**: Analytics requirements
- **db_layer_todo.md**: Implementation tasks
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
