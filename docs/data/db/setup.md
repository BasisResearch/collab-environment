# Database Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
source .venv-310/bin/activate
pip install psycopg2-binary duckdb sqlalchemy
```

Or use the requirements file:
```bash
pip install -r requirements-db.txt
```

### 2. Configure Environment (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your database settings
# DB_BACKEND=duckdb (or postgres)
# POSTGRES_DB=tracking_analytics
# POSTGRES_USER=dima
# etc.
```

Environment variables will be used as defaults. Command-line arguments override them.

Optional - install postgres server
```bash
docker run -v .data/pgdata:/pgdata -e PGDATA=/pgdata \
    -d --name timescaledb -p 127.0.0.1:5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg17
```

### 3. Initialize Database

**Option A: DuckDB (recommended for local development)**
```bash
python -m collab_env.data.db.init_database --backend duckdb --dbpath ./data/tracking.duckdb
```

**Option B: PostgreSQL (for production/Grafana)**
```bash
# Requires PostgreSQL server running
python -m collab_env.data.db.init_database --backend postgres
```

**Option C: Use Environment Variables**
```bash
# Set DB_BACKEND in .env, then run without arguments
python -m collab_env.data.db.init_database
```

### 4. Verify

```bash
# DuckDB
duckdb tracking.duckdb -c "SHOW TABLES;"

# PostgreSQL
psql tracking_analytics -c "\dt"
```

## Loading Data

After database initialization, load your simulation or tracking data:

### Load 3D Boids Simulation

```bash
# Load a complete simulation directory
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/hackathon-boid-small-200-sim_run-started-20250926-220926

# With specific backend
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/hackathon-boid-small-200-sim_run-started-20250926-220926 \
    --backend duckdb \
    --dbpath ./data/tracking.duckdb
```

The loader will:
- Create session metadata from config.yaml
- Load all episode-*.parquet files
- Extract observations (positions, velocities)
- Load extended properties (distances, mesh data)
- Filter out environment entities to avoid duplicate keys

**Note**: Loading large datasets takes time (~40 seconds per episode with 90K observations).

### Load Other Data Sources (Coming Soon)

```bash
# 2D Boids (not yet implemented)
python -m collab_env.data.db.db_loader --source boids2d --path data/2d_simulation.pt

# Tracking CSV (not yet implemented)
python -m collab_env.data.db.db_loader --source tracking --path data/tracking_session.csv
```

## Database Schema

The unified schema supports three data sources:
- **3D Boids**: Parquet files from simulations
- **2D Boids**: PyTorch .pt files
- **Tracking CSV**: Video tracking data

### Tables Created

1. **sessions** - Top-level grouping (simulation runs, fieldwork sessions)
2. **episodes** - Individual runs within a session
3. **agent_types** - Type definitions (agent, target, bird, etc.)
4. **observations** - Core time-series data (positions, velocities)
5. **property_categories** - Data source categories (boids_3d, boids_2d, tracking_csv, computed)
6. **property_definitions** - Extended property definitions (distances, accelerations, etc.)
7. **property_category_mapping** - M2M relationship between properties and categories
8. **extended_properties** - EAV storage for flexible properties

### Key Design Features

- **Composite Primary Keys**: Natural keys on observations `(episode_id, time_index, agent_id)`
- **EAV Pattern**: Flexible extended properties without hardcoded columns
- **Property Categories**: Group properties by data source type
- **Unified Interface**: Same schema works for PostgreSQL and DuckDB

## Backend Comparison

| Feature | PostgreSQL | DuckDB |
|---------|-----------|--------|
| **Setup** | Requires server | Zero-config file |
| **Use Case** | Production, Grafana | Local development, analytics |
| **Concurrency** | High | Single-writer |
| **Performance** | Good for OLTP | Excellent for OLAP |
| **SQL Compatibility** | Full SQL | Most SQL features |
| **Foreign Keys** | Full CASCADE support | Limited CASCADE |

## Switching Between Backends

The initialization script handles SQL dialect differences automatically:

```python
# PostgreSQL
python -m collab_env.data.init_database --backend postgres

# DuckDB
python -m collab_env.data.init_database --backend duckdb
```

**Automatic conversions**:
- `BIGSERIAL` → `BIGINT` (DuckDB)
- `JSONB` → `JSON` (DuckDB)
- `DOUBLE PRECISION` → `DOUBLE` (DuckDB)
- `ON DELETE CASCADE` → removed (DuckDB)
- `ON CONFLICT ... DO NOTHING` → removed (DuckDB)

## Usage Examples

### Python API

```python
import duckdb

# Connect to DuckDB
conn = duckdb.connect('tracking.duckdb')

# Query property categories
df = conn.execute("""
    SELECT * FROM property_categories
""").df()

# Query observations with extended properties
df = conn.execute("""
    SELECT
        o.episode_id,
        o.time_index,
        o.agent_id,
        o.x, o.y, o.z,
        pd.property_name,
        ep.value_float
    FROM observations o
    JOIN extended_properties ep ON o.observation_id = ep.observation_id
    JOIN property_definitions pd ON ep.property_id = pd.property_id
    WHERE o.episode_id = ?
""", ['episode-0-...']).df()

conn.close()
```

### PostgreSQL Connection

```python
import psycopg2

conn = psycopg2.connect(
    dbname='tracking_analytics',
    user='dima',
    host='localhost'
)

cur = conn.cursor()
cur.execute("SELECT * FROM property_categories")
rows = cur.fetchall()

conn.close()
```

## Grafana Setup

1. **Add Data Source**
   - Type: PostgreSQL
   - Host: `localhost:5432`
   - Database: `tracking_analytics`
   - User: `dima`
   - SSL Mode: `disable` (for local)

2. **Example Time-Series Query**
   ```sql
   SELECT
       to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
       sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0)) as speed,
       o.agent_id::text as metric
   FROM observations o
   JOIN episodes e ON o.episode_id = e.episode_id
   WHERE o.episode_id = $episode_id
     AND o.v_x IS NOT NULL
   ORDER BY o.time_index
   ```

3. **Example Heatmap Query**
   ```sql
   SELECT
       floor(x / 10) * 10 as x_bin,
       floor(y / 10) * 10 as y_bin,
       count(*) as value
   FROM observations
   WHERE episode_id = $episode_id
     AND time_index BETWEEN $__timeFrom() AND $__timeTo()
   GROUP BY x_bin, y_bin
   ```

## Next Steps

1. **Load Data**: Create data loader in `collab_env/data/db_loader.py`
2. **Query Backend**: Create query interface in `collab_env/data/db_backend.py`
3. **Dashboard Integration**: Connect existing dashboard to database
4. **Grafana Dashboards**: Create time-series visualizations

## Troubleshooting

### PostgreSQL Connection Error

**Error**: `psycopg2.OperationalError: could not connect to server`

**Solution**: Ensure PostgreSQL is running:
```bash
brew services start postgresql@14
# or
pg_ctl -D /usr/local/var/postgres start
```

### DuckDB File Locked

**Error**: `IO Error: Could not set lock on file`

**Solution**: Close all DuckDB connections and retry

### Schema File Not Found

**Error**: `Schema file not found: schema/01_core_tables.sql`

**Solution**: Run from project root:
```bash
cd /Users/dima/git/collab-environment
python -m collab_env.data.init_database --backend duckdb
```

## File Locations

- **Schema Definition**: `schema/*.sql`
- **Initialization Script**: `collab_env/data/init_database.py`
- **Documentation**: `schema/README.md`
- **Example Queries**: `schema/04_views_examples.sql`
- **Dependencies**: `requirements-db.txt`

## Database Files

- **DuckDB**: `tracking.duckdb` (in project root or custom path)
- **PostgreSQL**: Server-managed (usually `/usr/local/var/postgres/`)

## Performance Tips

1. **Use property categories** to filter only relevant extended properties
2. **Filter by episode_id** early in queries (indexed)
3. **Batch inserts** when loading data (10K+ rows at once)
4. **Consider materialized views** for frequently-accessed aggregations
5. **Use DuckDB for local analysis**, PostgreSQL for production serving
