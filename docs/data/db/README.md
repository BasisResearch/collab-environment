# Database Layer Documentation

Unified tracking analytics database supporting PostgreSQL and DuckDB.

## Quick Links

- **[Schema Documentation](../../../schema/README.md)** - Database schema details and SQL files
- **[Data Formats](data_formats.md)** - Source data documentation

## Overview

The database layer provides a unified interface for storing and querying tracking data from multiple sources:
- **3D Boids**: Parquet files from `collab_env.sim.boids`
- **2D Boids**: PyTorch `.pt` files from GNN training
- **Tracking CSV**: Real-world video tracking data

### Architecture

```
┌─────────────────────────────────────────┐
│   collab_env.data.db.config             │
│   - Environment variable configuration   │
│   - SQLAlchemy URL generation           │
└─────────────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
┌────────▼──────────┐  ┌───▼────────────────────┐
│  init_database.py │  │    db_loader.py        │
│  ─────────────── │  │    ──────────────       │
│  Initialize DB    │  │  Load data from:       │
│  - Create tables  │  │  - 3D boids (parquet)  │
│  - Seed data      │  │  - 2D boids (.pt) TODO │
│  - Verify setup   │  │  - CSV tracking TODO   │
└───────────────────┘  └────────────────────────┘
```

## Key Features

✅ **Unified Interface**: SQLAlchemy-based abstraction for both PostgreSQL and DuckDB
✅ **Flexible Schema**: EAV pattern for extended properties
✅ **Fast Loading**: Bulk inserts via pandas (~18s per 90K observations)
✅ **Environment Config**: Configure via environment variables or CLI args
✅ **Automatic Adaptation**: SQL dialect conversion for DuckDB
✅ **Environment Entities**: Full support for agents and environment entities with composite primary keys

## Quick Start

### 1. Install Dependencies

```bash
source .venv-310/bin/activate
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

Optional - install PostgreSQL server:
```bash
docker run -v ./data/pgdata:/pgdata -e PGDATA=/pgdata \
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
# This will drop and recreate the database 'tracking_analytics'
python -m collab_env.data.db.init_database --backend postgres

# Or, if you want to preserve existing database and only create tables:
python -m collab_env.data.db.init_database --backend postgres --no-drop
```

**Option C: Use Environment Variables**
```bash
# Set DB_BACKEND in .env, then run without arguments
python -m collab_env.data.db.init_database
```

### 4. Verify Installation

```bash
# DuckDB
duckdb ./data/tracking.duckdb -c "SHOW TABLES;"

# PostgreSQL
psql tracking_analytics -c "\dt"
```

## Loading Data

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
- Extract observations (positions, velocities) for all entity types (agents and environment)
- Load extended properties (distances, mesh data)

**Note**: Loading large datasets takes time (~18 seconds per episode with 90K observations).

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
3. **agent_types** - Type definitions (agent, env, target, bird, rat, gerbil)
4. **observations** - Core time-series data (positions, velocities)
5. **categories** - Session and property categories (boids_3d, boids_2d, tracking_csv, computed)
6. **property_definitions** - Extended property definitions (distances, accelerations, etc.)
7. **property_category_mapping** - M2M relationship between properties and categories
8. **extended_properties** - EAV storage for flexible properties

### Key Design Features

- **Composite Primary Keys**: Natural keys on observations `(episode_id, time_index, agent_id, agent_type_id)` - allows same agent_id for different entity types
- **EAV Pattern**: Flexible extended properties without hardcoded columns
- **Unified Categories**: Single categories table for both sessions and extended properties
- **Unified Interface**: Same schema works for PostgreSQL and DuckDB

For complete schema documentation, see [schema/README.md](../../../schema/README.md).

## Environment Entities Support

✅ **Fully Supported**: The database stores both agent entities and environment entities using a composite primary key.

### Schema Design

The observations table uses `agent_type_id` in the primary key to distinguish between different entity types with the same ID:

```sql
PRIMARY KEY (episode_id, time_index, agent_id, agent_type_id)
```

This allows the same `agent_id` to be used for different entity types:
- **Agent**: `episode='e1', time=0, agent_id=0, type='agent', x=10.0, y=20.0`
- **Environment**: `episode='e1', time=0, agent_id=0, type='env', x=100.0, y=200.0` ✅ No conflict

### Agent Types

The database supports multiple agent types through the `agent_types` table:

| Type ID | Type Name | Description |
|---------|-----------|-------------|
| `agent` | agent | Generic simulated agent (boid) |
| `env` | environment | Environment entity (walls, obstacles, boundaries) |
| `target` | target | Target object in simulation |
| `bird` | bird | Bird detected in video tracking |
| `rat` | rat | Rat detected in video tracking |
| `gerbil` | gerbil | Gerbil detected in video tracking |

### Querying Different Entity Types

```sql
-- Get only agents
SELECT * FROM observations WHERE agent_type_id = 'agent';

-- Get only environment entities
SELECT * FROM observations WHERE agent_type_id = 'env';

-- Get entities with same agent_id but different types
SELECT agent_id, agent_type_id, x, y, z
FROM observations
WHERE agent_id = 0 AND episode_id = 'episode-1'
ORDER BY time_index, agent_type_id;

-- Count entities by type
SELECT agent_type_id, COUNT(*)
FROM observations
GROUP BY agent_type_id;
```

### Benefits

✅ **Complete data representation** - all simulation entities are stored
✅ **Agent-environment interactions** - analyze spatial relationships
✅ **Scene boundaries** - available for visualizations
✅ **Natural schema** - uses existing agent_types mechanism
✅ **Flexible** - easily add new entity types

## Backend Comparison

| Feature | PostgreSQL | DuckDB |
|---------|-----------|--------|
| **Setup** | Requires server | Zero-config file |
| **Use Case** | Production, Grafana | Local development, analytics |
| **Concurrency** | High | Single-writer |
| **Performance** | Good for OLTP | Excellent for OLAP |
| **SQL Compatibility** | Full SQL | Most SQL features |
| **Foreign Keys** | Full CASCADE support | Limited CASCADE |

## Usage Examples

### Python API

```python
from collab_env.data.db.config import get_db_config
from collab_env.data.db.db_loader import DatabaseConnection

# Connect to database
config = get_db_config()  # Reads from environment
db = DatabaseConnection(config)
db.connect()

# Execute queries
result = db.fetch_all(
    "SELECT * FROM observations WHERE episode_id = :ep_id LIMIT 10",
    {'ep_id': 'episode-0-...'}
)

# Bulk insert DataFrame
import pandas as pd
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
db.insert_dataframe(df, 'my_table')

db.close()
```

### Direct Database Access

**DuckDB:**
```python
import duckdb

conn = duckdb.connect('tracking.duckdb')

# Query categories
df = conn.execute("""
    SELECT * FROM categories
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

**PostgreSQL:**
```python
import psycopg2

conn = psycopg2.connect(
    dbname='tracking_analytics',
    user='dima',
    host='localhost'
)

cur = conn.cursor()
cur.execute("SELECT * FROM categories")
rows = cur.fetchall()

conn.close()
```

## Grafana Integration ✅

**Status**: Prototype complete - ready to use!

Visualize your boid simulation data with ready-to-use Grafana dashboards and query templates.

### Grafana Quick Start

1. **Install Grafana**:

   ```bash
   # macOS
   brew install grafana
   brew services start grafana

   # Access at http://localhost:3000 (admin/admin)
   ```

2. **Configure Data Source**:
   - Type: PostgreSQL
   - Host: `localhost:5432`
   - Database: `tracking_analytics`
   - User: `postgres`
   - Password: `password`
   - SSL Mode: `disable` (for local)

3. **Import Dashboard**:
   - Go to **Dashboards** → **Import**
   - Upload: `docs/data/db/grafana_dashboard_template.json`
   - Select data source: `tracking_analytics`

### Available Dashboards

#### 1. Time Series Overview

- Agent speed over time (individual and average)
- Distance to target tracking
- Current statistics (stat panels)
- Episode selector variable

#### 2. Spatial Analysis

- Position density heatmaps
- Speed distribution histograms
- Agent state tables with color coding

#### 3. Time-Windowed Statistics

- 100-frame window aggregations
- Before/after t=500 comparisons
- Distance convergence analysis

### Documentation

- **[📖 Complete Integration Guide](grafana_integration.md)** - Setup, dashboard creation, troubleshooting
- **[📝 Query Library](grafana_queries.md)** - 30+ tested SQL queries for all visualizations
- **[📊 Dashboard Template](grafana_dashboard_template.json)** - Importable Grafana dashboard JSON

### Example Queries

#### Time-Series Speed

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
ORDER BY o.time_index
```

#### Spatial Heatmap

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
```

#### Distance to Target

```sql
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    avg(ep.value_float) as avg_distance
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = $episode_id
  AND pd.property_name = 'Distance to Target Center'
GROUP BY o.time_index, e.frame_rate
ORDER BY o.time_index
```

See [grafana_queries.md](grafana_queries.md) for the complete query library.

## Performance

### Loading Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Database init | ~2 seconds | 8 tables, seed data |
| Load 90K observations | ~18 seconds | Bulk insert via pandas |
| Load 10 episodes | ~3 minutes | 900K observations total |

### Optimization

- Uses pandas `to_sql()` for bulk inserts (2x faster than executemany)
- SQLAlchemy connection pooling
- Minimal indexes for fast writes
- Future: COPY command could be 10-100x faster (not yet implemented)

### Performance Tips

1. **Use categories** to filter only relevant extended properties and sessions
2. **Filter by episode_id** early in queries (indexed)
3. **Batch inserts** when loading data (10K+ rows at once)
4. **Consider materialized views** for frequently-accessed aggregations
5. **Use DuckDB for local analysis**, PostgreSQL for production serving

## Code Structure

```
collab_env/data/db/
├── __init__.py
├── config.py              # Environment variable configuration
├── init_database.py       # Database initialization
└── db_loader.py          # Data loading (3D boids complete)

schema/
├── 01_core_tables.sql           # Core dimension and fact tables
├── 02_extended_properties.sql  # EAV pattern for properties
├── 03_seed_data.sql            # Default agent types and properties
├── 04_views_examples.sql       # Example query templates
└── README.md                   # Schema documentation

docs/data/db/
├── README.md             # This file
└── data_formats.md      # Source data documentation
```

## Implementation Status

### Complete ✅

- [x] Database schema (EAV pattern with property categories)
- [x] PostgreSQL and DuckDB support
- [x] SQLAlchemy-based unified interface
- [x] Environment variable configuration
- [x] Database initialization with verification
- [x] 3D boids data loader (parquet files)
- [x] Batch loading with pandas to_sql
- [x] Environment entity support with composite primary keys
- [x] Extended properties loading (all 9 properties: distances + mesh coordinates)
- [x] Grafana dashboards (prototype with query library and templates)

### TODO ⏳

- [ ] 2D boids loader (PyTorch .pt files)
- [ ] Tracking CSV loader
- [ ] Query backend interface
- [ ] Dashboard integration as an additional data source
- [ ] Spatial queries and statistics, in a separate dashboard, see `./docs/dashboard/spatial_analysis.md`
- [ ] Advanced Grafana features (correlations, pairwise statistics, alerts)

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
python -m collab_env.data.db.init_database
```

## Configuration

Configure via environment variables (`.env` file) or command-line arguments.

### Environment Variables

```bash
# Database backend
DB_BACKEND=duckdb  # or postgres (default: duckdb)

# PostgreSQL settings
POSTGRES_DB=tracking_analytics      # Database name (default: tracking_analytics)
POSTGRES_USER=your_user             # Database user (default: $USER)
POSTGRES_PASSWORD=your_password     # Database password (default: none)
POSTGRES_HOST=localhost             # Database host (default: localhost)
POSTGRES_PORT=5432                  # Database port (default: 5432)

# DuckDB settings
DUCKDB_PATH=tracking.duckdb         # Database file path (default: tracking.duckdb)
DUCKDB_READ_ONLY=false              # Read-only mode (default: false)
```

**Important**: By default, `init_database.py` will **drop and recreate** the database. Use `--no-drop` to preserve existing database and only create tables.

See [.env.example](../../../.env.example) for complete template.

## Architecture Improvements

### SQLAlchemy Unification (2025-11-05)

Refactored entire database layer to use SQLAlchemy:

**Benefits**:
- 62% code reduction in database logic
- 2x faster data loading (18s vs 40s per episode)
- Unified interface across all database code
- Named parameters instead of positional
- Standard patterns for connection management

**Changes**:
- `init_database.py`: Merged 2 backend classes → 1 unified class
- `db_loader.py`: Replaced manual connections with SQLAlchemy
- Both use `create_engine()` and `text()` queries
- Both use `config.sqlalchemy_url()` for connections

## References

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [DuckDB Documentation](https://duckdb.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
