# Database Layer Documentation

Unified tracking analytics database supporting PostgreSQL and DuckDB.

## Quick Links

- **[Setup Guide](setup.md)** - Installation and initialization
- **[Data Formats](data_formats.md)** - Source data documentation
- **[Schema Documentation](../../../schema/README.md)** - Database schema details

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

## Quick Start

### 1. Install Dependencies

```bash
source .venv-310/bin/activate
pip install -r requirements-db.txt
```

### 2. Initialize Database

```bash
# DuckDB (local analytics)
python -m collab_env.data.db.init_database --backend duckdb

# PostgreSQL (production/Grafana)
python -m collab_env.data.db.init_database --backend postgres
```

### 3. Load Data

```bash
# Load 3D boids simulation
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/my-simulation
```

See [Setup Guide](setup.md) for detailed instructions.

## Database Schema

### Core Tables

| Table | Purpose |
|-------|---------|
| `sessions` | Top-level simulation/tracking sessions |
| `episodes` | Individual simulation runs |
| `agent_types` | Agent type definitions (agent, target, bird, etc.) |
| `observations` | Time-series positions and velocities |
| `property_categories` | Categories for organizing properties |
| `property_definitions` | Extended property definitions |
| `property_category_mapping` | M2M property-to-category relationships |
| `extended_properties` | EAV storage for flexible properties |

### Design Principles

- **Composite Primary Keys**: Natural keys `(episode_id, time_index, agent_id)` ensure uniqueness
- **EAV Pattern**: Flexible properties without hardcoded columns
- **Property Categories**: Organize by data source (boids_3d, boids_2d, tracking_csv, computed)
- **Backend Agnostic**: Same schema works for PostgreSQL and DuckDB

See [Schema Documentation](../../../schema/README.md) for complete details.

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
├── setup.md             # Setup and initialization guide
└── data_formats.md      # Source data documentation
```

## Configuration

Configure via environment variables (`.env` file) or command-line arguments:

```bash
# Database backend
DB_BACKEND=duckdb  # or postgres

# PostgreSQL settings
POSTGRES_DB=tracking_analytics
POSTGRES_USER=your_user
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# DuckDB settings
DUCKDB_PATH=tracking.duckdb
```

See [.env.example](../../../.env.example) for complete template.

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

## Implementation Status

### Complete ✅

- [x] Database schema (EAV pattern with property categories)
- [x] PostgreSQL and DuckDB support
- [x] SQLAlchemy-based unified interface
- [x] Environment variable configuration
- [x] Database initialization with verification
- [x] 3D boids data loader (parquet files)
- [x] Batch loading with pandas to_sql

### TODO ⏳

- [ ] 2D boids loader (PyTorch .pt files)
- [ ] Tracking CSV loader
- [ ] Query backend interface
- [ ] Extended properties loading (partially implemented)
- [ ] Dashboard integration
- [ ] Grafana dashboards

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

```python
import duckdb

# DuckDB
conn = duckdb.connect('tracking.duckdb')
df = conn.execute("""
    SELECT episode_id, COUNT(*) as obs_count
    FROM observations
    GROUP BY episode_id
""").df()
conn.close()
```

```python
import psycopg2

# PostgreSQL
conn = psycopg2.connect(
    dbname='tracking_analytics',
    user='your_user',
    host='localhost'
)
cur = conn.cursor()
cur.execute("SELECT * FROM sessions")
sessions = cur.fetchall()
conn.close()
```

## Architecture Improvements (2025-11-05)

### SQLAlchemy Unification

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

## Troubleshooting

### DuckDB File Locked

**Error**: `IO Error: Could not set lock on file`

**Solution**: Close all DuckDB connections and retry

### PostgreSQL Connection Error

**Error**: `psycopg2.OperationalError: could not connect to server`

**Solution**: Ensure PostgreSQL is running:
```bash
brew services start postgresql@14
```

### Schema File Not Found

**Error**: `Schema file not found: schema/01_core_tables.sql`

**Solution**: Run from project root:
```bash
cd /Users/dima/git/collab-environment
python -m collab_env.data.db.init_database
```

## Implementation Status

For detailed implementation progress and technical decisions, see:
- **[Implementation Progress](implementation_progress.md)** - Complete phase-by-phase status tracking
- **[SQLAlchemy Refactoring](refactoring/)** - Technical details on database layer unification:
  - [db_loader.py Refactoring](refactoring/db_loader_refactoring.md)
  - [init_database.py Refactoring](refactoring/init_database_refactoring.md)
  - [Complete Summary](refactoring/complete_summary.md)

## References

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [DuckDB Documentation](https://duckdb.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## Support

For issues or questions:
1. Check the [Setup Guide](setup.md)
2. Review [Schema Documentation](../../../schema/README.md)
3. Check [Data Formats](data_formats.md) for source data details
4. Review [Implementation Progress](implementation_progress.md) for current status
