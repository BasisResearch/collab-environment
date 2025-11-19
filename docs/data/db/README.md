# Database Layer Documentation

Unified tracking analytics database supporting PostgreSQL and DuckDB.

## Quick Links

**Data Layer:**
- **[Schema Documentation](../../../schema/README.md)** - Database schema details and SQL files
- **[Data Formats](data_formats.md)** - Source data documentation
- **[Cascading Deletes](cascading_deletes.md)** - How to safely delete data with automatic cleanup
- **[Data Loader Improvements](data_loader_plan.md)** - Planned data loading performance improvements (COPY, optimizations)

**Query & Visualization Layer:**
- **[Dashboard System](../dashboard/README.md)** - 🎯 **QueryBackend API + GUI Widgets** - Complete guide to querying and visualizing data
- **[Grafana Integration](../../dashboard/grafana/grafana_integration.md)** - Grafana dashboards and queries
- **[Grafana Query Library](../../dashboard/grafana/grafana_queries.md)** - 30+ tested SQL queries

**Historical:**
- **[Schema Refactoring](archive/schema_refactoring.md)** - ✅ COMPLETE (2025-11-08)
- **[Implementation Progress](implementation_progress.md)** - ⚠️ DEPRECATED - See current docs above

## Overview

The database layer provides a unified interface for storing and querying tracking data from multiple sources:
- **3D Boids**: Parquet files from `collab_env.sim.boids`
- **2D Boids**: PyTorch `.pt` files from GNN training
- **2D Boids GNN Rollout**: Pickle files with GNN model predictions (actual vs predicted trajectories)
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
│  - Seed data      │  │  - 2D boids (.pt) ✅   │
│  - Verify setup   │  │  - CSV tracking ✅     │
└───────────────────┘  └────────────────────────┘
```

## Key Features

✅ **Unified Interface**: SQLAlchemy-based abstraction for both PostgreSQL and DuckDB
✅ **Flexible Schema**: EAV pattern for extended properties
✅ **Fast Loading**: Bulk inserts via pandas (~18s per 90K observations)
✅ **Multi-Session Loading**: Load multiple datasets/simulations in a single transaction
✅ **Environment Config**: Configure via environment variables or CLI args
✅ **Automatic Adaptation**: SQL dialect conversion for DuckDB
✅ **Environment Entities**: Full support for agents and environment entities with composite primary keys
✅ **Cascading Deletes**: Automatic cleanup when deleting sessions, episodes, or categories

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

**Single Simulation:**
```bash
# Load a complete simulation directory (auto-detected from config.yaml)
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/hackathon-boid-small-200-sim_run-started-20250926-220926

# With specific backend
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/hackathon-boid-small-200-sim_run-started-20250926-220926 \
    --backend duckdb \
    --dbpath ./data/tracking.duckdb

# Load only first 5 episodes per simulation (useful for testing)
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/hackathon-boid-small-200-sim_run-started-20250926-220926 \
    --max_episodes_per_session 5
```

**Multiple Simulations (Bulk Loading):**
```bash
# Load all simulations from a parent directory in single transaction
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/

# Load only first 3 episodes from each simulation (useful for large datasets)
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/ \
    --max_episodes_per_session 3

# The loader auto-detects subdirectories with config.yaml files
# Progress: [1/5] Loading simulation_1...
#           [2/5] Loading simulation_2...
#           etc.
```

The 3D loader will:
- Create session metadata from config.yaml
- Load all episode-*.parquet files
- Extract observations (positions, velocities) for all entity types (agents and environment)
- Load extended properties (distances, mesh data)

### Load 2D Boids Simulation

**Single Dataset:**

```bash
# Load a single PyTorch .pt file
python -m collab_env.data.db.db_loader \
    --source boids2d \
    --path simulated_data/boid_food_basic.pt

# With specific backend
python -m collab_env.data.db.db_loader \
    --source boids2d \
    --path simulated_data/boid_food_basic.pt \
    --backend postgres

# Load only first 100 episodes (useful for large datasets)
python -m collab_env.data.db.db_loader \
    --source boids2d \
    --path simulated_data/boid_food_basic.pt \
    --max_episodes_per_session 100
```

**Multiple Datasets (Bulk Loading):**

```bash
# Load all .pt datasets from a directory in single transaction
python -m collab_env.data.db.db_loader \
    --source boids2d \
    --path simulated_data/

# Load only first 50 episodes from each dataset
python -m collab_env.data.db.db_loader \
    --source boids2d \
    --path simulated_data/ \
    --max_episodes_per_session 50

# The loader auto-discovers all .pt files (excluding *_config.pt)
# Progress: [1/16] Loading boid_food_basic.pt...
#           [2/16] Loading boid_food_basic_independent.pt...
#           etc.
```

The 2D loader will:

- Load PyTorch dataset from .pt file
- Load configuration from matching *_config.pt file
- Compute velocities from position differences: `v[t] = p[t+1] - p[t]`
- Scale coordinates from normalized [0,1] to scene coordinates [0, scene_size]
- Create 2D observations (z and v_z are NULL)
- **Compute extended properties** (if food is present in config):
  - `distance_to_food`: Euclidean distance from each boid to food location

**Performance**: 8,000+ observations/second for bulk loading

### Load Tracking CSV Data

**Single Session:**

```bash
# Load a single tracking session (auto-detected from aligned_frames/)
python -m collab_env.data.db.db_loader \
    --source tracking \
    --path data/processed_tracks/2024_06_01-session_0003

# With specific backend
python -m collab_env.data.db.db_loader \
    --source tracking \
    --path data/processed_tracks/2024_06_01-session_0003 \
    --backend postgres

# Load only first 2 camera episodes (useful for testing multi-camera setups)
python -m collab_env.data.db.db_loader \
    --source tracking \
    --path data/processed_tracks/2024_06_01-session_0003 \
    --max_episodes_per_session 2
```

**Multiple Sessions (Bulk Loading):**

```bash
# Load all tracking sessions from a parent directory in single transaction
python -m collab_env.data.db.db_loader \
    --source tracking \
    --path data/processed_tracks/

# Load only first camera from each session (faster for initial exploration)
python -m collab_env.data.db.db_loader \
    --source tracking \
    --path data/processed_tracks/ \
    --max_episodes_per_session 1

# The loader auto-discovers all session directories with aligned_frames/
# Progress: [1/8] Loading 2023_11_05-session_0001...
#           [2/8] Loading 2023_11_05-session_0002...
#           etc.
```

The tracking loader will:

- Load session metadata from optional `Metadata.yaml` file
- Discover all camera episodes in `aligned_frames/` subdirectory
- Read CSV files matching pattern `*_tracks.csv`
- Compute velocities from positions: `v[t] = (p[t+1] - p[t]) / dt` where `dt = frame_diff / frame_rate`
- Handle frame gaps by setting velocity to NaN when frames are missing
- Create 2D observations (z and v_z are NULL)
- Store velocities in pixels/second units

**CSV Format Requirements**:
- Required columns: `track_id`, `frame`, `x`, `y`
- Pixel coordinates (video resolution dependent)
- Frame numbers (0-indexed)
- Frame rate: 30 fps (default for video tracking)

**Performance**: ~5,000 observations/second for bulk loading

### Load GNN Rollout Data

**Single Rollout File:**

```bash
# Load a single rollout pickle file
python -m collab_env.data.db.db_loader \
    --source boids2d_rollout \
    --path trained_models/food/basic/n0_h1_vr0.5_s2/rollout_results/file260_foodbasic_n0_h1_vr0.5_s2_30.pkl

# With specific scene size (default: 480.0)
python -m collab_env.data.db.db_loader \
    --source boids2d_rollout \
    --path trained_models/food/basic/n0_h1_vr0.5_s2/rollout_results/file260_foodbasic_n0_h1_vr0.5_s2_30.pkl \
    --backend duckdb \
    --dbpath ./data/tracking.duckdb

# Load only first 100 trajectories (useful for large rollout files)
python -m collab_env.data.db.db_loader \
    --source boids2d_rollout \
    --path trained_models/food/basic/n0_h1_vr0.5_s2/rollout_results/file260_foodbasic_n0_h1_vr0.5_s2_30.pkl \
    --max_episodes_per_session 100
```

**Multiple Rollout Files (Bulk Loading):**

```bash
# Load all rollout files from a directory in single transaction
python -m collab_env.data.db.db_loader \
    --source boids2d_rollout \
    --path trained_models/food/basic/n0_h1_vr0.5_s2/rollout_results/

# Load only first 50 trajectories from each rollout file
python -m collab_env.data.db.db_loader \
    --source boids2d_rollout \
    --path trained_models/food/basic/n0_h1_vr0.5_s2/rollout_results/ \
    --max_episodes_per_session 50

# The loader auto-discovers all .pkl files in the directory
# Progress: [1/10] Loading file260_foodbasic_n0_h1_vr0.5_s2_30.pkl...
#           [2/10] Loading file260_foodbasic_n0_h1_vr0.5_s2_35.pkl...
#           etc.
```

The GNN rollout loader will:

- Load rollout pickle files containing GNN model predictions
- Create **paired episodes** for each trajectory:
  - **Actual episode**: Ground truth trajectory from test data (positions, velocities, accelerations only)
  - **Predicted episode**: GNN model prediction for same initial conditions (positions, velocities, accelerations, attention weights)
- Extract positions and velocities from rollout tensors
- Compute acceleration from velocity differences: `a[t] = v[t+1] - v[t]`
- Process multi-head attention weights (average across heads) - **predicted episodes only**
- Decompose attention into components (predicted episodes only):
  - `attn_weight_self`: Self-attention weight
  - `attn_weight_boid`: Sum of attention to other boid agents
  - `attn_weight_food`: Attention to food agent (if present)
- Compute spatial metrics:
  - **Actual episodes**: `distance_to_food` - distance to actual food position
  - **Predicted episodes**:
    - `distance_to_food_actual` - distance to TRUE food position (from actual episode)
    - `distance_to_food_predicted` - distance to GNN-predicted food position
- Store extended properties:
  - **Actual episodes**: acceleration_x, acceleration_y, distance_to_food (if food present)
  - **Predicted episodes**: acceleration_x, acceleration_y, distance_to_food_actual, distance_to_food_predicted (if food present), attn_weight_self, attn_weight_boid, attn_weight_food
- Handle CUDA tensors safely (automatic CPU mapping for CPU-only machines)
- Scale coordinates from normalized [0,1] to scene coordinates [0, scene_size]

**Episode ID Format**: `{session_id}-{trajectory_number:04d}-{actual|predicted}`

Example episode IDs:
- `file260_foodbasic_n0_h1_vr0.5_s2_30-0000-actual`
- `file260_foodbasic_n0_h1_vr0.5_s2_30-0000-predicted`
- `file260_foodbasic_n0_h1_vr0.5_s2_30-0001-actual`
- `file260_foodbasic_n0_h1_vr0.5_s2_30-0001-predicted`

**Pickle File Format Requirements**:
- Dictionary with keys: `x_actual`, `x_predicted`, `attn_weights` (optional)
- Position tensors: shape `(num_trajectories, num_timesteps, num_agents, 2)`
- Attention weights: shape `(num_trajectories, num_timesteps, num_agents, num_agents)` or `(num_trajectories, num_timesteps, num_heads, num_agents, num_agents)`
- Food agent detection: Last agent is food if 'food' in filename

**Performance**: ~1,850 observations/second (~25K observations per episode in ~68 seconds for 4 episodes)

**Use Cases**:
- **Model Evaluation**: Compare GNN predictions to ground truth in database
- **Error Analysis**: Query spatial/temporal patterns in prediction errors
- **Attention Analysis**: Visualize attention weight evolution over time
- **Ablation Studies**: Compare multiple model configurations via SQL queries

### Limiting Episodes Per Session

For testing, development, or working with large datasets, you can limit the number of episodes loaded per session using the `--max_episodes_per_session` option:

```bash
# Load only first 5 episodes from each session
python -m collab_env.data.db.db_loader \
    --source boids3d \
    --path simulated_data/hackathon/ \
    --max_episodes_per_session 5
```

**Use Cases:**
- **Quick Testing**: Verify database setup and loader functionality without waiting for full dataset
- **Development**: Iterate faster when developing queries or visualizations
- **Large Datasets**: Explore data structure before committing to full load
- **Sampling**: Create representative subset for analysis or demonstrations

**Behavior:**
- Applies independently to each session (not globally across all sessions)
- Episodes are loaded in order (0, 1, 2, ..., N-1)
- Progress logs show actual episodes loaded vs total available
- Works with all data sources: `boids3d`, `boids2d`, and `tracking`

**Example Output:**
```
Loading up to 5 out of 50 episodes in single transaction...
[1/50] Loading episode 0 from: episode-0.parquet
[2/50] Loading episode 1 from: episode-1.parquet
...
[5/50] Loading episode 4 from: episode-4.parquet
Reached maximum number of episodes (5) for session session-xyz, stopping...
Completed loading simulation: session-xyz (5 out of 50 episodes)
```

## Database Schema

The unified schema supports four data sources:
- **3D Boids**: Parquet files from simulations
- **2D Boids**: PyTorch .pt files
- **2D Boids GNN Rollout**: Pickle files with GNN model predictions
- **Tracking CSV**: Video tracking data

### Tables Created

1. **sessions** - Top-level grouping (simulation runs, fieldwork sessions)
2. **episodes** - Individual runs within a session
3. **agent_types** - Type definitions (agent, env, target, food, bird, rat, gerbil)
4. **observations** - Core time-series data (positions, velocities)
5. **categories** - Session and property categories (boids_3d, boids_2d, boids_2d_rollout, tracking_csv)
6. **property_definitions** - Extended property definitions (distances, accelerations, attention weights, etc.)
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
| `food` | food | Stationary food target in 2D boids simulation |
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

- **[📖 Complete Integration Guide](../../dashboard/grafana/grafana_integration.md)** - Setup, dashboard creation, troubleshooting
- **[📝 Query Library](../../dashboard/grafana/grafana_queries.md)** - 30+ tested SQL queries for all visualizations
- **[📊 Dashboard Template](../../dashboard/grafana/grafana_dashboard_template.json)** - Importable Grafana dashboard JSON

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

See [grafana_queries.md](../../dashboard/grafana/grafana_queries.md) for the complete query library.

## Performance

### Loading Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Database init | ~2 seconds | 8 tables, seed data |
| Load 90K observations (3D) | ~18 seconds | Single episode, bulk insert via pandas |
| Load 10 episodes (3D) | ~3 minutes | 900K observations total |
| Load 2,000 episodes (2D) | ~52 seconds | 420K observations, single transaction bulk loading |
| Bulk loading rate (2D) | 8,000+ obs/sec | Multiple datasets in single transaction |
| Load 4 rollout episodes (GNN) | ~68 seconds | 100K observations + extended properties |
| Bulk loading rate (GNN rollout) | ~1,850 obs/sec | Includes attention weights and accelerations |

### Optimization

- Uses pandas `to_sql()` for bulk inserts (2x faster than executemany)
- Single transaction for multi-session loading (commit only once at the end)
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
- [x] 2D boids data loader (PyTorch .pt files)
- [x] GNN rollout loader (pickle files with actual vs predicted trajectories)
- [x] Multi-session bulk loading (single transaction)
- [x] Batch loading with pandas to_sql
- [x] Environment entity support with composite primary keys
- [x] Extended properties loading (distances, mesh coordinates, accelerations, attention weights, distance to food)
- [x] Cascading deletes (PostgreSQL only, DuckDB limitation documented)
- [x] Grafana dashboards (prototype with query library and templates)

### TODO ⏳

**Data Loading:**

- [ ] PostgreSQL COPY optimization for 10-100x faster loading (see [data_loader_plan.md](data_loader_plan.md))

**Query & Visualization:**
- See **[Dashboard System](../dashboard/README.md)** for:
  - ✅ QueryBackend API (production ready)
  - ✅ Spatial analysis GUI (production ready)
  - ⏳ Unified widget architecture (planned - Phase 7)
  - ⏳ Property computation framework (planned - Phase 8)

**Advanced Features:**
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
