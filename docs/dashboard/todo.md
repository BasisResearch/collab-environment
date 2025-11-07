# Dashboard Extension & Category System Refactoring - TODO

**Created:** 2025-11-07
**Status:** Planning Phase - DO NOT IMPLEMENT YET

---

## Overview

This document outlines planned changes to:
1. Refactor the category system to be semantic rather than procedural
2. Add new pairwise analysis queries and widgets
3. Add distribution/histogram visualization widgets
4. Implement property computation for speed and acceleration

---

## 1. Category System Refactoring

### Current State Analysis

**Current Category Structure:**
```
categories table:
- boids_3d: 3D Boids Simulations
- boids_2d: 2D Boids Simulations
- tracking_csv: Real-World Tracking
- computed: Computed Properties  ← TO BE REMOVED
```

**Current Usage:**
- Sessions reference category_id: `boids_3d`, `boids_2d`, `tracking_csv` (NOT `computed`)
- Extended properties map to categories via M2M relationship
- Computed properties (speed, acceleration) map to BOTH:
  - `computed` category (procedural classification) ← PROBLEM
  - Specific session categories: `boids_3d`, `boids_2d`, `tracking_csv` (applicability)

**Problem Identified:**
- `computed` category is **procedural** (describes HOW property is created)
- Other categories are **data source** types (describes WHERE data comes from)
- This creates confusion and unnecessary complexity
- Properties don't need separate categorization - they're just properties!

### Proposed New Design: Maximally Flat Structure

**Extremely Simple Design:**
- **Categories** = Data source types ONLY (boids_3d, boids_2d, tracking_csv)
  - Only used for `sessions.category_id` to identify data source
- **Properties** = Flat list in `property_definitions`
  - No categorization, no mapping tables
- **Discovery** = Query what actually exists in `extended_properties`
  - Source of truth is always the data

**Rationale:**
- This is an **analytics/research database** - discover what's in the data, don't enforce schemas
- A property is a property, regardless of whether it's "computed" or "raw"
- If a property exists in `extended_properties`, it's available. Period.
- No need for M2M mapping - can always query to discover:
  - "What properties exist for this session?" → Query extended_properties
  - "What sessions use this property?" → Query extended_properties
- Simpler schema = easier to maintain, no sync issues

**Result:**
- **Remove** `computed` category entirely
- **Remove** `property_category_mapping` table entirely
- **Keep** session categories (boids_3d, boids_2d, tracking_csv) for sessions only
- **Widget categories** (spatial, temporal, behavioral) remain separate - UI organization only

**Tables:**
```
categories: Just 3 rows (boids_3d, boids_2d, tracking_csv)
sessions: References category_id for data source type
property_definitions: All available properties (flat list)
extended_properties: Actual property values (source of truth for what exists)
```

### Database Changes Required

**Files to Modify:**

1. **schema/02_extended_properties.sql**
   - **DROP TABLE** `property_category_mapping` entirely
   - Update comments to clarify property_definitions is a flat list

2. **schema/01_core_tables.sql**
   - Update COMMENT on categories table: "Categories for data source types (sessions only)"
   - No structural changes

3. **schema/03_seed_data.sql**
   - **Remove** 'computed' category from categories INSERT (only keep boids_3d, boids_2d, tracking_csv)
   - **Remove** all property_category_mapping INSERTs (table no longer exists)
   - Keep property_definitions INSERTs (unchanged)

4. **collab_env/data/db/init_database.py**
   - No changes needed (generic schema loader)

5. **collab_env/data/db/db_loader.py**
   - Verify session loading uses correct category_id (boids_3d, boids_2d, tracking_csv)
   - Ensure no references to 'computed' category anywhere
   - Add property computation logic (see Section 4)

**Property Discovery Pattern:**
```python
# Get available properties for a session (query actual data)
def get_session_properties(session_id):
    return db.query("""
        SELECT DISTINCT pd.property_id, pd.property_name, pd.description
        FROM property_definitions pd
        WHERE pd.property_id IN (
            SELECT DISTINCT ep.property_id
            FROM extended_properties ep
            JOIN observations o ON ep.observation_id = o.observation_id
            JOIN episodes e ON o.episode_id = e.episode_id
            WHERE e.session_id = :session_id
        )
        ORDER BY pd.property_id
    """, session_id=session_id)
```

### Migration Strategy

**Option A: Fresh Database Initialization (Recommended for Dev)**
- Update schema files
- Drop and recreate database
- Re-import data with new categories

**Option B: Migration Script (For Production Data)**
- Create migration SQL script:
  1. Insert new semantic categories
  2. Update property_category_mapping to use semantic categories
  3. Delete 'computed' category
- Test migration on copy of production database
- Apply to production

**Decision:** Start with Option A for development, create Option B script if needed later

---

## 2. Pairwise Analysis Queries

### 2.1 Pairwise Distances

**Query Design:**

**File:** `collab_env/data/db/queries/pairwise_analysis.sql` (NEW)

```sql
-- name: get_pairwise_distances
-- Compute pairwise Euclidean distances between agents at each time step
-- Warning: O(n²) computation per time step, very expensive for many agents
-- Only supports episode_id (single episode). Session-level disabled.
WITH agent_positions AS (
    SELECT
        time_index,
        agent_id,
        x,
        y,
        COALESCE(z, 0) as z  -- Handle 2D data
    FROM observations
    WHERE episode_id = :episode_id
      AND (:agent_type = 'all' OR agent_type_id = :agent_type)
      AND (:start_time IS NULL OR time_index >= :start_time)
      AND (:end_time IS NULL OR time_index <= :end_time)
)
SELECT
    a.time_index,
    a.agent_id as agent_i,
    b.agent_id as agent_j,
    sqrt(
        (a.x - b.x) * (a.x - b.x) +
        (a.y - b.y) * (a.y - b.y) +
        (a.z - b.z) * (a.z - b.z)
    ) as distance
FROM agent_positions a
JOIN agent_positions b
  ON a.time_index = b.time_index
  AND a.agent_id < b.agent_id  -- Avoid duplicates and self-pairs
ORDER BY a.time_index, a.agent_id, b.agent_id;
```

**QueryBackend Method:**

**File:** `collab_env/data/db/query_backend.py`

```python
def get_pairwise_distances(
    self,
    episode_id: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    agent_type: str = 'agent',
    **kwargs
) -> pd.DataFrame:
    """
    Compute pairwise distances between agents.

    Warning: O(n²) per time step. Very expensive for many agents.
    Note: Only supports episode-level analysis. Session-level disabled.

    Returns:
        DataFrame with columns: time_index, agent_i, agent_j, distance
    """
```

**Widget Implementation:**

**File:** `collab_env/dashboard/widgets/pairwise_distance_widget.py` (NEW)

- Category: "interaction"
- Visualization options:
  - Heatmap: agent_i × agent_j (averaged over time window)
  - Time series: Mean/min/max distance over time
  - Distribution: Histogram of all pairwise distances
- Widget-specific parameters:
  - Time averaging window
  - Distance threshold filter
  - Visualization type selector

### 2.2 Pairwise Relative Velocities

**Query Design:**

**File:** `collab_env/data/db/queries/pairwise_analysis.sql` (APPEND)

```sql
-- name: get_pairwise_relative_velocities
-- Compute pairwise relative velocity magnitudes between agents
-- Warning: O(n²) computation per time step, very expensive for many agents
-- Only supports episode_id (single episode). Session-level disabled.
WITH agent_velocities AS (
    SELECT
        time_index,
        agent_id,
        v_x,
        v_y,
        COALESCE(v_z, 0) as v_z  -- Handle 2D data
    FROM observations
    WHERE episode_id = :episode_id
      AND (:agent_type = 'all' OR agent_type_id = :agent_type)
      AND (:start_time IS NULL OR time_index >= :start_time)
      AND (:end_time IS NULL OR time_index <= :end_time)
      AND v_x IS NOT NULL
)
SELECT
    a.time_index,
    a.agent_id as agent_i,
    b.agent_id as agent_j,
    sqrt(
        (a.v_x - b.v_x) * (a.v_x - b.v_x) +
        (a.v_y - b.v_y) * (a.v_y - b.v_y) +
        (a.v_z - b.v_z) * (a.v_z - b.v_z)
    ) as relative_velocity
FROM agent_velocities a
JOIN agent_velocities b
  ON a.time_index = b.time_index
  AND a.agent_id < b.agent_id  -- Avoid duplicates
ORDER BY a.time_index, a.agent_id, b.agent_id;
```

**QueryBackend Method:**

Similar to pairwise distances, returns: time_index, agent_i, agent_j, relative_velocity

**Widget Implementation:**

**File:** `collab_env/dashboard/widgets/pairwise_velocity_widget.py` (NEW)

- Category: "interaction"
- Similar visualization options to pairwise distances
- Widget-specific parameters for velocity thresholds

### Testing Requirements

**Files:**
- `tests/dashboard/test_pairwise_analysis.py` (NEW)

**Test Coverage:**
- Query returns correct pairwise combinations (n agents → n(n-1)/2 pairs)
- No self-pairs (agent_i != agent_j)
- No duplicate pairs (enforced by agent_i < agent_j)
- Distance/velocity calculations correct for known test cases
- Episode-only restriction enforced
- Widget instantiation and UI controls
- Widget error handling for missing velocity data

---

## 3. Distribution/Histogram Widgets

### Purpose

Visualize distributions of scalar properties to understand data characteristics:
- Speed distributions (identify behavioral modes)
- Distance distributions (spatial preferences)
- Acceleration distributions (detect sudden movements)

### Query Design

**File:** `collab_env/data/db/queries/distributions.sql` (NEW)

**Generic Histogram Query:**
```sql
-- name: get_histogram
-- Compute histogram bins for any scalar property from observations or extended_properties
-- Supports both episode and session scopes
-- Parameters:
--   :column_name - Column name from observations table (e.g., 'x', 'y', 'z')
--   :property_id - Property ID from extended_properties (optional, mutually exclusive with column_name)
--   :num_bins - Number of histogram bins (default: 50)
--   :min_value - Minimum value for binning (NULL = auto from data)
--   :max_value - Maximum value for binning (NULL = auto from data)
```

**Specialized Queries for Common Cases:**

```sql
-- name: get_speed_distribution
-- Distribution of speed values (computed from v_x, v_y, v_z)

-- name: get_distance_distribution
-- Distribution of distance to target/boundary from extended_properties

-- name: get_acceleration_distribution
-- Distribution of acceleration magnitudes
```

### Widget Implementation

**File:** `collab_env/dashboard/widgets/distribution_widget.py` (NEW)

**Widget Features:**
- Category: "statistical"
- Property selector dropdown (speed, distance_to_target, acceleration, etc.)
- Number of bins slider (10-200)
- Histogram type: count, density, cumulative
- Session scope: Supported (aggregate histograms across episodes)
- Export data button

**Widget-Specific Parameters:**
```python
property_selector = param.Selector(
    default='speed',
    objects=['speed', 'distance_to_target_center', 'distance_to_scene_mesh',
             'acceleration_magnitude', 'x', 'y', 'z'],
    doc="Property to visualize"
)

num_bins = param.Integer(default=50, bounds=(10, 200))
histogram_type = param.Selector(default='count', objects=['count', 'density', 'cumulative'])
log_scale = param.Boolean(default=False, doc="Use log scale for y-axis")
```

**Visualization:**
- Bokeh histogram with hover tooltips
- Overlays: mean, median, quartiles
- Optional KDE (kernel density estimate) overlay

### Testing Requirements

**Files:**
- `tests/dashboard/test_distributions.py` (NEW)

**Test Coverage:**
- Histogram bins correctly computed
- Handles both observations columns and extended_properties
- Session aggregation works correctly
- Widget parameter validation
- Edge cases: empty data, single value, outliers

---

## 4. Property Computation

### 4.1 Speed Computation

**Properties to Compute:**
- `speed`: sqrt(v_x² + v_y² + v_z²) - scalar magnitude

**Prerequisite Check:**
- Must verify v_x, v_y exist in observations before computing
- v_z is optional (use 0 for 2D data)

**Implementation Location:**

**File:** `collab_env/data/db/property_computations.py` (NEW)

```python
def compute_speed(df: pd.DataFrame) -> pd.Series:
    """
    Compute speed from velocity components.

    Args:
        df: DataFrame with columns v_x, v_y, v_z (v_z optional)

    Returns:
        Series with speed values

    Raises:
        ValueError: If v_x or v_y not present
    """
    if 'v_x' not in df.columns or 'v_y' not in df.columns:
        raise ValueError("Velocity components v_x and v_y required")

    v_z = df['v_z'] if 'v_z' in df.columns else 0
    return np.sqrt(df['v_x']**2 + df['v_y']**2 + v_z**2)
```

### 4.2 Acceleration Computation

**Properties to Compute:**
- `acceleration_x`: Δv_x / Δt (finite difference)
- `acceleration_y`: Δv_y / Δt
- `acceleration_z`: Δv_z / Δt (if 3D)
- `acceleration_magnitude`: sqrt(a_x² + a_y² + a_z²)

**Implementation:**

```python
def compute_acceleration(df: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    """
    Compute acceleration components from velocity time series.

    Args:
        df: DataFrame with columns time_index, agent_id, v_x, v_y, v_z (sorted by time)
        dt: Time step (default: 1.0 frame)

    Returns:
        DataFrame with columns: acceleration_x, acceleration_y, acceleration_z, acceleration_magnitude

    Notes:
        - Uses forward difference: a(t) = (v(t+1) - v(t)) / dt
        - Last time step will have NaN (no future velocity)
        - Groups by agent_id to compute per-agent accelerations
    """
    # Group by agent and compute per-agent differences
    # Handle edge cases (first/last frames)
```

### 4.3 Integration Points

**A. During Data Import**

**File:** `collab_env/data/db/db_loader.py`

Add method:
```python
def compute_derived_properties(
    self,
    episode_id: str,
    compute_speed: bool = True,
    compute_acceleration: bool = True
) -> Dict[str, pd.Series]:
    """
    Compute derived properties for an episode.

    Returns:
        Dict mapping property_id to Series of values (indexed by observation_id)
    """
```

**Integration in load_episode():**
```python
# After loading observations
if compute_derived:
    derived_props = self.compute_derived_properties(episode_id)
    self.load_extended_properties(episode_id, derived_props)
```

**B. Standalone Utility**

**File:** `collab_env/data/db/compute_properties_cli.py` (NEW)

Command-line tool:
```bash
# Compute for specific episode
python -m collab_env.data.db.compute_properties_cli --episode-id ep_123 --properties speed,acceleration

# Compute for entire session
python -m collab_env.data.db.compute_properties_cli --session-id sess_456 --properties all

# Recompute (overwrite existing)
python -m collab_env.data.db.compute_properties_cli --session-id sess_456 --properties speed --overwrite
```

**Features:**
- Query existing observations from database
- Compute requested properties
- Insert into extended_properties table
- Progress bar for large datasets
- Dry-run mode to preview

### Testing Requirements

**Files:**
- `tests/data/test_property_computations.py` (NEW)
- `tests/data/test_compute_properties_cli.py` (NEW)

**Test Coverage:**
- Speed computation: 2D and 3D cases
- Acceleration: Forward/backward/central differences
- Edge cases: Missing velocities, single frame episodes
- CLI: Argument parsing, episode/session selection, overwrite behavior
- Database integration: Correct insertion into extended_properties

---

## 5. Implementation Plan & Order

### Phase 1: Category System Refactoring
**Priority:** HIGH (Foundation for other work)
**Estimated Effort:** 1-2 hours (MAXIMALLY SIMPLIFIED)

1. Update schema/02_extended_properties.sql:
   - DROP TABLE property_category_mapping
   - Update comments
2. Update schema/03_seed_data.sql:
   - Remove 'computed' category (keep only boids_3d, boids_2d, tracking_csv)
   - Remove ALL property_category_mapping INSERTs
3. Update comments in schema/01_core_tables.sql (clarify categories = session types only)
4. Grep codebase for any references to 'computed' category or property_category_mapping and remove
5. Test fresh database initialization
6. Update documentation

**Files:**
- schema/02_extended_properties.sql (DROP property_category_mapping table)
- schema/01_core_tables.sql (comments only)
- schema/03_seed_data.sql (remove computed category, remove all property mappings)
- docs/database/category_system.md (NEW - document maximally flat design)

**Testing:**
- Manual: Initialize fresh database
- Verify exactly 3 categories exist (boids_3d, boids_2d, tracking_csv)
- Verify property_category_mapping table does NOT exist
- Verify all properties still in property_definitions
- Verify existing data loaders still work

### Phase 2: Property Computation
**Priority:** HIGH (Needed for distributions)
**Estimated Effort:** 6-8 hours

1. Implement property_computations.py
2. Integrate into db_loader.py
3. Create standalone CLI tool
4. Add comprehensive tests
5. Compute properties for existing test datasets

**Files:**
- collab_env/data/db/property_computations.py (NEW)
- collab_env/data/db/db_loader.py (MODIFY)
- collab_env/data/db/compute_properties_cli.py (NEW)
- tests/data/test_property_computations.py (NEW)
- tests/data/test_compute_properties_cli.py (NEW)

**Testing:**
- Unit tests for computation functions
- Integration tests with database
- CLI argument validation

### Phase 3: Distribution Widgets
**Priority:** MEDIUM (Useful for data exploration)
**Estimated Effort:** 6-8 hours

1. Create distributions.sql queries
2. Add QueryBackend methods
3. Implement DistributionWidget
4. Add to analysis_widgets.yaml
5. Test with real data
6. Update documentation

**Files:**
- collab_env/data/db/queries/distributions.sql (NEW)
- collab_env/data/db/query_backend.py (MODIFY)
- collab_env/dashboard/widgets/distribution_widget.py (NEW)
- collab_env/dashboard/analysis_widgets.yaml (MODIFY)
- tests/dashboard/test_distributions.py (NEW)
- docs/dashboard/distribution_analysis.md (NEW)

**Testing:**
- Query correctness for various properties
- Widget rendering and parameter validation
- Session vs episode scope behavior

### Phase 4: Pairwise Analysis
**Priority:** MEDIUM-LOW (Specialized use case)
**Estimated Effort:** 10-12 hours

1. Create pairwise_analysis.sql queries
2. Add QueryBackend methods (get_pairwise_distances, get_pairwise_relative_velocities)
3. Implement PairwiseDistanceWidget
4. Implement PairwiseVelocityWidget
5. Add to analysis_widgets.yaml
6. Performance testing (O(n²) concerns)
7. Documentation and usage examples

**Files:**
- collab_env/data/db/queries/pairwise_analysis.sql (NEW)
- collab_env/data/db/query_backend.py (MODIFY)
- collab_env/dashboard/widgets/pairwise_distance_widget.py (NEW)
- collab_env/dashboard/widgets/pairwise_velocity_widget.py (NEW)
- collab_env/dashboard/analysis_widgets.yaml (MODIFY)
- tests/dashboard/test_pairwise_analysis.py (NEW)
- docs/dashboard/pairwise_analysis.md (NEW)

**Testing:**
- Correctness: Known test cases with 2-3 agents
- Performance: Benchmark with 10, 50, 100 agents
- Episode-only restriction
- Widget error handling

---

## 6. Success Criteria

### Category System
- [ ] 'computed' category removed from database
- [ ] property_category_mapping table dropped entirely
- [ ] Only 3 categories remain: boids_3d, boids_2d, tracking_csv (for sessions only)
- [ ] property_definitions remains as flat list of all properties
- [ ] Sessions still correctly reference category_id
- [ ] Existing data loading works with simplified schema
- [ ] Documentation updated with property discovery pattern

### Property Computation
- [ ] Speed computation works for 2D and 3D data
- [ ] Acceleration computation handles edge cases (first/last frames)
- [ ] Integration with data import pipeline functional
- [ ] Standalone CLI tool works for episodes and sessions
- [ ] All tests passing
- [ ] Properties correctly stored in extended_properties table

### Distribution Widgets
- [ ] Histogram queries return correct bin counts
- [ ] Widget renders for all supported properties
- [ ] Session aggregation works correctly
- [ ] User can switch properties, adjust bins, toggle log scale
- [ ] Export functionality works
- [ ] Widget added to dashboard and loads without errors

### Pairwise Analysis
- [ ] Queries return correct number of pairs (n choose 2)
- [ ] No duplicate or self-pairs
- [ ] Distance calculations accurate
- [ ] Relative velocity calculations accurate
- [ ] Widgets render visualizations correctly
- [ ] Performance acceptable for reasonable agent counts (<100)
- [ ] Episode-only restriction enforced
- [ ] Error messages clear for session scope attempts

---

## 7. Risks & Mitigations

### Risk: Breaking Existing Data
**Mitigation:**
- Test schema changes on copy of database first
- Create rollback migration script
- Document migration steps clearly

### Risk: Performance Issues with Pairwise Queries
**Mitigation:**
- Add query timeouts
- Document performance characteristics (O(n²))
- Add widget warnings for large agent counts
- Consider sampling/approximation for very large datasets

### Risk: Widget Complexity
**Mitigation:**
- Start with simple visualizations
- Iterate based on user feedback
- Provide sensible defaults
- Clear documentation and examples

### Risk: Property Computation Edge Cases
**Mitigation:**
- Comprehensive unit tests
- Validation checks (non-negative speed, etc.)
- Graceful handling of missing data
- Clear error messages

---

## 8. Open Questions

1. ~~**Category Naming:**~~ **RESOLVED** - Using flat property structure with only data source categories (boids_3d, boids_2d, tracking_csv)

2. **Session Scope for Distributions:** Should histograms aggregate across episodes in a session, or compute per-episode distributions?

3. **Pairwise Analysis Performance:** What's the maximum number of agents we need to support? Should we implement sampling/approximation?

4. **Acceleration Method:** Forward difference, backward difference, or central difference? Trade-offs?

5. **Property Computation Timing:** Should property computation be:
   - Automatic during import (default enabled)?
   - Optional flag during import?
   - Always manual via CLI tool?

6. **Widget Organization:** Should pairwise widgets be separate, or combine into single "Interaction Analysis" widget with tabs?

---

## Next Steps

1. **Review this plan** - Get feedback on approach, priorities, open questions
2. **Finalize category design** - Confirm semantic category names and mappings
3. **Begin Phase 1** - Update schema and test migration
4. **Iterate** - Implement phases sequentially, testing thoroughly

---

**End of TODO Document**
