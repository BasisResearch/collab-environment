-- =============================================================================
-- 04_views_examples.sql
-- Example queries and view templates (NOT created by default)
-- Use these as templates for application queries or create views as needed
-- =============================================================================

-- =============================================================================
-- EXAMPLE 1: Query observations with extended properties for a session
-- =============================================================================

/*
-- Get all observations with extended properties for a session's episodes
SELECT
    o.episode_id,
    o.time_index,
    o.agent_id,
    o.agent_type_id,
    o.x, o.y, o.z,
    o.v_x, o.v_y, o.v_z,
    pd.property_name,
    ep.value_float as property_value,
    pd.unit
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
JOIN property_definitions pd ON ep.property_id = pd.property_id
WHERE o.episode_id = 'episode-0-...'
ORDER BY o.time_index, o.agent_id, pd.property_name;
*/

-- =============================================================================
-- EXAMPLE 2: Get available properties for a session (property discovery)
-- =============================================================================

/*
-- Discover all properties that exist for a session (query actual data)
SELECT DISTINCT
    pd.property_id,
    pd.property_name,
    pd.data_type,
    pd.description,
    pd.unit
FROM property_definitions pd
WHERE pd.property_id IN (
    SELECT DISTINCT ep.property_id
    FROM extended_properties ep
    JOIN observations o ON ep.observation_id = o.observation_id
    JOIN episodes e ON o.episode_id = e.episode_id
    WHERE e.session_id = 'session-...'
)
ORDER BY pd.property_name;
*/

-- =============================================================================
-- EXAMPLE 3: Pivot extended properties to columns (dynamic)
-- =============================================================================

/*
-- Pivot specific properties for an episode
-- Note: This requires knowing which properties to pivot
-- In practice, generate this SQL dynamically based on available properties

SELECT
    o.observation_id,
    o.episode_id,
    o.time_index,
    o.agent_id,
    o.x, o.y, o.z,
    o.v_x, o.v_y, o.v_z,
    MAX(CASE WHEN ep.property_id = 'distance_to_target_center' THEN ep.value_float END) as distance_to_target_center,
    MAX(CASE WHEN ep.property_id = 'distance_to_target_mesh' THEN ep.value_float END) as distance_to_target_mesh,
    MAX(CASE WHEN ep.property_id = 'distance_to_scene_mesh' THEN ep.value_float END) as distance_to_scene_mesh,
    MAX(CASE WHEN ep.property_id = 'speed' THEN ep.value_float END) as speed
FROM observations o
LEFT JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE o.episode_id = 'episode-0-...'
GROUP BY o.observation_id, o.episode_id, o.time_index, o.agent_id,
         o.x, o.y, o.z, o.v_x, o.v_y, o.v_z
ORDER BY o.time_index, o.agent_id;
*/

-- =============================================================================
-- EXAMPLE 4: Spatial heatmap (binned density)
-- =============================================================================

/*
-- 2D spatial density with velocity averaging
SELECT
    floor(x / 10) * 10 as x_bin,
    floor(y / 10) * 10 as y_bin,
    count(*) as density,
    avg(v_x) as avg_vx,
    avg(v_y) as avg_vy
FROM observations
WHERE episode_id = 'episode-0-...'
  AND time_index BETWEEN 500 AND 1000
GROUP BY x_bin, y_bin
HAVING count(*) > 5
ORDER BY density DESC;
*/

-- =============================================================================
-- EXAMPLE 5: Time-series velocity statistics
-- =============================================================================

/*
-- Moving time window statistics (100-frame windows)
SELECT
    floor(time_index / 100) * 100 as time_window,
    agent_type_id,
    count(*) as n_observations,
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed,
    stddev_pop(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as std_speed,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY sqrt(v_x*v_x + v_y*v_y)) as median_speed
FROM observations
WHERE episode_id = 'episode-0-...'
  AND v_x IS NOT NULL
GROUP BY time_window, agent_type_id
ORDER BY time_window, agent_type_id;
*/

-- =============================================================================
-- EXAMPLE 6: Distance to target over time (with extended properties)
-- =============================================================================

/*
-- Average distance to target per time window
SELECT
    floor(o.time_index / 100) * 100 as time_window,
    avg(ep.value_float) as avg_distance_to_target,
    stddev_pop(ep.value_float) as std_distance_to_target,
    count(*) as n_observations
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE o.episode_id = 'episode-0-...'
  AND ep.property_id = 'distance_to_target_center'
  AND o.time_index > 500  -- After target appears
GROUP BY time_window
ORDER BY time_window;
*/

-- =============================================================================
-- EXAMPLE 7: Agent correlation analysis (pairwise distances)
-- =============================================================================

/*
-- Correlation between agents' distances to target
WITH agent_distances AS (
    SELECT
        o.time_index,
        o.agent_id,
        ep.value_float as dist_to_target
    FROM observations o
    JOIN extended_properties ep ON o.observation_id = ep.observation_id
    WHERE o.episode_id = 'episode-0-...'
      AND ep.property_id = 'distance_to_target_center'
)
SELECT
    a.agent_id as agent_i,
    b.agent_id as agent_j,
    corr(a.dist_to_target, b.dist_to_target) as distance_correlation,
    count(*) as n_samples
FROM agent_distances a
JOIN agent_distances b
  ON a.time_index = b.time_index
  AND a.agent_id < b.agent_id
GROUP BY a.agent_id, b.agent_id
HAVING count(*) > 100
ORDER BY distance_correlation DESC;
*/

-- =============================================================================
-- EXAMPLE 8: Grafana-friendly time-series query
-- =============================================================================

/*
-- Speed over time (suitable for Grafana time-series panel)
SELECT
    to_timestamp(o.time_index * (1.0 / e.frame_rate)) as time,
    o.agent_id,
    sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0)) as speed
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
WHERE o.episode_id = 'episode-0-...'
  AND o.v_x IS NOT NULL
ORDER BY o.time_index, o.agent_id;
*/

-- =============================================================================
-- EXAMPLE 9: Aggregate across episodes in a session
-- =============================================================================

/*
-- Average speed across all episodes in a session
SELECT
    s.session_name,
    e.episode_number,
    avg(sqrt(o.v_x*o.v_x + o.v_y*o.v_y + COALESCE(o.v_z*o.v_z, 0))) as avg_speed,
    count(DISTINCT o.agent_id) as num_agents,
    max(o.time_index) as max_time
FROM observations o
JOIN episodes e ON o.episode_id = e.episode_id
JOIN sessions s ON e.session_id = s.session_id
WHERE s.session_id = 'hackathon-boid-small-200-align-cohesion_sim_run-started-20250926-214330'
  AND o.v_x IS NOT NULL
GROUP BY s.session_name, e.episode_number
ORDER BY e.episode_number;
*/

-- =============================================================================
-- EXAMPLE 10: Database statistics
-- =============================================================================

/*
-- Check row counts and sizes
SELECT
    'sessions' as table_name,
    count(*) as row_count
FROM sessions
UNION ALL
SELECT 'episodes', count(*) FROM episodes
UNION ALL
SELECT 'observations', count(*) FROM observations
UNION ALL
SELECT 'extended_properties', count(*) FROM extended_properties
UNION ALL
SELECT 'agent_types', count(*) FROM agent_types
UNION ALL
SELECT 'property_definitions', count(*) FROM property_definitions
ORDER BY table_name;
*/

-- =============================================================================
-- OPTIONAL: Create a view for common 3D boids queries
-- =============================================================================

/*
-- Uncomment to create a view for 3D boids with common properties
CREATE VIEW boids_3d_observations AS
SELECT
    o.observation_id,
    o.episode_id,
    o.time_index,
    o.agent_id,
    o.agent_type_id,
    o.x, o.y, o.z,
    o.v_x, o.v_y, o.v_z,
    MAX(CASE WHEN ep.property_id = 'distance_to_target_center' THEN ep.value_float END) as distance_to_target_center,
    MAX(CASE WHEN ep.property_id = 'distance_to_target_mesh' THEN ep.value_float END) as distance_to_target_mesh,
    MAX(CASE WHEN ep.property_id = 'distance_to_scene_mesh' THEN ep.value_float END) as distance_to_scene_mesh
FROM observations o
LEFT JOIN extended_properties ep ON o.observation_id = ep.observation_id
GROUP BY o.observation_id, o.episode_id, o.time_index, o.agent_id, o.agent_type_id,
         o.x, o.y, o.z, o.v_x, o.v_y, o.v_z;

-- Then query the view:
-- SELECT * FROM boids_3d_observations WHERE episode_id = 'episode-0-...' AND time_index BETWEEN 500 AND 1000;
*/

-- =============================================================================
-- END OF EXAMPLES
-- =============================================================================
