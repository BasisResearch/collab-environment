-- Spatial analysis queries for 3D boids data

-- name: get_spatial_heatmap
-- Get spatial density heatmap with configurable binning in 3D
-- Returns binned positions with density counts and average velocities
-- Supports both episode_id (single episode) and session_id (all episodes in session)
SELECT
    floor(x / :bin_size) * :bin_size as x_bin,
    floor(y / :bin_size) * :bin_size as y_bin,
    floor(z / :bin_size) * :bin_size as z_bin,
    count(*) as density,
    avg(v_x) as avg_vx,
    avg(v_y) as avg_vy,
    avg(v_z) as avg_vz
FROM observations
WHERE ((:episode_id IS NOT NULL AND episode_id = :episode_id)
    OR (:session_id IS NOT NULL AND episode_id IN (
        SELECT episode_id FROM episodes WHERE session_id = :session_id
    )))
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
GROUP BY x_bin, y_bin, z_bin
HAVING count(*) > :min_count
ORDER BY x_bin, y_bin, z_bin;


-- name: get_velocity_heatmap
-- Get velocity field on spatial grid (for quiver plots)
-- Returns average velocities per spatial bin
SELECT
    floor(x / :bin_size) * :bin_size as x_bin,
    floor(y / :bin_size) * :bin_size as y_bin,
    count(*) as count,
    avg(v_x) as avg_vx,
    avg(v_y) as avg_vy,
    avg(v_z) as avg_vz,
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed
FROM observations
WHERE episode_id = :episode_id
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
  AND v_x IS NOT NULL
GROUP BY x_bin, y_bin
HAVING count(*) > :min_count
ORDER BY x_bin, y_bin;


-- name: get_velocity_distribution
-- Get raw velocity vectors for distribution analysis
-- Returns individual observations with velocity components and speed
SELECT
    agent_id,
    time_index,
    v_x,
    v_y,
    v_z,
    sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0)) as speed
FROM observations
WHERE episode_id = :episode_id
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
  AND v_x IS NOT NULL
ORDER BY time_index, agent_id;


-- name: get_speed_statistics
-- Compute speed statistics over time windows
-- Returns aggregated statistics per time window
-- Supports both episode_id (single episode) and session_id (all episodes in session)
SELECT
    floor(time_index / :window_size) * :window_size as time_window,
    count(*) as n_observations,
    avg(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as avg_speed,
    stddev_pop(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as std_speed,
    min(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as min_speed,
    max(sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as max_speed,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0))) as median_speed
FROM observations
WHERE ((:episode_id IS NOT NULL AND episode_id = :episode_id)
    OR (:session_id IS NOT NULL AND episode_id IN (
        SELECT episode_id FROM episodes WHERE session_id = :session_id
    )))
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
  AND v_x IS NOT NULL
GROUP BY time_window
ORDER BY time_window;


-- name: get_distance_to_target
-- Compute distance to target statistics over time windows
-- Joins with extended_properties to get distance_to_target_center
-- Supports both episode_id (single episode) and session_id (all episodes in session)
SELECT
    floor(o.time_index / :window_size) * :window_size as time_window,
    count(*) as n_observations,
    avg(ep.value_float) as avg_distance,
    stddev_pop(ep.value_float) as std_distance,
    min(ep.value_float) as min_distance,
    max(ep.value_float) as max_distance
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE ((:episode_id IS NOT NULL AND o.episode_id = :episode_id)
    OR (:session_id IS NOT NULL AND o.episode_id IN (
        SELECT episode_id FROM episodes WHERE session_id = :session_id
    )))
  AND (:start_time IS NULL OR o.time_index >= :start_time)
  AND (:end_time IS NULL OR o.time_index <= :end_time)
  AND ep.property_id = 'distance_to_target_center'
  AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
GROUP BY time_window
ORDER BY time_window;


-- name: get_distance_to_boundary
-- Compute distance to scene boundary statistics over time windows
-- Joins with extended_properties to get distance_to_scene_mesh
-- Supports both episode_id (single episode) and session_id (all episodes in session)
SELECT
    floor(o.time_index / :window_size) * :window_size as time_window,
    count(*) as n_observations,
    avg(ep.value_float) as avg_distance,
    stddev_pop(ep.value_float) as std_distance,
    min(ep.value_float) as min_distance,
    max(ep.value_float) as max_distance
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE ((:episode_id IS NOT NULL AND o.episode_id = :episode_id)
    OR (:session_id IS NOT NULL AND o.episode_id IN (
        SELECT episode_id FROM episodes WHERE session_id = :session_id
    )))
  AND (:start_time IS NULL OR o.time_index >= :start_time)
  AND (:end_time IS NULL OR o.time_index <= :end_time)
  AND ep.property_id = 'distance_to_scene_mesh'
  AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
GROUP BY time_window
ORDER BY time_window;
