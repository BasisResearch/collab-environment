-- Spatial analysis queries for 3D boids data

-- name: get_spatial_heatmap_episode
-- Get spatial density heatmap with configurable binning in 3D (episode scope)
-- Returns binned positions with density counts and average velocities
-- Optimized version without NULL checks (episode scope only)
SELECT
    floor(x / :bin_size) * :bin_size as x_bin,
    floor(y / :bin_size) * :bin_size as y_bin,
    floor(z / :bin_size) * :bin_size as z_bin,
    count(*) as density,
    avg(v_x) as avg_vx,
    avg(v_y) as avg_vy,
    avg(v_z) as avg_vz
FROM observations
WHERE episode_id = :episode_id
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
GROUP BY x_bin, y_bin, z_bin
HAVING count(*) > :min_count
ORDER BY x_bin, y_bin, z_bin;


-- name: get_spatial_heatmap_session
-- Get spatial density heatmap with configurable binning in 3D (session scope)
-- Returns binned positions with density counts and average velocities
-- Optimized version without NULL checks (session scope only)
SELECT
    floor(x / :bin_size) * :bin_size as x_bin,
    floor(y / :bin_size) * :bin_size as y_bin,
    floor(z / :bin_size) * :bin_size as z_bin,
    count(*) as density,
    avg(v_x) as avg_vx,
    avg(v_y) as avg_vy,
    avg(v_z) as avg_vz
FROM observations
WHERE episode_id IN (
        SELECT episode_id FROM episodes WHERE session_id = :session_id
    )
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
GROUP BY x_bin, y_bin, z_bin
HAVING count(*) > :min_count
ORDER BY x_bin, y_bin, z_bin;


