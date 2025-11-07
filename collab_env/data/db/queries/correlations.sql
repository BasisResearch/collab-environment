-- Correlation queries for agent interactions

-- name: get_velocity_correlations
-- Compute pairwise velocity correlations between agents
-- Warning: O(n²) computation, can be slow for many agents
-- Only supports episode_id (single episode). Session-level correlation disabled.
WITH agent_velocities AS (
    SELECT
        time_index,
        agent_id,
        v_x,
        v_y,
        v_z
    FROM observations
    WHERE episode_id = :episode_id
      AND (:agent_type = 'all' OR agent_type_id = :agent_type)
      AND (:start_time IS NULL OR time_index >= :start_time)
      AND (:end_time IS NULL OR time_index <= :end_time)
      AND v_x IS NOT NULL
)
SELECT
    a.agent_id as agent_i,
    b.agent_id as agent_j,
    corr(a.v_x, b.v_x) as v_x_correlation,
    corr(a.v_y, b.v_y) as v_y_correlation,
    corr(a.v_z, b.v_z) as v_z_correlation,
    count(*) as n_samples
FROM agent_velocities a
JOIN agent_velocities b
  ON a.time_index = b.time_index
  AND a.agent_id < b.agent_id
GROUP BY a.agent_id, b.agent_id
HAVING count(*) > :min_samples
ORDER BY v_x_correlation DESC;


-- name: get_distance_correlations
-- Compute pairwise distance-to-target correlations between agents
-- Warning: O(n²) computation, can be slow for many agents
-- Only supports episode_id (single episode). Session-level correlation disabled.
WITH agent_distances AS (
    SELECT
        o.time_index,
        o.agent_id,
        ep.value_float as dist_to_target
    FROM observations o
    JOIN extended_properties ep ON o.observation_id = ep.observation_id
    WHERE o.episode_id = :episode_id
      AND ep.property_id = 'distance_to_target_center'
      AND o.agent_type_id = 'agent'
      AND (:start_time IS NULL OR o.time_index >= :start_time)
      AND (:end_time IS NULL OR o.time_index <= :end_time)
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
HAVING count(*) > :min_samples
ORDER BY distance_correlation DESC;
