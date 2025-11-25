-- Basic Data Viewer queries for comprehensive episode visualization
-- Provides general-purpose queries that work with any extended property

-- name: get_episode_tracks
-- Get position and velocity data for all agents in an episode for animation
-- Returns raw trajectory data for time-based animation
SELECT
    agent_id,
    time_index,
    x,
    y,
    z,
    v_x,
    v_y,
    v_z,
    sqrt(v_x*v_x + v_y*v_y + COALESCE(v_z*v_z, 0)) as speed
FROM observations
WHERE episode_id = :episode_id
  AND (:start_time IS NULL OR time_index >= :start_time)
  AND (:end_time IS NULL OR time_index <= :end_time)
  AND (:agent_type = 'all' OR agent_type_id = :agent_type)
ORDER BY time_index, agent_id;


-- name: get_extended_properties_timeseries
-- Get aggregated time series for any extended properties
-- Property-agnostic query - gets all properties, filter in Python if needed
-- Returns windowed statistics for all properties (median + configurable quantile bands)
SELECT
    floor(o.time_index / :window_size) * :window_size as time_window,
    ep.property_id,
    count(*) as n_observations,
    avg(ep.value_float) as avg_value,
    stddev_pop(ep.value_float) as std_value,
    min(ep.value_float) as min_value,
    max(ep.value_float) as max_value,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY ep.value_float) as median_value,
    percentile_cont(:lower_quantile) WITHIN GROUP (ORDER BY ep.value_float) as q_lower,
    percentile_cont(:upper_quantile) WITHIN GROUP (ORDER BY ep.value_float) as q_upper
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE o.episode_id = :episode_id
  AND (:start_time IS NULL OR o.time_index >= :start_time)
  AND (:end_time IS NULL OR o.time_index <= :end_time)
  AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
  AND ep.value_float IS NOT NULL
GROUP BY time_window, ep.property_id
ORDER BY time_window, ep.property_id;


-- name: get_property_distributions_episode
-- Get raw property values for histogram generation (episode scope)
-- Property-agnostic query - gets all properties, filter in Python if needed
-- Returns individual property values for distribution analysis
-- Optimized version without NULL checks (episode scope only)
SELECT
    ep.property_id,
    ep.value_float
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE o.episode_id = :episode_id
  AND (:start_time IS NULL OR o.time_index >= :start_time)
  AND (:end_time IS NULL OR o.time_index <= :end_time)
  AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
  AND ep.value_float IS NOT NULL
ORDER BY ep.property_id, ep.value_float;


-- name: get_property_distributions_session
-- Get raw property values for histogram generation (session scope)
-- Property-agnostic query - gets all properties, filter in Python if needed
-- Returns individual property values for distribution analysis
-- Optimized version without NULL checks (session scope only)
SELECT
    ep.property_id,
    ep.value_float
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE o.episode_id IN (
        SELECT episode_id FROM episodes WHERE session_id = :session_id
    )
  AND (:start_time IS NULL OR o.time_index >= :start_time)
  AND (:end_time IS NULL OR o.time_index <= :end_time)
  AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
  AND ep.value_float IS NOT NULL
ORDER BY ep.property_id, ep.value_float;


-- name: get_available_properties_episode
-- Get list of available extended properties for a single episode
-- Returns property metadata for UI selection
-- Optimized version without NULL checks (episode scope only)
SELECT
    pd.property_id,
    pd.property_name,
    pd.description,
    pd.unit,
    pd.data_type
FROM property_definitions pd
WHERE pd.property_id IN (
    SELECT DISTINCT ep.property_id
    FROM extended_properties ep
    JOIN observations o ON ep.observation_id = o.observation_id
    WHERE o.episode_id = :episode_id
      AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
)
ORDER BY pd.property_name;


-- name: get_available_properties_session
-- Get list of available extended properties for all episodes in a session
-- Returns property metadata for UI selection
-- Optimized version without NULL checks (session scope only)
SELECT
    pd.property_id,
    pd.property_name,
    pd.description,
    pd.unit,
    pd.data_type
FROM property_definitions pd
WHERE pd.property_id IN (
    SELECT DISTINCT ep.property_id
    FROM extended_properties ep
    JOIN observations o ON ep.observation_id = o.observation_id
    WHERE o.episode_id IN (
        SELECT episode_id FROM episodes WHERE session_id = :session_id
    )
    AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
)
ORDER BY pd.property_name;


-- name: get_extended_properties_raw
-- Get raw (unaggregated) extended property values with time indices
-- For overlaying individual agent trajectories on time series plots
-- Returns all raw observations in long format for line plotting per agent
SELECT
    o.time_index,
    o.agent_id,
    ep.property_id,
    ep.value_float
FROM observations o
JOIN extended_properties ep ON o.observation_id = ep.observation_id
WHERE o.episode_id = :episode_id
  AND (:start_time IS NULL OR o.time_index >= :start_time)
  AND (:end_time IS NULL OR o.time_index <= :end_time)
  AND (:agent_type = 'all' OR o.agent_type_id = :agent_type)
  AND ep.value_float IS NOT NULL
ORDER BY ep.property_id, o.agent_id, o.time_index;