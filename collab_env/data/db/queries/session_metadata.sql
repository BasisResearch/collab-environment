-- Session and episode metadata queries

-- name: get_categories
-- Get list of all categories
SELECT
    category_id,
    category_name,
    description
FROM categories
ORDER BY category_name;


-- name: get_sessions
-- Get list of all sessions, optionally filtered by category
SELECT
    s.session_id,
    s.session_name,
    s.category_id,
    s.created_at,
    s.config
FROM sessions s
WHERE (:category_id IS NULL OR s.category_id = :category_id)
ORDER BY s.created_at DESC;


-- name: get_episodes
-- Get all episodes for a given session
SELECT
    e.episode_id,
    e.episode_number,
    e.num_frames,
    e.num_agents,
    e.frame_rate,
    e.file_path
FROM episodes e
WHERE e.session_id = :session_id
ORDER BY e.episode_number;


-- name: get_episode_metadata
-- Get detailed metadata for a single episode
SELECT
    e.episode_id,
    e.session_id,
    e.episode_number,
    e.num_frames,
    e.num_agents,
    e.frame_rate,
    e.file_path,
    s.session_name,
    s.category_id,
    s.config
FROM episodes e
JOIN sessions s ON e.session_id = s.session_id
WHERE e.episode_id = :episode_id;


-- name: get_agent_types
-- Get distinct agent types for an episode
SELECT DISTINCT agent_type_id
FROM observations
WHERE episode_id = :episode_id
ORDER BY agent_type_id;


-- name: get_agent_types_for_session
-- Get distinct agent types across all episodes in a session
SELECT DISTINCT agent_type_id
FROM observations
WHERE episode_id IN (
    SELECT episode_id FROM episodes WHERE session_id = :session_id
)
ORDER BY agent_type_id;
