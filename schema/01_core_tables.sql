-- =============================================================================
-- 01_core_tables.sql
-- Core dimension and fact tables for tracking analytics
-- PostgreSQL-compatible
-- =============================================================================

-- =============================================================================
-- DIMENSION TABLES
-- =============================================================================

CREATE TABLE sessions (
    session_id VARCHAR PRIMARY KEY,
    session_name VARCHAR NOT NULL,
    data_source VARCHAR NOT NULL,  -- boids_3d, boids_2d, tracking_csv
    category VARCHAR NOT NULL,      -- simulated, birds, rats, gerbils
    created_at TIMESTAMP DEFAULT now(),
    config JSONB,                   -- Full configuration from YAML/config.pt
    metadata JSONB                  -- Environment, mesh paths, notes
);

CREATE INDEX idx_sessions_source ON sessions(data_source);

COMMENT ON TABLE sessions IS 'Top-level container for related episodes (simulation run or fieldwork session)';
COMMENT ON COLUMN sessions.config IS 'Full configuration as JSON (from YAML or .pt config)';
COMMENT ON COLUMN sessions.metadata IS 'Additional metadata: notes, environment, mesh references';

-- =============================================================================

CREATE TABLE episodes (
    episode_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL,
    num_frames INTEGER NOT NULL,
    num_agents INTEGER NOT NULL,
    frame_rate DOUBLE PRECISION DEFAULT 30.0,
    file_path VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_episodes_session ON episodes(session_id);

COMMENT ON TABLE episodes IS 'Single simulation run or video tracking session';
COMMENT ON COLUMN episodes.num_frames IS 'Total number of timesteps/frames in episode';
COMMENT ON COLUMN episodes.frame_rate IS 'Frames per second (typically 30)';

-- =============================================================================

CREATE TABLE agent_types (
    type_id VARCHAR PRIMARY KEY,
    type_name VARCHAR NOT NULL,
    description TEXT
);

COMMENT ON TABLE agent_types IS 'Agent/track type definitions (agent, target, bird, rat, etc.)';

-- =============================================================================
-- FACT TABLE - Core observations (positions and velocities only)
-- =============================================================================

CREATE TABLE observations (
    -- Surrogate key for foreign key references
    observation_id BIGSERIAL UNIQUE NOT NULL,

    -- Natural composite primary key
    episode_id VARCHAR NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    time_index INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,

    -- Dimensions
    agent_type_id VARCHAR NOT NULL REFERENCES agent_types(type_id),

    -- Core spatial data (required)
    x DOUBLE PRECISION NOT NULL,
    y DOUBLE PRECISION NOT NULL,
    z DOUBLE PRECISION,  -- NULL for 2D data

    -- Core velocity data (optional, may be computed)
    v_x DOUBLE PRECISION,
    v_y DOUBLE PRECISION,
    v_z DOUBLE PRECISION,

    -- Tracking metadata (for real-world data)
    confidence DOUBLE PRECISION,      -- Detection confidence [0-1]
    detection_class VARCHAR,          -- Detected object class

    -- Primary key: ensures unique (episode, time, agent) tuple
    PRIMARY KEY (episode_id, time_index, agent_id)
);

-- Essential indexes
CREATE INDEX idx_obs_episode ON observations(episode_id);
CREATE INDEX idx_obs_episode_time ON observations(episode_id, time_index);

COMMENT ON TABLE observations IS 'Core time-series data: positions and velocities. Extended properties in separate table.';
COMMENT ON COLUMN observations.observation_id IS 'Surrogate key for foreign key references (auto-increment)';
COMMENT ON COLUMN observations.time_index IS 'Frame number / timestep (0-indexed)';
COMMENT ON COLUMN observations.agent_id IS 'Agent ID within episode';
COMMENT ON COLUMN observations.z IS 'Z position (NULL for 2D data)';
COMMENT ON COLUMN observations.v_x IS 'X velocity component (may be NULL if not stored)';

-- =============================================================================
-- END OF CORE TABLES
-- =============================================================================
