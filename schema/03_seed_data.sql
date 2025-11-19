-- =============================================================================
-- 03_seed_data.sql
-- Seed data: default agent types, property categories, and property definitions
-- PostgreSQL-compatible
-- =============================================================================

-- =============================================================================
-- AGENT TYPES - Common agent/track types
-- =============================================================================

INSERT INTO agent_types (type_id, type_name, description) VALUES
    ('agent', 'agent', 'Generic simulated agent (boid)'),
    ('env', 'environment', 'Environment entity (walls, obstacles, boundaries)'),
    ('target', 'target', 'Target object in simulation'),
    ('food', 'food', 'Stationary food target in 2D boids simulation'),
    ('bird', 'bird', 'Bird detected in video tracking'),
    ('rat', 'rat', 'Rat detected in video tracking'),
    ('gerbil', 'gerbil', 'Gerbil detected in video tracking')
ON CONFLICT (type_id) DO NOTHING;

-- =============================================================================
-- CATEGORIES - Session and property categories
-- =============================================================================

INSERT INTO categories (category_id, category_name, description) VALUES
    ('boids_3d', '3D Boids Simulations', 'Sessions from 3D boid simulations'),
    ('boids_2d', '2D Boids Simulations', 'Sessions from 2D boid simulations'),
    ('boids_2d_rollout', '2D Boids GNN Rollout', 'GNN model predictions on 2D boids test data'),
    ('tracking_csv', 'Real-World Tracking', 'Sessions from video tracking (CSV data)')
ON CONFLICT (category_id) DO NOTHING;

-- =============================================================================
-- PROPERTY DEFINITIONS - Extended properties
-- =============================================================================

-- 3D Boids properties
INSERT INTO property_definitions (property_id, property_name, data_type, description, unit) VALUES
    -- Distance metrics
    ('distance_to_target_center', 'Distance to Target Center', 'float', 'Euclidean distance to target centroid', 'scene_units'),
    ('distance_to_target_mesh', 'Distance to Target Mesh', 'float', 'Distance to nearest point on target mesh', 'scene_units'),
    ('distance_to_scene_mesh', 'Distance to Scene Mesh', 'float', 'Distance to scene boundary mesh', 'scene_units'),

    -- Target mesh closest point (stored as separate components)
    ('target_mesh_closest_x', 'Target Mesh Closest Point X', 'float', 'X coordinate of closest point on target mesh', 'scene_units'),
    ('target_mesh_closest_y', 'Target Mesh Closest Point Y', 'float', 'Y coordinate of closest point on target mesh', 'scene_units'),
    ('target_mesh_closest_z', 'Target Mesh Closest Point Z', 'float', 'Z coordinate of closest point on target mesh', 'scene_units'),

    -- Scene mesh closest point
    ('scene_mesh_closest_x', 'Scene Mesh Closest Point X', 'float', 'X coordinate of closest point on scene mesh', 'scene_units'),
    ('scene_mesh_closest_y', 'Scene Mesh Closest Point Y', 'float', 'Y coordinate of closest point on scene mesh', 'scene_units'),
    ('scene_mesh_closest_z', 'Scene Mesh Closest Point Z', 'float', 'Z coordinate of closest point on scene mesh', 'scene_units')
ON CONFLICT (property_id) DO NOTHING;

-- Tracking CSV properties
INSERT INTO property_definitions (property_id, property_name, data_type, description, unit) VALUES
    ('bbox_x1', 'Bounding Box X1', 'float', 'Top-left X coordinate of bounding box', 'pixels'),
    ('bbox_y1', 'Bounding Box Y1', 'float', 'Top-left Y coordinate of bounding box', 'pixels'),
    ('bbox_x2', 'Bounding Box X2', 'float', 'Bottom-right X coordinate of bounding box', 'pixels'),
    ('bbox_y2', 'Bounding Box Y2', 'float', 'Bottom-right Y coordinate of bounding box', 'pixels')
ON CONFLICT (property_id) DO NOTHING;

-- Computed properties
INSERT INTO property_definitions (property_id, property_name, data_type, description, unit) VALUES
    ('acceleration_x', 'Acceleration X', 'float', 'X component of acceleration (computed from velocity)', 'scene_units/frame^2'),
    ('acceleration_y', 'Acceleration Y', 'float', 'Y component of acceleration (computed from velocity)', 'scene_units/frame^2'),
    ('acceleration_z', 'Acceleration Z', 'float', 'Z component of acceleration (computed from velocity)', 'scene_units/frame^2'),
    ('speed', 'Speed', 'float', 'Magnitude of velocity vector', 'scene_units/frame'),
    ('acceleration_magnitude', 'Acceleration Magnitude', 'float', 'Magnitude of acceleration vector', 'scene_units/frame^2')
ON CONFLICT (property_id) DO NOTHING;

-- Tracking metadata properties (moved from observations table)
INSERT INTO property_definitions (property_id, property_name, data_type, description, unit) VALUES
    ('confidence', 'Detection Confidence', 'float', 'Detection confidence score from tracking', 'probability'),
    ('detection_class', 'Detection Class', 'string', 'Detected object class label', 'label')
ON CONFLICT (property_id) DO NOTHING;

-- GNN Attention weight properties
INSERT INTO property_definitions (property_id, property_name, data_type, description, unit) VALUES
    ('attn_weight_self', 'Self Attention Weight', 'float', 'Agent self-attention weight', 'dimensionless'),
    ('attn_weight_boid', 'Boid Attention Weight', 'float', 'Sum of attention to other boid agents', 'dimensionless'),
    ('attn_weight_food', 'Food Attention Weight', 'float', 'Attention to food agent', 'dimensionless')
ON CONFLICT (property_id) DO NOTHING;

-- 2D Boids distance metrics
INSERT INTO property_definitions (property_id, property_name, data_type, description, unit) VALUES
    ('distance_to_food', 'Distance to Food', 'float', 'Euclidean distance from boid to food location', 'scene_units'),
    ('distance_to_food_actual', 'Distance to Actual Food', 'float', 'Euclidean distance from boid to true food location (from actual episode) - predicted episodes only', 'scene_units'),
    ('distance_to_food_predicted', 'Distance to Predicted Food', 'float', 'Euclidean distance from boid to GNN-predicted food location - predicted episodes only', 'scene_units')
ON CONFLICT (property_id) DO NOTHING;

-- =============================================================================
-- END OF SEED DATA
-- =============================================================================
