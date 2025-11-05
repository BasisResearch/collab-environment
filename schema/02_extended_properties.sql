-- =============================================================================
-- 02_extended_properties.sql
-- Extended properties EAV schema with category-based organization
-- PostgreSQL-compatible
-- =============================================================================

-- =============================================================================
-- PROPERTY CATEGORIES - Organize properties by data source type
-- =============================================================================

CREATE TABLE property_categories (
    category_id VARCHAR PRIMARY KEY,
    category_name VARCHAR NOT NULL,
    description TEXT
);

COMMENT ON TABLE property_categories IS 'Categories for grouping properties by data source type';
COMMENT ON COLUMN property_categories.category_id IS 'Unique category identifier (e.g., boids_3d, tracking_csv)';

-- =============================================================================
-- PROPERTY DEFINITIONS - Define available extended properties
-- =============================================================================

CREATE TABLE property_definitions (
    property_id VARCHAR PRIMARY KEY,
    property_name VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,     -- float, vector, string
    description TEXT,
    unit VARCHAR                     -- meters, pixels, m/s^2, etc.
);

COMMENT ON TABLE property_definitions IS 'Defines all available extended properties';
COMMENT ON COLUMN property_definitions.data_type IS 'Data type: float, vector, string';
COMMENT ON COLUMN property_definitions.unit IS 'Unit of measurement (e.g., scene_units, pixels, m/s^2)';

-- =============================================================================
-- PROPERTY CATEGORY MAPPING - M2M relationship
-- =============================================================================

CREATE TABLE property_category_mapping (
    property_id VARCHAR NOT NULL REFERENCES property_definitions(property_id) ON DELETE CASCADE,
    category_id VARCHAR NOT NULL REFERENCES property_categories(category_id) ON DELETE CASCADE,

    PRIMARY KEY (property_id, category_id)
);

CREATE INDEX idx_prop_cat_mapping_category ON property_category_mapping(category_id);

COMMENT ON TABLE property_category_mapping IS 'Maps properties to categories (M2M relationship)';

-- =============================================================================
-- EXTENDED PROPERTIES - EAV table for flexible property storage
-- =============================================================================

CREATE TABLE extended_properties (
    observation_id BIGINT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
    property_id VARCHAR NOT NULL REFERENCES property_definitions(property_id),

    value_float DOUBLE PRECISION,   -- For numeric properties
    value_text TEXT,                -- For strings, arrays (JSON), etc.

    PRIMARY KEY (observation_id, property_id)
);

CREATE INDEX idx_ext_props_observation ON extended_properties(observation_id);
CREATE INDEX idx_ext_props_property ON extended_properties(property_id);

COMMENT ON TABLE extended_properties IS 'EAV table for flexible extended properties';
COMMENT ON COLUMN extended_properties.value_float IS 'Use for numeric properties (distances, accelerations, etc.)';
COMMENT ON COLUMN extended_properties.value_text IS 'Use for strings, arrays (stored as JSON), etc.';

-- =============================================================================
-- END OF EXTENDED PROPERTIES
-- =============================================================================
