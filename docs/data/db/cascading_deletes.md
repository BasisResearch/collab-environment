# Cascading Deletes

## Overview

The database schema supports cascading deletes with the following hierarchy:

```
category
  └─> session (CASCADE)
        └─> episode (CASCADE)
              ├─> observation (CASCADE)
              └─> extended_property → observation (CASCADE)
```

When you delete a parent record, all child records are automatically deleted.

## Cascade Chain

### 1. Delete Category
Deletes:
- All sessions in that category
- All episodes in those sessions
- All observations in those episodes
- All extended properties for those observations

### 2. Delete Session
Deletes:
- All episodes in that session
- All observations in those episodes
- All extended properties for those observations

### 3. Delete Episode
Deletes:
- All observations in that episode
- All extended properties for those observations

## Usage Examples

### Delete a Session (and all its data)

```sql
-- This will automatically delete all episodes, observations, and extended properties
DELETE FROM sessions WHERE session_id = 'session-2d-boid_food_basic';
```

### Delete a Category (and all its data)

```sql
-- This will automatically delete all sessions, episodes, observations, and extended properties
DELETE FROM categories WHERE category_id = 'boids_2d';
```

### Delete an Episode (and all its data)

```sql
-- This will automatically delete all observations and extended properties
DELETE FROM episodes WHERE episode_id = 'episode-0001-session-2d-boid_food_basic';
```

## Database Support

### PostgreSQL ✅
**Full support for cascading deletes.**

The schema includes `ON DELETE CASCADE` constraints on all foreign keys:
- `sessions.category_id` → `categories(category_id)` ON DELETE CASCADE
- `episodes.session_id` → `sessions(session_id)` ON DELETE CASCADE
- `observations.episode_id` → `episodes(episode_id)` ON DELETE CASCADE
- `extended_properties.observation_id` → `observations(observation_id)` ON DELETE CASCADE

### DuckDB ❌
**No support for cascading deletes.**

DuckDB does not support `ON DELETE CASCADE` as of version 1.4.1. The error message is:
```
Parser Error: FOREIGN KEY constraints cannot use CASCADE, SET NULL or SET DEFAULT
```

For DuckDB, you must manually delete child records before deleting parent records:

```sql
-- Manual cascade delete for DuckDB
-- Delete in reverse order of dependencies

-- 1. Delete extended properties first
DELETE FROM extended_properties
WHERE observation_id IN (
    SELECT o.observation_id
    FROM observations o
    JOIN episodes e ON o.episode_id = e.episode_id
    WHERE e.session_id = 'session-2d-boid_food_basic'
);

-- 2. Delete observations
DELETE FROM observations
WHERE episode_id IN (
    SELECT episode_id
    FROM episodes
    WHERE session_id = 'session-2d-boid_food_basic'
);

-- 3. Delete episodes
DELETE FROM episodes WHERE session_id = 'session-2d-boid_food_basic';

-- 4. Finally delete the session
DELETE FROM sessions WHERE session_id = 'session-2d-boid_food_basic';
```

## Testing

### Manual Test (PostgreSQL)

Run the manual cascade delete test to verify everything works:

```bash
python test_manual_cascade_delete.py
```

This creates a test category with session, episode, observations, and extended properties, then deletes the category and verifies all child data was cascade deleted.

Expected output:
```
✅ CASCADE DELETE SUCCESSFUL!
   All related data was automatically deleted:
   - Category deleted
   - Session cascade deleted
   - Episode cascade deleted
   - Observations cascade deleted
   - Extended properties cascade deleted
```

## Best Practices

1. **Be Careful**: Cascade deletes are permanent and affect all child records
2. **Use Transactions**: Wrap deletes in transactions so you can rollback if needed
3. **Backup First**: Always backup before bulk deletions
4. **Verify Counts**: Check row counts before and after to ensure expected behavior

### Example with Transaction

```sql
BEGIN;

-- Check what will be deleted
SELECT COUNT(*) FROM sessions WHERE category_id = 'boids_2d';
SELECT COUNT(*) FROM episodes e
  JOIN sessions s ON e.session_id = s.session_id
  WHERE s.category_id = 'boids_2d';

-- If counts look correct, proceed
DELETE FROM categories WHERE category_id = 'boids_2d';

-- Verify deletion
SELECT COUNT(*) FROM sessions WHERE category_id = 'boids_2d';

-- If everything looks good, commit
COMMIT;

-- Or rollback if something went wrong
-- ROLLBACK;
```

## Future Work

- Add API endpoints for safe deletion with confirmation
- Add soft deletes (mark as deleted instead of physical delete)
- Add deletion audit log
- Add restore from backup functionality
