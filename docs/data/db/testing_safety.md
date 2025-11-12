# Testing Safety Guidelines

## 🚨 CRITICAL: Database Safety

**NEVER run tests against production databases!**

Tests will:
- Drop ALL tables with CASCADE
- Delete ALL data
- Recreate schema from scratch

## Test Database Setup

### PostgreSQL Test Database

**Required:** Database name MUST end with `_test`

```bash
# Create dedicated test database
createdb tracking_analytics_test

# Set environment variables for testing
export POSTGRES_DB=tracking_analytics_test
export POSTGRES_USER=your_user
export POSTGRES_PASSWORD=your_password
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
```

### Safety Checks

The test fixtures include safety checks that will **refuse to run** if:

1. Database name is in the forbidden list:
   - `tracking_analytics` (production)
   - `production`
   - `prod`
   - `main`

2. Database name doesn't end with `_test`

**Example error:**
```
SKIPPED [1] tests/db/conftest.py:87: SAFETY: Database name 'tracking_analytics' must end with '_test'.
Example: tracking_analytics_test.
This prevents accidentally running destructive tests on production.
```

### DuckDB

DuckDB tests automatically use temporary files - no production risk.

## Running Tests Safely

### 1. Check Your Environment

```bash
# Verify test database is configured
echo $POSTGRES_DB
# Should output: tracking_analytics_test (or similar with _test suffix)

# NOT: tracking_analytics, production, etc.
```

### 2. Create Test Database

```bash
# PostgreSQL
createdb tracking_analytics_test

# Grant permissions if needed
psql -d tracking_analytics_test -c "GRANT ALL PRIVILEGES ON SCHEMA public TO your_user;"
```

### 3. Run Tests

```bash
# Run all database tests
pytest tests/db/ -v

# Run specific test file
pytest tests/db/test_2d_boids_loader.py -v

# Run only DuckDB tests (no PostgreSQL setup needed)
pytest tests/db/ -v -k duckdb
```

## What Tests Do

### Setup (per test)
1. Connect to database
2. **DROP ALL TABLES** with CASCADE
3. Recreate schema from SQL files
4. Insert seed data

### Test Execution
1. Load test data
2. Verify data
3. Test functionality

### Cleanup (per test)
1. Delete test data
2. Tables remain for next test

## Production Safety Checklist

- [ ] Test database name ends with `_test`
- [ ] Test database is separate from production
- [ ] Production credentials are NOT in test environment
- [ ] `.env` file has test database configuration
- [ ] CI/CD uses separate test database

## Example .env for Testing

```bash
# Production database (read-only for analysis)
POSTGRES_DB=tracking_analytics
POSTGRES_USER=readonly_user
POSTGRES_PASSWORD=***
POSTGRES_HOST=production.example.com

# Test database (for pytest)
TEST_POSTGRES_DB=tracking_analytics_test
TEST_POSTGRES_USER=test_user
TEST_POSTGRES_PASSWORD=***
TEST_POSTGRES_HOST=localhost
```

## Common Mistakes to Avoid

❌ **DON'T:**
```bash
# Running tests with production DB
export POSTGRES_DB=tracking_analytics
pytest tests/db/  # DANGEROUS!
```

✅ **DO:**
```bash
# Always use test database
export POSTGRES_DB=tracking_analytics_test
pytest tests/db/  # Safe
```

❌ **DON'T:**
```bash
# Testing on main production database
POSTGRES_DB=production pytest tests/db/
```

✅ **DO:**
```bash
# Use dedicated test database
POSTGRES_DB=myapp_test pytest tests/db/
```

## CI/CD Configuration

Ensure your CI/CD pipeline uses test databases:

```yaml
# GitHub Actions example
env:
  POSTGRES_DB: tracking_analytics_test
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: postgres
  POSTGRES_HOST: localhost

services:
  postgres:
    image: postgres:16
    env:
      POSTGRES_DB: tracking_analytics_test
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
```

## Recovery from Mistakes

If you accidentally ran tests on production:

1. **Stop immediately** - Don't run more tests
2. **Restore from backup** - Use your most recent backup
3. **Review environment** - Check `env` variables
4. **Update safety checks** - Add your DB name to forbidden list
5. **Test on test DB first** - Always verify configuration

## Adding New Tests

When writing new tests:

1. Use the `backend_config` or `postgres_initialized` fixtures
2. Safety checks are automatic
3. Document any special database requirements
4. Test on test database first
5. Never hardcode production credentials

## Questions?

- "Can I skip the `_test` suffix?" - **NO.** This is a critical safety feature.
- "Can I use my production DB for read-only tests?" - **NO.** Tests drop tables.
- "What if I need real data?" - Export from production, import to test DB.
- "Can I disable safety checks?" - **NO.** They exist for a reason.
