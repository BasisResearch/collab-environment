# 🚨 TESTING SAFETY - READ THIS FIRST! 🚨

## CRITICAL WARNING

**Database tests will DROP ALL TABLES and DELETE ALL DATA!**

## Before Running Tests

### ✅ SAFE: Using Test Database
```bash
export POSTGRES_DB=tracking_analytics_test  # Ends with _test
pytest tests/db/
```

### ❌ DANGEROUS: Using Production Database
```bash
export POSTGRES_DB=tracking_analytics  # Production database!
pytest tests/db/  # ← This will DELETE ALL YOUR DATA!
```

## Safety Checks

Tests include automatic safety checks that will REFUSE to run if:

1. Database name is `tracking_analytics`, `production`, `prod`, or `main`
2. Database name doesn't end with `_test`

**If tests are skipped, this is PROTECTING YOUR DATA!**

## Setup Test Database

```bash
# Create test database
createdb tracking_analytics_test

# Configure environment
export POSTGRES_DB=tracking_analytics_test
export POSTGRES_USER=your_user
export POSTGRES_PASSWORD=your_password

# Now tests are safe to run
pytest tests/db/ -v
```

## Full Documentation

See [docs/data/db/testing_safety.md](docs/data/db/testing_safety.md) for complete testing guidelines.

## Quick Reference

| Database Name | Tests Will Run? | Safe? |
|---------------|-----------------|-------|
| `tracking_analytics_test` | ✅ Yes | ✅ Safe |
| `mydb_test` | ✅ Yes | ✅ Safe |
| `tracking_analytics` | ❌ BLOCKED | 🚨 Would destroy data |
| `production` | ❌ BLOCKED | 🚨 Would destroy data |
| `mydb` | ❌ BLOCKED | 🚨 No _test suffix |

**When in doubt, tests being skipped is GOOD - it means your production data is protected!**
