# Cloud Deployment Scripts

Automated scripts for deploying the spatial analysis dashboard to Google Cloud Run with Cloud SQL.

## Quick Start

```bash
# 1. Configure project (edit if needed)
vim scripts/deploy/config.sh

# 2. Create Cloud SQL instance
./scripts/deploy/setup_cloud_sql.sh

# 3. Initialize database schema
./scripts/deploy/init_database.sh

# 4. Load data locally
./scripts/deploy/dev_local_cloudsql.sh

# 5. Deploy dashboard to Cloud Run
./scripts/deploy/build_and_deploy.sh
```

## Scripts

### Core Setup

**`setup_cloud_sql.sh`**
- Creates Cloud SQL PostgreSQL instance
- Creates database
- Stores password in Secret Manager
- One-time setup

**`init_database.sh`**
- Initializes database schema
- Downloads Cloud SQL Auth Proxy if needed
- Connects via proxy and runs init_database.py
- Run after setup_cloud_sql.sh

### Development

**`dev_local_cloudsql.sh`**
- Interactive script for local development
- Starts Cloud SQL Auth Proxy
- Options to:
  - Load data (db_loader)
  - Start dashboard locally
  - Open Python shell
  - Export environment variables

**`stop_proxy.sh`**
- Stops Cloud SQL Auth Proxy
- Run when done with local development

### Deployment

**`build_and_deploy.sh`**
- Builds Docker image
- Pushes to Google Container Registry
- Deploys to Cloud Run
- Configures Cloud SQL connection
- Use for initial deployment and updates

### Configuration

**`config.sh`**
- Centralized configuration
- Sourced by all scripts
- Edit to customize:
  - Project ID, region
  - Instance names
  - Database tier
  - Cloud Run resources

**`Dockerfile.dashboard`**
- Docker image definition
- Installs dependencies
- Runs Panel dashboard
- Used by build_and_deploy.sh

## Typical Workflows

### Initial Setup

```bash
# One-time setup
./scripts/deploy/setup_cloud_sql.sh
./scripts/deploy/init_database.sh
./scripts/deploy/build_and_deploy.sh
```

### Daily Development

```bash
# Terminal 1: Start proxy and load data
./scripts/deploy/dev_local_cloudsql.sh
# Choose option 1 (Load data)

# Terminal 2: Run dashboard locally
source .venv-310/bin/activate
source .envrc
export DB_BACKEND=postgres
export POSTGRES_HOST=localhost
export POSTGRES_DB=tracking_analytics
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=$(gcloud secrets versions access latest --secret=postgres-password)
panel serve collab_env/dashboard/spatial_analysis_app.py --show --dev

# When done
./scripts/deploy/stop_proxy.sh
```

### Update Deployment

```bash
# After code changes
./scripts/deploy/build_and_deploy.sh
```

## Configuration

Edit `scripts/deploy/config.sh` to customize:

```bash
# Google Cloud
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export INSTANCE_NAME="spatial-analysis-db"

# Database
export DB_NAME="tracking_analytics"
export DB_USER="postgres"
export DB_TIER="db-g1-small"  # or db-f1-micro for testing

# Cloud Run
export SERVICE_NAME="spatial-analysis-dashboard"
export MEMORY="2Gi"
export CPU="2"
export TIMEOUT="3600"
```

## Troubleshooting

### Proxy won't connect

```bash
# Check instance is running
gcloud sql instances describe spatial-analysis-db

# Test proxy with verbose logging
./cloud-sql-proxy PROJECT_ID:REGION:INSTANCE_NAME --verbose
```

### Cloud Run can't access database

```bash
# Verify Cloud SQL connection
gcloud run services describe spatial-analysis-dashboard \
    --region us-central1 \
    --format="value(spec.template.spec.containers[0].env)"

# Check logs
gcloud run logs read spatial-analysis-dashboard --limit 50
```

### Password issues

```bash
# Verify secret exists
gcloud secrets versions access latest --secret=postgres-password

# Reset if needed
echo -n "new_password" | gcloud secrets versions add postgres-password --data-file=-
gcloud sql users set-password postgres --instance=spatial-analysis-db --password="new_password"
```

## Cost Management

```bash
# Stop instance when not in use (saves ~$25/month)
gcloud sql instances patch spatial-analysis-db --activation-policy=NEVER

# Restart when needed
gcloud sql instances patch spatial-analysis-db --activation-policy=ALWAYS

# Use smaller tier for testing
# Edit config.sh: export DB_TIER="db-f1-micro"
```

## Clean Up

```bash
# Delete Cloud Run service
gcloud run services delete spatial-analysis-dashboard --region us-central1

# Delete Cloud SQL instance (WARNING: destroys all data)
gcloud sql instances delete spatial-analysis-db

# Delete secrets
gcloud secrets delete postgres-password
```

## See Also

- [Complete Setup Guide](../../docs/dashboard/CLOUD_SETUP.md)
- [Database Documentation](../../docs/data/db/README.md)
- [Dashboard Widgets](../../docs/data/dashboard/README.md)
