# Cloud Deployment Scripts

Automated scripts for deploying the spatial analysis dashboard to Google Cloud Run with Cloud SQL.

## Quick Start

```bash
# 1. Configure project (edit if needed)
vim scripts/deploy/config.sh

# 2. Create Cloud SQL instance (see caveat below)
./scripts/deploy/setup_cloud_sql.sh

# 3. Start proxy, init database, and load data locally
./scripts/deploy/start_proxy.sh
# In another terminal:
source scripts/deploy/config.sh
python -m collab_env.data.db.init_database --backend postgres
python -m collab_env.data.db.db_loader --source boids2d --path simulated_data/boid_food_basic.pt

# 4. Deploy dashboard to Cloud Run
./scripts/deploy/build_and_deploy.sh
```

## Scripts

### Core Setup

**`setup_cloud_sql.sh`**
- Creates Cloud SQL PostgreSQL 17 instance, database, and user
- Stores password in Secret Manager
- Grants IAM permissions (Cloud SQL Client role)
- One-time setup
- **Note:** The `gcloud sql instances create` command may fail; you may need to create the instance manually via the Cloud Console

**`start_proxy.sh`**
- Starts Cloud SQL Auth Proxy on `PROXY_PORT` (default 5433)
- Requires `GOOGLE_APPLICATION_CREDENTIALS` to be set
- Run in a dedicated terminal; leave running during local dev

### Deployment

**`build_and_deploy.sh`**
- Submits build to Cloud Build via `cloudbuild.yaml`
- Grants IAM roles (Cloud Build -> Cloud Run admin, service account user, secret access)
- Grants public access (`allUsers` invoker role) to the Cloud Run service
- Use for initial deployment and updates

**`cloudbuild.yaml`**
- Cloud Build configuration used by `build_and_deploy.sh`
- Builds Docker image from `Dockerfile.dashboard`
- Pushes to Google Container Registry (`gcr.io`)
- Deploys to Cloud Run with Cloud SQL connection and secrets

### Configuration

**`config.sh`**
- Centralized configuration sourced by all scripts
- Reads `PROJECT_ID` from `gcloud config` by default
- Fetches password from Secret Manager
- Sets all `POSTGRES_*` and `DB_*` environment variables for local use
- Edit to customize project, region, instance names, database tier, Cloud Run resources

**`Dockerfile.dashboard`**
- Python 3.10 slim image
- Installs deps via `uv` from `requirements-db.txt`
- Runs `panel serve collab_env/dashboard/spatial_analysis_app.py`
- Used by `cloudbuild.yaml`

## Typical Workflows

### Initial Setup

```bash
# One-time: create Cloud SQL instance
./scripts/deploy/setup_cloud_sql.sh

# Start proxy (in a dedicated terminal)
./scripts/deploy/start_proxy.sh

# In another terminal: init database and deploy
source scripts/deploy/config.sh
python -m collab_env.data.db.init_database --backend postgres
./scripts/deploy/build_and_deploy.sh
```

### Daily Development

```bash
# Terminal 1: Start proxy
./scripts/deploy/start_proxy.sh

# Terminal 2: Source config (sets all DB env vars) and work
source scripts/deploy/config.sh

# Load data
python -m collab_env.data.db.db_loader --source boids2d --path simulated_data/boid_food_basic.pt

# Run dashboard locally (proxy uses port 5433 by default)
panel serve collab_env/dashboard/spatial_analysis_app.py --show --dev

# When done: Ctrl-C the proxy in Terminal 1
```

### Update Deployment

```bash
# After code changes
./scripts/deploy/build_and_deploy.sh
```

## Configuration

Edit `scripts/deploy/config.sh` to customize. Defaults:

```bash
# Google Cloud (PROJECT_ID reads from gcloud config by default)
export REGION="us-central1"
export INSTANCE_NAME="spatial-analysis-db"

# Database
export DB_NAME="tracking_analytics"
export DB_USER="postgres"
export DB_TIER="db-g1-small"  # or db-f1-micro for testing

# Local proxy
export PROXY_PORT="5433"  # avoids conflict with local postgres on 5432

# Cloud Run
export SERVICE_NAME="spatial-analysis-dashboard"
export MEMORY="5Gi"
export CPU="2"
export TIMEOUT="3600"
```

## Troubleshooting

### Proxy won't connect

```bash
# Check instance is running
gcloud sql instances describe spatial-analysis-db

# Test proxy with verbose logging
cloud-sql-proxy PROJECT_ID:REGION:INSTANCE_NAME --verbose
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
