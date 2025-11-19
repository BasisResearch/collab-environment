# Cloud Deployment Setup

Production setup for spatial analysis dashboard with Cloud SQL and Cloud Run.

## Architecture

```
┌─────────────────┐
│   Cloud SQL     │  ← Managed PostgreSQL/TimescaleDB
│  (PostgreSQL)   │
└────────▲────────┘
         │
    ┌────┴─────┐
    │          │
┌───┴────┐  ┌──┴─────────┐
│ Local  │  │ Cloud Run  │
│ Dev +  │  │ Dashboard  │
│ Loader │  │ (read-only)│
└────────┘  └────────────┘
```

**Benefits:**
- ✅ Centralized database in the cloud
- ✅ Load data from local machine (fast upload from source files)
- ✅ View/analyze from anywhere via Cloud Run dashboard
- ✅ Multiple users can access the same data

---

## Prerequisites

- Google Cloud Project with billing enabled
- `.envrc` with `GOOGLE_APPLICATION_CREDENTIALS` set
- Local Python environment: `.venv-310`

```bash
# Load credentials
source .envrc

# Verify authentication
gcloud auth list
```

---

## 1. Create Cloud SQL Instance

```bash
# Set variables
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export INSTANCE_NAME="spatial-analysis-db"
export DB_NAME="tracking_analytics"
export DB_USER="postgres"
export DB_PASSWORD="$(openssl rand -base64 32)"

# Create PostgreSQL instance
gcloud sql instances create ${INSTANCE_NAME} \
    --database-version=POSTGRES_15 \
    --tier=db-g1-small \
    --region=${REGION} \
    --database-flags=max_connections=100 \
    --backup-start-time=03:00

# Create database
gcloud sql databases create ${DB_NAME} \
    --instance=${INSTANCE_NAME}

# Set password
gcloud sql users set-password ${DB_USER} \
    --instance=${INSTANCE_NAME} \
    --password="${DB_PASSWORD}"

# Store password in Secret Manager
echo -n "${DB_PASSWORD}" | gcloud secrets create postgres-password \
    --data-file=- \
    --replication-policy="automatic"

echo "✅ Cloud SQL instance created: ${INSTANCE_NAME}"
echo "📝 Database password stored in Secret Manager: postgres-password"
```

### Initialize Database Schema

```bash
# Download and start Cloud SQL Auth Proxy
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.13.0/cloud-sql-proxy.darwin.amd64
chmod +x cloud-sql-proxy

# Start proxy in background
./cloud-sql-proxy ${PROJECT_ID}:${REGION}:${INSTANCE_NAME} &
PROXY_PID=$!

# Wait for connection
sleep 3

# Initialize database
export DB_BACKEND=postgres
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=${DB_NAME}
export POSTGRES_USER=${DB_USER}
export POSTGRES_PASSWORD=$(gcloud secrets versions access latest --secret=postgres-password)

source .venv-310/bin/activate
python -m collab_env.data.db.init_database --backend postgres

# Stop proxy
kill $PROXY_PID

echo "✅ Database schema initialized"
```

---

## 2. Local Development Setup

### Install Cloud SQL Auth Proxy

```bash
# Download proxy (macOS ARM64)
curl -o cloud-sql-proxy \
    https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.13.0/cloud-sql-proxy.darwin.arm64

# OR macOS Intel
curl -o cloud-sql-proxy \
    https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.13.0/cloud-sql-proxy.darwin.amd64

# OR Linux
curl -o cloud-sql-proxy \
    https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.13.0/cloud-sql-proxy.linux.amd64

chmod +x cloud-sql-proxy
mv cloud-sql-proxy ~/bin/  # or keep in project root
```

### Configure Local Environment

Add to `.envrc`:

```bash
# Cloud SQL Configuration
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export INSTANCE_NAME="spatial-analysis-db"

# Database connection (via Cloud SQL Auth Proxy)
export DB_BACKEND=postgres
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=tracking_analytics
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=$(gcloud secrets versions access latest --secret=postgres-password 2>/dev/null || echo "password")

# Cloud SQL connection string for convenience
export CLOUD_SQL_INSTANCE="${PROJECT_ID}:${REGION}:${INSTANCE_NAME}"
```

Reload environment:
```bash
direnv allow  # if using direnv
# OR
source .envrc
```

### Daily Workflow

**Terminal 1: Start Cloud SQL Proxy**
```bash
source .envrc
./cloud-sql-proxy ${CLOUD_SQL_INSTANCE}

# Leave running...
```

**Terminal 2: Load Data**
```bash
source .envrc
source .venv-310/bin/activate

# Load 2D boids data
python -m collab_env.data.db.db_loader \
    --source boids2d \
    --path simulated_data/boid_food_basic.pt

# Load GNN rollout data
python -m collab_env.data.db.db_loader \
    --source boids2d_rollout \
    --path trained_models/food/basic/n0_h1_vr0.5_s2/rollout_results/
```

**Terminal 3: Run Dashboard Locally**
```bash
source .envrc
source .venv-310/bin/activate

panel serve collab_env/dashboard/spatial_analysis_app.py \
    --show \
    --dev \
    --port 5007
```

Open: http://localhost:5007

---

## 3. Deploy Dashboard to Cloud Run

### Create Dockerfile

Create `Dockerfile.dashboard`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-db.txt .
COPY pyproject.toml .
COPY README.rst .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-db.txt
RUN pip install --no-cache-dir -e .

# Copy application code
COPY collab_env/ /app/collab_env/
COPY schema/ /app/schema/

# Environment
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run dashboard
CMD panel serve collab_env/dashboard/spatial_analysis_app.py \
    --address 0.0.0.0 \
    --port ${PORT} \
    --allow-websocket-origin="*" \
    --num-threads 4
```

### Build and Deploy

```bash
# Set variables (if not already set)
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export INSTANCE_NAME="spatial-analysis-db"

# Build and submit to Artifact Registry
gcloud builds submit \
    --tag gcr.io/${PROJECT_ID}/spatial-dashboard \
    -f Dockerfile.dashboard

# Grant Cloud Run service account access to secrets
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding postgres-password \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Deploy to Cloud Run
gcloud run deploy spatial-analysis-dashboard \
    --image gcr.io/${PROJECT_ID}/spatial-dashboard \
    --region ${REGION} \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --add-cloudsql-instances ${PROJECT_ID}:${REGION}:${INSTANCE_NAME} \
    --set-env-vars="DB_BACKEND=postgres,POSTGRES_DB=tracking_analytics,POSTGRES_USER=postgres" \
    --set-env-vars="POSTGRES_HOST=/cloudsql/${PROJECT_ID}:${REGION}:${INSTANCE_NAME}" \
    --set-secrets="POSTGRES_PASSWORD=postgres-password:latest"

# Get the URL
gcloud run services describe spatial-analysis-dashboard \
    --region ${REGION} \
    --format="value(status.url)"
```

Your dashboard is now live! 🚀

---

## 4. Update and Redeploy

```bash
# Rebuild and deploy
gcloud builds submit --tag gcr.io/${PROJECT_ID}/spatial-dashboard -f Dockerfile.dashboard

gcloud run deploy spatial-analysis-dashboard \
    --image gcr.io/${PROJECT_ID}/spatial-dashboard \
    --region ${REGION}
```

---

## 5. Maintenance

### View Logs

```bash
# Dashboard logs
gcloud run logs read spatial-analysis-dashboard --region ${REGION} --limit 50

# Database logs
gcloud sql operations list --instance ${INSTANCE_NAME} --limit 10
```

### Database Backup

```bash
# Manual backup
gcloud sql backups create --instance ${INSTANCE_NAME}

# List backups
gcloud sql backups list --instance ${INSTANCE_NAME}

# Restore from backup
gcloud sql backups restore BACKUP_ID --backup-instance ${INSTANCE_NAME}
```

### Connect to Database Directly

```bash
# Via Cloud SQL Proxy
./cloud-sql-proxy ${PROJECT_ID}:${REGION}:${INSTANCE_NAME}

# In another terminal
psql "host=localhost port=5432 dbname=tracking_analytics user=postgres"
```

### Delete Session Data

```bash
# Connect via proxy, then:
source .venv-310/bin/activate
python -c "
from collab_env.data.db.query_backend import QueryBackend
qb = QueryBackend()
# List sessions
print(qb.get_sessions())
# Delete specific session (cascading delete)
# qb.delete_session('session-id-here')
"
```

---

## Cost Estimation

**Cloud SQL (db-g1-small):**
- Instance: ~$25-35/month
- Storage: ~$0.17/GB/month
- Backups: ~$0.08/GB/month

**Cloud Run:**
- Free tier: 2M requests/month
- Beyond free: $0.00002400/request
- Idle: $0 (scales to zero)

**Total:** ~$30-40/month for small-medium usage

**Cost optimization:**
- Use `db-f1-micro` ($7/month) for testing
- Stop Cloud SQL instance when not in use: `gcloud sql instances patch ${INSTANCE_NAME} --activation-policy=NEVER`
- Restart: `gcloud sql instances patch ${INSTANCE_NAME} --activation-policy=ALWAYS`

---

## Troubleshooting

### Cloud SQL Proxy Connection Failed

```bash
# Check if instance is running
gcloud sql instances describe ${INSTANCE_NAME} --format="value(state)"

# Check connection name
gcloud sql instances describe ${INSTANCE_NAME} --format="value(connectionName)"

# Test with verbose logging
./cloud-sql-proxy ${PROJECT_ID}:${REGION}:${INSTANCE_NAME} --verbose
```

### Cloud Run Can't Connect to Database

```bash
# Verify Cloud SQL connection is configured
gcloud run services describe spatial-analysis-dashboard \
    --region ${REGION} \
    --format="value(spec.template.spec.containers[0].env)"

# Check service account permissions
gcloud sql instances describe ${INSTANCE_NAME} \
    --format="value(serviceAccountEmailAddress)"
```

### Password Not Found

```bash
# Verify secret exists
gcloud secrets versions access latest --secret=postgres-password

# Recreate if needed
echo -n "your_password" | gcloud secrets versions add postgres-password --data-file=-
```

---

## Security Best Practices

1. **Private IP (Optional):** Configure Cloud SQL with private IP for VPC-only access
2. **IAM Authentication:** Use Cloud SQL IAM authentication instead of passwords
3. **Restricted Access:** Add `--no-allow-unauthenticated` to Cloud Run for internal-only access
4. **Audit Logs:** Enable Cloud SQL audit logging for compliance

---

## References

- [Cloud SQL for PostgreSQL](https://cloud.google.com/sql/docs/postgres)
- [Cloud SQL Auth Proxy](https://cloud.google.com/sql/docs/postgres/connect-auth-proxy)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Database Schema](../data/db/README.md)
- [Dashboard Widgets](../data/dashboard/README.md)
