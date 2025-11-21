#!/bin/bash
# Deployment configuration
# Source this file in other scripts: source scripts/deploy/config.sh

# Google Cloud Configuration
export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-us-central1}"
export INSTANCE_NAME="${INSTANCE_NAME:-spatial-analysis-db}"
export DB_NAME="${DB_NAME:-tracking_analytics}"
export DB_USER="${DB_USER:-postgres}"

# Local proxy port (use 5433 if you have local postgres on 5432)
export PROXY_PORT="${PROXY_PORT:-5433}"

# Get password from Secret Manager
export POSTGRES_PASSWORD=$(gcloud secrets versions access latest --secret=postgres-password --project=${PROJECT_ID})

# Set database connection environment variables
export DB_BACKEND=postgres
export POSTGRES_HOST=localhost
export POSTGRES_PORT=${PROXY_PORT}
export POSTGRES_DB=${DB_NAME}
export POSTGRES_USER=${DB_USER}

# Cloud Run Configuration
export SERVICE_NAME="${SERVICE_NAME:-spatial-analysis-dashboard}"
export MEMORY="${MEMORY:-5Gi}"
export CPU="${CPU:-2}"
export TIMEOUT="${TIMEOUT:-3600}"

# Database tier (db-f1-micro, db-g1-small, db-n1-standard-1, etc.)
export DB_TIER="${DB_TIER:-db-g1-small}"

# Cloud SQL connection string
export CLOUD_SQL_INSTANCE="${PROJECT_ID}:${REGION}:${INSTANCE_NAME}"

# Validate configuration
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID not set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Colors for output
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export RED='\033[0;31m'
export NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}
