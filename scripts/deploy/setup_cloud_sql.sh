#!/bin/bash
# Create and configure Cloud SQL instance

set -e

# Load configuration
source "$(dirname "$0")/config.sh"

echo "=========================================="
echo "Cloud SQL Instance Setup"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Instance: ${INSTANCE_NAME}"
echo "Database: ${DB_NAME}"
echo "Tier: ${DB_TIER}"
echo "=========================================="
echo ""

# Check if instance already exists
if gcloud sql instances describe ${INSTANCE_NAME} --project=${PROJECT_ID} &>/dev/null; then
    log_warn "Instance ${INSTANCE_NAME} already exists"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    # Generate secure password
    export DB_PASSWORD="$(openssl rand -base64 32)"

    ########################################################################
    #### WARNING: THIS DOESN"T WORK. CREATE MANUALLY IN THE WEB UI. ####
    ########################################################################
    log_info "Creating Cloud SQL instance (this may take 5-10 minutes)..."
    gcloud sql instances create ${INSTANCE_NAME} \
        --database-version=POSTGRES_17 \
        --tier=${DB_TIER} \
        --region=${REGION} \
        --database-flags=max_connections=100 \
        --backup-start-time=03:00 \
        --project=${PROJECT_ID}

    log_info "Cloud SQL instance created: ${INSTANCE_NAME}"
    ########################################################################

    # Create database
    log_info "Creating database: ${DB_NAME}"
    gcloud sql databases create ${DB_NAME} \
        --instance=${INSTANCE_NAME} \
        --project=${PROJECT_ID}

    # Set password
    log_info "Setting database password..."
    gcloud sql users set-password ${DB_USER} \
        --instance=${INSTANCE_NAME} \
        --password="${DB_PASSWORD}" \
        --project=${PROJECT_ID}

    # Store password in Secret Manager
    log_info "Storing password in Secret Manager..."

    # Check if secret exists
    if gcloud secrets describe postgres-password --project=${PROJECT_ID} &>/dev/null; then
        log_warn "Secret postgres-password already exists, adding new version..."
        echo -n "${DB_PASSWORD}" | gcloud secrets versions add postgres-password \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo -n "${DB_PASSWORD}" | gcloud secrets create postgres-password \
            --data-file=- \
            --replication-policy="automatic" \
            --project=${PROJECT_ID}
    fi

    log_info "Password stored in Secret Manager: postgres-password"

    # Grant IAM permissions
    log_info "Granting Cloud SQL Client permissions..."

    # Try user account first
    USER_EMAIL=$(gcloud config get-value account 2>/dev/null || echo "")
    if [ -n "$USER_EMAIL" ]; then
        gcloud projects add-iam-policy-binding ${PROJECT_ID} \
            --member="user:${USER_EMAIL}" \
            --role="roles/cloudsql.client" \
            --condition=None 2>/dev/null || log_warn "Permission already granted or failed"
        log_info "Granted Cloud SQL Client role to: ${USER_EMAIL}"
    fi

    # Also grant to service account if available
    if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
        SA_EMAIL=$(cat ${GOOGLE_APPLICATION_CREDENTIALS} | python3 -c "import sys, json; print(json.load(sys.stdin)['client_email'])" 2>/dev/null || echo "")
        if [ -n "$SA_EMAIL" ]; then
            gcloud projects add-iam-policy-binding ${PROJECT_ID} \
                --member="serviceAccount:${SA_EMAIL}" \
                --role="roles/cloudsql.client" \
                --condition=None 2>/dev/null || log_warn "Permission already granted or failed"
            log_info "Granted Cloud SQL Client role to: ${SA_EMAIL}"
        fi
    fi

    echo ""
    echo "=========================================="
    log_info "Cloud SQL setup complete!"
    echo "=========================================="
    echo ""
    echo "Connection details:"
    echo "  Instance: ${CLOUD_SQL_INSTANCE}"
    echo "  Database: ${DB_NAME}"
    echo "  User: ${DB_USER}"
    echo "  Password: Stored in Secret Manager (postgres-password)"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./scripts/deploy/init_database.sh"
    echo "  2. Load data with: ./scripts/deploy/dev_local_cloudsql.sh"
    echo "  3. Deploy dashboard: ./scripts/deploy/build_and_deploy.sh"
fi
