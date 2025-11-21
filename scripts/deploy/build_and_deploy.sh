#!/bin/bash
# Build Docker image and deploy to Cloud Run

set -e

# Load configuration
source "$(dirname "$0")/config.sh"

echo "=========================================="
echo "Build and Deploy to Cloud Run"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Image: gcr.io/${PROJECT_ID}/spatial-dashboard"
echo "=========================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# Get project number for service accounts
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")

# Ensure Cloud Build service account can deploy to Cloud Run
log_info "Ensuring Cloud Build can deploy to Cloud Run..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin" \
    --condition=None 2>/dev/null || log_warn "Cloud Build permission already granted"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser" \
    --condition=None 2>/dev/null || log_warn "Service Account User permission already granted"

# Ensure Cloud Run service account has secret access
log_info "Ensuring Cloud Run service account has secret access..."
gcloud secrets add-iam-policy-binding postgres-password \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --project=${PROJECT_ID} \
    --condition=None 2>/dev/null || log_warn "Secret access already granted"

# Build, push, and deploy (all in one step)
log_info "Building Docker image and deploying to Cloud Run..."
log_info "This will take 3-5 minutes..."

gcloud builds submit \
    --config=scripts/deploy/cloudbuild.yaml \
    --project=${PROJECT_ID}

# Explicitly grant public access (allUsers can invoke)
log_info "Ensuring public access to Cloud Run service..."
gcloud run services add-iam-policy-binding ${SERVICE_NAME} \
    --region=${REGION} \
    --member="allUsers" \
    --role="roles/run.invoker" \
    --project=${PROJECT_ID} 2>/dev/null || log_warn "Public access already granted"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --project=${PROJECT_ID} \
    --format="value(status.url)")

echo ""
echo "=========================================="
log_info "Deployment complete!"
echo "=========================================="
echo ""
echo "Dashboard URL: ${SERVICE_URL}"
echo ""
echo "Useful commands:"
echo "  View logs: gcloud run services logs read ${SERVICE_NAME} --region ${REGION} --limit 50"
echo "  Update: ./scripts/deploy/build_and_deploy.sh"
echo "  Delete: gcloud run services delete ${SERVICE_NAME} --region ${REGION}"
