# Deploy Tracking Studio to Cloud Run

## Context
The tracking studio NiceGUI app is implemented in `collab_env/tracking_studio/`. The Cloud Run infra (`Dockerfile.tracking-studio`, `cloudbuild.yaml`) exists but has several blockers that prevent a successful deployment.

## Blockers Found

### 1. Wrong script path in Dockerfile CMD (CRITICAL)
- **File**: `Dockerfile.tracking-studio:59`
- CMD is `python scripts/run_tracking_studio.py` but the file lives at `scripts/tracking/run_tracking_studio.py`
- Container will crash on startup

### 2. `config-local/` not available in Cloud Build (CRITICAL)
- **File**: `Dockerfile.tracking-studio:34`
- `COPY config-local/ config/` will fail because `config-local/` is in `.gitignore`
- Cloud Build clones the repo, so gitignored files aren't available

### 3. GCSClient doesn't support Application Default Credentials (CRITICAL)
- **File**: `collab_env/data/gcs_utils.py:33`
- `GCSClient.__init__()` asserts the credentials file exists and uses `service_account.Credentials.from_service_account_file()`
- On Cloud Run, the recommended auth is ADC via the service account - no credential file needed
- Need to add ADC fallback to `GCSClient`

### 4. `reload=True` in production
- **File**: `collab_env/tracking_studio/app.py:800`
- NiceGUI's reload mode uses file watchers and a different startup method, which can cause issues in containers
- Should be `False` in production (controlled by env var)

### 5. Roboflow API key needs Secret Manager
- **File**: `cloudbuild.yaml:33`
- Currently uses `--set-env-vars=ROBOFLOW_API_KEY=${_ROBOFLOW_API_KEY}` (build substitution)
- Key should be stored in GCP Secret Manager for security

## Plan

### Step 1: Fix Dockerfile
In `Dockerfile.tracking-studio`:
- Fix CMD path: `scripts/run_tracking_studio.py` -> `scripts/tracking/run_tracking_studio.py`
- Remove `COPY config-local/ config/` (not available in Cloud Build, ADC replaces it)
- Add `ENV NICEGUI_RELOAD=false`

### Step 2: Add ADC support to GCSClient
In `collab_env/data/gcs_utils.py`, modify `GCSClient.__init__()`:
- If credentials_path is provided and file exists -> use service account file (current behavior)
- If credentials_path is `None` or file doesn't exist -> fall back to ADC:
  - `storage.Client(project=project_id)` (ADC auto-detected)
  - `gcsfs.GCSFileSystem(project=project_id, token='google_default')`
- Replace `assert os.path.exists()` with conditional logic
- Log which auth method is being used

### Step 3: Update tracking studio app for production
In `collab_env/tracking_studio/app.py`:
- `get_credentials_path()`: return `None` when env var not set and default path doesn't exist (triggers ADC in GCSClient)
- `ui.run(reload=...)`: use `os.getenv("NICEGUI_RELOAD", "true").lower() == "true"` so Dockerfile can disable it

### Step 4: Set up Roboflow API key in Secret Manager
In `cloudbuild.yaml`, change the deploy step to use `--set-secrets` instead of `--set-env-vars` for the API key:
```yaml
- '--set-secrets=ROBOFLOW_API_KEY=roboflow-api-key:latest'
```
This references a secret named `roboflow-api-key` in Secret Manager.

**Manual prerequisite** (run once before first deploy):
```bash
# Create the secret
echo -n "YOUR_ROBOFLOW_KEY" | gcloud secrets create roboflow-api-key --data-file=-

# Grant Cloud Run service account access
gcloud secrets add-iam-policy-binding roboflow-api-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Step 5: Deploy
```bash
gcloud builds submit --config=cloudbuild.yaml
```
No substitution needed since the Roboflow key comes from Secret Manager.

## Files to Modify
1. `Dockerfile.tracking-studio` - fix CMD path, remove config-local COPY, add NICEGUI_RELOAD=false
2. `collab_env/data/gcs_utils.py` - add ADC fallback in GCSClient.__init__()
3. `collab_env/tracking_studio/app.py` - update get_credentials_path(), make reload configurable
4. `cloudbuild.yaml` - switch ROBOFLOW_API_KEY from substitution to Secret Manager

## Verification
1. Build Docker image locally: `docker build -f Dockerfile.tracking-studio -t tracking-studio .`
2. Run locally: `docker run -p 8080:8080 tracking-studio` (GCS will be disabled without creds, that's fine)
3. Verify the app loads at http://localhost:8080 with upload + YOLO working
4. Create secret in Secret Manager (manual one-time step)
5. Deploy via `gcloud builds submit --config=cloudbuild.yaml`
6. Verify the Cloud Run URL serves the app and GCS browsing works
