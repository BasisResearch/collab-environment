#!/bin/bash

# Pass all arguments after --args to the dashboard app
# Usage: ./scripts/dev_dashboard.sh
# Or with custom buckets: ./scripts/dev_dashboard.sh --curated-bucket my_curated --processed-bucket my_processed

cd collab_env/dashboard

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    # No arguments, run with defaults
    panel serve dashboard_app.py --dev --show --port 5007
else
    # Arguments provided, pass them through with --args
    panel serve dashboard_app.py --dev --show --port 5007 --args "$@"
fi