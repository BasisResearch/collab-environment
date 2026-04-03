#!/bin/bash

# source $(dirname $0)/deploy/config.sh

panel serve collab_env/dashboard/spatial_analysis_app.py --dev --show --port 5008 --static-dirs dashboard-static=collab_env/dashboard/static