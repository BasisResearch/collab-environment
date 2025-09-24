#!/bin/bash

INCLUDED_NOTEBOOKS="docs/"

# Array of notebooks to exclude
EXCLUDED_NOTEBOOKS=(
    "docs/data/gcloud_example_download.ipynb"
    "docs/gnn/0b-Select_frames_2D.ipynb"
    "docs/gnn/ModelSelection.ipynb"
    "docs/gnn/ModelSelection-food.ipynb"
    "docs/alignment/align.ipynb"
    "docs/alignment/reprojection.ipynb"
)

# Build the ignore flags
IGNORE_FLAGS=""
for notebook in "${EXCLUDED_NOTEBOOKS[@]}"; do
    IGNORE_FLAGS="$IGNORE_FLAGS --ignore $notebook"
done

CI=1 pytest --nbval-lax --dist loadscope --nbval-cell-timeout=600 -n auto $INCLUDED_NOTEBOOKS $IGNORE_FLAGS
