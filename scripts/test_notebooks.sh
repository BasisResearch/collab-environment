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
    "docs/tracking/full_pipeline.ipynb"
)

# Notebooks requiring GCS credentials - excluded when SKIP_GCS_TESTS is set
if [ -n "${SKIP_GCS_TESTS:-}" ]; then
    EXCLUDED_NOTEBOOKS+=(
        "docs/data/gcloud_bucket_manipulation.ipynb"
    )
fi

# Build the ignore flags
IGNORE_FLAGS=""
for notebook in "${EXCLUDED_NOTEBOOKS[@]}"; do
    IGNORE_FLAGS="$IGNORE_FLAGS --ignore $notebook"
done

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS $IGNORE_FLAGS
