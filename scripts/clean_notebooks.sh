#!/bin/bash
set -euxo pipefail

SRC="docs/"

# Array of notebooks to exclude
EXCLUDED_NOTEBOOKS=(
    "docs/gnn/ModelSelection.ipynb"
    "docs/gnn/ModelSelection-food.ipynb"
    "docs/gnn/0b-Select_frames_2D.ipynb"
)

# Build --exclude flags for ruff
RUFF_EXCLUDES=""
for nb in "${EXCLUDED_NOTEBOOKS[@]}"; do
    RUFF_EXCLUDES="$RUFF_EXCLUDES --exclude $nb"
done

ruff check --fix $SRC $RUFF_EXCLUDES
ruff format $SRC $RUFF_EXCLUDES
