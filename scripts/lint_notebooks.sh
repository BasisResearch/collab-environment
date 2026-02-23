#!/bin/bash
set -euxo pipefail

SRC="docs/"

# Array of notebooks to exclude
EXCLUDED_NOTEBOOKS=(
    "docs/gnn/ModelSelection.ipynb"
    "docs/gnn/ModelSelection-food.ipynb"
    "docs/gnn/0b-Select_frames_2D.ipynb"
)

# Build --exclude flags for ruff and --nbqa-exclude pattern for nbqa
RUFF_EXCLUDES=""
NBQA_EXCLUDE_PATTERN=""
for nb in "${EXCLUDED_NOTEBOOKS[@]}"; do
    RUFF_EXCLUDES="$RUFF_EXCLUDES --exclude $nb"
    NBQA_EXCLUDE_PATTERN="${NBQA_EXCLUDE_PATTERN:+$NBQA_EXCLUDE_PATTERN|}$nb"
done

nbqa mypy $SRC --nbqa-exclude "$NBQA_EXCLUDE_PATTERN"
ruff check $SRC $RUFF_EXCLUDES
ruff format --check $SRC $RUFF_EXCLUDES
