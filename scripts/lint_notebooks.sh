#!/bin/bash

INCLUDED_NOTEBOOKS="docs/"

# Array of notebooks to exclude
EXCLUDED_NOTEBOOKS=(
    "docs/gnn/ModelSelection.ipynb"
    "docs/gnn/ModelSelection-food.ipynb"
    "docs/gnn/0b-Select_frames_2D.ipynb"
)

# Build the nbqa-exclude pattern (regex with | separator)
EXCLUDE_PATTERN=$(printf "|%s" "${EXCLUDED_NOTEBOOKS[@]}")
EXCLUDE_PATTERN=${EXCLUDE_PATTERN:1}  # Remove the leading '|'

nbqa mypy $INCLUDED_NOTEBOOKS --nbqa-exclude "$EXCLUDE_PATTERN"
nbqa isort --check --diff $INCLUDED_NOTEBOOKS --nbqa-exclude "$EXCLUDE_PATTERN"
nbqa black --check $INCLUDED_NOTEBOOKS --nbqa-exclude "$EXCLUDE_PATTERN"
nbqa flake8 $INCLUDED_NOTEBOOKS --nbqa-exclude "$EXCLUDE_PATTERN"