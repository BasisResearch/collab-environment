#!/bin/bash

INCLUDED_NOTEBOOKS="docs/"
EXCLUDED_NOTEBOOKS="docs/data/gcloud_example_download.ipynb"

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS --ignore $EXCLUDED_NOTEBOOKS
