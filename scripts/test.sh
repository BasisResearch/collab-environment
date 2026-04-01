#!/bin/bash
set -euxo pipefail

CI=1 pytest tests/ -n auto
