#!/bin/bash
set -euxo pipefail

SRC="tests/ collab_env/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
