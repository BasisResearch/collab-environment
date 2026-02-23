#!/bin/bash
set -euxo pipefail

SRC="collab_env tests"

mypy $SRC
ruff check $SRC
ruff format --check $SRC
