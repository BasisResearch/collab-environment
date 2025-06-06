#!/bin/bash
set -euxo pipefail

SRC="tests/ splat/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
