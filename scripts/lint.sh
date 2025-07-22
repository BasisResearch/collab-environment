#!/bin/bash
set -euxo pipefail

SRC="tests/ environment/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
