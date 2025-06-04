#!/bin/bash
set -euxo pipefail

SRC="tests/ module/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
