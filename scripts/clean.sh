#!/bin/bash
set -euxo pipefail

SRC="tests"
ruff check --fix $SRC
ruff format $SRC