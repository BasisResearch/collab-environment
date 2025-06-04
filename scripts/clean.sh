#!/bin/bash
set -euxo pipefail

SRC="module tests"
ruff check --fix $SRC
ruff format $SRC
