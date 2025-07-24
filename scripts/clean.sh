#!/bin/bash
set -euxo pipefail

SRC="collab_env tests"
ruff check --fix $SRC
ruff format $SRC