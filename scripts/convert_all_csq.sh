#!/bin/bash
# Convert all CSQ files in a directory to AVI/MP4.
# Usage: ./scripts/convert_all_csq.sh [DIR]
# Default DIR: /Users/ralph/Downloads/100_FLIR

set -e

DIR="${1:-/Users/ralph/Downloads/100_FLIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
source .venv/bin/activate

shopt -s nullglob
for f in "$DIR"/*.csq; do
  out="${f%.csq}.avi"
  echo "Converting: $f -> $out"
  python -m collab_env.tracking.csq convert "$f" "$out" --max-length 99999
done

echo "Done."
