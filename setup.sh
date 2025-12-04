#!/bin/bash
set -e

# Parse arguments
DEV_MODE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) DEV_MODE=true ;;
        -h|--help)
            echo "Usage: bash setup.sh [--dev]"
            echo ""
            echo "Options:"
            echo "  --dev    Install with development dependencies (testing, linting)"
            echo "  -h, --help  Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Use uv pip if available, otherwise fall back to regular pip
pip() {
    if command -v uv &> /dev/null; then
        uv pip "$@"
    else
        command pip "$@"
    fi
}

# Now install hloc
base_dir=$(pwd)
submodule_dir="./submodules/hloc"

# Install the package
cd $base_dir
if [ "$DEV_MODE" = true ]; then
    echo "Installing with development dependencies..."
    pip install -e ".[dev]"
else
    pip install -e .
fi

if [ ! -d "$submodule_dir" ]; then
    git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git $submodule_dir
fi

cd $submodule_dir
# git checkout v1.4
git submodule update --init --recursive
pip install -e . --no-cache-dir
