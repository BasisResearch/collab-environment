#!/bin/bash
set -e

# Now install hloc
base_dir=$(pwd)
submodule_dir="./submodules/hloc"

# Install the package
cd $base_dir
pip install -e .

if [ ! -d "$submodule_dir" ]; then
    git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git $submodule_dir
fi

cd $submodule_dir
# git checkout v1.4
git submodule update --init --recursive
pip install -e . --no-cache-dir
