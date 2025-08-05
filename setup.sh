#!/bin/bash
set -e

# Install the package
pip install -e .

# Now install hloc
submodule_dir="./submodules/hloc"

if [ ! -d "$submodule_dir" ]; then
    git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git $submodule_dir
fi

cd $submodule_dir
git submodule update --init --recursive
pip install -e . --no-cache-dir