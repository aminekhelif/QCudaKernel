#!/usr/bin/env bash
set -e

# This script builds the project, runs tests, and generates performance plots.

rm -rf build
mkdir build
cd build
cmake ..
make -j

# Run tests (which generate results.csv)
ctest --output-on-failure

cd ..
python3 plot_results.py
echo "All done. Check 'build/results.csv' and 'performance.png'."