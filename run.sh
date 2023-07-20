#!/bin/bash

# Step 1: Generate whl
echo "Generating whl..."
python setup.py sdist bdist_wheel

# Step 2: Generate pip installable package
echo "Generating pip installable package..."
pip install .

# Step 3: Clean up
echo "Cleaning up..."
rm -rf build
rm -rf dist
rm -rf global_utils.egg-info

echo "Done!"
