#!/bin/bash

# This will be '/' on Code Ocean and '/app' in local docker-compose
export CODEOCEAN_BASE_DIR="${CODEOCEAN_BASE_DIR:-/}"

# Add the base directory to Python's path to enable module imports
export PYTHONPATH="${PYTHONPATH}:${CODEOCEAN_BASE_DIR}"

# Change to the base code directory
cd "${CODEOCEAN_BASE_DIR}/code"

echo "Running neural network example..."
python nn_example.py
