#!/bin/bash
set -e
cd "$(dirname "$0")"

ENV_NAME="cuda-challenge"

# Activate conda environment
if conda info --envs 2>/dev/null | grep -q "$ENV_NAME"; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "ERROR: Conda environment '$ENV_NAME' not found. Run ./setup.sh first."
    exit 1
fi

python benchmark.py "$@"
