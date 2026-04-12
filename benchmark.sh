#!/bin/bash
set -e
cd "$(dirname "$0")"
python benchmark.py "$@"
