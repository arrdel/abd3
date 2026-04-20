#!/usr/bin/env zsh
# Run the ABD3 feasibility training run (tiny model on WikiText-2)
# Usage: ./scripts/run_feasibility.sh

set -euo pipefail
cd "$(dirname "$(dirname "$0")")" # cd to project root

# Use Python module entrypoint to ensure package imports resolve
python -u -m abd3.main --config-name=feasibility
