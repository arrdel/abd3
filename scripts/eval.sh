#!/bin/bash
# ABD3 Evaluation Script
# Usage: bash scripts/eval.sh eval.checkpoint_path=<path>

set -e

python -m abd3.main \
    mode=sample \
    algo.adaptive_stopping=true \
    "$@"
