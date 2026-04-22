#!/bin/bash
# ABD3 Training Script
# Usage: bash scripts/train.sh

set -e

python -m abd3.main \
    mode=train \
    block_size=4 \
    algo.mixed_block_sizes=true \
    algo.self_conditioning=true \
    algo.time_conditioning=true \
    training.max_steps=500000 \
    loader.batch_size=64 \
    "$@"
