#!/usr/bin/env bash
# Master training pipeline — chains Phase 4 (WT103 ablations) → Phase 5 (OWT main).
# Designed for a single tmux window that runs uninterrupted for ~3 weeks.
#
# Usage:
#   tmux new-window -t abd3 -n pipeline './scripts/run_training_pipeline.sh'

set -euo pipefail
cd "$(dirname "$(dirname "$0")")"

source /home/achinda1/miniconda3/etc/profile.d/conda.sh
conda activate abd3

STAMP="$(date +%Y%m%d-%H%M%S)"
MASTER_LOG="logs/pipeline_${STAMP}.log"
mkdir -p logs
echo "[pipeline] master log: ${MASTER_LOG}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========================================================================"
echo "[$(date '+%F %T')] TRAINING PIPELINE START"
echo "========================================================================"

./scripts/run_phase4_wt103.sh
echo "[$(date '+%F %T')] Phase 4 done; starting Phase 5"
./scripts/run_phase5_owt.sh

echo "========================================================================"
echo "[$(date '+%F %T')] TRAINING PIPELINE COMPLETE"
echo "========================================================================"
