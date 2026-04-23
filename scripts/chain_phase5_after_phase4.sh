#!/usr/bin/env bash
# Watchdog: waits for the currently-running phase4 orchestrator to exit, then
# automatically launches phase5. Intended to run in its own tmux window so the
# user gets continuous GPU utilization across phase4 → phase5 without manual
# intervention.
#
# Detection: polls pgrep for `run_phase4_wt103.sh`. When the parent orchestrator
# is gone for 5 consecutive checks (≥5 minutes), phase5 is kicked off.

set -euo pipefail
cd "$(dirname "$(dirname "$0")")"

source /home/achinda1/miniconda3/etc/profile.d/conda.sh
conda activate abd3

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/chain_phase5_${STAMP}.log"
exec > >(tee -a "${LOG}") 2>&1

echo "[chain] watcher started at $(date)"
echo "[chain] watching for run_phase4_wt103.sh to exit ..."

gone_streak=0
REQUIRED_GONE=5   # 5 × 60s = 5 min of being gone before we trust it's done
INTERVAL=60

while true; do
    if pgrep -f "run_phase4_wt103.sh" > /dev/null; then
        gone_streak=0
        echo "[chain] $(date +%T)  phase4 still running ..."
    else
        gone_streak=$((gone_streak + 1))
        echo "[chain] $(date +%T)  phase4 not seen (${gone_streak}/${REQUIRED_GONE})"
        if [[ ${gone_streak} -ge ${REQUIRED_GONE} ]]; then
            break
        fi
    fi
    sleep "${INTERVAL}"
done

echo "[chain] phase4 exited; waiting 120 s for GPU memory to drain, then starting phase5"
sleep 120

exec ./scripts/run_phase5_owt.sh
