#!/usr/bin/env bash
# ==============================================================================
# retry_rdr_only.sh
#
# One-shot: closes out the feasibility suite by re-running the
# `ablation_rdr_only` experiment that was terminated by host-wide CUDA OOM
# during the original orchestrator pass.
#
# - First blocks on `wait_for_gpus.sh` until every GPU has >= 8 GiB free.
# - Then launches rdr_only with the same retry-once semantics the main
#   orchestrator uses, so a one-off DataLoader/NCCL hiccup doesn't abort.
#
# Intended to be launched inside the `abd3` tmux runner pane:
#   tmux send-keys -t abd3:runner "./scripts/retry_rdr_only.sh" C-m
# ==============================================================================
set -euo pipefail
cd "$(dirname "$(dirname "$0")")"

source /home/achinda1/miniconda3/etc/profile.d/conda.sh
conda activate abd3

mkdir -p outputs logs checkpoints

STAMP="$(date +%Y%m%d-%H%M%S)"
SUITE_LOG="logs/retry_rdr_only_${STAMP}.log"
echo "[retry_rdr_only] logging to ${SUITE_LOG}"
exec > >(tee -a "${SUITE_LOG}") 2>&1

echo "[retry_rdr_only] starting at $(date)"
echo "[retry_rdr_only] initial GPU state:"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv

# 8 GiB threshold, poll every 30s, up to 7 days, need 3 consecutive OK polls
# (≈90s of sustained availability) before firing — shields us from tenants
# who briefly release memory and immediately reclaim it.
./scripts/wait_for_gpus.sh 8 30 168 3

NAME="ablation_rdr_only"
ALGO_OVERRIDE="algo=rdr_only"

run_one() {
    local attempt="$1"
    local run_dir="outputs/${NAME}"
    local ckpt_dir="checkpoints/${NAME}"
    local run_log="logs/${NAME}_${STAMP}_try${attempt}.log"

    rm -rf "${run_dir}"
    mkdir -p "${run_dir}" "${ckpt_dir}"

    echo
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTING ${NAME} (attempt ${attempt})"
    echo "  algo override    : ${ALGO_OVERRIDE}"
    echo "  hydra run dir    : ${run_dir}"
    echo "  checkpoint dir   : ${ckpt_dir}"
    echo "  per-run log file : ${run_log}"
    echo "========================================================================"

    python -u -m abd3.main \
        --config-name=feasibility \
        ${ALGO_OVERRIDE} \
        training.val_check_interval=1.0 \
        loader.num_workers=0 \
        checkpointing.save_dir="${ckpt_dir}" \
        hydra.run.dir="${run_dir}" \
        2>&1 | tee "${run_log}"

    return ${PIPESTATUS[0]}
}

if run_one 1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED ${NAME} (ok, attempt 1)"
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] ${NAME} failed on attempt 1; waiting 60s then checking GPU state before retry"
sleep 60
./scripts/wait_for_gpus.sh 8 30 168 3

if run_one 2; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED ${NAME} (ok, attempt 2)"
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FATAL] ${NAME} failed twice; leaving logs at ${SUITE_LOG}"
exit 1
