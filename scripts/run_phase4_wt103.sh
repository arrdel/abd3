#!/usr/bin/env bash
# Phase 4 — WikiText-103 ablation suite (A0-A6 × 1 seed × 200 k steps).
#
# Each run uses ALL 8 GPUs via DDP (max utilization) and takes ≈ 12-24 h on
# 8× A6000 with dit_small. The 7 runs therefore take ≈ 4-7 days wall-clock.
#
# Matrix (mirrors todo.md §6.1):
#   A0  full ABD3                     algo=abd3
#   A1  − two-stream                  algo=abd3 algo.cross_attn=false
#   A2  − self-conditioning           algo=abd3 algo.self_conditioning=false algo.self_cond_prob=0
#   A3  − mixed-B                     algo=abd3 algo.mixed_block_sizes=false
#   A4  − adaptive stopping           algo=abd3 algo.adaptive_stopping=false
#   A5  − per-block σ                 algo=abd3 algo.time_conditioning=false
#   A6  BD3-LMs equivalent            algo=baseline
#
# Usage:
#   tmux new-window -t abd3 -n phase4 './scripts/run_phase4_wt103.sh'
#   tmux attach -t abd3

set -euo pipefail
cd "$(dirname "$(dirname "$0")")"

source /home/achinda1/miniconda3/etc/profile.d/conda.sh
conda activate abd3

mkdir -p outputs logs checkpoints/phase4

STAMP="$(date +%Y%m%d-%H%M%S)"
ORCHESTRATOR_LOG="logs/phase4_wt103_${STAMP}.log"
echo "[phase4] logging to ${ORCHESTRATOR_LOG}"
exec > >(tee -a "${ORCHESTRATOR_LOG}") 2>&1

# NCCL hardening (survives transient network hiccups on shared boxes).
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

run_one() {
    local name="$1"
    local algo_overrides="$2"
    local attempt="$3"

    local run_dir="outputs/phase4/${name}"
    local ckpt_dir="checkpoints/phase4/${name}"
    local run_log="logs/phase4_${name}_${STAMP}_try${attempt}.log"

    rm -rf "${run_dir}"
    mkdir -p "${run_dir}" "${ckpt_dir}"

    echo
    echo "========================================================================"
    echo "[$(date '+%F %T')] STARTING phase4/${name} (attempt ${attempt})"
    echo "  overrides : ${algo_overrides}"
    echo "  hydra dir : ${run_dir}"
    echo "  ckpt dir  : ${ckpt_dir}"
    echo "  log       : ${run_log}"
    echo "========================================================================"

    python -u -m abd3.main \
        --config-name=phase4_wt103 \
        ${algo_overrides} \
        checkpointing.save_dir="${ckpt_dir}" \
        wandb.name="abd3-wt103-${name}-s${STAMP}" \
        hydra.run.dir="${run_dir}" \
        2>&1 | tee "${run_log}"

    return ${PIPESTATUS[0]}
}

run_experiment() {
    local name="$1"
    local algo_overrides="$2"

    # Gate on cluster-wide GPU headroom (8 GPUs, 30 GiB each free, up to 24h wait).
    ./scripts/wait_for_gpus.sh 8 30 24

    if run_one "${name}" "${algo_overrides}" 1; then
        echo "[$(date '+%F %T')] FINISHED phase4/${name} (ok, attempt 1)"
        return 0
    fi

    echo "[$(date '+%F %T')] [WARN] phase4/${name} failed attempt 1; retrying once"
    sleep 30
    ./scripts/wait_for_gpus.sh 8 30 6

    if run_one "${name}" "${algo_overrides}" 2; then
        echo "[$(date '+%F %T')] FINISHED phase4/${name} (ok, attempt 2)"
        return 0
    fi

    echo "[$(date '+%F %T')] [FATAL] phase4/${name} failed twice; skipping to next ablation"
    return 1
}

echo "[phase4] starting suite at $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ------------------------------------------------------------------------
# Ablation matrix (A0-A6). Order: start with A0 (full) so we get the headline
# number ASAP, then peel off one innovation at a time, ending with A6 (baseline).
# ------------------------------------------------------------------------
run_experiment "a0_full"          ""                                                || true
run_experiment "a1_no_twostream"  "algo.cross_attn=false"                            || true
run_experiment "a2_no_selfcond"   "algo.self_conditioning=false algo.self_cond_prob=0" || true
run_experiment "a3_no_mixedb"     "algo.mixed_block_sizes=false"                     || true
run_experiment "a4_no_adastop"    "algo.adaptive_stopping=false"                     || true
run_experiment "a5_no_perblocks"  "algo.time_conditioning=false"                     || true
run_experiment "a6_baseline"      "algo=baseline"                                    || true

echo
echo "========================================================================"
echo "[$(date '+%F %T')] PHASE 4 ABLATION SUITE COMPLETE"
echo "========================================================================"

# Summary of which runs produced a final checkpoint.
for name in a0_full a1_no_twostream a2_no_selfcond a3_no_mixedb a4_no_adastop a5_no_perblocks a6_baseline; do
    ck_dir="checkpoints/phase4/${name}"
    latest="$(ls -t "${ck_dir}"/*.ckpt 2>/dev/null | head -1 || true)"
    if [[ -n "${latest}" ]]; then
        echo "  [ok] ${name} → ${latest}"
    else
        echo "  [MISSING] ${name} — no checkpoint produced"
    fi
done
