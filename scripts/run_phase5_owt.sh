#!/usr/bin/env bash
# Phase 5 — Headline OpenWebText runs (all 8 GPUs, DDP).
#
# Queue (all dit_small, 500 k steps, seq 1024, bs 8 × 8 GPUs × accum 2 = 512 eff.):
#   5.1  ABD3 full              algo=abd3       (HEADLINE paper number)
#   5.2  BD3-LMs equivalent     algo=baseline   (primary comparison)
#   5.3  RDR-only               algo=rdr_only   (ablation context for main runs)
#
# Each run ≈ 6-7 days on 8× A6000. The whole queue ≈ 3 weeks of continuous
# compute. Results land in checkpoints/phase5/{abd3,baseline,rdr_only}/.
#
# Usage:
#   tmux new-window -t abd3 -n phase5 './scripts/run_phase5_owt.sh'
# or chain from phase4:
#   (./scripts/run_phase4_wt103.sh && ./scripts/run_phase5_owt.sh) 2>&1 | tee ...

set -euo pipefail
cd "$(dirname "$(dirname "$0")")"

source /home/achinda1/miniconda3/etc/profile.d/conda.sh
conda activate abd3

mkdir -p outputs logs checkpoints/phase5

STAMP="$(date +%Y%m%d-%H%M%S)"
ORCHESTRATOR_LOG="logs/phase5_owt_${STAMP}.log"
echo "[phase5] logging to ${ORCHESTRATOR_LOG}"
exec > >(tee -a "${ORCHESTRATOR_LOG}") 2>&1

export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

run_one() {
    local name="$1"
    local algo_overrides="$2"
    local attempt="$3"

    local run_dir="outputs/phase5/${name}"
    local ckpt_dir="checkpoints/phase5/${name}"
    local run_log="logs/phase5_${name}_${STAMP}_try${attempt}.log"

    # Note: we do NOT rm -rf ckpt_dir — Phase 5 supports resume on retry.
    rm -rf "${run_dir}"
    mkdir -p "${run_dir}" "${ckpt_dir}"

    # Resume if a checkpoint already exists (best-effort).
    local resume_flag=""
    local latest_ckpt
    latest_ckpt="$(ls -t "${ckpt_dir}"/*.ckpt 2>/dev/null | head -1 || true)"
    if [[ -n "${latest_ckpt}" ]]; then
        resume_flag="eval.checkpoint_path=${latest_ckpt}"
        echo "[phase5/${name}] resuming from ${latest_ckpt}"
    fi

    echo
    echo "========================================================================"
    echo "[$(date '+%F %T')] STARTING phase5/${name} (attempt ${attempt})"
    echo "  overrides : ${algo_overrides} ${resume_flag}"
    echo "  hydra dir : ${run_dir}"
    echo "  ckpt dir  : ${ckpt_dir}"
    echo "  log       : ${run_log}"
    echo "========================================================================"

    python -u -m abd3.main \
        --config-name=phase5_owt_small \
        ${algo_overrides} \
        ${resume_flag} \
        checkpointing.save_dir="${ckpt_dir}" \
        wandb.name="abd3-owt-${name}-s${STAMP}" \
        hydra.run.dir="${run_dir}" \
        2>&1 | tee "${run_log}"

    return ${PIPESTATUS[0]}
}

run_experiment() {
    local name="$1"
    local algo_overrides="$2"

    ./scripts/wait_for_gpus.sh 8 30 48

    if run_one "${name}" "${algo_overrides}" 1; then
        echo "[$(date '+%F %T')] FINISHED phase5/${name} (ok, attempt 1)"
        return 0
    fi

    echo "[$(date '+%F %T')] [WARN] phase5/${name} failed attempt 1; retrying once"
    sleep 60
    ./scripts/wait_for_gpus.sh 8 30 12

    if run_one "${name}" "${algo_overrides}" 2; then
        echo "[$(date '+%F %T')] FINISHED phase5/${name} (ok, attempt 2)"
        return 0
    fi

    echo "[$(date '+%F %T')] [FATAL] phase5/${name} failed twice; skipping"
    return 1
}

echo "[phase5] starting suite at $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ------------------------------------------------------------------------
# Order: headline ABD3 first, then baseline (direct comparison), then RDR-only.
# ------------------------------------------------------------------------
run_experiment "abd3"     "algo=abd3"     || true
run_experiment "baseline" "algo=baseline" || true
run_experiment "rdr_only" "algo=rdr_only" || true

echo
echo "========================================================================"
echo "[$(date '+%F %T')] PHASE 5 OWT SUITE COMPLETE"
echo "========================================================================"

for name in abd3 baseline rdr_only; do
    ck_dir="checkpoints/phase5/${name}"
    latest="$(ls -t "${ck_dir}"/*.ckpt 2>/dev/null | head -1 || true)"
    if [[ -n "${latest}" ]]; then
        echo "  [ok] ${name} → ${latest}"
    else
        echo "  [MISSING] ${name} — no checkpoint produced"
    fi
done
