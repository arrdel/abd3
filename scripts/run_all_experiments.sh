#!/usr/bin/env bash
# Orchestrator: runs the feasibility experiment followed by the multi-GPU
# ablation matrix. Each experiment is 5k steps on 8 GPUs using the feasibility
# hyperparameters; only `algo` differs across ablations.
#
# Experiments:
#   1. feasibility        - full ABD3 (reference run)
#   2. ablation_baseline  - BD3-LMs style, no RDR, no mixed blocks
#   3. ablation_rdr_only  - RDR (self-cond + adaptive stop), no mixed blocks
#
# Usage:
#   ./scripts/run_all_experiments.sh
#
# Intended to be launched inside the `abd3` tmux session so progress is
# observable via `tmux attach -t abd3`.

set -euo pipefail
cd "$(dirname "$(dirname "$0")")"

source /home/achinda1/miniconda3/etc/profile.d/conda.sh
conda activate abd3

mkdir -p outputs logs

STAMP="$(date +%Y%m%d-%H%M%S)"
ORCHESTRATOR_LOG="logs/orchestrator_${STAMP}.log"
echo "[orchestrator] logging to ${ORCHESTRATOR_LOG}"
exec > >(tee -a "${ORCHESTRATOR_LOG}") 2>&1

run_one() {
    local name="$1"
    local algo_override="$2"
    local attempt="$3"

    local run_dir="outputs/${name}"
    local ckpt_dir="checkpoints/${name}"
    local run_log="logs/${name}_${STAMP}_try${attempt}.log"

    # Clean previous attempt's run dir so CSVLogger/Hydra don't fight over version_* or hparams
    rm -rf "${run_dir}"
    mkdir -p "${run_dir}" "${ckpt_dir}"

    echo
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTING experiment: ${name} (attempt ${attempt})"
    echo "  algo override    : ${algo_override}"
    echo "  hydra run dir    : ${run_dir}"
    echo "  checkpoint dir   : ${ckpt_dir}"
    echo "  per-run log file : ${run_log}"
    echo "========================================================================"

    # loader.num_workers=0 avoids the torch DataLoader signal handler that
    # turns any unrelated worker crash (e.g. host OOM on a shared box) into a
    # fatal `signal: Aborted` in the main process.
    python -u -m abd3.main \
        --config-name=feasibility \
        ${algo_override} \
        training.val_check_interval=1.0 \
        loader.num_workers=0 \
        checkpointing.save_dir="${ckpt_dir}" \
        hydra.run.dir="${run_dir}" \
        2>&1 | tee "${run_log}"

    return ${PIPESTATUS[0]}
}

run_experiment() {
    local name="$1"
    local algo_override="$2"

    if run_one "${name}" "${algo_override}" 1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED experiment: ${name} (ok, attempt 1)"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] experiment '${name}' failed on attempt 1; retrying once"
    sleep 10

    if run_one "${name}" "${algo_override}" 2; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED experiment: ${name} (ok, attempt 2)"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FATAL] experiment '${name}' failed twice; aborting suite"
    exit 1
}

echo "[orchestrator] starting suite at $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

run_experiment "feasibility"       ""
run_experiment "ablation_baseline" "algo=baseline"
run_experiment "ablation_rdr_only" "algo=rdr_only"

echo
echo "========================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
