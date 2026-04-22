#!/usr/bin/env bash
# ==============================================================================
# wait_for_gpus.sh
#
# Polls `nvidia-smi` until every visible GPU has at least MIN_FREE_GIB of free
# memory, then exits 0. This is the safety valve for shared multi-tenant boxes
# where launching DDP against saturated GPUs would crash with a near-immediate
# OOM in `torch.cuda.set_device()`.
#
# Usage:
#   ./scripts/wait_for_gpus.sh [MIN_FREE_GIB] [POLL_SECONDS] [TIMEOUT_HOURS]
#
# Defaults:
#   MIN_FREE_GIB = 8      (threshold for every GPU)
#   POLL_SECONDS = 30
#   TIMEOUT_HOURS = 24    (gives up eventually so we never wait forever)
#
# Respects CUDA_VISIBLE_DEVICES when set (only those GPUs are required to be
# free). Progress is printed every poll; the final decision line is always
# flushed.
# ==============================================================================
set -euo pipefail

MIN_FREE_GIB="${1:-8}"
POLL_SECONDS="${2:-30}"
TIMEOUT_HOURS="${3:-24}"

MIN_FREE_MIB=$(( MIN_FREE_GIB * 1024 ))
MAX_ITERS=$(( TIMEOUT_HOURS * 3600 / POLL_SECONDS ))

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    QUERY_ARGS=(-i "${CUDA_VISIBLE_DEVICES}")
    SCOPE="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    QUERY_ARGS=()
    SCOPE="all visible GPUs"
fi

echo "[wait_for_gpus] waiting for ${SCOPE} to each have >= ${MIN_FREE_GIB} GiB free"
echo "[wait_for_gpus] polling every ${POLL_SECONDS}s, giving up after ${TIMEOUT_HOURS}h"

for iter in $(seq 1 "${MAX_ITERS}"); do
    # Returns one line per GPU: "<index>, <free MiB>"
    readarray -t rows < <(
        nvidia-smi "${QUERY_ARGS[@]}" \
                   --query-gpu=index,memory.free \
                   --format=csv,noheader,nounits
    )

    all_ok=1
    min_free=999999
    min_idx=-1
    summary=""
    for row in "${rows[@]}"; do
        idx="${row%%,*}"
        free_mib="${row##*, }"
        idx="${idx// /}"
        free_mib="${free_mib// /}"
        if (( free_mib < min_free )); then
            min_free="${free_mib}"
            min_idx="${idx}"
        fi
        summary="${summary}${idx}:$((free_mib/1024))GiB "
        if (( free_mib < MIN_FREE_MIB )); then
            all_ok=0
        fi
    done

    ts="$(date '+%H:%M:%S')"
    if (( all_ok == 1 )); then
        echo "[wait_for_gpus] ${ts} OK: all GPUs have >= ${MIN_FREE_GIB} GiB free  (${summary})"
        exit 0
    fi

    echo "[wait_for_gpus] ${ts} waiting... min-free GPU ${min_idx}: $((min_free/1024)) GiB  (${summary})"
    sleep "${POLL_SECONDS}"
done

echo "[wait_for_gpus] ERROR: timed out after ${TIMEOUT_HOURS}h without ${MIN_FREE_GIB} GiB free on every GPU"
exit 1
