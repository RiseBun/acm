#!/usr/bin/env bash
# Durable one-job watcher for P1 follow-up experiments.
#
# Usage:
#   bash scripts/p1_mount_watch_job.sh A|B|C
#
# This script is meant to be launched with nohup. It waits for the required
# FUSE mount(s), then runs exactly one job.
set -euo pipefail

cd "$(dirname "$0")/.."

JOB="${1:?usage: $0 A|B|C}"
PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
RUN_ROOT="experiments/p1_full_gpu"
LOG_DIR="${RUN_ROOT}/logs"
PID_DIR="${RUN_ROOT}/pids"
DONE_DIR="${RUN_ROOT}/done"
mkdir -p "${LOG_DIR}" "${PID_DIR}" "${DONE_DIR}"

NUPLAN_ROOT="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split"
NUSC_ROOT="/mnt/datasets/e2e-nuscenes/20260302"
NUPLAN_INDEX="$(pwd)/experiments/nuplan_50k_balanced_paths_seed7.json"
NUPLAN_STAGE0="experiments/nuplan_stage0_50k_seed7"
NUPLAN_STAGE1="experiments/nuplan_stage1_50k"
CACHE_DIR="experiments/_token_cache"
SEEDS=(7 42 123)

echo "$$" > "${PID_DIR}/${JOB}.pid"

wait_for_mount() {
    local path="$1"
    local label="$2"
    until ls "${path}" >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] ${JOB}: waiting for ${label}: ${path}"
        sleep 300
    done
    echo "[$(date '+%F %T')] ${JOB}: ${label} available"
}

run_a() {
    wait_for_mount "${NUPLAN_ROOT}" "nuPlan mount"

    local vis_ckpt="${NUPLAN_STAGE0}/object_relation_decoupled_visibility/model.pt"
    if [[ ! -f "${vis_ckpt}" ]]; then
        echo "[$(date '+%F %T')] A: running missing Stage0 decoupled+visibility warm-start"
        env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" run_stage0_table3.py \
            --config configs/debug_mvp.json \
            --dataset nuplan \
            --nuplan-root "${NUPLAN_ROOT}" \
            --nuplan-num-samples 50000 \
            --nuplan-index-json "${NUPLAN_INDEX}" \
            --nuplan-workers 32 \
            --nuplan-lazy \
            --loader-workers 32 \
            --variant object_relation_decoupled_visibility \
            --epochs 10 \
            --batch-size 128 \
            --lr-scale 4 \
            --output-dir "${NUPLAN_STAGE0}" \
            --seed 7
    else
        echo "[$(date '+%F %T')] A: found Stage0 warm-start ${vis_ckpt}"
    fi

    echo "[$(date '+%F %T')] A: launching 3 Stage1 seeds concurrently"
    local pids=()
    for seed in "${SEEDS[@]}"; do
        local seed_log="${LOG_DIR}/A_stage1_50k_seed${seed}_wm_decoupled.log"
        env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" run_stage1_table4.py \
            --config configs/debug_mvp.json \
            --dataset nuplan \
            --nuplan-root "${NUPLAN_ROOT}" \
            --nuplan-num-samples 50000 \
            --nuplan-index-json "${NUPLAN_INDEX}" \
            --nuplan-workers 32 \
            --nuplan-lazy \
            --loader-workers 32 \
            --condition wm_decoupled \
            --stage0-root "${NUPLAN_STAGE0}" \
            --epochs 10 \
            --batch-size 128 \
            --lr-scale 4 \
            --horizon 5 \
            --entropy-beta 0.003 \
            --action-sample-clip 5 \
            --output-dir "${NUPLAN_STAGE1}" \
            --seed "${seed}" \
            > "${seed_log}" 2>&1 &
        pids+=("$!")
        sleep 5
    done
    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            fail=$((fail + 1))
        fi
    done
    [[ "${fail}" -eq 0 ]]
}

run_b() {
    wait_for_mount "${NUPLAN_ROOT}" "nuPlan mount"
    wait_for_mount "${NUSC_ROOT}" "nuScenes mount"
    env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/cross_dataset_eval.py \
        --out-dir experiments/cross_dataset_eval \
        --num-scenes 700 \
        --token-cache-dir "${CACHE_DIR}" \
        --nuplan-root "${NUPLAN_ROOT}" \
        --nuplan-num-samples 50000 \
        --nuplan-index-json "${NUPLAN_INDEX}" \
        --nuplan-lazy \
        --nuplan-workers 32 \
        --loader-workers 32 \
        --batch-size 128 \
        --horizon 5
}

run_c() {
    wait_for_mount "${NUPLAN_ROOT}" "nuPlan mount"
    env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/plot_paper_fig_imagination_nuplan.py \
        --nuplan-root "${NUPLAN_ROOT}" \
        --nuplan-num-samples 50000 \
        --nuplan-index-json "${NUPLAN_INDEX}" \
        --stage1-root "${NUPLAN_STAGE1}/seed7" \
        --batch-size 128 \
        --loader-workers 32 \
        --horizon 5 \
        --max-val-samples 5000
}

echo "[$(date '+%F %T')] ${JOB}: watcher started"
case "${JOB}" in
    A) run_a ;;
    B) run_b ;;
    C) run_c ;;
    *) echo "unknown job: ${JOB}" >&2; exit 2 ;;
esac
echo "[$(date '+%F %T')] ${JOB}: complete"
touch "${DONE_DIR}/${JOB}.done"
