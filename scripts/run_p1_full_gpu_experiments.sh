#!/usr/bin/env bash
# Launch the remaining P1/P2 experiments aggressively on the single H20.
#
# The runner is mount-aware: nuScenes tasks can often use the token cache, while
# nuPlan tasks wait until the FUSE mount comes back. Long jobs write logs under
# experiments/p1_full_gpu/logs.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
ROOT="$(pwd)"
RUN_ROOT="experiments/p1_full_gpu"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

NUPLAN_ROOT="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split"
NUSC_ROOT="/mnt/datasets/e2e-nuscenes/20260302"
NUPLAN_INDEX="experiments/nuplan_50k_balanced_paths_seed7.json"
NUPLAN_STAGE0="experiments/nuplan_stage0_50k_seed7"
NUPLAN_STAGE1="experiments/nuplan_stage1_50k"
NUSC_STAGE1="experiments/stage1_pilot_x"
CACHE_DIR="experiments/_token_cache"

SEEDS=(7 42 123)

wait_for_mount() {
    local path="$1"
    local label="$2"
    local log="$3"
    until ls "${path}" >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] waiting for ${label}: ${path}" | tee -a "${log}"
        sleep 300
    done
    echo "[$(date '+%F %T')] ${label} is available" | tee -a "${log}"
}

run_a_nuplan_decoupled_vis() {
    local log="${LOG_DIR}/A_nuplan_wm_decoupled_vis_orchestrator.log"
    echo "[$(date '+%F %T')] A: start orchestrator" | tee -a "${log}"
    wait_for_mount "${NUPLAN_ROOT}" "nuPlan mount" "${log}"

    local vis_ckpt="${NUPLAN_STAGE0}/object_relation_decoupled_visibility/model.pt"
    if [[ ! -f "${vis_ckpt}" ]]; then
        echo "[$(date '+%F %T')] A: missing ${vis_ckpt}; running Stage0 vis warm-start" | tee -a "${log}"
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
            --seed 7 \
            > "${LOG_DIR}/A_stage0_vis_50k.log" 2>&1
    else
        echo "[$(date '+%F %T')] A: found Stage0 vis warm-start ${vis_ckpt}" | tee -a "${log}"
    fi

    echo "[$(date '+%F %T')] A: launching 3 Stage1 seeds concurrently" | tee -a "${log}"
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
    echo "[$(date '+%F %T')] A: Stage1 finished; failures=${fail}" | tee -a "${log}"
    return "${fail}"
}

run_b_cross_dataset_eval() {
    local log="${LOG_DIR}/B_cross_dataset_eval.log"
    echo "[$(date '+%F %T')] B: start orchestrator" | tee -a "${log}"
    wait_for_mount "${NUPLAN_ROOT}" "nuPlan mount" "${log}"
    # nuScenes usually resolves from token cache, but keep the mount check so a
    # cache miss fails here instead of halfway through the eval.
    wait_for_mount "${NUSC_ROOT}" "nuScenes mount" "${log}"
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
        --horizon 5 \
        > "${log}" 2>&1
}

run_c_fig4b() {
    local log="${LOG_DIR}/C_fig4b_nuplan_lane_conflict.log"
    echo "[$(date '+%F %T')] C: start orchestrator" | tee -a "${log}"
    wait_for_mount "${NUPLAN_ROOT}" "nuPlan mount" "${log}"
    env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/plot_paper_fig_imagination_nuplan.py \
        --nuplan-root "${NUPLAN_ROOT}" \
        --nuplan-num-samples 50000 \
        --nuplan-index-json "${NUPLAN_INDEX}" \
        --stage1-root "${NUPLAN_STAGE1}/seed7" \
        --batch-size 128 \
        --loader-workers 32 \
        --horizon 5 \
        --max-val-samples 5000 \
        > "${log}" 2>&1
}

run_d_horizon_sensitivity() {
    local log="${LOG_DIR}/D_horizon_sensitivity_orchestrator.log"
    echo "[$(date '+%F %T')] D: launching K=3/5/7 nuScenes evals" | tee -a "${log}"
    local pids=()
    for horizon in 3 5 7; do
        local out_dir="experiments/horizon_sensitivity_nuscenes/k${horizon}"
        local hlog="${LOG_DIR}/D_nuscenes_horizon_k${horizon}.log"
        mkdir -p "${out_dir}"
        env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/plot_paper_fig_imagination_nuscenes.py \
            --stage1-root "${NUSC_STAGE1}/seed7" \
            --out-dir "${out_dir}" \
            --num-scenes 700 \
            --token-cache-dir "${CACHE_DIR}" \
            --batch-size 128 \
            --horizon "${horizon}" \
            --conditions wm_object wm_decoupled \
            > "${hlog}" 2>&1 &
        pids+=("$!")
        sleep 3
    done
    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            fail=$((fail + 1))
        fi
    done
    echo "[$(date '+%F %T')] D: finished; failures=${fail}" | tee -a "${log}"
    return "${fail}"
}

run_a_nuplan_decoupled_vis &
PID_A=$!
run_b_cross_dataset_eval &
PID_B=$!
run_c_fig4b &
PID_C=$!
run_d_horizon_sensitivity &
PID_D=$!

echo "Launched orchestrators:"
echo "  A=${PID_A}  B=${PID_B}  C=${PID_C}  D=${PID_D}"
echo "Logs: ${ROOT}/${LOG_DIR}"

FAIL=0
for pid in "${PID_A}" "${PID_B}" "${PID_C}" "${PID_D}"; do
    if ! wait "${pid}"; then
        FAIL=$((FAIL + 1))
    fi
done

echo "[$(date '+%F %T')] all orchestrators finished; failures=${FAIL}"
exit "${FAIL}"
