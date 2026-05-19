#!/usr/bin/env bash
# A2 smoke: nuPlan shared-relation Stage-1 baseline.
#
# Pipeline:
#   1. Train/evaluate Stage-0 `object_relation` warm-start on nuPlan 20k seed7.
#   2. Train/evaluate Stage-1 `wm_naive` (object_relation + WM) on nuPlan 20k seed7.
#
# This closes the mechanism chain against typed decoupling:
# shared relation top-k vs typed dyn/rel budget under the same downstream setup.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
ROOT="$(pwd)"
LOG_DIR="experiments/a2_shared_relation_smoke/logs"
mkdir -p "${LOG_DIR}"

NUPLAN_ROOT="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split"
NUPLAN_INDEX="${ROOT}/experiments/nuplan_20k_balanced_paths_seed7.json"
STAGE0_ROOT="experiments/nuplan_stage0_20k_seed7"
STAGE1_ROOT="experiments/nuplan_stage1_shared_relation_20k"

echo "[$(date '+%F %T')] A2 smoke start"
echo "[$(date '+%F %T')] checking nuPlan mount: ${NUPLAN_ROOT}"
ls "${NUPLAN_ROOT}" >/dev/null

if [[ ! -f "${STAGE0_ROOT}/object_relation/model.pt" ]]; then
    echo "[$(date '+%F %T')] training Stage0 object_relation warm-start"
    env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" run_stage0_table3.py \
        --config configs/debug_mvp.json \
        --dataset nuplan \
        --nuplan-root "${NUPLAN_ROOT}" \
        --nuplan-num-samples 20000 \
        --nuplan-index-json "${NUPLAN_INDEX}" \
        --nuplan-workers 32 \
        --nuplan-lazy \
        --loader-workers 32 \
        --variant object_relation \
        --epochs 10 \
        --batch-size 128 \
        --lr-scale 4 \
        --output-dir "${STAGE0_ROOT}" \
        --seed 7 \
        2>&1 | tee "${LOG_DIR}/stage0_object_relation_20k_seed7.log"
else
    echo "[$(date '+%F %T')] found Stage0 checkpoint: ${STAGE0_ROOT}/object_relation/model.pt"
fi

echo "[$(date '+%F %T')] training Stage1 wm_naive seed7 smoke"
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "${PY}" run_stage1_table4.py \
    --config configs/debug_mvp.json \
    --dataset nuplan \
    --nuplan-root "${NUPLAN_ROOT}" \
    --nuplan-num-samples 20000 \
    --nuplan-index-json "${NUPLAN_INDEX}" \
    --nuplan-workers 32 \
    --nuplan-lazy \
    --loader-workers 32 \
    --condition wm_naive \
    --stage0-root "${STAGE0_ROOT}" \
    --epochs 10 \
    --batch-size 128 \
    --lr-scale 4 \
    --horizon 5 \
    --entropy-beta 0.003 \
    --action-sample-clip 5 \
    --output-dir "${STAGE1_ROOT}" \
    --seed 7 \
    2>&1 | tee "${LOG_DIR}/stage1_wm_naive_20k_seed7.log"

echo "[$(date '+%F %T')] A2 smoke complete"
