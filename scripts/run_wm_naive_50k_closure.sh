#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/volumes/cpfs/prediction/lipeinan/code"
PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
INDEX_JSON="${ROOT}/experiments/nuplan_50k_balanced_paths_seed7.json"
LOG_DIR="${ROOT}/experiments/closure_wm_naive_50k_logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

echo "[closure] root=${ROOT}"
echo "[closure] python=${PY}"
echo "[closure] index=${INDEX_JSON}"

if [[ ! -f "experiments/nuplan_stage0_50k_seed7/object_relation/model.pt" ]]; then
  echo "[closure] Stage0 object_relation 50k warm-start is missing; training it first."
  "${PY}" run_stage0_table3.py \
    --dataset nuplan \
    --nuplan-num-samples 50000 \
    --nuplan-index-json "${INDEX_JSON}" \
    --nuplan-lazy \
    --loader-workers 64 \
    --variant object_relation \
    --epochs 10 \
    --batch-size 128 \
    --output-dir experiments/nuplan_stage0_50k_seed7 \
    --seed 7 \
    2>&1 | tee "${LOG_DIR}/stage0_object_relation_50k_seed7.log"
else
  echo "[closure] Stage0 object_relation 50k warm-start already exists; skipping Stage0."
fi

for seed in 7 42 123; do
  echo "[closure] Stage1 wm_naive 50k seed=${seed}"
  "${PY}" run_stage1_table4.py \
    --dataset nuplan \
    --nuplan-num-samples 50000 \
    --nuplan-index-json "${INDEX_JSON}" \
    --nuplan-lazy \
    --loader-workers 32 \
    --condition wm_naive \
    --seed "${seed}" \
    --epochs 10 \
    --batch-size 128 \
    --horizon 5 \
    --stage0-root experiments/nuplan_stage0_50k_seed7 \
    --output-dir experiments/nuplan_stage1_shared_relation_50k \
    --entropy-beta 0.003 \
    --action-sample-clip 5.0 \
    2>&1 | tee "${LOG_DIR}/stage1_wm_naive_50k_seed${seed}.log"
done

echo "[closure] Selection diagnostic for wm_naive 50k"
"${PY}" scripts/selection_diagnostic.py \
  --seeds 7 42 123 \
  --conditions wm_naive \
  --checkpoint-root experiments/nuplan_stage1_shared_relation_50k \
  --nuplan-num-samples 50000 \
  --nuplan-index-json "${INDEX_JSON}" \
  --output-dir experiments/selection_diagnostic_shared_relation_50k \
  --batch-size 128 \
  --loader-workers 8 \
  2>&1 | tee "${LOG_DIR}/selection_diagnostic_wm_naive_50k.log"

echo "[closure] done"
echo "[closure] Stage1 summary artifacts under experiments/nuplan_stage1_shared_relation_50k"
echo "[closure] Diagnostic artifacts under experiments/selection_diagnostic_shared_relation_50k"
