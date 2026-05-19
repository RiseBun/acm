#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/volumes/cpfs/prediction/lipeinan/code"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/plantf/bin/python}"
SPLIT_SIZE="${1:-50}"
SEED="${2:-7}"
BASELINES="${BASELINES:-pdm}"
MAX_WORKERS="${MAX_WORKERS:-8}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/experiments/nuplan_official_closed_loop_p0}"
TOKENS_JSON="${TOKENS_JSON:-${OUT_ROOT}/scenario_subsets/nuplan_official_closed_loop_${SPLIT_SIZE}_scenarios.json}"

export PYTHONPATH="${ROOT}/src:${ROOT}/cangku/nuplan-devkit:${ROOT}/cangku/navsim:${PYTHONPATH:-}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/mnt/datasets/e2e-nuplan/20260302}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/datasets/e2e-nuplan/20260302/maps}"

if [[ ! -f "${TOKENS_JSON}" ]]; then
  "${PYTHON_BIN}" scripts/prepare_nuplan_closed_loop_subset.py \
    --subset-sizes 50 200 500 \
    --pool-size 500 \
    --output-dir "${OUT_ROOT}/scenario_subsets"
fi

JOB_NAME="p0_${SPLIT_SIZE}_seed${SEED}_door_bc_${BASELINES// /_}"

"${PYTHON_BIN}" scripts/run_nuplan_closed_loop_mvp.py \
  --seed "${SEED}" \
  --conditions wm_object wm_decoupled_no_vis bc \
  --baselines ${BASELINES} \
  --scenario-filter all_scenarios \
  --scenario-tokens-json "${TOKENS_JSON}" \
  --limit-total-scenarios "${SPLIT_SIZE}" \
  --worker single_machine_thread_pool \
  --worker-max-workers "${MAX_WORKERS}" \
  --output-dir "${OUT_ROOT}/runs" \
  --job-name "${JOB_NAME}"

RUN_DIR="${OUT_ROOT}/runs/doorrl_closed_loop_mvp/${JOB_NAME}"
"${PYTHON_BIN}" scripts/summarize_nuplan_official_closed_loop_p0.py \
  "${RUN_DIR}" \
  --name "nuplan_official_closed_loop_${SPLIT_SIZE}_seed${SEED}"
