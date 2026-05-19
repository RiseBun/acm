#!/usr/bin/env bash
# Stage 1 pilot X: multi-seed verification of the v3 ranking.
#
# Runs 3 seeds x 3 conditions = 9 Stage-1 pilots IN PARALLEL on a single H20.
# Justification for parallel:
#   * token cache already built -> each run starts in ~6 s, not 18 min
#   * v3 training at bs=128 uses <2% GPU util; 9 concurrent processes still
#     fit comfortably in 97 GiB and under full 192-core host
# Hyperparameters are v3 (entropy=0.003, clip=5, bs=128, lr_scale=4).

set -euo pipefail
cd "$(dirname "$0")/.."

PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
OUT_ROOT="experiments/stage1_pilot_x"
EPOCHS=10
BATCH_SIZE=128
LR_SCALE=4
NUM_SCENES=700
HORIZON=5
ENTROPY_BETA=0.003
ACTION_CLIP=5.0
CACHE_DIR="experiments/_token_cache"

SEEDS=(7 42 123)
CONDS=(wm_object wm_decoupled wm_decoupled_no_vis)

mkdir -p "${OUT_ROOT}/logs"

echo "================================================================"
echo "X: 3 seeds x 3 conditions, v3 hparams, cache reuse, parallel"
echo "================================================================"

PIDS=()
for SEED in "${SEEDS[@]}"; do
    for COND in "${CONDS[@]}"; do
        LOG="${OUT_ROOT}/logs/x_seed${SEED}_${COND}.log"
        echo "[launch] seed=${SEED} cond=${COND} -> ${LOG}"
        nohup env PYTHONPATH=src "${PY}" run_stage1_table4.py \
            --config configs/debug_mvp.json \
            --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
            --condition "${COND}" \
            --seed "${SEED}" \
            --num-scenes "${NUM_SCENES}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --lr-scale "${LR_SCALE}" \
            --horizon "${HORIZON}" \
            --output-dir "${OUT_ROOT}" \
            --stage0-root experiments/table3_fair_fix2_seed7 \
            --entropy-beta "${ENTROPY_BETA}" \
            --action-sample-clip "${ACTION_CLIP}" \
            --token-cache-dir "${CACHE_DIR}" \
            > "${LOG}" 2>&1 &
        PIDS+=($!)
        # Light stagger to avoid an I/O thundering herd on the 951 MB cache.
        sleep 2
    done
done

echo ""
echo "Launched ${#PIDS[@]} jobs: ${PIDS[*]}"
echo "Waiting for all to finish..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=$((FAIL+1))
        echo "  pid ${pid} failed (see logs)"
    fi
done

echo ""
echo "All jobs done; failures=${FAIL}"
