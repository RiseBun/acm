#!/usr/bin/env bash
# Compact status report for P1/P2 follow-up jobs.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ROOT="experiments/p1_full_gpu"
LOG_DIR="${RUN_ROOT}/logs"
PID_DIR="${RUN_ROOT}/pids"
DONE_DIR="${RUN_ROOT}/done"

echo "== mounts =="
if ls /mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split >/dev/null 2>&1; then
    echo "nuPlan:   ok"
else
    echo "nuPlan:   unavailable"
fi
if ls /mnt/datasets/e2e-nuscenes/20260302 >/dev/null 2>&1; then
    echo "nuScenes: ok"
else
    echo "nuScenes: unavailable"
fi

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true

echo
echo "== watcher jobs =="
for job in A B C; do
    pid_file="${PID_DIR}/${job}.pid"
    done_file="${DONE_DIR}/${job}.done"
    log_file="${LOG_DIR}/${job}_watch.log"
    status="not launched"
    if [[ -f "${done_file}" ]]; then
        status="done"
    elif [[ -f "${pid_file}" ]] && ps -p "$(cat "${pid_file}")" >/dev/null 2>&1; then
        status="running pid=$(cat "${pid_file}")"
    elif [[ -f "${pid_file}" ]]; then
        status="stopped pid=$(cat "${pid_file}")"
    fi
    echo "${job}: ${status}"
    if [[ -f "${log_file}" ]]; then
        echo "  last: $(tail -n 1 "${log_file}")"
    fi
done

echo
echo "== horizon sensitivity =="
for k in 3 5 7; do
    summary="experiments/horizon_sensitivity_nuscenes/k${k}/fig3_imagination_nuscenes_summary.json"
    if [[ -f "${summary}" ]]; then
        echo "K=${k}: complete (${summary})"
    else
        echo "K=${k}: missing"
    fi
done
