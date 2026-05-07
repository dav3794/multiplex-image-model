#!/usr/bin/env bash
set -euo pipefail

# Run ABMIL CV for all embedding models.
#
# Usage:
#   ./run_abmil_all_models.sh
#   CONTINUE_ON_ERROR=1 ./run_abmil_all_models.sh
#   PYTHON_BIN=python3 ./run_abmil_all_models.sh
#
# Notes:
# - `src/run_abmil_v2.py` controls the output directory (RESULTS_DIR).
# - Per-crop embeddings use z-score fit on each CV train fold only (`--zscore cv_train`).
# - Logs are saved under `results/logs/`.

PYTHON_BIN="${PYTHON_BIN:-python}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

MODELS=(
  immuvis
)

ZSCORE_MODES=(
  cv_train
)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

echo "Python: ${PYTHON_BIN}"
echo "Repo:   ${ROOT_DIR}"
echo "Models: ${MODELS[*]}"
echo "Zscore: ${ZSCORE_MODES[*]}"
echo "Logs:   ${LOG_DIR}"
echo

for model in "${MODELS[@]}"; do
  for zscore_mode in "${ZSCORE_MODES[@]}"; do
    log_file="${LOG_DIR}/run_abmil_${model}_${zscore_mode}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Running model: ${model} (zscore=${zscore_mode}) ==="
    echo "Log: ${log_file}"

    set +e
    "${PYTHON_BIN}" "${ROOT_DIR}/src/run_abmil.py" --model "${model}" --zscore "${zscore_mode}" 2>&1 | tee "${log_file}"
    exit_code="${PIPESTATUS[0]}"
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
      echo "FAILED: ${model} (zscore=${zscore_mode}, exit ${exit_code})"
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "Continuing because CONTINUE_ON_ERROR=1"
        echo
        continue
      fi
      exit "${exit_code}"
    fi

    echo "OK: ${model} (zscore=${zscore_mode})"
    echo
  done
done

echo "All models finished."
