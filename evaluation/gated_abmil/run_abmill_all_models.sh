#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

MODELS=(
  immuvis_475
  virtues_new
  vitm200
)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

echo "Python: ${PYTHON_BIN}"
echo "Repo:   ${ROOT_DIR}"
echo "Models: ${MODELS[*]}"
echo "Logs:   ${LOG_DIR}"
echo

for model in "${MODELS[@]}"; do
  log_file="${LOG_DIR}/run_abmil_${model}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== Running model: ${model} ==="
  echo "Log: ${log_file}"

  set +e
  "${PYTHON_BIN}" "${ROOT_DIR}/src/run_abmil.py" --model "${model}" 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "FAILED: ${model} (exit ${exit_code})"
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "Continuing because CONTINUE_ON_ERROR=1"
      echo
      continue
    fi
    exit "${exit_code}"
  fi

  echo "OK: ${model}"
  echo
done

echo "All models finished."
