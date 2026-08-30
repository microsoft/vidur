#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_SCRIPTS=(
  "$SCRIPT_DIR/llumnix_slurm_job.sh"
  "$SCRIPT_DIR/compare_schedulers_slurm_job.sh"
  "$SCRIPT_DIR/compare_llumnix_priority_slurm_job.sh"
)

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found in PATH. Run this on a SLURM node/login host." >&2
  exit 1
fi

for job_script in "${JOB_SCRIPTS[@]}"; do
  if [[ ! -f "$job_script" ]]; then
    echo "Warning: job script not found: $job_script" >&2
    continue
  fi
  if [[ ! -x "$job_script" ]]; then
    echo "Warning: job script not executable, attempting to submit anyway: $job_script" >&2
  fi
  echo "Submitting $job_script"
  sbatch "$job_script"
done
