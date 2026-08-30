#!/bin/bash
#SBATCH --job-name=llumnix_priority_cmp
#SBATCH --partition=savio4_htc
#SBATCH --account=fc_cosi
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/llumnix_priority_cmp_%j.out
#SBATCH --error=logs/llumnix_priority_cmp_%j.err

set -euo pipefail

mkdir -p logs

module load python/3.10

source .venv/bin/activate

echo "HOST=$(hostname) JOBID=$SLURM_JOB_ID" >&2

# Pass through any CLI args to the compare script so runs can be customized.
python3 scripts/compare_llumnix_priority.py "$@"
