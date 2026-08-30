#!/bin/bash
#SBATCH --job-name=compare_schedulers
#SBATCH --partition=savio4_htc
#SBATCH --account=fc_cosi
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/compare_schedulers_%j.out
#SBATCH --error=logs/compare_schedulers_%j.err

mkdir -p logs

module load python/3.10

source .venv/bin/activate

echo "HOST=$(hostname) JOBID=$SLURM_JOB_ID" >&2

python3 scripts/compare_schedulers.py
