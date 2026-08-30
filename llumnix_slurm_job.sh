#!/bin/bash
#SBATCH --job-name=llumnix_slurm_job
#SBATCH --partition=savio4_htc
#SBATCH --account=fc_cosi
#SBATCH --time=02:00:00
#SBATCH --array=0-119
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err

mkdir -p logs

module load python/3.10

source .venv/bin/activate

echo "HOST=$(hostname) JOBID=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID" >&2

# Run Llumnix plots and Llumnix vs LOR comparisons in parallel for this task index.
# Plots stay Llumnix-only; compare runs paired Llumnix/LOR metrics (no plots).
python3 run_tests.py --mode plots --index "$SLURM_ARRAY_TASK_ID" &
PLOTS_PID=$!

python3 run_tests.py --mode compare --index "$SLURM_ARRAY_TASK_ID" &
COMPARE_PID=$!

wait $PLOTS_PID $COMPARE_PID
