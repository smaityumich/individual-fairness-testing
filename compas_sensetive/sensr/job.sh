#!/bin/bash
#SBATCH --job-name=test-std-node
#SBATCH --output=logs/compas_sensr_%A_%a.out
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=02:00:00
#SBATCH --account=yuekai1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=smaity@umich.edu
#SBATCH --partition=standard
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

python3 run_sensr_compas.py $SLURM_ARRAY_TASK_ID
