#!/bin/bash
#SBATCH --job-name=multi_job
#SBATCH --output=multi_job_%j_%a.out
#SBATCH --error=multi_job_%j_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --array=1-5

echo "=========================================="
echo "Array Job Example - Task $SLURM_ARRAY_TASK_ID"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

echo "Running task $SLURM_ARRAY_TASK_ID..."
nvidia-smi -L

echo "Simulating work..."
sleep 10

echo ""
echo "End time: $(date)"
echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
