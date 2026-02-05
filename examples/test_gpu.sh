#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

echo "=========================================="
echo "GPU Test Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

echo "=========================================="
echo "NVIDIA Driver Information"
echo "=========================================="
nvidia-smi
echo ""

echo "=========================================="
echo "GPU Devices"
echo "=========================================="
nvidia-smi -L
echo ""

echo "=========================================="
echo "CUDA Version"
echo "=========================================="
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv
echo ""

echo "=========================================="
echo "GPU Utilization"
echo "=========================================="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu --format=csv
echo ""

echo "End time: $(date)"
echo "Job completed successfully!"
