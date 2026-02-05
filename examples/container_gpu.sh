#!/bin/bash
#SBATCH --job-name=container_gpu
#SBATCH --output=container_gpu_%j.out
#SBATCH --error=container_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

echo "=========================================="
echo "GPU Container Test Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

echo "=========================================="
echo "Testing NVIDIA Container Toolkit"
echo "=========================================="

echo "Running nvidia-smi in CUDA container..."
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  nvidia/cuda:13.0.0-base-ubi9 \
  nvidia-smi

echo ""
echo "=========================================="
echo "Listing GPU devices in container"
echo "=========================================="
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  nvidia/cuda:13.0.0-base-ubi9 \
  nvidia-smi -L

echo ""
echo "End time: $(date)"
echo "Job completed successfully!"
