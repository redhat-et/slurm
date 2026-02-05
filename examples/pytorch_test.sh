#!/bin/bash
#SBATCH --job-name=pytorch_test
#SBATCH --output=pytorch_test_%j.out
#SBATCH --error=pytorch_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

echo "=========================================="
echo "PyTorch GPU Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

echo "Running PyTorch in container..."
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('cuDNN version:', torch.backends.cudnn.version())
if torch.cuda.is_available():
    print('GPU device:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())
    # Simple GPU test
    x = torch.rand(5, 3).cuda()
    print('Created tensor on GPU:', x.device)
    print('Tensor shape:', x.shape)
else:
    print('CUDA not available!')
"

echo ""
echo "End time: $(date)"
echo "Job completed successfully!"
