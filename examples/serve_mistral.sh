#!/bin/bash
#SBATCH --job-name=vllm-serve-mistral
#SBATCH --output=/home/%u/outputs/serve_mistral_%j.out
#SBATCH --error=/home/%u/outputs/serve_mistral_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=slinky

set -euo pipefail

MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_CACHE="/opt/models"
PORT=8001
GPU_MEM_UTILIZATION=0.9
MAX_MODEL_LEN=8192

echo "=========================================="
echo "vLLM Model Serving: Mistral 7B"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Model:     $MODEL_ID"
echo "Port:      $PORT"
echo "Start:     $(date)"
echo ""

# ── Environment ──────────────────────────────────────────────────────
export HF_HOME="$MODEL_CACHE"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── GPU info ─────────────────────────────────────────────────────────
echo "GPU Information:"
nvidia-smi -L
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# ── Install dependencies (idempotent) ───────────────────────────────
export LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/nvidia/cufile/lib
echo "Checking dependencies..."
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Please install required dependencies - run as root:"
    echo '1. `dnf install -y python3 python3-pip python3-devel`'
    echo '2. `pip3 install vllm huggingface_hub accelerate transformers`'
    exit 1
else
    echo "vLLM $(python3 -c 'import vllm; print(vllm.__version__)') is installed"
fi
echo ""

# ── Prepare directories ─────────────────────────────────────────────
mkdir -p "$MODEL_CACHE"
mkdir -p /var/log/vllm

# ── Download model (idempotent) ──────────────────────────────────────
echo "Ensuring model is cached..."
python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/opt/models")
model_id = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
cache_dir = os.environ.get("MODEL_CACHE", "/opt/models")

print(f"Model:     {model_id}")
print(f"Cache dir: {cache_dir}")

try:
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
        max_workers=4,
    )
    print(f"Model ready at: {path}")
except Exception as e:
    print(f"Error downloading model: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
echo ""

# ── Verify model cache ──────────────────────────────────────────────
echo "Model cache:"
ls -lh "$MODEL_CACHE/models--${MODEL_ID//\//-}/" 2>/dev/null \
    || ls -lh "$MODEL_CACHE/models--${MODEL_ID//\/\//-}/" 2>/dev/null \
    || echo "WARNING: Could not list model cache directory"
echo ""

# ── Start vLLM server ───────────────────────────────────────────────
echo "Starting vLLM server..."
echo "API will be available at: http://$SLURMD_NODENAME:$PORT"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --disable-log-requests

echo ""
echo "vLLM server stopped at $(date)"
