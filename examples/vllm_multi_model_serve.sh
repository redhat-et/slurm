#!/bin/bash

# Script to start all three vLLM models simultaneously on the cluster
# Each model will run on a different compute node

echo "=========================================="
echo "Multi-Model vLLM Deployment"
echo "=========================================="
echo ""

JOBS_DIR="$HOME/vllm-jobs"

if [ ! -d "$JOBS_DIR" ]; then
    echo "Error: vLLM jobs directory not found at $JOBS_DIR"
    echo "Please run the vLLM deployment playbook first:"
    echo "  ansible-playbook deploy-vllm-models.yml"
    exit 1
fi

cd "$JOBS_DIR" || exit 1

echo "Submitting all model serving jobs..."
echo ""

# Submit Phi-3
echo "1. Submitting Phi-3 Mini..."
PHI3_JOB=$(sbatch --parsable serve_phi3.sh)
echo "   Job ID: $PHI3_JOB"

# Submit Mistral
echo "2. Submitting Mistral 7B..."
MISTRAL_JOB=$(sbatch --parsable serve_mistral.sh)
echo "   Job ID: $MISTRAL_JOB"

# Submit Llama 3.1
echo "3. Submitting Llama 3.1 8B..."
LLAMA_JOB=$(sbatch --parsable serve_llama3.sh)
echo "   Job ID: $LLAMA_JOB"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Job IDs:"
echo "  Phi-3:    $PHI3_JOB"
echo "  Mistral:  $MISTRAL_JOB"
echo "  Llama3:   $LLAMA_JOB"
echo ""
echo "Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "Or use:"
echo "  watch -n 1 'squeue -u \$USER'"
echo ""
echo "Once running (status = R), find nodes:"
echo "  squeue -u \$USER -o \"%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R\""
echo ""
echo "Models will be available at:"
echo "  Phi-3:    http://<node>:8000/v1/completions"
echo "  Mistral:  http://<node>:8001/v1/completions"
echo "  Llama3:   http://<node>:8002/v1/completions"
echo ""
echo "Test endpoints when ready:"
echo "  python3 test_inference.py phi3 <node-name>"
echo ""
