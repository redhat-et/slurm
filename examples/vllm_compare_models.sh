#!/bin/bash
#SBATCH --job-name=compare-models
#SBATCH --output=outputs/compare_models_%j.out
#SBATCH --error=outputs/compare_models_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

echo "=========================================="
echo "LLM Model Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

export HF_HOME=/opt/models
export CUDA_VISIBLE_DEVICES=0

# Test prompt
TEST_PROMPT="Explain the concept of cloud computing in exactly 3 sentences."

echo "Test Prompt: $TEST_PROMPT"
echo ""
echo "=========================================="
echo ""

# Function to test a model
test_model() {
    local model_name=$1
    local model_id=$2
    local max_len=$3

    echo "Testing: $model_name"
    echo "Model ID: $model_id"
    echo "---"

    python3 <<PYTHON_EOF
from vllm import LLM, SamplingParams
import time
import os

os.environ['HF_HOME'] = '/opt/models'

# Load model
print(f"Loading {model_id}...")
start = time.time()
llm = LLM(
    model="$model_id",
    dtype="auto",
    max_model_len=$max_len,
    gpu_memory_utilization=0.85,
    trust_remote_code=True
)
load_time = time.time() - start
print(f"Load time: {load_time:.2f}s\n")

# Generate
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=150
)

prompt = "$TEST_PROMPT"
start = time.time()
outputs = llm.generate([prompt], sampling_params)
gen_time = time.time() - start

response = outputs[0].outputs[0].text
tokens = len(outputs[0].outputs[0].token_ids)
tokens_per_sec = tokens / gen_time

print(f"Response: {response}\n")
print(f"Generation time: {gen_time:.2f}s")
print(f"Tokens generated: {tokens}")
print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")

# Cleanup to free memory
del llm
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
PYTHON_EOF

    echo ""
    echo "=========================================="
    echo ""
}

# Test all three models sequentially
echo "Running comparison across all models..."
echo ""

test_model "Phi-3 Mini (3.8B)" "microsoft/Phi-3-mini-4k-instruct" 4096

test_model "Mistral 7B Instruct" "mistralai/Mistral-7B-Instruct-v0.3" 8192

test_model "Llama 3.1 8B Instruct" "meta-llama/Meta-Llama-3.1-8B-Instruct" 8192

echo "=========================================="
echo "Comparison Complete"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Summary: All three models tested with identical prompt"
echo "Review the output above to compare:"
echo "  - Response quality"
echo "  - Load time"
echo "  - Generation speed"
echo "  - Tokens per second"
