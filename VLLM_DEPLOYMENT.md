# vLLM Model Deployment Guide

This guide covers deploying and using Large Language Models (LLMs) with vLLM on your Slurm GPU cluster.

## Overview

The optional vLLM playbook deploys three production-ready language models optimized for the NVIDIA L4 GPUs in your cluster:

1. **Phi-3 Mini (3.8B)** - Fast, efficient small model for quick responses
2. **Mistral 7B Instruct** - Excellent general-purpose instruction-following model
3. **Llama 3.1 8B Instruct** - State-of-the-art model with advanced reasoning

## Prerequisites

**Required Infrastructure:**
- Slurm cluster already deployed (`playbook.yml` and `continue-slurm-install.yml` completed)
- NVIDIA GPU instances (tested on g6.2xlarge with L4 GPUs)
- NVIDIA drivers and CUDA toolkit installed
- Internet connectivity on compute nodes (for downloading models)
- ~50GB free disk space for model weights

**Required Files:**
- `deploy-vllm-models.yml` - vLLM deployment playbook
- `group_vars/vllm.yml` - Model configuration
- `inventory/hosts` - Ansible inventory file

**Optional:**
- HuggingFace account (for gated models like Llama)

## Quick Start

### 1. Deploy vLLM and Models

```bash
# Run the optional vLLM deployment playbook
ansible-playbook deploy-vllm-models.yml
```

This playbook performs three phases:

**Phase 1: System-wide vLLM Installation (all nodes)**
- Installs Python 3.12 development packages
- Creates directories: `/opt/vllm`, `/opt/models`, `/var/log/vllm`
- Installs vLLM 0.15.1 and dependencies (`huggingface_hub`, `accelerate`, `transformers`)
- Configures HuggingFace cache location

**Phase 2: Model Download (compute nodes only)**
- Downloads model weights to `/opt/models/` using `snapshot_download()`
- Models are cached in HuggingFace format:
  ```
  /opt/models/
  ├── models--microsoft--Phi-3-mini-4k-instruct/
  └── models--mistralai--Mistral-7B-Instruct-v0.3/
  ```
- Downloads in parallel with 4 workers, resumes interrupted downloads
- Sets ownership to `slurm` user (~15-20 minutes total)

**Phase 3: Job Script Creation (controller only)**
- Creates `/home/slurm/vllm-jobs/` directory
- Generates Slurm batch scripts for serving and batch inference
- Creates `vllm-manager.sh` management CLI
- Creates documentation files

### 2. Access the Cluster

```bash
ssh -i ~/.ssh/slurm-cluster-key.pem ec2-user@<controller-ip>
cd ~/vllm-jobs
```

### 3. Start Serving a Model

```bash
# Using the management script
./vllm-manager.sh serve phi3

# Or manually
sbatch serve_phi3.sh
```

### 4. Test the Model

```bash
# Wait for job to start (model loading takes ~2-4 minutes)
squeue -u $USER

# Find the node where it's running
NODE=$(squeue -u $USER -h -o "%N" | head -1)

# Test the endpoint
python3 test_inference.py phi3 $NODE
```

**What happens during job startup:**

1. **Slurm schedules the job** - Allocates 1 GPU, 8 CPU cores, assigns to available compute node

2. **Environment setup** - Sets `HF_HOME=/opt/models`, configures CUDA devices

3. **Model loading phases:**
   - Configuration loading (~5 seconds) - Reads model config, determines architecture
   - Weight loading (~1-2 minutes) - Loads safetensors from cache, allocates GPU memory
   - Compilation (~1-2 minutes) - torch.compile optimization, CUDA graph generation
   - **Total startup time: 2-4 minutes** depending on model size

4. **Server ready** - Binds to 0.0.0.0:port, accepts API requests

**Expected resource usage (Mistral 7B):**
- GPU Memory: ~21 GB / 23 GB (91% utilization)
- System Memory: ~950 MB
- Startup Time: ~4 minutes

### Deployment Verification

After running the deployment playbook, verify the installation:

```bash
# Check vLLM installation on all nodes
ansible all -i inventory/hosts -m shell -a "python3 -c 'import vllm; print(vllm.__version__)'" -b

# Verify model cache on compute nodes
ansible slurm_compute -i inventory/hosts -m shell \
  -a "ls -lh /opt/models/models--*/snapshots" -b

# Verify job scripts created on controller
ssh ec2-user@<controller-ip>
ls -lh /home/slurm/vllm-jobs/
```

**Expected output:**
- vLLM version: 0.15.1
- Model directories exist in `/opt/models/`
- Job scripts present in `/home/slurm/vllm-jobs/`

## Deployed Models

### Phi-3 Mini (3.8B)

**Best for**: Fast responses, low latency, chat applications

```bash
./vllm-manager.sh serve phi3
```

- **Model ID**: microsoft/Phi-3-mini-4k-instruct
- **Parameters**: 3.8 billion
- **Context**: 4,096 tokens
- **Memory**: ~8GB VRAM
- **Speed**: Fastest of the three models
- **Port**: 8000

**Use cases**:
- Quick Q&A
- Chatbots
- Low-latency applications
- Code completion

### Mistral 7B Instruct

**Best for**: General-purpose instruction following, reasoning

```bash
./vllm-manager.sh serve mistral
```

- **Model ID**: mistralai/Mistral-7B-Instruct-v0.3
- **Parameters**: 7 billion
- **Context**: 32,768 tokens (using 8,192 for memory efficiency)
- **Memory**: ~14GB VRAM
- **Speed**: Fast
- **Port**: 8001

**Use cases**:
- General chat and assistance
- Instruction following
- Reasoning tasks
- Content generation

### Llama 3.1 8B Instruct

**Best for**: Advanced reasoning, long context, multilingual

```bash
./vllm-manager.sh serve llama3
```

- **Model ID**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Parameters**: 8 billion
- **Context**: 131,072 tokens (using 8,192 for memory efficiency)
- **Memory**: ~16GB VRAM
- **Speed**: Moderate
- **Port**: 8002

**Use cases**:
- Complex reasoning
- Multilingual tasks
- Long-form content
- Advanced instruction following

## Usage Modes

### Mode 1: API Serving (Long-Running)

Start a model as an OpenAI-compatible API server:

```bash
# Start the server
./vllm-manager.sh serve mistral

# Check status
./vllm-manager.sh list

# The job will run for up to 12 hours or until cancelled
```

Once running, you can query it:

```bash
# Find the node
NODE=$(squeue -u $USER -h -o "%N" | head -1)

# Query via curl
curl http://$NODE:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "Explain machine learning in simple terms:",
    "max_tokens": 200,
    "temperature": 0.7
  }'

# Chat completion
curl http://$NODE:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

### Mode 2: Batch Inference (One-Time)

Run batch inference on a set of prompts:

```bash
# Submit batch job
./vllm-manager.sh batch phi3

# Results will be saved to outputs/
```

The batch scripts include example prompts and save results to files.

### Mode 3: Custom Jobs

Create your own Slurm job script:

```bash
#!/bin/bash
#SBATCH --job-name=my-llm-job
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

export HF_HOME=/opt/models

python3 <<EOF
from vllm import LLM, SamplingParams

llm = LLM(model="microsoft/Phi-3-mini-4k-instruct")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

prompts = ["Your custom prompts here"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
EOF
```

## Management Commands

### vllm-manager.sh CLI

```bash
# Serve models
./vllm-manager.sh serve <model_name>

# Run batch inference
./vllm-manager.sh batch <model_name>

# List running jobs
./vllm-manager.sh list

# Cancel a job
./vllm-manager.sh cancel <job_id>
```

### Slurm Commands

```bash
# View queue
squeue -u $USER

# View job details
scontrol show job <job_id>

# View job output
tail -f ~/vllm-jobs/outputs/serve_phi3_<jobid>.out

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Running Multiple Models Simultaneously

You can run all three models at once (one per compute node):

```bash
# Submit all models
./vllm-manager.sh serve phi3
./vllm-manager.sh serve mistral
./vllm-manager.sh serve llama3

# Check they're running
./vllm-manager.sh list

# Each will be on a different node with its own GPU
```

Access them on their respective nodes:

```bash
# Find nodes
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R"

# Query different models
curl http://slurm-node-1:8000/v1/completions -d '...'  # Phi-3
curl http://slurm-node-2:8001/v1/completions -d '...'  # Mistral
```

## API Examples

### Testing Endpoints

**From controller using curl:**
```bash
# Find the compute node running your job
NODE=$(squeue -u $USER -h -o "%N" | head -1)

# List available models
curl -s http://$NODE:8000/v1/models | jq

# Simple completion
curl -s http://$NODE:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "Explain GPU computing in one sentence:",
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq
```

**Using Ansible from remote:**
```bash
# Test model endpoint
ansible slurm_controller -i inventory/hosts -m shell \
  -a "curl -s http://slurm-node-1:8001/v1/models" -b

# Test completion
ansible slurm_controller -i inventory/hosts -m shell \
  -a 'curl -s http://slurm-node-1:8001/v1/completions -H "Content-Type: application/json" -d "{\"model\": \"mistralai/Mistral-7B-Instruct-v0.3\", \"prompt\": \"Hello\", \"max_tokens\": 50}"' -b
```

### Python with OpenAI Library

```python
from openai import OpenAI

# Point to your vLLM server
client = OpenAI(
    base_url="http://slurm-node-1:8000/v1",
    api_key="not-needed"  # vLLM doesn't require API keys by default
)

# Chat completion
response = client.chat.completions.create(
    model="microsoft/Phi-3-mini-4k-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about GPUs."}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Python with Requests

```python
import requests
import json

url = "http://slurm-node-1:8000/v1/chat/completions"

payload = {
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "messages": [
        {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 100
}

response = requests.post(url, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

### Curl

```bash
curl http://slurm-node-1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "messages": [
      {"role": "user", "content": "Tell me a joke."}
    ]
  }' | jq .
```

## Performance Tuning

### GPU Memory Management

Each model has default GPU memory utilization settings in `group_vars/vllm.yml`:

```yaml
- name: phi3
  gpu_memory_utilization: 0.85  # Use 85% of GPU memory

- name: mistral
  gpu_memory_utilization: 0.90  # Use 90% of GPU memory
```

Adjust these if you encounter OOM errors or want to optimize:

```bash
# In the job script, modify:
--gpu-memory-utilization 0.95  # Use more memory (better performance)
--gpu-memory-utilization 0.70  # Use less memory (safer)
```

### Context Length

Default context lengths are conservative for memory efficiency:

```bash
# Increase context length (requires more memory)
--max-model-len 16384

# Decrease for lower memory usage
--max-model-len 4096
```

### Batch Size and Throughput

For batch inference, larger batches = better throughput:

```python
# In your Python script
llm = LLM(
    model="...",
    max_num_seqs=256,  # More concurrent sequences
    max_num_batched_tokens=8192  # Larger batches
)
```

### Quantization

For larger models, use quantization:

```bash
# AWQ quantization (4-bit)
--quantization awq

# GPTQ quantization (4-bit)
--quantization gptq

# Or use pre-quantized models from HuggingFace
```

## Performance Benchmarks

### Model Loading Times

**Phi-3 Mini (3.8B):**
- Configuration: ~5 seconds
- Weight loading: ~60 seconds
- torch.compile: ~1 minute
- **Total startup: ~2 minutes**

**Mistral 7B:**
- Configuration: ~5 seconds
- Weight loading: ~113 seconds
- torch.compile: ~2 minutes
- **Total startup: ~4 minutes**

**Llama 3.1 8B:**
- Configuration: ~5 seconds
- Weight loading: ~120 seconds
- torch.compile: ~2 minutes
- **Total startup: ~4-5 minutes**

### Resource Usage (on L4 GPU)

| Model | GPU Memory | System Memory | Tokens/sec |
|-------|-----------|---------------|------------|
| Phi-3 Mini | ~8 GB | ~600 MB | 80-100 |
| Mistral 7B | ~21 GB | ~950 MB | 40-60 |
| Llama 3.1 8B | ~22 GB | ~1 GB | 35-55 |

### Inference Latency

- **First token latency:** 100-200ms
- **Subsequent tokens:** 20-30ms each
- **Throughput:** Varies by sequence length and batch size

## Monitoring

### GPU Utilization

**Direct SSH to compute node:**
```bash
# SSH to compute node
ssh slurm-node-1

# Watch GPU usage
watch -n 1 nvidia-smi

# Check vLLM process
ps aux | grep vllm
```

**Using Ansible from controller (for remote monitoring):**
```bash
# Check GPU status across all compute nodes
ansible slurm_compute -i inventory/hosts -m shell \
  -a "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv" \
  -b

# Check vLLM processes
ansible slurm_compute -i inventory/hosts -m shell \
  -a "ps aux | grep vllm | grep -v grep" -b

# Check specific node
ansible slurm_compute -i inventory/hosts -m shell \
  -a "nvidia-smi" -b --limit "slurm-node-1"
```

**Interpreting GPU metrics:**
- `memory.used`: Model size + KV cache (e.g., 21 GB = ~13.5 GB model + ~7 GB cache)
- `utilization.gpu`: 0% when idle, 80-100% during inference
- `temperature`: Should stay below 80°C

### Job Logs

**On controller node:**
```bash
# View real-time logs
tail -f ~/vllm-jobs/outputs/serve_phi3_*.out

# Search logs
grep -i "error" ~/vllm-jobs/outputs/*.err

# Check model loading
grep -i "loading" ~/vllm-jobs/outputs/serve_*.out
```

**Using Ansible from remote (logs are on compute nodes):**
```bash
# Check output logs on specific compute node
ansible slurm_compute -i inventory/hosts -m shell \
  -a "tail -100 /home/slurm/vllm-jobs/outputs/serve_mistral_*.out" \
  -b --limit "slurm-node-1"

# Check for errors
ansible slurm_compute -i inventory/hosts -m shell \
  -a "cat /home/slurm/vllm-jobs/outputs/serve_mistral_*.err" -b
```

**Key log indicators for successful startup:**
```
INFO: vLLM version 0.15.1
INFO: Resolved architecture: MistralForCausalLM
INFO: Loading weights took 113.27 seconds
INFO: Model loading took 13.51 GiB memory
INFO: Uvicorn running on http://0.0.0.0:8001
```

### API Metrics

vLLM exposes Prometheus metrics at `/metrics`:

```bash
curl http://slurm-node-1:8000/metrics
```

## Troubleshooting

### Model Not Loading

**Symptom**: Job fails with "model not found" error

**Solution**:
```bash
# Check if model was downloaded
ssh slurm-node-1
ls -lh /opt/models/models--*/

# Re-download if needed
cd /tmp
python3 <<EOF
from huggingface_hub import snapshot_download
snapshot_download("microsoft/Phi-3-mini-4k-instruct", cache_dir="/opt/models")
EOF
```

### Out of Memory

**Symptom**: "CUDA out of memory" error

**Solutions**:
1. Reduce GPU memory utilization:
   ```bash
   --gpu-memory-utilization 0.70
   ```

2. Reduce context length:
   ```bash
   --max-model-len 4096
   ```

3. Use a smaller model (Phi-3 instead of Llama-3)

4. Enable quantization:
   ```bash
   --quantization awq
   ```

### Job Completes Immediately Without Loading

**Symptom**: Job shows COMPLETED status in seconds, no API responding

**Cause**: Gated repository requires HuggingFace authentication

**Solution**:
- Use ungated models (Phi-3, Mistral work without authentication)
- For gated models (Llama), set HuggingFace token (see Advanced Usage)

### Job Stuck in Queue

**Symptom**: Job shows PD (pending) state

**Check**:
```bash
# See why it's pending
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R"

# Check node availability
sinfo -Nel

# Check detailed job info
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Start,End,NodeList

# Resume down nodes if needed
sudo scontrol update nodename=slurm-node-1 state=resume
```

### API Not Responding

**Symptom**: Cannot connect to model endpoint

**Solutions**:
1. Check job is running:
   ```bash
   squeue -u $USER
   ```

2. Check if server started (takes 1-2 minutes):
   ```bash
   tail ~/vllm-jobs/outputs/serve_phi3_*.out
   # Look for "Uvicorn running on"
   ```

3. Test locally on the compute node:
   ```bash
   ssh slurm-node-1
   curl http://localhost:8000/v1/models
   ```

4. Check firewall/security groups allow port access

### Slow Inference

**Causes & Solutions**:

1. **Small batch size**: Increase concurrent requests
2. **Context too long**: Reduce max_model_len
3. **Temperature too high**: Lower temperature for faster sampling
4. **GPU not utilized**: Check nvidia-smi, ensure jobs are using GPU

## Advanced Usage

### Using Gated Models

Some models (like Llama) require HuggingFace authentication:

```bash
# On all compute nodes
export HF_TOKEN="your_huggingface_token"

# Or set in job script
#SBATCH --export=HF_TOKEN=your_token

# Login (one time)
huggingface-cli login
```

### Multi-GPU Inference

For larger models, use tensor parallelism:

```bash
# Request multiple GPUs (if using larger instance types)
#SBATCH --gres=gpu:2

# Enable tensor parallelism
--tensor-parallel-size 2
```

### Custom Model from Local Path

```bash
# Download model to shared location
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('model/name', local_dir='/opt/models/my-model')
"

# Use local path in job
--model /opt/models/my-model
```

### Streaming Responses

```python
import requests
import json

url = "http://slurm-node-1:8000/v1/chat/completions"

payload = {
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "messages": [{"role": "user", "content": "Write a story."}],
    "stream": True
}

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            print(json.loads(line.decode().replace("data: ", "")))
```

## Cost Optimization

### Auto-Shutdown

Models consume GPU resources even when idle. Cancel when not needed:

```bash
# Set shorter time limits
#SBATCH --time=02:00:00

# Or cancel manually when done
scancel <job_id>
```

### On-Demand vs. Always-On

**Option 1: On-Demand** (Recommended for development)
- Start models when needed
- Cancel after use
- Lower cost

**Option 2: Always-On** (For production)
- Keep models running
- Faster response
- Higher cost but better availability

### Resource Sharing

Run multiple models on the cluster, not all simultaneously:

```bash
# Submit jobs with dependencies
JOB1=$(sbatch --parsable serve_phi3.sh)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 serve_mistral.sh)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 serve_llama3.sh)
```

## Integration Examples

### REST API Service

Build a simple service that routes to your models:

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

MODELS = {
    'phi3': 'http://slurm-node-1:8000',
    'mistral': 'http://slurm-node-2:8001',
    'llama3': 'http://slurm-node-1:8002'  # If running multiple on same node
}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    model = data.get('model', 'phi3')
    prompt = data.get('prompt')

    response = requests.post(
        f"{MODELS[model]}/v1/completions",
        json={'prompt': prompt, 'max_tokens': 200}
    )

    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### LangChain Integration

```python
from langchain.llms import VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_base="http://slurm-node-1:8000/v1",
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_tokens=256,
    temperature=0.7
)

response = llm("What is the meaning of life?")
print(response)
```

## Model Management

### Adding New Models

Edit `group_vars/vllm.yml`:

```yaml
vllm_models:
  - name: custom
    model_id: organization/model-name
    description: "Description"
    context_length: 4096
    use_case: "Use case description"
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    port: 8003
```

Re-run the playbook:

```bash
ansible-playbook deploy-vllm-models.yml
```

### Updating Models

```bash
# Remove cached model
sudo rm -rf /opt/models/models--organization--model-name

# Re-download
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('organization/model-name', cache_dir='/opt/models')
"
```

### Disk Space Management

```bash
# Check model cache size
du -sh /opt/models/

# Remove unused models
sudo rm -rf /opt/models/models--unused--model

# Clean HuggingFace cache
rm -rf /opt/models/downloads/*
```

## Best Practices

1. **Start Small**: Test with Phi-3 before moving to larger models
2. **Monitor Memory**: Use nvidia-smi to track GPU utilization
3. **Set Time Limits**: Don't run serving jobs indefinitely in development
4. **Test Locally**: Use batch mode first to verify model works
5. **Log Everything**: Keep job outputs for debugging
6. **Version Control**: Document which model versions you're using
7. **Security**: In production, add authentication to APIs
8. **Backup**: Keep important inference results

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## Support

For issues:
1. Check ~/vllm-jobs/README.md on the cluster
2. Review job logs in ~/vllm-jobs/outputs/
3. Check main project documentation
4. Open an issue on GitHub
